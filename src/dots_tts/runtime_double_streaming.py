from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from dots_tts.data.pipelines.tts_pipeline import TTS_INTERLEAVE_PREFIX
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.logging import categorized_log as logc
from dots_tts.utils.util import get_dtype

DOUBLE_STREAMING_TEMPLATE_NAMES = frozenset({"tts_interleave"})


class DoubleStreamingSession:
    """Incremental interleave session for text-token to audio-chunk generation."""

    def __init__(
        self,
        runtime: DotsTtsRuntime,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str = "tts_interleave",
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int | None = None,
    ) -> None:
        normalized_prompt_text = runtime._process_prompt_text(prompt_text)
        if template_name not in DOUBLE_STREAMING_TEMPLATE_NAMES:
            raise ValueError(
                f"Unknown double streaming template_name={template_name!r}. "
                f"Expected one of {sorted(DOUBLE_STREAMING_TEMPLATE_NAMES)}."
            )
        if normalized_prompt_text:
            raise ValueError(
                "tts_interleave double streaming does not support prompt_text."
            )

        ode_method, num_steps, guidance_scale = runtime.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        self.runtime = runtime
        self.model = runtime.model
        self.device = runtime.device
        self.template_name = template_name
        self.ode_method = ode_method
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.speaker_scale = float(speaker_scale)
        self.eos_threshold = float(eos_threshold)
        self.max_generate_length = runtime.max_generate_length
        if initial_silence_audio_tokens is None:
            initial_silence_audio_tokens = 1
        self._initial_silence_audio_tokens = max(
            0,
            min(10, int(initial_silence_audio_tokens or 0)),
        )

        self._dtype = get_dtype(runtime.precision)
        self._use_amp = self.device.type == "cuda" and self._dtype in {
            torch.float16,
            torch.bfloat16,
        }
        self._prefix_token_ids = tuple(
            self.model.tokenizer.encode(
                TTS_INTERLEAVE_PREFIX,
                add_special_tokens=False,
            )
        )
        self._pending_token_ids = list(self._prefix_token_ids)
        self._state = self.model._allocate_generate_state(
            max_audio_patch_count=self.max_generate_length,
            device=self.device,
            dtype=self._dtype,
        )
        self._vocoder_inference = self.model._get_vocoder_inference()
        self._vocoder_state = self._vocoder_inference.init_stream_state(
            batch_size=1,
            chunk_size=self.model.core.latent_patch_size,
        )
        self._g_cond = None
        self._started = False
        self._text_finished = False
        self._closed = False
        self._decoded_patch_count = 0

        if prompt_audio_path is not None:
            self._prepare_ref_audio_only_conditioning(prompt_audio_path)

        logger.debug(
            logc(
                "stream",
                "Double streaming session started: template_name={} prefix_token_count={} precision={} "
                "ode_method={} num_steps={} guidance_scale={} speaker_scale={} max_audio_patch_count={} "
                "initial_silence_audio_tokens={} has_ref_audio_only={}",
            ),
            self.template_name,
            len(self._prefix_token_ids),
            runtime.precision,
            self.ode_method,
            self.num_steps,
            self.guidance_scale,
            self.speaker_scale,
            self.max_generate_length,
            self._initial_silence_audio_tokens,
            self._g_cond is not None,
        )

    @property
    def is_finished(self) -> bool:
        return self._closed

    def push_text_token(self, text_token: int) -> torch.Tensor | None:
        self._ensure_active()
        if self._text_finished:
            raise RuntimeError("Cannot push text tokens after finish_text().")
        if self._state.end_flag:
            raise RuntimeError(
                "Double streaming generation has already reached EOS. "
                "Call finish_text() to flush the remaining audio tail."
            )

        token_id = int(text_token)
        if not self._started:
            chunk_token_ids = [*self._pending_token_ids, token_id]
            self._pending_token_ids.clear()
            self._started = True
        else:
            chunk_token_ids = [token_id]

        self._consume_text_chunk(chunk_token_ids)
        return self._decode_audio_chunk()

    def finish_text(self):
        self._ensure_active()

        if not self._state.end_flag:
            if not self._text_finished:
                text_end_chunk = [self.model.core.text_cond_end_id]
                if not self._started:
                    text_end_chunk = [*self._pending_token_ids, *text_end_chunk]
                    self._pending_token_ids.clear()
                    self._started = True
                self._consume_text_chunk(text_end_chunk)
                self._text_finished = True

            while not self._state.end_flag:
                audio_chunk = self._decode_audio_chunk(continue_audio_span=True)
                if audio_chunk is not None:
                    yield audio_chunk
        else:
            self._text_finished = True

        final_chunk = self._vocoder_inference.flush(self._vocoder_state)
        self._closed = True
        logger.info(
            logc("stream", "Double streaming session finished: decoded_patch_count={}"),
            self._decoded_patch_count,
        )
        if final_chunk.size(-1) > 0:
            yield final_chunk

    def _ensure_active(self) -> None:
        if self._closed:
            raise RuntimeError("Double streaming session is already closed.")

    def _prepare_ref_audio_only_conditioning(self, prompt_audio_path: str) -> None:
        cache = getattr(self.runtime, "_double_streaming_prompt_g_cond_cache", None)
        if cache is None:
            cache = {}
            setattr(self.runtime, "_double_streaming_prompt_g_cond_cache", cache)
        prompt_cache_key = (
            str(Path(prompt_audio_path).expanduser().resolve()),
            str(self.device),
            str(self._dtype),
            self.speaker_scale,
        )
        cached_g_cond = cache.get(prompt_cache_key)
        if cached_g_cond is None:
            prompt_audio = self.runtime._load_prompt_audio(prompt_audio_path)
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._use_amp,
                ):
                    prompt_conditioning = self.model._prepare_prompt_conditioning(
                        prompt_audio,
                        use_prompt_prefill=False,
                        speaker_scale=self.speaker_scale,
                    )
            cached_g_cond = prompt_conditioning.g_cond.detach()
            cache[prompt_cache_key] = cached_g_cond
            logger.debug(
                logc(
                    "cache",
                    "Double streaming prompt conditioning cached: path={} device={} "
                    "dtype={} speaker_scale={}",
                ),
                prompt_cache_key[0],
                self.device,
                self._dtype,
                self.speaker_scale,
            )
        else:
            logger.debug(
                logc(
                    "cache",
                    "Double streaming prompt conditioning cache hit: path={} device={} "
                    "dtype={} speaker_scale={}",
                ),
                prompt_cache_key[0],
                self.device,
                self._dtype,
                self.speaker_scale,
            )
        self._g_cond = cached_g_cond

    def _consume_text_chunk(self, token_ids: list[int]) -> None:
        schedule = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                self.model._consume_text_schedule(
                    schedule,
                    position=0,
                    next_audio_position=schedule.size(1),
                    state=self._state,
                )

    def _get_initial_silence_audio_patch(
        self,
        patch_index: int,
        audio_patch: torch.Tensor,
    ) -> torch.Tensor:
        cache = getattr(
            self.runtime, "_double_streaming_silence_audio_patch_cache", None
        )
        if cache is None:
            cache = {}
            setattr(self.runtime, "_double_streaming_silence_audio_patch_cache", cache)

        cache_count = 10
        patch_size = int(self.model.core.latent_patch_size)
        key = (
            str(self.device),
            str(self._dtype),
            patch_size,
            int(audio_patch.size(-1)),
            cache_count,
        )
        cached_patches = cache.get(key)
        if cached_patches is None:
            hop_size = int(getattr(self.model.vocoder, "hop_size", 1))
            zero_samples = cache_count * patch_size * hop_size
            zero_audio = torch.zeros(
                (1, 1, zero_samples),
                device=self.device,
                dtype=torch.float32,
            )
            silence_latents = self._vocoder_inference.extract_latents(zero_audio)
            silence_latents, _ = torch.split(
                silence_latents,
                int(audio_patch.size(-1)),
                dim=1,
            )
            silence_latents = silence_latents.transpose(1, 2)
            target_frames = cache_count * patch_size
            if silence_latents.size(1) < target_frames:
                silence_latents = torch.cat(
                    [
                        silence_latents,
                        silence_latents.new_zeros(
                            (
                                silence_latents.size(0),
                                target_frames - silence_latents.size(1),
                                silence_latents.size(2),
                            )
                        ),
                    ],
                    dim=1,
                )
            silence_latents = silence_latents[:, :target_frames, :]
            cached_patches = self.model.core.io_helper.normalize(silence_latents)
            cached_patches = cached_patches.to(
                device=self.device, dtype=audio_patch.dtype
            )
            cached_patches = cached_patches.reshape(
                1,
                cache_count,
                patch_size,
                int(audio_patch.size(-1)),
            ).detach()
            cache[key] = cached_patches
            logger.debug(
                logc(
                    "cache",
                    "Double streaming initial silence cache built: patches={} patch_size={} "
                    "hop_size={} device={} dtype={}",
                ),
                cache_count,
                patch_size,
                hop_size,
                self.device,
                audio_patch.dtype,
            )
        return cached_patches[:, int(patch_index)].clone()

    def _consume_audio_patch(self, audio_patch: torch.Tensor) -> None:
        self.model._consume_audio_patch(self._state, audio_patch=audio_patch)

    def _decode_audio_chunk(
        self, *, continue_audio_span: bool = False
    ) -> torch.Tensor | None:
        if self._decoded_patch_count >= self.max_generate_length:
            raise RuntimeError(
                "Double streaming exceeded max_generate_length before reaching EOS."
            )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                stop_after_current_audio = self.model._should_stop_after_current_audio(
                    self._state,
                    eos_threshold=self.eos_threshold,
                )
                audio_patch = self.model._decode_next_audio(
                    self._state,
                    g_cond=self._g_cond,
                    ode_method=self.ode_method,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                if self._decoded_patch_count < self._initial_silence_audio_tokens:
                    audio_patch = self._get_initial_silence_audio_patch(
                        self._decoded_patch_count,
                        audio_patch,
                    )
                self._consume_audio_patch(audio_patch)
                if continue_audio_span:
                    self.model._append_hidden_chunk(
                        self._state, self._state.llm_hiddens
                    )
                self._decoded_patch_count += 1
                latent_patch = self.model.core.io_helper.denormalize(audio_patch)
                audio_chunk = self._vocoder_inference.stream_step(
                    latent_patch,
                    self._vocoder_state,
                    optimize=self.model._optimize_enabled,
                    profile_step=self._decoded_patch_count,
                )
                if stop_after_current_audio:
                    self._state.end_flag = True

        if audio_chunk.size(-1) == 0:
            return None
        return audio_chunk


class DotsTtsRuntimeDoubleStreaming(DotsTtsRuntime):
    def start_double_streaming(
        self,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str = "tts_interleave",
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int | None = None,
    ) -> DoubleStreamingSession:
        return DoubleStreamingSession(
            self,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            eos_threshold=eos_threshold,
            initial_silence_audio_tokens=initial_silence_audio_tokens,
        )


__all__ = ["DotsTtsRuntimeDoubleStreaming", "DoubleStreamingSession"]
