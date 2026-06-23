from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, TypedDict

import librosa
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from dots_tts.data.pipelines.tokenizing import build_generation_schedule
from dots_tts.data.pipelines.tts_pipeline import (
    DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
    DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    DEFAULT_TRAIN_TEMPLATE,
)
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.utils.audio import high_quality_resample
from dots_tts.utils.profiling import (
    InferenceProfiler,
    activate_inference_profiler,
    inference_profiling,
    log_inference_profile,
)
from dots_tts.utils.text import (
    attach_language_tag,
    detect,
    normalize_language_code,
    normalize_text,
)
from dots_tts.utils.util import get_dtype

RUNTIME_TEMPLATE_BY_NAME = {
    "tts": DEFAULT_TRAIN_TEMPLATE,
    "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    "tts_interleave": DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
}


class RuntimeInputs(TypedDict, total=False):
    fid: str
    language: str
    text: str
    prompt_text: str
    template_name: str
    generation_schedule: torch.Tensor
    prompt_audio: torch.Tensor


class DotsTtsRuntime:
    # region Lifecycle and pretrained loading
    def __init__(
        self,
        model: DotsTtsModel,
        pretrained_path: Path,
        *,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
    ):
        self.model = model
        self.pretrained_path = pretrained_path
        self.precision = precision
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)
        if self.device.type == "cuda" and self.precision.lower() in {
            "fp32",
            "torch.float32",
            "float32",
        }:
            torch.set_float32_matmul_precision("high")
        target_dtype = get_dtype(self.precision)
        self.model.core.to(dtype=target_dtype)
        self.model = self.model.to(self.device).eval()
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.model.set_optimize(self.optimize)
        self.sample_rate = int(self.model.config.vocoder.sample_rate)
        if self.optimize and hasattr(self.model, "run_warmup"):
            self.model.run_warmup(
                max_generate_length=self.max_generate_length,
                precision=self.precision,
            )
        logger.info(
            "Runtime initialized: pretrained_path={} device={} sample_rate={} "
            "precision={} "
            "optimize={} max_audio_patch_count={}",
            self.pretrained_path,
            self.device,
            self.sample_rate,
            self.precision,
            self.optimize,
            self.max_generate_length,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
    ) -> DotsTtsRuntime:
        logger.info(
            "Runtime load started: model={} revision={} cache_dir={} precision={}",
            model_name_or_path,
            revision,
            cache_dir,
            precision,
        )
        pretrained_path = cls._resolve_pretrained_path(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        loaded_model = DotsTtsModel.from_pretrained(pretrained_path)
        logger.info("Runtime load completed: pretrained_path={}", pretrained_path)
        return cls(
            model=loaded_model,
            pretrained_path=pretrained_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
        )

    @classmethod
    def _resolve_pretrained_path(
        cls,
        model_name_or_path: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> Path:
        logger.info(
            "Resolving pretrained path: model={} revision={} cache_dir={}",
            model_name_or_path,
            revision,
            cache_dir,
        )
        resolved_path = Path(model_name_or_path).expanduser().resolve()
        if resolved_path.exists():
            logger.info("Using local pretrained directory: path={}", resolved_path)
            return resolved_path

        logger.info(
            "Downloading pretrained snapshot: repo_id={} revision={} cache_dir={}",
            model_name_or_path,
            revision,
            cache_dir,
        )
        snapshot_dir = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        resolved_path = Path(snapshot_dir).expanduser().resolve()
        logger.info("Pretrained snapshot ready: path={}", resolved_path)
        return resolved_path
    # endregion Lifecycle and pretrained loading

    # region Request normalization and metadata
    @staticmethod
    def _build_request_id(
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str,
        language: str | None = None,
    ) -> str:
        payload = {
            "text": text,
            "prompt_audio_path": prompt_audio_path,
            "prompt_text": prompt_text,
            "template_name": template_name,
        }
        if language is not None:
            payload["language"] = language
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def _load_prompt_audio(
        self,
        prompt_audio_path: str,
    ) -> torch.Tensor:
        logger.info("Loading prompt audio: path={}", prompt_audio_path)
        prompt_audio, sample_rate = librosa.load(prompt_audio_path, sr=None, mono=True)
        prompt_audio = librosa.effects.trim(prompt_audio, top_db=30)[0]
        prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0)
        prompt_audio = high_quality_resample(
            prompt_audio,
            orig_sr=sample_rate,
            target_sr=self.sample_rate,
        )
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        logger.info(
            "Prompt audio loaded: path={} original_sample_rate={} resampled_sample_rate={} "
            "samples={}",
            prompt_audio_path,
            sample_rate,
            self.sample_rate,
            prompt_audio.shape[-1],
        )
        return prompt_audio

    def _resolve_language(
        self,
        language: str | None,
        *,
        text: str,
    ) -> str | None:
        if language is None:
            return None

        stripped = language.strip()
        if not stripped or stripped.lower() == "none":
            return None
        if stripped.lower() == "auto_detect":
            return normalize_language_code(detect(text))

        normalized_language = normalize_language_code(stripped)
        if normalized_language is None:
            raise ValueError(
                f"Unsupported language={language!r}. "
                "Expected 'none', 'auto_detect', or a valid language code/name. Supported languages include: 'EN', 'ZH', etc."
            )
        return normalized_language

    def _process_prompt_text(
        self,
        prompt_text: str | None,
        *,
        language: str | None = None,
    ) -> str:
        if prompt_text is None:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""

        prompt_text += "\n"
        if language is not None:
            prompt_text = attach_language_tag(prompt_text, language)
        return prompt_text

    def _process_text(
        self,
        text: str,
        *,
        language: str | None = None,
        normalize: bool = False,
    ) -> tuple[str, str | None]:
        stripped = text.strip()
        if normalize:
            stripped = normalize_text(stripped)
        resolved_language = self._resolve_language(language, text=stripped)
        return stripped, resolved_language

    def _estimate_prompt_audio_patch_count(
        self,
        *,
        prompt_audio: torch.Tensor | None,
        prompt_text: str,
    ) -> int:
        if prompt_audio is None or not prompt_text:
            return 0
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        prompt_samples = int(prompt_audio.shape[-1])
        return (prompt_samples + samples_per_patch - 1) // samples_per_patch
    # endregion Request normalization and metadata

    # region Generation schedule assembly
    def _normalize_template_name(self, template_name: str | None) -> str:
        if template_name is None:
            return "tts"
        if template_name not in RUNTIME_TEMPLATE_BY_NAME:
            raise ValueError(
                f"Unknown template_name={template_name!r}. "
                f"Expected one of {sorted(RUNTIME_TEMPLATE_BY_NAME)}."
            )
        return template_name

    def _prepare_inputs(
        self,
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str | None,
        language: str | None = None,
        normalize_text: bool = False,
    ) -> RuntimeInputs:
        normalized_template_name = self._normalize_template_name(template_name)
        template = RUNTIME_TEMPLATE_BY_NAME[normalized_template_name]
        if prompt_text and not prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")

        normalized_text, normalized_language = self._process_text(
            text,
            language=language,
            normalize=normalize_text,
        )
        normalized_prompt_text = self._process_prompt_text(
            prompt_text,
            language=normalized_language,
        )
        if normalized_language is not None and not normalized_prompt_text:
            normalized_text = attach_language_tag(normalized_text, normalized_language)
        inputs: RuntimeInputs = {
            "fid": self._build_request_id(
                text=normalized_text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text,
                template_name=normalized_template_name,
                language=normalized_language,
            ),
            "language": normalized_language or "",
            "text": normalized_text,
            "prompt_text": normalized_prompt_text,
            "template_name": normalized_template_name,
        }

        if prompt_audio_path:
            inputs["prompt_audio"] = self._load_prompt_audio(prompt_audio_path)
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=inputs.get("prompt_audio"),
            prompt_text=normalized_prompt_text,
        )
        if (
            prompt_audio_patch_count > 0
            and self.max_generate_length <= prompt_audio_patch_count
        ):
            raise ValueError(
                "max_generate_length must exceed prompt audio patch count when prompt_text is provided: "
                f"max_generate_length={self.max_generate_length} "
                f"prompt_audio_patch_count={prompt_audio_patch_count}."
            )

        schedule_spec = build_generation_schedule(
            text=f"{normalized_prompt_text}{normalized_text}",
            tokenizer=self.model.tokenizer,
            template=template,
            max_audio_tokens=self.max_generate_length,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        inputs["generation_schedule"] = schedule.unsqueeze(0)
        logger.info(
            "Inputs prepared: request_id={} template_name={} "
            "language={} text_len={} prompt_text_len={} schedule_length={} "
            "prompt_audio_patch_count={} max_audio_patch_count={} has_prompt_audio={}",
            inputs["fid"],
            normalized_template_name,
            normalized_language,
            len(normalized_text),
            len(normalized_prompt_text),
            schedule.numel(),
            prompt_audio_patch_count,
            self.max_generate_length,
            bool(prompt_audio_path),
        )
        return inputs
    # endregion Generation schedule assembly

    # region Public generation APIs
    def generate_stream(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        normalize_text: bool = False,
        profile_inference: bool = False,
    ) -> Iterator[torch.Tensor]:
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
        )
        logger.info(
            "Streaming generation started: request_id={} text_len={} has_prompt_audio={} "
            "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
            "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={}",
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
        )
        start_time = time.time()
        emitted_samples = 0
        chunk_count = 0
        profiler: InferenceProfiler | None = None
        try:
            profiler = (
                InferenceProfiler(self.device) if profile_inference else None
            )
            stream = self.model.generate_audio_stream(
                inputs,
                precision=self.precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
            )
            while True:
                try:
                    with activate_inference_profiler(profiler):
                        chunk = next(stream)
                except StopIteration:
                    break
                emitted_samples += int(chunk.shape[-1])
                chunk_count += 1
                yield chunk
        except Exception:
            logger.exception(
                "Streaming generation failed: request_id={}",
                inputs["fid"],
            )
            raise
        time_used = time.time() - start_time
        duration_seconds = emitted_samples / self.sample_rate
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profile_inference and profiler is not None:
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiler.summary(duration_seconds=duration_seconds),
                duration_seconds=duration_seconds,
            )
        logger.info(
            "Streaming generation finished: request_id={} chunk_count={} elapsed_seconds={:.3f} "
            "audio_seconds={:.3f} rtf={:.4f} sample_rate={}",
            inputs["fid"],
            chunk_count,
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )

    def generate(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        normalize_text: bool = False,
        profile_inference: bool = False,
    ) -> dict[str, Any]:
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
        )
        logger.info(
            "Generation started: request_id={} text_len={} has_prompt_audio={} "
            "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
            "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={}",
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
        )
        start_time = time.time()
        profiling = None
        try:
            with inference_profiling(
                enabled=profile_inference,
                device=self.device,
            ) as profiler:
                audio = self.model.generate_audio(
                    inputs,
                    precision=self.precision,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=speaker_scale,
                )
        except Exception:
            logger.exception("Generation failed: request_id={}", inputs["fid"])
            raise
        time_used = time.time() - start_time
        duration_seconds = audio.shape[-1] / self.sample_rate
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profiler is not None:
            profiling = profiler.summary(duration_seconds=duration_seconds)
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiling,
                duration_seconds=duration_seconds,
            )
        logger.info(
            "Generation completed: request_id={} elapsed_seconds={:.3f} audio_seconds={:.3f} "
            "rtf={:.4f} sample_rate={}",
            inputs["fid"],
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )
        return {
            "fid": inputs["fid"],
            "audio": audio,
            "sample_rate": self.sample_rate,
            "time_used": time_used,
            "rtf": rtf,
            "profiling": profiling,
        }
    # endregion Public generation APIs
