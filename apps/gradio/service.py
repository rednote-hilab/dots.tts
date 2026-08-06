from __future__ import annotations

import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from apps.gradio.constants import (  # noqa: E402
    DEFAULT_EXECUTION_MODE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HOST,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_NUM_STEPS,
    DEFAULT_ODE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
    DEFAULT_PROMPT_MAPPING_FILE,
    DEFAULT_PROMPT_NAME,
    DEFAULT_PROMPT_NONE,
    DEFAULT_PROMPT_SOURCE_DIR,
    DEFAULT_PROMPTS_DIR,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
    DEFAULT_WARMUP_TEXT,
    PROMPT_AUDIO_SUFFIXES,
)
from apps.gradio.languages import (  # noqa: E402
    SUPPORTED_LANGUAGE_CODE_BY_NAME,
    build_language_choice_items,
)
from dots_tts.runtime import DotsTtsRuntime  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402

ExecutionMode = Literal["generate", "generate_stream"]
GRADIO_SYNTHESIS_MODE_CHOICES = (
    ("tts", "tts"),
    ("instruct_tts", "instruction_tts"),
    ("instruct_tts_general", "text_to_audio"),
)
GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES = tuple(
    value for _, value in GRADIO_SYNTHESIS_MODE_CHOICES
)


@dataclass(frozen=True)
class PromptPreset:
    name: str
    audio_path: str
    prompt_text: str


def _is_prompt_asset(path: Path) -> bool:
    return path.is_file() and (
        path.name == "prompt_text" or path.suffix.lower() in PROMPT_AUDIO_SUFFIXES
    )


def sync_default_prompt_library(
    source_dir: Path = DEFAULT_PROMPT_SOURCE_DIR,
    target_dir: Path = DEFAULT_PROMPTS_DIR,
) -> None:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        logger.info(
            "Prompt library sync skipped: source_dir={} does not exist.",
            source_dir,
        )
        return

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Prompt library sync started: source_dir={} target_dir={}",
        source_dir,
        target_dir,
    )

    source_assets = {
        asset.name: asset for asset in sorted(source_dir.iterdir()) if _is_prompt_asset(asset)
    }
    copied_count = 0
    for asset_name, source_asset in source_assets.items():
        target_asset = target_dir / asset_name
        if (
            not target_asset.exists()
            or target_asset.stat().st_size != source_asset.stat().st_size
            or target_asset.stat().st_mtime_ns != source_asset.stat().st_mtime_ns
        ):
            shutil.copy2(source_asset, target_asset)
            copied_count += 1

    removed_count = 0
    for target_asset in sorted(target_dir.iterdir()):
        if _is_prompt_asset(target_asset) and target_asset.name not in source_assets:
            target_asset.unlink(missing_ok=True)
            removed_count += 1
    logger.info(
        "Prompt library sync completed: copied_assets={} removed_assets={} "
        "available_assets={}",
        copied_count,
        removed_count,
        len(source_assets),
    )


def _load_prompt_text_map(mapping_file: Path) -> dict[str, str]:
    if not mapping_file.is_file():
        return {}

    prompt_text_map: dict[str, str] = {}
    with mapping_file.open(encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            name, text = line.split("|", 1)
            prompt_text_map[name.strip()] = text.strip()
    return prompt_text_map


def discover_prompt_presets(
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    mapping_file: Path = DEFAULT_PROMPT_MAPPING_FILE,
) -> tuple[PromptPreset, ...]:
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.is_dir():
        return ()

    prompt_text_map = _load_prompt_text_map(Path(mapping_file))
    prompt_audio_paths = [
        audio_path
        for audio_path in sorted(prompts_dir.iterdir(), key=lambda path: (path.stem == "child", path.stem))
        if audio_path.is_file() and audio_path.suffix.lower() in PROMPT_AUDIO_SUFFIXES
    ]
    return tuple(
        PromptPreset(
            name=audio_path.stem,
            audio_path=str(audio_path.resolve()),
            prompt_text=prompt_text_map.get(audio_path.stem, ""),
        )
        for audio_path in prompt_audio_paths
    )


def build_prompt_choice_items(
    prompt_presets: tuple[PromptPreset, ...],
) -> list[tuple[str, str]]:
    return [("No Preset", DEFAULT_PROMPT_NONE), *[(preset.name, preset.name) for preset in prompt_presets]]


def resolve_default_prompt_selection(
    prompt_presets: tuple[PromptPreset, ...],
    default_prompt_name: str = DEFAULT_PROMPT_NAME,
) -> tuple[str, str | None, str]:
    if not prompt_presets:
        return DEFAULT_PROMPT_NONE, None, ""

    preset_by_name = {preset.name: preset for preset in prompt_presets}
    selected_name = default_prompt_name if default_prompt_name in preset_by_name else prompt_presets[0].name
    selected_preset = preset_by_name[selected_name]
    return selected_name, selected_preset.audio_path, selected_preset.prompt_text


def resolve_prompt_selection(
    prompt_name: str,
    prompt_presets: tuple[PromptPreset, ...],
) -> tuple[str | None, str]:
    if prompt_name == DEFAULT_PROMPT_NONE:
        return None, ""

    for preset in prompt_presets:
        if preset.name == prompt_name:
            return preset.audio_path, preset.prompt_text
    return None, ""


def discover_local_model_choices(repo_root: Path = REPO_ROOT) -> list[str]:
    model_root = Path(repo_root) / "pretrained_models"
    if not model_root.is_dir():
        return []
    return sorted(
        path.relative_to(repo_root).as_posix()
        for path in model_root.glob("**/model")
        if path.is_dir()
    )


def resolve_model_name_or_path(model_name_or_path: str, repo_root: Path = REPO_ROOT) -> str:
    normalized = model_name_or_path.strip()
    if not normalized:
        raise ValueError("model_name_or_path 不能为空。")

    direct_path = Path(normalized).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    repo_relative_path = Path(repo_root) / normalized
    if repo_relative_path.exists():
        return str(repo_relative_path.resolve())

    return normalized


def default_model_name_or_path(repo_root: Path = REPO_ROOT) -> str:
    discovered = discover_local_model_choices(repo_root=repo_root)
    if not discovered:
        return ""
    return discovered[0]


@dataclass(frozen=True)
class GradioAppConfig:
    host: str
    port: int
    execution_mode: ExecutionMode
    precision: str
    optimize: bool
    output_dir: Path
    prompts_dir: Path
    output_retention_count: int
    max_generate_length: int
    default_model_name_or_path: str
    prompt_presets: tuple[PromptPreset, ...]
    default_prompt_name: str
    default_prompt_audio_path: str | None
    default_prompt_text: str
    default_precision: str
    default_num_steps: int
    default_guidance_scale: float
    default_speaker_scale: float
    default_max_generate_length: int
    local_model_choices: tuple[str, ...]
    repo_root: Path = REPO_ROOT


def build_gradio_app_config(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE,
    precision: str = DEFAULT_PRECISION,
    optimize: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_retention_count: int = DEFAULT_OUTPUT_RETENTION,
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
    model_name_or_path: str | None = None,
    default_prompt_name: str = DEFAULT_PROMPT_NAME,
    default_precision: str = DEFAULT_PRECISION,
    default_num_steps: int = DEFAULT_NUM_STEPS,
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    default_speaker_scale: float = DEFAULT_SPEAKER_SCALE,
    default_max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
    repo_root: Path = REPO_ROOT,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    prompt_source_dir: Path = DEFAULT_PROMPT_SOURCE_DIR,
) -> GradioAppConfig:
    sync_default_prompt_library(
        source_dir=prompt_source_dir,
        target_dir=prompts_dir,
    )
    discovered_models = discover_local_model_choices(repo_root=repo_root)
    prompt_presets = discover_prompt_presets(
        prompts_dir=prompts_dir,
        mapping_file=prompts_dir / "prompt_text",
    )
    resolved_default_prompt_name, default_prompt_audio_path, default_prompt_text = (
        resolve_default_prompt_selection(
            prompt_presets,
            default_prompt_name=default_prompt_name,
        )
    )
    selected_model_name_or_path = (
        model_name_or_path.strip()
        if model_name_or_path is not None
        else default_model_name_or_path(repo_root=repo_root)
    )
    if not selected_model_name_or_path:
        raise ValueError("No default model found. Please pass --model-name-or-path.")
    if execution_mode not in ("generate", "generate_stream"):
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    resolved_max_generate_length = int(max_generate_length)
    if resolved_max_generate_length <= 0:
        raise ValueError("max_generate_length must be positive.")
    resolved_precision = precision.strip() or DEFAULT_PRECISION
    logger.info(
        "Gradio app config prepared: host={} port={} output_dir={} "
        "output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={} "
        "default_model_name_or_path={} prompt_preset_count={} language_count={} local_model_choice_count={}",
        host,
        port,
        output_dir,
        output_retention_count,
        resolved_max_generate_length,
        execution_mode,
        resolved_precision,
        bool(optimize),
        selected_model_name_or_path,
        len(prompt_presets),
        len(SUPPORTED_LANGUAGE_CODE_BY_NAME),
        len(discovered_models),
    )
    return GradioAppConfig(
        host=host,
        port=int(port),
        execution_mode=execution_mode,
        precision=resolved_precision,
        optimize=bool(optimize),
        output_dir=Path(output_dir),
        prompts_dir=Path(prompts_dir),
        output_retention_count=int(output_retention_count),
        max_generate_length=resolved_max_generate_length,
        default_model_name_or_path=selected_model_name_or_path,
        prompt_presets=prompt_presets,
        default_prompt_name=resolved_default_prompt_name,
        default_prompt_audio_path=default_prompt_audio_path,
        default_prompt_text=default_prompt_text,
        default_precision=default_precision,
        default_num_steps=int(default_num_steps),
        default_guidance_scale=float(default_guidance_scale),
        default_speaker_scale=float(default_speaker_scale),
        default_max_generate_length=int(default_max_generate_length),
        local_model_choices=tuple(discovered_models),
        repo_root=repo_root,
    )


@dataclass(frozen=True)
class SynthesisRequest:
    model_name_or_path: str
    text: str
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE
    template_name: str = "tts"
    language: str | None = None
    ode_method: str = DEFAULT_ODE_METHOD
    num_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    speaker_scale: float = DEFAULT_SPEAKER_SCALE
    normalize_text: bool = False
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: str
    metrics: dict[str, Any]
    status: str


class GradioAppService:
    def __init__(self, config: GradioAppConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runtime: DotsTtsRuntime | None = None
        self._runtime_model_name_or_path: str | None = None
        logger.info(
            "Gradio service initialized: output_dir={} default_model_name_or_path={} "
            "output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={}",
            self.config.output_dir,
            self.config.default_model_name_or_path,
            self.config.output_retention_count,
            self.config.max_generate_length,
            self.config.execution_mode,
            self.config.precision,
            self.config.optimize,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.config.repo_root),
            "default_model_name_or_path": self.config.default_model_name_or_path,
            "local_model_choices": list(self.config.local_model_choices),
            "prompts_dir": str(self.config.prompts_dir),
            "prompt_preset_names": [preset.name for preset in self.config.prompt_presets],
            "default_prompt_name": self.config.default_prompt_name,
            "output_dir": str(self.config.output_dir),
            "output_retention_count": self.config.output_retention_count,
            "configured_max_generate_length": self.config.max_generate_length,
            "configured_execution_mode": self.config.execution_mode,
            "configured_precision": self.config.precision,
            "optimize": self.config.optimize,
            "loaded_model_name_or_path": self._runtime_model_name_or_path,
            "loaded_max_generate_length": (
                self.config.max_generate_length if self._runtime is not None else None
            ),
            "loaded_precision": (
                self.config.precision if self._runtime is not None else None
            ),
            "model_loaded": self._runtime is not None,
            "host": self.config.host,
            "port": self.config.port,
            "default_precision": self.config.default_precision,
            "default_num_steps": self.config.default_num_steps,
            "default_guidance_scale": self.config.default_guidance_scale,
            "default_speaker_scale": self.config.default_speaker_scale,
            "default_max_generate_length": self.config.default_max_generate_length,
            "supported_languages": build_language_choice_items()[1:],
            "supported_template_names": list(GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES),
        }

    def _get_runtime(
        self,
        model_name_or_path: str,
    ) -> tuple[DotsTtsRuntime, str]:
        resolved_model_name_or_path = resolve_model_name_or_path(
            model_name_or_path,
            repo_root=self.config.repo_root,
        )
        if (
            self._runtime is None
            or self._runtime_model_name_or_path != resolved_model_name_or_path
        ):
            logger.info(
                "Gradio runtime cache miss: requested_model={} resolved_model={} "
                "max_generate_length={} execution_mode={} precision={} optimize={}",
                model_name_or_path,
                resolved_model_name_or_path,
                self.config.max_generate_length,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
            )
            self._runtime = DotsTtsRuntime.from_pretrained(
                resolved_model_name_or_path,
                precision=self.config.precision,
                optimize=self.config.optimize,
                max_generate_length=self.config.max_generate_length,
            )
            self._runtime_model_name_or_path = resolved_model_name_or_path
        else:
            logger.info(
                "Gradio runtime cache hit: requested_model={} resolved_model={} "
                "max_generate_length={} execution_mode={} precision={} optimize={}",
                model_name_or_path,
                resolved_model_name_or_path,
                self.config.max_generate_length,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
            )
        return self._runtime, resolved_model_name_or_path

    def sampling_controls(self) -> dict[str, Any]:
        """Return the fixed model's artifact-resolved sampling UI contract."""

        runtime, _resolved_model_name_or_path = self._get_runtime(
            self.config.default_model_name_or_path
        )
        sampling = runtime.model.config.sampling
        if sampling is None:
            return {
                "solver": runtime.model.core.mode,
                "locked": False,
                "ode_method": DEFAULT_ODE_METHOD,
                "num_steps": self.config.default_num_steps,
                "guidance_scale": self.config.default_guidance_scale,
            }
        return {
            "solver": sampling.solver,
            "locked": True,
            "ode_method": sampling.ode_method,
            "num_steps": sampling.num_steps,
            "guidance_scale": sampling.guidance_scale,
        }

    def _build_stream_request_id(
        self,
        runtime: DotsTtsRuntime,
        request: SynthesisRequest,
    ) -> str:
        normalized_text, normalized_language = runtime._process_text(  # noqa: SLF001
            request.text,
            language=request.language,
            normalize=request.normalize_text,
        )
        normalized_prompt_text = runtime._process_prompt_text(  # noqa: SLF001
            request.prompt_text,
            language=normalized_language,
        )
        if normalized_language is not None and not normalized_prompt_text:
            from dots_tts.utils.text import attach_language_tag  # noqa: PLC0415

            normalized_text = attach_language_tag(
                normalized_text,
                normalized_language,
            )
        request_id_kwargs = {
            "text": normalized_text,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_text": normalized_prompt_text,
            "template_name": request.template_name,
        }
        if normalized_language is not None:
            request_id_kwargs["language"] = normalized_language
        return runtime._build_request_id(  # noqa: SLF001
            **request_id_kwargs,
        )

    @staticmethod
    def _build_runtime_generate_kwargs(request: SynthesisRequest) -> dict[str, Any]:
        runtime_kwargs: dict[str, Any] = {
            "text": request.text,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_text": request.prompt_text,
            "template_name": request.template_name,
            "ode_method": request.ode_method,
            "num_steps": request.num_steps,
            "guidance_scale": request.guidance_scale,
            "speaker_scale": request.speaker_scale,
            "normalize_text": request.normalize_text,
        }
        if request.language is not None:
            runtime_kwargs["language"] = request.language
        return runtime_kwargs

    def _run_stream_generation(
        self,
        runtime: DotsTtsRuntime,
        request: SynthesisRequest,
    ) -> dict[str, Any]:
        start_time = time.time()
        chunks = [
            chunk.detach().float().cpu()
            for chunk in runtime.generate_stream(
                **self._build_runtime_generate_kwargs(request)
            )
        ]
        if not chunks:
            raise ValueError("流式生成未返回任何音频块。")

        audio = torch.cat(chunks, dim=-1)
        elapsed_seconds = time.time() - start_time
        audio_seconds = audio.shape[-1] / runtime.sample_rate
        rtf = elapsed_seconds / audio_seconds if audio_seconds > 0 else float("inf")
        return {
            "fid": self._build_stream_request_id(runtime, request),
            "audio": audio,
            "sample_rate": runtime.sample_rate,
            "time_used": elapsed_seconds,
            "rtf": rtf,
            "chunk_count": len(chunks),
        }

    def warmup(self, text: str | None = None) -> dict[str, Any]:
        warmup_text = (text or "").strip() or DEFAULT_WARMUP_TEXT.strip()
        if not warmup_text:
            raise ValueError("DEFAULT_WARMUP_TEXT 不能为空。")

        with self._lock:
            logger.info(
                "Gradio warmup requested: default_model_name_or_path={} execution_mode={} precision={} optimize={} seed={}",
                self.config.default_model_name_or_path,
                self.config.execution_mode,
                self.config.precision,
                self.config.optimize,
                DEFAULT_SEED,
            )
            try:
                seed_everything(DEFAULT_SEED)
                runtime, resolved_model_name_or_path = self._get_runtime(
                    self.config.default_model_name_or_path,
                )
                ode_method, num_steps, guidance_scale = (
                    runtime.resolve_sampling_options(
                        ode_method=None,
                        num_steps=None,
                        guidance_scale=None,
                    )
                )
                warmup_request = SynthesisRequest(
                    model_name_or_path=self.config.default_model_name_or_path,
                    text=warmup_text,
                    execution_mode=self.config.execution_mode,
                    template_name="tts",
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=self.config.default_speaker_scale,
                    normalize_text=False,
                    seed=DEFAULT_SEED,
                )
                request_id = self._build_stream_request_id(runtime, warmup_request)
                if self.config.execution_mode == "generate_stream":
                    result = self._run_stream_generation(runtime, warmup_request)
                else:
                    start_time = time.time()
                    result = runtime.generate(**self._build_runtime_generate_kwargs(warmup_request))
                    result["time_used"] = time.time() - start_time
                    result["chunk_count"] = 1
                audio_samples = int(result["audio"].shape[-1])
            except Exception:
                logger.exception(
                    "Gradio warmup failed: default_model_name_or_path={}",
                    self.config.default_model_name_or_path,
                )
                raise
            audio_seconds = audio_samples / runtime.sample_rate
            metrics = {
                "request_id": request_id,
                "execution_mode": self.config.execution_mode,
                "chunk_count": int(result["chunk_count"]),
                "resolved_model_name_or_path": resolved_model_name_or_path,
                "sample_rate": runtime.sample_rate,
                "elapsed_seconds": round(float(result["time_used"]), 3),
                "audio_seconds": round(float(audio_seconds), 3),
                "rtf": round(float(result["rtf"]), 4),
                "seed": DEFAULT_SEED,
                "text": warmup_text,
            }
            logger.info(
                "Gradio warmup ready: request_id={} execution_mode={} resolved_model_name_or_path={}",
                metrics["request_id"],
                metrics["execution_mode"],
                metrics["resolved_model_name_or_path"],
            )
            return metrics

    def _normalize_request(self, request: SynthesisRequest) -> SynthesisRequest:
        normalized_text = request.text.strip()
        if not normalized_text:
            raise ValueError("text 不能为空。")

        normalized_prompt_audio_path = request.prompt_audio_path or None
        normalized_prompt_text = (request.prompt_text or "").strip() or None
        if normalized_prompt_text and not normalized_prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")
        normalized_template_name = request.template_name.strip() or "tts"
        if normalized_template_name not in GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES:
            raise ValueError(
                f"Unsupported template_name={normalized_template_name!r}. "
                f"Expected one of {list(GRADIO_SYNTHESIS_MODE_TEMPLATE_NAMES)}."
            )
        normalized_language = (request.language or "").strip() or None
        supported_language_codes = set(SUPPORTED_LANGUAGE_CODE_BY_NAME.values())
        if (
            normalized_language is not None
            and normalized_language not in supported_language_codes
        ):
            raise ValueError(
                f"Unsupported language={normalized_language!r}. "
                f"Expected one of {sorted(supported_language_codes)}."
            )

        resolved_seed = int(request.seed)
        return SynthesisRequest(
            model_name_or_path=request.model_name_or_path.strip(),
            text=normalized_text,
            prompt_audio_path=normalized_prompt_audio_path,
            prompt_text=normalized_prompt_text,
            execution_mode=request.execution_mode,
            template_name=normalized_template_name,
            language=normalized_language,
            ode_method=request.ode_method.strip() or DEFAULT_ODE_METHOD,
            num_steps=int(request.num_steps),
            guidance_scale=float(request.guidance_scale),
            speaker_scale=float(request.speaker_scale),
            normalize_text=bool(request.normalize_text),
            seed=resolved_seed,
        )

    def _build_output_path(self) -> Path:
        output_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
        return self.config.output_dir / output_name

    def _cleanup_outputs(self) -> None:
        if self.config.output_retention_count <= 0:
            return

        wav_files = sorted(
            self.config.output_dir.glob("*.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed_count = 0
        for stale_file in wav_files[self.config.output_retention_count :]:
            stale_file.unlink(missing_ok=True)
            removed_count += 1
        if removed_count > 0:
            logger.info(
                "Gradio output cleanup completed: removed_files={} retention_limit={}",
                removed_count,
                self.config.output_retention_count,
            )

    @staticmethod
    def _waveform_to_numpy(audio: torch.Tensor):
        waveform = audio.detach().float().cpu().squeeze()
        if waveform.ndim == 0:
            raise ValueError("生成音频为空。")
        return waveform.numpy()

    def _write_audio(self, audio: torch.Tensor, sample_rate: int) -> str:
        output_path = self._build_output_path()
        logger.info(
            "Writing synthesized audio: output_path={} sample_rate={} samples={}",
            output_path,
            sample_rate,
            audio.shape[-1],
        )
        sf.write(output_path, self._waveform_to_numpy(audio), sample_rate)
        self._cleanup_outputs()
        logger.info("Synthesized audio written: output_path={}", output_path)
        return str(output_path)

    def generate(self, request: SynthesisRequest) -> SynthesisResult:
        normalized_request = self._normalize_request(request)

        with self._lock:
            try:
                seed_everything(normalized_request.seed)
                runtime, resolved_model_name_or_path = self._get_runtime(
                    normalized_request.model_name_or_path,
                )
                logger.info(
                    "Gradio request accepted: resolved_model_name_or_path={} execution_mode={} seed={}",
                    resolved_model_name_or_path,
                    normalized_request.execution_mode,
                    normalized_request.seed,
                )
                if normalized_request.execution_mode == "generate_stream":
                    result = self._run_stream_generation(runtime, normalized_request)
                else:
                    result = runtime.generate(
                        **self._build_runtime_generate_kwargs(normalized_request)
                    )
                    result["chunk_count"] = 1
                audio_path = self._write_audio(result["audio"], result["sample_rate"])
            except Exception:
                logger.exception(
                    "Gradio request failed: model_name_or_path={} execution_mode={} text_len={} has_prompt_audio={} has_prompt_text={} template_name={} language={} "
                    "precision={} ode_method={} num_steps={} guidance_scale={} speaker_scale={} max_generate_length={} "
                    "normalize_text={} seed={}",
                    normalized_request.model_name_or_path,
                    normalized_request.execution_mode,
                    len(normalized_request.text),
                    bool(normalized_request.prompt_audio_path),
                    bool(normalized_request.prompt_text),
                    normalized_request.template_name,
                    normalized_request.language,
                    self.config.precision,
                    normalized_request.ode_method,
                    normalized_request.num_steps,
                    normalized_request.guidance_scale,
                    normalized_request.speaker_scale,
                    self.config.max_generate_length,
                    normalized_request.normalize_text,
                    normalized_request.seed,
                )
                raise
            audio_seconds = result["audio"].shape[-1] / result["sample_rate"]
            metrics = {
                "request_id": result["fid"],
                "execution_mode": normalized_request.execution_mode,
                "chunk_count": int(result["chunk_count"]),
                "template_name": normalized_request.template_name,
                "language": normalized_request.language,
                "resolved_model_name_or_path": resolved_model_name_or_path,
                "sample_rate": result["sample_rate"],
                "elapsed_seconds": round(float(result["time_used"]), 3),
                "audio_seconds": round(float(audio_seconds), 3),
                "rtf": round(float(result["rtf"]), 4),
                "seed": normalized_request.seed,
                "output_path": audio_path,
            }
            logger.info(
                "Gradio request output ready: request_id={} execution_mode={} resolved_model_name_or_path={} output_path={}",
                metrics["request_id"],
                metrics["execution_mode"],
                metrics["resolved_model_name_or_path"],
                metrics["output_path"],
            )
            status = (
                f"完成：{Path(audio_path).name} | "
                f"模式 {metrics['execution_mode']} | "
                f"耗时 {metrics['elapsed_seconds']}s | "
                f"音频 {metrics['audio_seconds']}s | "
                f"RTF {metrics['rtf']}"
            )
            return SynthesisResult(
                audio_path=audio_path,
                metrics=metrics,
                status=status,
            )
