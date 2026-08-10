from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from apps.gradio.service import PromptPreset
from apps.edit_playground.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HOST,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_NOISE_PRESETS_DIR,
    DEFAULT_NUM_STEPS,
    DEFAULT_ODE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
    REPO_ROOT,
    SUPPORTED_ODE_METHODS,
)
from apps.edit_playground.editing import CompiledEdit, EditOperation, compile_edit
from dots_tts.data.edit_instruction import (
    EditXVectorMode,
    normalize_edit_xvector_mode,
    resolve_edit_use_xvector,
)
from dots_tts.edit_runtime import DotsTtsEditRuntime
from dots_tts.utils.util import seed_everything


class RuntimeFactory(Protocol):
    def __call__(self, model_name_or_path: str, **kwargs: Any) -> Any: ...


def _resolve_edit_request_use_xvector(
    mode: EditXVectorMode,
    instruction: str,
    *,
    allowed: bool,
) -> bool:
    """Resolve the request mode, with source capability as the final gate."""

    return bool(
        allowed
        and resolve_edit_use_xvector(
            normalize_edit_xvector_mode(mode),
            instruction,
        )
    )


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    model_name_or_path: str
    compile_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("model id and label must not be empty.")
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty.")


def model_spec_id(label: str, model_name_or_path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip(".-").lower()
    digest = hashlib.sha1(
        f"{label}\0{model_name_or_path}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:8]
    return f"{slug or 'model'}-{digest}"


@dataclass(frozen=True)
class V2ServiceConfig:
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    models: tuple[ModelSpec, ...] = ()
    default_model_id: str | None = None
    compiler_cache_root: Path | None = None
    noise_presets_dir: Path = DEFAULT_NOISE_PRESETS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prompt_presets: tuple[PromptPreset, ...] = ()
    output_retention_count: int = DEFAULT_OUTPUT_RETENTION
    precision: str = DEFAULT_PRECISION
    optimize: bool = True
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    repo_root: Path = REPO_ROOT

    def __post_init__(self) -> None:
        models = self.models
        if not models:
            path = self.model_name_or_path.strip()
            if not path:
                raise ValueError("model_name_or_path must not be empty.")
            models = (
                ModelSpec(
                    id=model_spec_id("Default", path),
                    label="Default",
                    model_name_or_path=path,
                ),
            )
            object.__setattr__(self, "models", models)
        ids = [item.id for item in models]
        labels = [item.label for item in models]
        if len(set(ids)) != len(ids):
            raise ValueError("model ids must be unique.")
        if len(set(labels)) != len(labels):
            raise ValueError("model labels must be unique.")
        default_model_id = self.default_model_id or models[0].id
        if default_model_id not in set(ids):
            raise ValueError(f"unknown default_model_id: {default_model_id!r}")
        object.__setattr__(self, "default_model_id", default_model_id)
        if self.compiler_cache_root is None:
            object.__setattr__(
                self,
                "compiler_cache_root",
                Path(self.output_dir) / "compiler-cache",
            )
        if self.output_retention_count < 0:
            raise ValueError("output_retention_count must be non-negative.")
        if self.max_generate_length <= 0 or self.max_sequence_length <= 0:
            raise ValueError("generation length limits must be positive.")


def build_service_config(
    *,
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
    models: tuple[ModelSpec, ...] = (),
    default_model_id: str | None = None,
    compiler_cache_root: Path | None = None,
    noise_presets_dir: Path = DEFAULT_NOISE_PRESETS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prompts_dir: Path | None = None,
    output_retention_count: int = DEFAULT_OUTPUT_RETENTION,
    precision: str = DEFAULT_PRECISION,
    optimize: bool = True,
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    repo_root: Path = REPO_ROOT,
) -> V2ServiceConfig:
    del prompts_dir
    return V2ServiceConfig(
        model_name_or_path=model_name_or_path,
        models=models,
        default_model_id=default_model_id,
        compiler_cache_root=compiler_cache_root,
        noise_presets_dir=Path(noise_presets_dir),
        output_dir=Path(output_dir),
        prompt_presets=(),
        output_retention_count=output_retention_count,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
        max_sequence_length=max_sequence_length,
        host=host,
        port=port,
        repo_root=Path(repo_root),
    )


@dataclass(frozen=True)
class GenerationSettings:
    ode_method: str = DEFAULT_ODE_METHOD
    num_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    speaker_scale: float = DEFAULT_SPEAKER_SCALE
    seed: int = DEFAULT_SEED

    def normalized(self) -> GenerationSettings:
        ode_method = self.ode_method.strip()
        if ode_method not in SUPPORTED_ODE_METHODS:
            raise ValueError(
                f"ode_method must be one of: {', '.join(SUPPORTED_ODE_METHODS)}."
            )
        if int(self.num_steps) <= 0:
            raise ValueError("num_steps must be positive.")
        if float(self.guidance_scale) < 0 or float(self.speaker_scale) < 0:
            raise ValueError("generation scales must be non-negative.")
        return GenerationSettings(
            ode_method=ode_method,
            num_steps=int(self.num_steps),
            guidance_scale=float(self.guidance_scale),
            speaker_scale=float(self.speaker_scale),
            seed=int(self.seed),
        )


@dataclass(frozen=True)
class TTSRequest:
    text: str
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    prompt_revision_id: str | None = None
    model_id: str | None = None
    use_xvector: bool = True
    settings: GenerationSettings = field(default_factory=GenerationSettings)


@dataclass(frozen=True)
class EditRequest:
    source_audio_path: str
    source_text: str
    operations: Sequence[EditOperation | dict[str, Any]] = ()
    compiled_edit: CompiledEdit | None = None
    source_revision_id: str | None = None
    model_id: str | None = None
    use_xvector: EditXVectorMode = "auto"
    xvector_allowed: bool = True
    settings: GenerationSettings = field(default_factory=GenerationSettings)


@dataclass(frozen=True)
class GenerationResult:
    audio_path: str
    text: str
    parent_id: str | None
    metadata: dict[str, Any]
    compiled_edit: CompiledEdit | None = None


def _default_runtime_factory(
    model_name_or_path: str, **kwargs: Any
) -> DotsTtsEditRuntime:
    return DotsTtsEditRuntime.from_pretrained(model_name_or_path, **kwargs)


def _safe_session_id(session_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id.strip()).strip(".-")
    if not normalized:
        raise ValueError("session_id must contain at least one safe character.")
    return normalized[:96]


def resolve_model_name_or_path(
    model_name_or_path: str, repo_root: Path = REPO_ROOT
) -> str:
    value = model_name_or_path.strip()
    if not value:
        raise ValueError("model_name_or_path must not be empty.")
    direct = Path(value).expanduser()
    if direct.exists():
        return str(direct.resolve())
    relative = Path(repo_root) / value
    if relative.exists():
        return str(relative.resolve())
    return value


def validate_local_model_artifacts(
    model_name_or_path: str, repo_root: Path = REPO_ROOT
) -> None:
    resolved = Path(resolve_model_name_or_path(model_name_or_path, repo_root))
    if not resolved.exists():
        return  # A Hugging Face model id may be resolved by the runtime.
    if not resolved.is_dir():
        raise ValueError(f"model path is not a directory: {resolved}")
    required = (
        "config.json",
        "latent_stats.pt",
        "llm_config.json",
        "model.safetensors",
        "vocoder.safetensors",
        "speaker_encoder.safetensors",
    )
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise ValueError(
            f"model directory is missing required files: {', '.join(missing)}"
        )


class StudioService:
    """Thread-safe structured generation service used only by the v2 UI."""

    def __init__(
        self,
        config: V2ServiceConfig,
        *,
        runtime_factory: RuntimeFactory = _default_runtime_factory,
        seed_fn: Callable[[int], None] = seed_everything,
    ) -> None:
        self.config = config
        for spec in self.config.models:
            validate_local_model_artifacts(
                spec.model_name_or_path,
                repo_root=self.config.repo_root,
            )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_factory = runtime_factory
        self._seed_fn = seed_fn
        self._runtimes: dict[str, Any] = {}
        self._resolved_model_paths: dict[str, str] = {}
        self._model_status: dict[str, dict[str, Any]] = {
            spec.id: {"status": "pending", "error": None} for spec in self.config.models
        }
        self._status_lock = threading.RLock()
        self._lock = threading.RLock()

    @property
    def prompt_presets(self) -> tuple[PromptPreset, ...]:
        return self.config.prompt_presets

    @property
    def runtime_loaded(self) -> bool:
        with self._status_lock:
            return bool(self._runtimes)

    @property
    def default_model_id(self) -> str:
        assert self.config.default_model_id is not None
        return self.config.default_model_id

    def get_model_spec(self, model_id: str | None = None) -> ModelSpec:
        selected = model_id or self.default_model_id
        for spec in self.config.models:
            if spec.id == selected:
                return spec
        raise ValueError(f"Unknown model: {selected!r}")

    def model_catalog(self) -> list[dict[str, Any]]:
        with self._status_lock:
            return [
                {
                    "id": spec.id,
                    "label": spec.label,
                    "path": resolve_model_name_or_path(
                        spec.model_name_or_path,
                        repo_root=self.config.repo_root,
                    ),
                    "status": self._model_status[spec.id]["status"],
                    "error": self._model_status[spec.id]["error"],
                    "runtime_loaded": spec.id in self._runtimes,
                }
                for spec in self.config.models
            ]

    def _set_model_status(
        self,
        model_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._status_lock:
            self._model_status[model_id] = {"status": status, "error": error}

    def _compile_cache_dir(self, spec: ModelSpec, resolved: str) -> Path:
        if spec.compile_cache_dir is not None:
            return Path(spec.compile_cache_dir)
        digest = hashlib.sha1(
            resolved.encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        assert self.config.compiler_cache_root is not None
        return Path(self.config.compiler_cache_root) / digest

    @contextmanager
    def _compiler_cache_environment(self, spec: ModelSpec, resolved: str):
        cache_root = self._compile_cache_dir(spec, resolved)
        values = {
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
        }
        previous = {key: os.environ.get(key) for key in values}
        for key, value in values.items():
            Path(value).mkdir(parents=True, exist_ok=True)
            os.environ[key] = value
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _get_runtime(self, model_id: str | None = None) -> tuple[Any, ModelSpec, str]:
        spec = self.get_model_spec(model_id)
        resolved = resolve_model_name_or_path(
            spec.model_name_or_path,
            repo_root=self.config.repo_root,
        )
        if spec.id not in self._runtimes:
            self._set_model_status(spec.id, "loading")
            try:
                with self._compiler_cache_environment(spec, resolved):
                    runtime = self._runtime_factory(
                        resolved,
                        precision=self.config.precision,
                        optimize=self.config.optimize,
                        max_generate_length=self.config.max_generate_length,
                    )
            except Exception as exc:
                self._set_model_status(spec.id, "error", str(exc))
                raise
            with self._status_lock:
                self._runtimes[spec.id] = runtime
                self._resolved_model_paths[spec.id] = resolved
            self._set_model_status(spec.id, "ready")
        return self._runtimes[spec.id], spec, resolved

    def warmup(
        self,
        text: str = "预热文本",
        *,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Load the configured runtime and perform a history-free TTS warmup."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("warmup text must not be empty.")
        settings = GenerationSettings().normalized()
        with self._lock:
            self._seed_fn(settings.seed)
            runtime, spec, model_path = self._get_runtime(model_id)
            self._set_model_status(spec.id, "warming")
            started_at = time.time()
            try:
                with self._compiler_cache_environment(spec, model_path):
                    result = runtime.generate(
                        text=normalized_text,
                        prompt_audio_path=None,
                        prompt_text=None,
                        template_name="tts",
                        normalize_text=False,
                        **self._runtime_settings(settings),
                    )
            except Exception as exc:
                self._set_model_status(spec.id, "error", str(exc))
                raise
            self._set_model_status(spec.id, "ready")
            elapsed = float(
                result.get(
                    "elapsed_seconds",
                    result.get("time_used", time.time() - started_at),
                )
            )
            waveform = self._waveform_to_numpy(result["audio"])
            sample_rate = int(result["sample_rate"])
            return {
                "request_id": result.get("request_id", result.get("fid")),
                "model_id": spec.id,
                "model_label": spec.label,
                "model": model_path,
                "sample_rate": sample_rate,
                "duration_seconds": float(
                    result.get("duration_seconds", waveform.size / sample_rate)
                ),
                "elapsed_seconds": elapsed,
            }

    def warmup_all(self, text: str = "预热文本") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        for spec in self.config.models:
            try:
                results.append(self.warmup(text, model_id=spec.id))
            except Exception as exc:
                errors.append(f"{spec.label}: {exc}")
        if errors and not results:
            raise RuntimeError("; ".join(errors))
        return results

    def load_all(self) -> list[dict[str, Any]]:
        """Load every configured runtime without running generation.

        This deliberately skips generation and is useful when startup should
        validate all configured checkpoints without paying warmup latency.
        """

        loaded: list[dict[str, Any]] = []
        errors: list[str] = []
        with self._lock:
            for spec in self.config.models:
                try:
                    _, resolved_spec, resolved = self._get_runtime(spec.id)
                except Exception as exc:
                    errors.append(f"{spec.label}: {exc}")
                    continue
                loaded.append(
                    {
                        "model_id": resolved_spec.id,
                        "model_label": resolved_spec.label,
                        "model": resolved,
                    }
                )
        if errors:
            raise RuntimeError("; ".join(errors))
        return loaded

    @staticmethod
    def _validate_audio_path(value: str | None, *, field_name: str) -> str | None:
        normalized = (value or "").strip()
        if not normalized:
            return None
        path = Path(normalized).expanduser()
        if not path.is_file():
            raise ValueError(f"{field_name} does not exist: {path}")
        return str(path.resolve())

    @staticmethod
    def _waveform_to_numpy(audio: Any) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("generated audio must be a non-empty mono waveform.")
        if not np.isfinite(waveform).all():
            raise ValueError("generated audio contains non-finite samples.")
        return waveform

    def _write_audio(self, session_id: str, audio: Any, sample_rate: int) -> str:
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be positive.")
        session_dir = self.config.output_dir / _safe_session_id(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        output_path = session_dir / (
            f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}.wav"
        )
        sf.write(output_path, self._waveform_to_numpy(audio), int(sample_rate))
        self._cleanup_session_outputs(session_dir)
        return str(output_path)

    def _cleanup_session_outputs(self, session_dir: Path) -> None:
        limit = self.config.output_retention_count
        if limit <= 0:
            return
        wav_files = sorted(
            session_dir.glob("*.wav"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in wav_files[limit:]:
            path.unlink(missing_ok=True)

    @staticmethod
    def _runtime_settings(settings: GenerationSettings) -> dict[str, Any]:
        values: dict[str, Any] = {
            "ode_method": settings.ode_method,
            "num_steps": settings.num_steps,
            "guidance_scale": settings.guidance_scale,
            "speaker_scale": settings.speaker_scale,
        }
        return values

    def _finish_result(
        self,
        *,
        session_id: str,
        kind: str,
        text: str,
        parent_id: str | None,
        settings: GenerationSettings,
        runtime_result: dict[str, Any],
        model_name_or_path: str,
        model_id: str,
        model_label: str,
        compiled_edit: CompiledEdit | None = None,
        use_xvector: bool = True,
        generation_extra: dict[str, Any] | None = None,
    ) -> GenerationResult:
        audio = runtime_result["audio"]
        sample_rate = int(runtime_result["sample_rate"])
        audio_path = self._write_audio(session_id, audio, sample_rate)
        waveform = self._waveform_to_numpy(audio)
        duration = float(
            runtime_result.get("duration_seconds", waveform.size / sample_rate)
        )
        request_id = runtime_result.get("request_id", runtime_result.get("fid"))
        elapsed = runtime_result.get("elapsed_seconds", runtime_result.get("time_used"))
        metadata: dict[str, Any] = {
            "kind": kind,
            "model": model_name_or_path,
            "model_id": model_id,
            "model_label": model_label,
            "request_id": request_id,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "generation": {
                "ode_method": settings.ode_method,
                "num_steps": settings.num_steps,
                "guidance_scale": settings.guidance_scale,
                "speaker_scale": settings.speaker_scale,
                "use_xvector": use_xvector,
                "seed": settings.seed,
            },
        }
        if generation_extra:
            metadata["generation"].update(generation_extra)
        if elapsed is not None:
            metadata["elapsed_seconds"] = float(elapsed)
        if runtime_result.get("rtf") is not None:
            metadata["rtf"] = float(runtime_result["rtf"])
        return GenerationResult(
            audio_path=audio_path,
            text=text,
            parent_id=parent_id,
            metadata=metadata,
            compiled_edit=compiled_edit,
        )

    def generate_tts(self, session_id: str, request: TTSRequest) -> GenerationResult:
        _safe_session_id(session_id)
        text = request.text.strip()
        if not text:
            raise ValueError("Target Transcript must not be empty.")
        prompt_audio = self._validate_audio_path(
            request.prompt_audio_path,
            field_name="prompt_audio_path",
        )
        prompt_text = (request.prompt_text or "").strip() or None
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError(
                "Prompt Audio and Prompt Transcript must be provided together."
            )
        settings = request.settings.normalized()
        use_xvector = bool(request.use_xvector and prompt_audio is not None)
        if not use_xvector:
            settings = replace(settings, speaker_scale=0.0)
        with self._lock:
            self._seed_fn(settings.seed)
            runtime, spec, model_path = self._get_runtime(request.model_id)
            kwargs = {
                "text": text,
                "prompt_audio_path": prompt_audio,
                "prompt_text": prompt_text,
                "template_name": "tts",
                "normalize_text": False,
                "use_xvector": use_xvector,
                **self._runtime_settings(settings),
            }
            with self._compiler_cache_environment(spec, model_path):
                result = runtime.generate(**kwargs)
            return self._finish_result(
                session_id=session_id,
                kind="tts",
                text=text,
                parent_id=request.prompt_revision_id,
                settings=settings,
                runtime_result=result,
                model_name_or_path=model_path,
                model_id=spec.id,
                model_label=spec.label,
                use_xvector=use_xvector,
            )

    def generate_edit(self, session_id: str, request: EditRequest) -> GenerationResult:
        _safe_session_id(session_id)
        source_text = request.source_text
        if not source_text.strip():
            raise ValueError("Source Transcript must not be empty.")
        source_audio = self._validate_audio_path(
            request.source_audio_path,
            field_name="source_audio_path",
        )
        if source_audio is None:  # Kept explicit for optimized Python builds.
            raise ValueError("source_audio_path must not be empty.")
        if request.compiled_edit is None and not request.operations:
            raise ValueError("edit generation requires at least one operation.")
        compiled = request.compiled_edit or compile_edit(
            source_text, request.operations
        )
        settings = request.settings.normalized()
        use_xvector = _resolve_edit_request_use_xvector(
            request.use_xvector,
            compiled.instruction,
            allowed=request.xvector_allowed,
        )
        if not use_xvector:
            settings = replace(settings, speaker_scale=0.0)
        with self._lock:
            self._seed_fn(settings.seed)
            runtime, spec, model_path = self._get_runtime(request.model_id)
            with self._compiler_cache_environment(spec, model_path):
                kwargs = {
                    "source_text": source_text,
                    "target_text": compiled.target_text,
                    "source_audio_path": source_audio,
                    "instruction": compiled.instruction,
                    "use_xvector": use_xvector,
                    **self._runtime_settings(settings),
                }
                result = runtime.generate_edit(
                    **kwargs,
                )
            return self._finish_result(
                session_id=session_id,
                kind="edit",
                text=compiled.target_text,
                parent_id=request.source_revision_id,
                settings=settings,
                runtime_result=result,
                model_name_or_path=model_path,
                model_id=spec.id,
                model_label=spec.label,
                compiled_edit=compiled,
                use_xvector=use_xvector,
            )
