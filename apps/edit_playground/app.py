"""Launch the React-based DotsTTS Edit Playground."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from apps.edit_playground.constants import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME_OR_PATH,
    DEFAULT_NOISE_PRESETS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
)
from apps.edit_playground.history import GenerationHistoryStore  # noqa: E402
from apps.edit_playground.recognition import (  # noqa: E402
    Qwen3AsrRecognizer,
    RecognitionJobStore,
    execute_recognition,
)
from apps.edit_playground.service import (  # noqa: E402
    ModelSpec,
    StudioService,
    _safe_session_id,  # noqa: E402
    build_service_config,
    model_spec_id,
)
from apps.edit_playground.studio_server import (  # noqa: E402
    StartupState,
    StudioSessionStore,
    create_studio_app,
)

FRONTEND_ROOT = REPO_ROOT / "apps" / "edit_playground" / "frontend"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DotsTTS Edit Playground")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Register a model; repeat to preload multiple checkpoints.",
    )
    parser.add_argument("--default-model", default=None, metavar="LABEL_OR_ID")
    parser.add_argument("--compiler-cache-root", type=Path, default=None)
    parser.add_argument(
        "--noise-presets-dir",
        type=Path,
        default=DEFAULT_NOISE_PRESETS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-retention-count", type=int, default=DEFAULT_OUTPUT_RETENTION
    )
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable runtime optimization (default: enabled).",
    )
    parser.add_argument(
        "--max-generate-length", type=int, default=DEFAULT_MAX_GENERATE_LENGTH
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Enable Recognition with a local path or Hugging Face Qwen3-ASR model.",
    )
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument(
        "--uploads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow users to upload audio and noise files (default: enabled).",
    )
    parser.add_argument(
        "--rebuild-frontend",
        action="store_true",
        help=(
            "Install frontend dependencies and rebuild the bundle with Node.js 20+."
        ),
    )
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=None,
        help="Persist generation history under this directory.",
    )
    parser.add_argument("--history-retention-days", type=int, default=90)
    args = parser.parse_args(argv)
    if args.model and args.model_name_or_path:
        parser.error("--model cannot be combined with --model-name-or-path")
    if not args.uploads and args.asr_model:
        parser.error("--asr-model requires uploads; remove --no-uploads.")
    if not args.model and not args.model_name_or_path:
        args.model_name_or_path = DEFAULT_MODEL_NAME_OR_PATH
    return args


def parse_model_specs(values: list[str]) -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    for value in values:
        label, separator, path = value.partition("=")
        label = label.strip()
        path = path.strip()
        if not separator or not label or not path:
            raise ValueError("--model must use LABEL=PATH.")
        specs.append(
            ModelSpec(
                id=model_spec_id(label, path),
                label=label,
                model_name_or_path=path,
            )
        )
    return tuple(specs)


def resolve_default_model_id(
    specs: tuple[ModelSpec, ...],
    requested: str | None,
) -> str | None:
    if not specs:
        return None
    if requested is None:
        return specs[0].id
    for spec in specs:
        if requested in {spec.id, spec.label}:
            return spec.id
    raise ValueError(f"Unknown default model: {requested!r}")


def frontend_fingerprint(root: Path = FRONTEND_ROOT) -> str:
    digest = hashlib.sha256()
    candidates = [root / "package.json", root / "package-lock.json"]
    candidates.extend(sorted((root / "src").rglob("*")))
    for path in candidates:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_frontend_built(root: Path = FRONTEND_ROOT) -> Path:
    dist = root / "dist"
    stamp = dist / ".source-fingerprint"
    fingerprint = frontend_fingerprint(root)
    if (dist / "index.html").is_file() and stamp.is_file():
        if stamp.read_text(encoding="utf-8").strip() == fingerprint:
            return dist
    if not shutil_which("npm"):
        raise RuntimeError(
            "The edit_playground frontend needs Node.js and npm. "
            "Install Node 20+ and rerun."
        )
    lockfile = root / "package-lock.json"
    dependency_stamp = root / "node_modules" / ".dots-package-lock"
    lock_fingerprint = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    dependencies_current = (
        dependency_stamp.is_file()
        and dependency_stamp.read_text(encoding="utf-8").strip() == lock_fingerprint
    )
    if not dependencies_current:
        subprocess.run(["npm", "ci"], cwd=root, check=True)
        dependency_stamp.write_text(lock_fingerprint, encoding="utf-8")
    subprocess.run(["npm", "run", "build"], cwd=root, check=True)
    dist.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fingerprint, encoding="utf-8")
    return dist


def resolve_frontend_dist(
    *,
    rebuild: bool,
    root: Path = FRONTEND_ROOT,
) -> Path:
    """Return a usable frontend bundle, building it only when explicitly requested."""

    if rebuild:
        return ensure_frontend_built(root)
    dist = root / "dist"
    index = dist / "index.html"
    if index.is_file():
        return dist
    raise RuntimeError(
        f"Frontend build is missing: {index}. Build it with "
        f"`cd {root} && npm ci && npm run build`, or restart with "
        "--rebuild-frontend. Node.js 20+ is required."
    )


def shutil_which(program: str) -> str | None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for raw in paths:
        candidate = Path(raw) / program
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_specs = parse_model_specs(args.model)
    default_model_id = resolve_default_model_id(model_specs, args.default_model)
    frontend_dist = resolve_frontend_dist(rebuild=args.rebuild_frontend)
    service = StudioService(
        build_service_config(
            model_name_or_path=args.model_name_or_path or DEFAULT_MODEL_NAME_OR_PATH,
            models=model_specs,
            default_model_id=default_model_id,
            compiler_cache_root=args.compiler_cache_root,
            noise_presets_dir=args.noise_presets_dir,
            output_dir=args.output_dir,
            output_retention_count=args.output_retention_count,
            precision=args.precision,
            optimize=args.optimize,
            max_generate_length=args.max_generate_length,
            max_sequence_length=args.max_sequence_length,
            host=args.host,
            port=args.port,
        )
    )
    startup = StartupState(
        frontend="ready",
        model="ready" if args.skip_warmup else "warming",
        warmup="skipped" if args.skip_warmup else "running",
    )

    def warm_model() -> None:
        try:
            service.warmup_all()
        except Exception as exc:  # surfaced in the loaded application
            startup.set("model", "error", str(exc))
            startup.set_warmup("error")
        else:
            startup.set_warmup("complete")
            startup.set("model", "ready")

    if not args.skip_warmup:
        threading.Thread(target=warm_model, daemon=True, name="studio-model").start()

    recognition_jobs = None
    recognizer = None
    if args.asr_model:
        recognizer = Qwen3AsrRecognizer(args.asr_model, device=args.asr_device)
        recognition_jobs = RecognitionJobStore(
            args.output_dir / ".recognition-jobs",
            save_upload=StudioSessionStore._store_upload,
            validate_duration=StudioSessionStore._validate_uploaded_duration,
            safe_session_id=_safe_session_id,
        )

    def recognize(job_id: str) -> str:
        assert recognition_jobs is not None and recognizer is not None
        return execute_recognition(job_id, recognition_jobs, recognizer)

    history_store = (
        GenerationHistoryStore(
            args.history_root,
            retention_days=args.history_retention_days,
        )
        if args.history_root is not None
        else None
    )
    if history_store is not None:
        history_store.cleanup_expired()

    app, _ = create_studio_app(
        service,
        frontend_dist=frontend_dist,
        startup_state=startup,
        recognition_jobs=recognition_jobs,
        gpu_recognition_handler=recognize if recognition_jobs else None,
        history_store=history_store,
        uploads_enabled=args.uploads,
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=args.inbrowser,
        show_error=True,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
