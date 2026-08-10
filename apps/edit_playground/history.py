"""Filesystem-backed, storage-agnostic playground generation history."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from apps.edit_playground.service import GenerationResult
from apps.edit_playground.state import AudioReference

_EVENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def history_allowed(reference: AudioReference | None) -> bool:
    """Return whether every user-uploaded dependency consented to retention."""

    if reference is None:
        return True
    return all(segment.history_consent for segment in reference.segments) and (
        reference.noise_overlay is None or reference.noise_overlay.history_consent
    )


class GenerationHistoryStore:
    """Persist immutable generation events under a local or mounted root."""

    def __init__(self, root: Path, *, retention_days: int = 90) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(retention_days)
        if self.retention_days <= 0:
            raise ValueError("history retention days must be positive")
        self._lock = threading.RLock()

    def event_dir(self, event_id: str) -> Path:
        if not _EVENT_ID_RE.fullmatch(event_id):
            raise ValueError("invalid history event id")
        return self.root / event_id

    @staticmethod
    def _copy_asset(source: str, destination: Path) -> dict[str, Any]:
        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return {
            "path": destination.name,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }

    @staticmethod
    def _reference_payload(reference: AudioReference | None) -> dict[str, Any] | None:
        if reference is None:
            return None
        return {
            "text": reference.text,
            "revision_id": reference.revision_id,
            "duration_seconds": reference.duration_seconds,
            "use_xvector": reference.use_xvector,
            "noise": (
                {
                    "kind": reference.noise_overlay.kind,
                    "item_id": reference.noise_overlay.item_id,
                    "snr_db": reference.noise_overlay.snr_db,
                    "crop_start": reference.noise_overlay.crop_start,
                    "tiled": reference.noise_overlay.tiled,
                }
                if reference.noise_overlay is not None
                else None
            ),
            "segments": [
                {
                    "id": segment.id,
                    "transcript": segment.original_transcript,
                    "duration_seconds": segment.duration_seconds,
                    "revision_id": segment.revision_id,
                    "multiple_speakers": segment.multiple_speakers,
                    "allow_xvector": segment.allow_xvector,
                }
                for segment in reference.segments
            ],
        }

    def record_success(
        self,
        *,
        event_id: str,
        session_id: str,
        result: GenerationResult,
        prompt: AudioReference | None,
        source: AudioReference | None,
    ) -> Path:
        """Write one complete event once; an existing event is idempotent."""

        destination = self.event_dir(event_id)
        manifest_path = destination / "manifest.json"
        with self._lock:
            if manifest_path.is_file():
                return manifest_path
            temporary = self.root / f".{event_id}.{os.getpid()}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            try:
                assets: dict[str, Any] = {
                    "output": self._copy_asset(
                        result.audio_path, temporary / "output.wav"
                    )
                }
                if source is not None:
                    assets["source"] = self._copy_asset(
                        source.audio_path, temporary / "source.wav"
                    )
                if prompt is not None:
                    assets["prompt"] = self._copy_asset(
                        prompt.audio_path, temporary / "prompt.wav"
                    )
                compiled = result.compiled_edit
                manifest = {
                    "schema_version": 1,
                    "event_id": event_id,
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                    "kind": str(result.metadata.get("kind", "")),
                    "model": {
                        "id": result.metadata.get("model_id"),
                        "label": result.metadata.get("model_label"),
                        "path": result.metadata.get("model"),
                    },
                    "generation": _plain(result.metadata.get("generation", {})),
                    "metrics": {
                        key: result.metadata[key]
                        for key in (
                            "duration_seconds",
                            "sample_rate",
                            "elapsed_seconds",
                            "rtf",
                        )
                        if key in result.metadata
                    },
                    "prompt": self._reference_payload(prompt),
                    "source": self._reference_payload(source),
                    "target_text": result.text,
                    "instruction": compiled.instruction if compiled else None,
                    "assets": assets,
                }
                encoded = (
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8")
                (temporary / "manifest.json").write_bytes(encoded)
                os.replace(temporary, destination)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return manifest_path

    def read(self, event_id: str) -> dict[str, Any]:
        path = self.event_dir(event_id) / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("event_id") != event_id:
            raise ValueError("invalid history manifest")
        return value

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        threshold = (now or datetime.now(timezone.utc)) - timedelta(
            days=self.retention_days
        )
        removed = 0
        with self._lock:
            for path in self.root.iterdir():
                if not path.is_dir() or not _EVENT_ID_RE.fullmatch(path.name):
                    continue
                manifest_path = path / "manifest.json"
                try:
                    value = json.loads(manifest_path.read_text(encoding="utf-8"))
                    created = datetime.fromisoformat(str(value["created_at"]))
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                if created < threshold:
                    shutil.rmtree(path)
                    removed += 1
        return removed
