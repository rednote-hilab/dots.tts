"""Temporary upload jobs and Qwen3-ASR inference for the playground."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import UploadFile


class Recognizer(Protocol):
    def transcribe(self, audio_path: str) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class RecognitionJob:
    id: str
    session_id: str
    audio_path: str


class RecognitionJobStore:
    """Own short-lived recognition uploads without exposing filesystem paths."""

    def __init__(
        self,
        root: Path,
        *,
        save_upload: Any,
        validate_duration: Any,
        safe_session_id: Any,
    ) -> None:
        self.root = root
        self._save_upload = save_upload
        self._validate_duration = validate_duration
        self._safe_session_id = safe_session_id
        self._jobs: dict[str, RecognitionJob] = {}
        self._lock = threading.RLock()

    def create(self, session_id: str, upload: UploadFile) -> dict[str, str]:
        safe_session = self._safe_session_id(session_id)
        suffix = Path(upload.filename or "audio.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}:
            suffix = ".wav"
        job_id = uuid.uuid4().hex
        directory = self.root / safe_session
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{job_id}{suffix}"
        self._save_upload(upload, output)
        try:
            self._validate_duration(output)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        with self._lock:
            self._jobs[job_id] = RecognitionJob(
                id=job_id,
                session_id=safe_session,
                audio_path=str(output),
            )
        return {"job_id": job_id}

    def resolve(self, job_id: str) -> RecognitionJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise ValueError("Unknown or expired recognition job.") from exc

    def cancel(self, session_id: str, job_id: str) -> None:
        safe_session = self._safe_session_id(session_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.session_id != safe_session:
                raise ValueError("Unknown or expired recognition job.")
            self._jobs.pop(job_id)
        Path(job.audio_path).unlink(missing_ok=True)


class Qwen3AsrRecognizer:
    """Thin Transformers-backend adapter; vLLM is intentionally not used."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            model_name_or_path,
            # qwen-asr 0.0.6 leaves the audio feature tensor in FP32. Loading
            # the projection weights as BF16 therefore fails at inference with
            # a matmul dtype mismatch on L20/A10 GPUs.
            dtype=torch.float32,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio_path: str) -> dict[str, str]:
        result = self.model.transcribe(
            audio=audio_path,
            language=None,
            return_time_stamps=False,
        )[0]
        text = str(result.text).strip()
        if not text:
            raise ValueError("No speech was recognized.")
        return {"text": text, "language": str(result.language).strip()}


def execute_recognition(
    job_id: str,
    jobs: RecognitionJobStore,
    recognizer: Recognizer,
) -> str:
    job = jobs.resolve(job_id)
    return json.dumps(recognizer.transcribe(job.audio_path), ensure_ascii=False)
