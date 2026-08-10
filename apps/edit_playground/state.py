"""Immutable, session-local state for Gradio v2 audio revisions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

DESTINATIONS = frozenset({"prompt", "edit_source"})
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "ancestor",
        "ancestors",
        "child",
        "children",
        "instruction",
        "operations",
        "parent",
        "parent_node",
        "prompt_audio",
        "prompt_text",
        "source_audio",
        "source_text",
    }
)


class StateValidationError(ValueError):
    """Raised for invalid session routing or revision state."""


class FrozenDict(Mapping[str, Any]):
    """Small read-only mapping that remains compatible with ``gr.State``.

    ``types.MappingProxyType`` is read-only but cannot be deep-copied.  Gradio
    deep-copies a state's initial value, so immutable session objects use this
    mapping instead.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._values = dict(values or {})

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"FrozenDict({self._values!r})"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._values.items())))

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict":
        # Values have already been recursively frozen into immutable objects.
        memo[id(self)] = self
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _validate_metadata(value: Any) -> None:
    """Keep generation metadata local to the node that owns it."""

    if isinstance(
        value,
        (
            AudioSegment,
            AudioReference,
            UploadedAudio,
            NoiseUpload,
            NoiseOverlay,
            RevisionNode,
        ),
    ):
        raise StateValidationError(
            "revision metadata cannot embed audio references or nodes"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_METADATA_KEYS:
                raise StateValidationError(
                    f"revision metadata cannot contain {str(key)!r}"
                )
            _validate_metadata(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_metadata(item)


@dataclass(frozen=True, slots=True)
class NoiseOverlay:
    item_id: str
    kind: str
    noise_path: str
    snr_db: int
    crop_start: int = 0
    tiled: bool = False
    history_consent: bool = True

    def __post_init__(self) -> None:
        if not self.item_id or not self.noise_path:
            raise StateValidationError("noise overlay requires an item and audio path")
        if self.kind not in {"presets", "uploads"}:
            raise StateValidationError("unknown noise overlay kind")
        if not 0 <= int(self.snr_db) <= 20:
            raise StateValidationError("noise SNR must be between 0 and 20 dB")


@dataclass(frozen=True, slots=True)
class NoiseUpload:
    id: str
    name: str
    audio_path: str
    duration_seconds: float
    history_consent: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.id or not self.name or not self.audio_path:
            raise StateValidationError("noise upload requires id, name and audio path")
        if self.duration_seconds <= 0:
            raise StateValidationError("noise upload duration must be positive")


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One immutable source component and its ingestion-time transcript."""

    id: str
    audio_path: str
    original_transcript: str
    duration_seconds: float = 0.0
    revision_id: str | None = None
    multiple_speakers: bool = False
    allow_xvector: bool = True
    history_consent: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.audio_path:
            raise StateValidationError("audio segment requires id and audio_path")
        if not isinstance(self.original_transcript, str):
            raise StateValidationError("segment transcript must be a string")
        if self.duration_seconds < 0:
            raise StateValidationError("segment duration must be non-negative")


def _legacy_segment_id(audio_path: str, revision_id: str | None) -> str:
    import hashlib

    value = f"{audio_path}\0{revision_id or ''}"
    digest = hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"segment-{digest}"


@dataclass(frozen=True, slots=True)
class AudioReference:
    """Ephemeral audio/text pair held by an input slot."""

    audio_path: str
    text: str
    revision_id: str | None = None
    segments: tuple[AudioSegment, ...] = ()
    clean_audio_path: str | None = None
    noise_overlay: NoiseOverlay | None = None
    preset_name: str | None = None

    def __post_init__(self) -> None:
        if not self.audio_path:
            raise StateValidationError("audio reference requires audio_path")
        if not isinstance(self.text, str):
            raise StateValidationError("audio reference text must be a string")
        if not self.segments:
            object.__setattr__(
                self,
                "segments",
                (
                    AudioSegment(
                        id=_legacy_segment_id(self.audio_path, self.revision_id),
                        audio_path=self.audio_path,
                        original_transcript=self.text,
                        revision_id=self.revision_id,
                    ),
                ),
            )
        if self.clean_audio_path is None:
            object.__setattr__(self, "clean_audio_path", self.audio_path)
        if len(self.segments) > 3:
            raise StateValidationError("Edit Source supports at most 3 segments")
        if len({segment.id for segment in self.segments}) != len(self.segments):
            raise StateValidationError("audio segment ids must be unique")

    @property
    def duration_seconds(self) -> float:
        return sum(segment.duration_seconds for segment in self.segments)

    @property
    def use_xvector(self) -> bool:
        return (
            len(self.segments) == 1
            and self.segments[0].allow_xvector
            and not self.segments[0].multiple_speakers
            and self.noise_overlay is None
        )

    @property
    def composition_version(self) -> str:
        import hashlib

        value = "\0".join(
            [
                self.audio_path,
                str(self.clean_audio_path),
                self.text,
                str(self.noise_overlay),
                str(self.preset_name),
            ]
            + [
                f"{item.id}:{item.audio_path}:{item.original_transcript}:"
                f"{item.multiple_speakers}:{item.allow_xvector}:"
                f"{item.history_consent}"
                for item in self.segments
            ]
        )
        return hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class UploadedAudio:
    """One reusable upload owned by a browser session."""

    id: str
    name: str
    audio_path: str
    transcript: str
    duration_seconds: float
    multiple_speakers: bool = False
    allow_xvector: bool = True
    history_consent: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.id or not self.name or not self.audio_path:
            raise StateValidationError("uploaded audio requires id and audio_path")
        if not self.transcript.strip():
            raise StateValidationError("uploaded audio transcript must not be empty")
        if self.duration_seconds < 0:
            raise StateValidationError("uploaded audio duration must be non-negative")
        if self.multiple_speakers and self.allow_xvector:
            object.__setattr__(self, "allow_xvector", False)

    def as_reference(self) -> AudioReference:
        segment = AudioSegment(
            id=uuid4().hex,
            audio_path=self.audio_path,
            original_transcript=self.transcript,
            duration_seconds=self.duration_seconds,
            multiple_speakers=self.multiple_speakers,
            allow_xvector=self.allow_xvector,
            history_consent=self.history_consent,
        )
        return AudioReference(
            audio_path=self.audio_path,
            text=self.transcript,
            segments=(segment,),
        )


@dataclass(frozen=True, slots=True)
class RevisionNode:
    """One immutable Markov node; it never embeds another node or its inputs."""

    id: str
    audio_path: str
    text: str
    parent_id: str | None
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.audio_path:
            raise StateValidationError("revision requires id and audio_path")
        if not isinstance(self.text, str):
            raise StateValidationError("revision text must be a string")
        _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))

    @classmethod
    def create(
        cls,
        *,
        audio_path: str,
        text: str,
        parent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        revision_id: str | None = None,
        created_at: str | None = None,
    ) -> "RevisionNode":
        return cls(
            id=revision_id or uuid4().hex,
            audio_path=audio_path,
            text=text,
            parent_id=parent_id,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def as_reference(self) -> AudioReference:
        return AudioReference(
            audio_path=self.audio_path,
            text=self.text,
            revision_id=self.id,
            segments=(
                AudioSegment(
                    id=_legacy_segment_id(self.audio_path, self.id),
                    audio_path=self.audio_path,
                    original_transcript=self.text,
                    revision_id=self.id,
                    history_consent=bool(self.metadata.get("history_consent", True)),
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class SessionState:
    """State owned by exactly one browser session."""

    revisions: tuple[RevisionNode, ...] = ()
    uploads: tuple[UploadedAudio, ...] = ()
    noise_uploads: tuple[NoiseUpload, ...] = ()
    prompt: AudioReference | None = None
    edit_source: AudioReference | None = None
    selected_model_id: str | None = None

    def find_revision(self, revision_id: str) -> RevisionNode:
        for revision in self.revisions:
            if revision.id == revision_id:
                return revision
        raise StateValidationError(f"unknown revision: {revision_id!r}")

    def find_upload(self, upload_id: str) -> UploadedAudio:
        for upload in self.uploads:
            if upload.id == upload_id:
                return upload
        raise StateValidationError(f"unknown upload: {upload_id!r}")

    def add_upload(self, upload: UploadedAudio) -> "SessionState":
        if any(existing.id == upload.id for existing in self.uploads):
            raise StateValidationError(f"duplicate upload id: {upload.id!r}")
        return replace(self, uploads=self.uploads + (upload,))

    def find_noise_upload(self, upload_id: str) -> NoiseUpload:
        for upload in self.noise_uploads:
            if upload.id == upload_id:
                return upload
        raise StateValidationError(f"unknown noise upload: {upload_id!r}")

    def add_noise_upload(self, upload: NoiseUpload) -> "SessionState":
        if any(existing.id == upload.id for existing in self.noise_uploads):
            raise StateValidationError(f"duplicate noise upload id: {upload.id!r}")
        return replace(self, noise_uploads=self.noise_uploads + (upload,))

    def resolve_reference(self, source: str) -> AudioReference:
        if source == "prompt":
            if self.prompt is None:
                raise StateValidationError("prompt slot is empty")
            return self.prompt
        if source == "edit_source":
            if self.edit_source is None:
                raise StateValidationError("edit source slot is empty")
            return self.edit_source
        if self.edit_source is not None:
            for segment in self.edit_source.segments:
                if segment.id == source:
                    return AudioReference(
                        audio_path=segment.audio_path,
                        text=segment.original_transcript,
                        revision_id=segment.revision_id,
                        segments=(segment,),
                    )
        return self.find_revision(source).as_reference()

    def with_reference(
        self, destination: str, reference: AudioReference | None
    ) -> "SessionState":
        if destination not in DESTINATIONS:
            raise StateValidationError(f"unknown audio destination: {destination!r}")
        return replace(self, **{destination: reference})

    def route(self, source: str, destination: str) -> "SessionState":
        """Atomically copy both audio and text to an input destination."""

        reference = self.resolve_reference(source)
        if destination == "prompt" and len(reference.segments) > 1:
            flattened = AudioSegment(
                id=uuid4().hex,
                audio_path=reference.audio_path,
                original_transcript=reference.text,
                duration_seconds=reference.duration_seconds,
                allow_xvector=False,
                history_consent=all(
                    segment.history_consent for segment in reference.segments
                ),
            )
            reference = AudioReference(
                reference.audio_path,
                reference.text,
                segments=(flattened,),
                clean_audio_path=reference.clean_audio_path,
                noise_overlay=reference.noise_overlay,
            )
        return self.with_reference(destination, reference)

    def bind_upload(
        self,
        destination: str,
        audio_path: str,
        text: str = "",
        *,
        duration_seconds: float = 0.0,
        multiple_speakers: bool = False,
        history_consent: bool = True,
    ) -> "SessionState":
        """Atomically bind a local upload and its transcript."""

        segment = AudioSegment(
            id=_legacy_segment_id(audio_path, None),
            audio_path=audio_path,
            original_transcript=text,
            duration_seconds=duration_seconds,
            multiple_speakers=multiple_speakers,
            allow_xvector=not multiple_speakers,
            history_consent=history_consent,
        )
        return self.with_reference(
            destination,
            AudioReference(audio_path=audio_path, text=text, segments=(segment,)),
        )

    def set_reference_text(self, destination: str, text: str) -> "SessionState":
        reference = self.resolve_reference(destination)
        segments = reference.segments
        if len(segments) == 1:
            segments = (replace(segments[0], original_transcript=text),)
        return self.with_reference(
            destination,
            AudioReference(
                audio_path=reference.audio_path,
                text=text,
                revision_id=reference.revision_id,
                segments=segments,
            ),
        )

    def add_revision(
        self, revision: RevisionNode, *, max_revisions: int = 20
    ) -> "SessionState":
        if max_revisions < 1:
            raise StateValidationError("max_revisions must be at least 1")
        if any(existing.id == revision.id for existing in self.revisions):
            raise StateValidationError(f"duplicate revision id: {revision.id!r}")
        # Parent IDs deliberately need not resolve: retained Markov nodes may
        # outlive their pruned parents.
        retained = (self.revisions + (revision,))[-max_revisions:]
        return replace(self, revisions=retained)


def route_audio_reference(
    state: SessionState, source: str, destination: str
) -> SessionState:
    """Functional wrapper used directly by Gradio callbacks."""

    return state.route(source, destination)


def remove_pruned_audio(
    previous: SessionState,
    current: SessionState,
    *,
    managed_root: str | Path,
) -> tuple[Path, ...]:
    """Delete pruned generated files, never uploads outside ``managed_root``.

    State pruning happens before this optional cleanup.  Dangling ``parent_id``
    values are intentionally retained.
    """

    root = Path(managed_root).resolve()
    retained_ids = {revision.id for revision in current.revisions}
    removed: list[Path] = []
    for revision in previous.revisions:
        if revision.id in retained_ids:
            continue
        path = Path(revision.audio_path).resolve()
        if path != root and root not in path.parents:
            continue
        if path.is_file():
            path.unlink()
            removed.append(path)
    return tuple(removed)
