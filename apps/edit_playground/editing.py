"""Pure construction of DotsTTS structured edit instructions.

Positions are Unicode code-point offsets into ``source_text``.  This module does
not accept an instruction string from the caller: tags are always constructed
from validated, structured operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from dots_tts.data.edit_instruction import (
    TaggedTextContractError,
    escape_text,
    normalize_target_parts,
    validate_literal_text,
)

EMOTION_TYPES = frozenset(
    {"happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"}
)
PAUSE_ACTIONS = frozenset({"ins", "red"})
PAUSE_LEVELS = frozenset({"1", "2", "3"})
PITCH_SEMITONES = frozenset({-9, -7, -5, 5, 7, 9})
EMOTION_LEVELS = frozenset({1, 2, 3})
DEFAULT_EMOTION_LEVEL = 2

_KIND_ALIASES = {
    "sub": "replace",
    "replacement": "replace",
    "del": "delete",
    "deletion": "delete",
    "ins": "insert",
    "insertion": "insert",
    "emo": "emotion",
    "edit_emotion": "emotion",
    "edit_text_replace": "replace",
    "edit_text_delete": "delete",
    "edit_text_insert": "insert",
    "edit_pitch": "pitch",
    "edit_rate": "rate",
    "edit_pause": "pause",
}
_SPAN_KINDS = frozenset({"replace", "delete", "emotion", "pitch", "rate"})
_POINT_KINDS = frozenset({"insert", "pause"})
_ALL_KINDS = _SPAN_KINDS | _POINT_KINDS


class EditValidationError(ValueError):
    """Raised when structured operations cannot be safely compiled."""


def _freeze_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(params))


@dataclass(frozen=True, slots=True)
class EditOperation:
    """One edit in source-text coordinates (a half-open ``[start, end)`` span)."""

    id: str
    kind: str
    start: int
    end: int
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _KIND_ALIASES.get(self.kind, self.kind))
        object.__setattr__(self, "params", _freeze_params(self.params))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EditOperation":
        return cls(
            id=str(value.get("id", "")),
            kind=str(value.get("kind", "")),
            start=value.get("start"),  # type: ignore[arg-type]
            end=value.get("end", value.get("start")),  # type: ignore[arg-type]
            params=value.get("params") or {},
        )


@dataclass(frozen=True, slots=True)
class CompiledEdit:
    """Plain target text and the model-facing structured instruction."""

    target_text: str
    instruction: str


def _coerce_operations(
    operations: Iterable[EditOperation | Mapping[str, Any]],
) -> tuple[EditOperation, ...]:
    return tuple(
        operation
        if isinstance(operation, EditOperation)
        else EditOperation.from_mapping(operation)
        for operation in operations
    )


def _required_text(operation: EditOperation, *keys: str) -> str:
    for key in keys:
        if key in operation.params:
            value = operation.params[key]
            if isinstance(value, str) and value:
                return value
            raise EditValidationError(
                f"{operation.kind} parameter {key!r} must be non-empty text"
            )
    raise EditValidationError(f"{operation.kind} requires parameter {keys[0]!r}")


def _validate_literal(value: str, *, label: str) -> None:
    try:
        validate_literal_text(value, label=label)
    except TaggedTextContractError as exc:
        raise EditValidationError(str(exc)) from exc


def _validate_parameters(operation: EditOperation) -> None:
    kind = operation.kind
    params = operation.params
    if kind == "replace":
        target = _required_text(operation, "target", "targ", "text")
        _validate_literal(target, label="Replacement text")
    elif kind == "insert":
        target = _required_text(operation, "text", "target", "targ")
        _validate_literal(target, label="Inserted text")
    elif kind == "delete":
        return
    elif kind == "emotion":
        desc = params.get("desc", params.get("description"))
        emotion_type = params.get("type")
        if desc is not None:
            if not isinstance(desc, str) or not desc.strip():
                raise EditValidationError("emotion desc must be non-empty text")
            if emotion_type is not None or "level" in params:
                raise EditValidationError(
                    "emotion desc cannot be combined with type or level"
                )
            return
        if emotion_type not in EMOTION_TYPES:
            raise EditValidationError(f"unsupported emotion type: {emotion_type!r}")
        level = params.get("level", DEFAULT_EMOTION_LEVEL)
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level not in EMOTION_LEVELS
        ):
            raise EditValidationError("emotion level must be 1, 2, or 3")
    elif kind == "pitch":
        value = params.get("semitones")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in PITCH_SEMITONES
        ):
            raise EditValidationError(
                "pitch semitones must be one of -9, -7, -5, 5, 7, or 9"
            )
    elif kind == "rate":
        value = params.get("factor")
        try:
            factor = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise EditValidationError("rate factor must be numeric") from None
        if not factor.is_finite():
            raise EditValidationError("rate factor must be finite")
        if factor * 100 != (factor * 100).to_integral_value():
            raise EditValidationError("rate factor must use 0.01 increments")
        if not (Decimal("0.55") <= factor <= Decimal("1.95")) or factor == Decimal(
            "1.00"
        ):
            raise EditValidationError("rate factor must be 0.55..0.99 or 1.01..1.95")
    elif kind == "pause":
        action = params.get("act", params.get("action"))
        if action not in PAUSE_ACTIONS:
            raise EditValidationError(f"unsupported pause action: {action!r}")
        level = str(params.get("level", ""))
        if level not in PAUSE_LEVELS:
            raise EditValidationError(f"unsupported pause level: {level!r}")


def validate_operations(
    source_text: str,
    operations: Iterable[EditOperation | Mapping[str, Any]],
) -> tuple[EditOperation, ...]:
    """Validate and return operations in deterministic source order.

    A point operation may sit on a span boundary, but not strictly inside a
    span.  Two point operations at the same coordinate are rejected so their
    generated ordering can never be ambiguous.
    """

    if not isinstance(source_text, str):
        raise EditValidationError("source_text must be text")
    _validate_literal(source_text, label="Source Transcript")
    result = _coerce_operations(operations)
    ids: set[str] = set()
    source_length = len(source_text)
    for operation in result:
        if not operation.id:
            raise EditValidationError("every operation requires a non-empty id")
        if operation.id in ids:
            raise EditValidationError(f"duplicate operation id: {operation.id!r}")
        ids.add(operation.id)
        if operation.kind not in _ALL_KINDS:
            raise EditValidationError(f"unsupported edit kind: {operation.kind!r}")
        if (
            isinstance(operation.start, bool)
            or isinstance(operation.end, bool)
            or not isinstance(operation.start, int)
            or not isinstance(operation.end, int)
        ):
            raise EditValidationError("operation positions must be integers")
        if not 0 <= operation.start <= operation.end <= source_length:
            raise EditValidationError(
                f"operation {operation.id!r} is outside source text (length {source_length})"
            )
        if operation.kind in _SPAN_KINDS and operation.start == operation.end:
            raise EditValidationError(
                f"{operation.kind} requires a non-empty selection"
            )
        if operation.kind in _POINT_KINDS and operation.start != operation.end:
            raise EditValidationError(
                f"{operation.kind} must be placed at one cursor position"
            )
        _validate_parameters(operation)

    spans = sorted(
        (op for op in result if op.kind in _SPAN_KINDS),
        key=lambda op: (op.start, op.end),
    )
    for previous, current in zip(spans, spans[1:]):
        if current.start < previous.end:
            raise EditValidationError(
                f"operations {previous.id!r} and {current.id!r} overlap or nest"
            )

    point_positions: set[int] = set()
    for point in (op for op in result if op.kind in _POINT_KINDS):
        if point.start in point_positions:
            raise EditValidationError(
                f"multiple point operations at position {point.start}"
            )
        point_positions.add(point.start)
        for span in spans:
            if span.start < point.start < span.end:
                raise EditValidationError(
                    f"point operation {point.id!r} lies inside span {span.id!r}"
                )

    # Points precede a span that begins at the same cursor position.  Stable IDs
    # make the result independent of UI list order even though duplicate points
    # are already forbidden.
    return tuple(
        sorted(
            result,
            key=lambda op: (
                op.start,
                0 if op.kind in _POINT_KINDS else 1,
                op.end,
                op.id,
            ),
        )
    )


def _xml(value: Any) -> str:
    return escape_text(str(value))


def _format_rate(value: Any) -> str:
    formatted = format(Decimal(str(value)), "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _compile_operation(operation: EditOperation, selected: str) -> tuple[str, str]:
    """Return ``(instruction_fragment, target_fragment)``."""

    params = operation.params
    escaped_selected = _xml(selected)
    if operation.kind == "replace":
        target = _required_text(operation, "target", "targ", "text")
        return f'<sub targ="{_xml(target)}">{escaped_selected}</sub>', target
    if operation.kind == "insert":
        target = _required_text(operation, "text", "target", "targ")
        return f"<ins>{_xml(target)}</ins>", target
    if operation.kind == "delete":
        return f"<del>{escaped_selected}</del>", ""
    if operation.kind == "emotion":
        desc = params.get("desc", params.get("description"))
        if desc is not None:
            return f'<emo desc="{_xml(desc)}">{escaped_selected}</emo>', selected
        level = params.get("level", DEFAULT_EMOTION_LEVEL)
        return (
            f'<emo type="{_xml(params["type"])}" level="{level}">'
            f"{escaped_selected}</emo>",
            selected,
        )
    if operation.kind == "pitch":
        return (
            f"<pitch, semitones={params['semitones']}>{escaped_selected}</pitch>",
            selected,
        )
    if operation.kind == "rate":
        return (
            f"<rate, factor={_format_rate(params['factor'])}>{escaped_selected}</rate>",
            selected,
        )
    if operation.kind == "pause":
        action = params.get("act", params.get("action"))
        return f'<pause act="{action}" level="{params["level"]}"/>', ""
    raise AssertionError(f"unreachable edit kind: {operation.kind}")


def compile_edit(
    source_text: str,
    operations: Sequence[EditOperation | Mapping[str, Any]],
) -> CompiledEdit:
    """Compile validated operations into target text and a tagged instruction."""

    ordered = validate_operations(source_text, operations)
    instruction_parts: list[str] = []
    cursor = 0
    for operation in ordered:
        if operation.start > cursor:
            plain = source_text[cursor : operation.start]
            instruction_parts.append(_xml(plain))
            cursor = operation.start
        selected = source_text[operation.start : operation.end]
        instruction, _ = _compile_operation(operation, selected)
        instruction_parts.append(instruction)
        cursor = operation.end
    if cursor < len(source_text):
        plain = source_text[cursor:]
        instruction_parts.append(_xml(plain))
    has_text_edit = any(
        operation.kind in {"replace", "delete", "insert"} for operation in ordered
    )
    return CompiledEdit(
        target_text=(
            normalize_target_parts(_target_surface_parts(source_text, ordered))
            if has_text_edit
            else source_text
        ),
        instruction="".join(instruction_parts),
    )


def _target_surface_parts(
    source_text: str, operations: Sequence[EditOperation]
) -> list[str]:
    """Project only text edits, keeping non-text tags transparent to spacing."""

    parts: list[str] = []
    unchanged: list[str] = []
    cursor = 0
    for operation in operations:
        if operation.start > cursor:
            unchanged.append(source_text[cursor : operation.start])
        selected = source_text[operation.start : operation.end]
        if operation.kind in {"replace", "delete", "insert"}:
            if unchanged:
                parts.append("".join(unchanged))
                unchanged.clear()
            if operation.kind == "replace":
                parts.append(_required_text(operation, "target", "targ", "text"))
            elif operation.kind == "insert":
                parts.append(_required_text(operation, "text", "target", "targ"))
        else:
            unchanged.append(selected)
        cursor = operation.end
    if cursor < len(source_text):
        unchanged.append(source_text[cursor:])
    if unchanged:
        parts.append("".join(unchanged))
    return parts
