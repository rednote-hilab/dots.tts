"""Structured starter operations for the built-in Edit Source presets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from apps.edit_playground.editing import EditOperation, validate_operations


@dataclass(frozen=True, slots=True)
class PresetOperationSpec:
    kind: str
    anchor: str
    params: Mapping[str, Any]
    point_after: bool = False
    segment_index: int = 0


@dataclass(frozen=True, slots=True)
class PresetNoiseSpec:
    item_id: str
    snr_db: int
    crop_start: int


@dataclass(frozen=True, slots=True)
class EditSourcePresetSpec:
    name: str
    segment_presets: tuple[str, ...]
    enhance: bool = False
    noise: PresetNoiseSpec | None = None


EDIT_SOURCE_PRESET_SPECS: tuple[EditSourcePresetSpec, ...] = (
    *(
        EditSourcePresetSpec(name, (name,))
        for name in (
            "text_en",
            "text_zh",
            "emotion_en",
            "pitch_zh",
            "rate_zh",
            "pause_en",
            "compositional_zh",
        )
    ),
    EditSourcePresetSpec(
        "noise_preservation_en",
        ("text_en",),
        noise=PresetNoiseSpec(
            item_id="noise-82c33999bde42d2b",
            snr_db=8,
            crop_start=50501,
        ),
    ),
    EditSourcePresetSpec(
        "denoising_zh",
        ("text_zh",),
        enhance=True,
        noise=PresetNoiseSpec(
            item_id="noise-82c33999bde42d2b",
            snr_db=5,
            crop_start=166114,
        ),
    ),
    EditSourcePresetSpec(
        "multi_speaker_zh",
        ("male_zh", "female_zh"),
    ),
)

EDIT_SOURCE_PRESET_BY_NAME = {item.name: item for item in EDIT_SOURCE_PRESET_SPECS}


PRESET_OPERATION_SPECS: Mapping[str, Sequence[PresetOperationSpec]] = {
    "text_en": (
        PresetOperationSpec(
            "replace",
            "wish",
            {"target": "long"},
        ),
        PresetOperationSpec("replace", "get", {"target": "obtain"}),
    ),
    "text_zh": (
        PresetOperationSpec("replace", "任何奖牌", {"target": "金牌"}),
        PresetOperationSpec(
            "insert",
            "。",
            {"text": "但多名选手进入了决赛。"},
            point_after=True,
        ),
    ),
    "emotion_en": (
        PresetOperationSpec(
            "emotion",
            "To be or not to be, that is the question; whether 'tis nobler in the mind to suffer the slings and arrows-What? No, Hamlet speaking.",
            {"type": "melancholic", "level": 2},
        ),
    ),
    "pitch_zh": (
        PresetOperationSpec("pitch", "就来谈谈", {"semitones": -5}),
        PresetOperationSpec("pitch", "种族问题", {"semitones": 9}),
    ),
    "rate_zh": (
        PresetOperationSpec("rate", "九四零七五一幺九七三九", {"factor": 1.19}),
    ),
    "pause_en": (
        PresetOperationSpec(
            "pause",
            "I shouldn't wonder- ",
            {"act": "red", "level": 2},
            point_after=True,
        ),
    ),
    "compositional_zh": (
        PresetOperationSpec(
            "pause",
            "法团",
            {"act": "ins", "level": 2},
            point_after=True,
        ),
        PresetOperationSpec(
            "replace",
            "向时任特首的梁振英",
            {"target": "向相关部门"},
        ),
    ),
    "noise_preservation_en": (
        PresetOperationSpec("replace", "wish", {"target": "hope"}),
    ),
    "multi_speaker_zh": (
        PresetOperationSpec("replace", "红头发", {"target": "蓝头发"}),
    ),
}


def build_preset_operations(preset_name: str, source_text: str) -> list[dict[str, Any]]:
    """Resolve stable text anchors into validated source-code-point operations."""

    operations: list[EditOperation] = []
    for index, spec in enumerate(PRESET_OPERATION_SPECS.get(preset_name, ())):
        if spec.anchor:
            start = source_text.find(spec.anchor)
            if start < 0 or source_text.find(spec.anchor, start + 1) >= 0:
                return []
            end = start + len(spec.anchor)
        else:
            start = end = 0
        if spec.kind in {"insert", "pause"}:
            position = end if spec.point_after else start
            start = end = position
        operations.append(
            EditOperation(
                id=f"preset-{preset_name}-{index + 1}",
                kind=spec.kind,
                start=start,
                end=end,
                params=spec.params,
            )
        )
    validated = validate_operations(source_text, operations)
    return [
        {
            "id": operation.id,
            "kind": operation.kind,
            "start": operation.start,
            "end": operation.end,
            "params": dict(operation.params),
        }
        for operation in validated
    ]


def build_segmented_preset_operations(
    preset_name: str,
    segment_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Resolve preset operations while preserving their segment ownership."""

    resolved: list[dict[str, Any]] = []
    for index, spec in enumerate(PRESET_OPERATION_SPECS.get(preset_name, ())):
        if not 0 <= spec.segment_index < len(segment_texts):
            return []
        source_text = segment_texts[spec.segment_index]
        if spec.anchor:
            start = source_text.find(spec.anchor)
            if start < 0 or source_text.find(spec.anchor, start + 1) >= 0:
                return []
            end = start + len(spec.anchor)
        else:
            start = end = 0
        if spec.kind in {"insert", "pause"}:
            position = end if spec.point_after else start
            start = end = position
        operation = EditOperation(
            id=f"preset-{preset_name}-{index + 1}",
            kind=spec.kind,
            start=start,
            end=end,
            params=spec.params,
        )
        validated = validate_operations(source_text, [operation])
        item = validated[0]
        resolved.append(
            {
                "id": item.id,
                "kind": item.kind,
                "start": item.start,
                "end": item.end,
                "params": dict(item.params),
                "segment_index": spec.segment_index,
            }
        )
    return resolved
