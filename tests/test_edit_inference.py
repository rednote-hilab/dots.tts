from __future__ import annotations

from collections import OrderedDict
from types import MethodType, SimpleNamespace

import pytest
import soundfile as sf
import torch
from safetensors.torch import save_file

from dots_tts.data.edit_instruction import (
    instruction_operation_tags,
    normalize_edit_xvector_mode,
    render_source_text,
    render_target_text,
    resolve_edit_use_xvector,
)
from dots_tts.data.pipelines.tokenizing import build_edit_generation_schedule
from dots_tts.edit_cli import parse_args
from dots_tts.models.dots_tts import model as model_module
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.audio import prepare_edit_source_audio
from dots_tts.utils.tokenizer import (
    AUDIO_GEN_END_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self._next = 10
        self._text: dict[str, int] = {}
        self.encode_calls: list[str] = []
        self._special = {
            AUDIO_GEN_START_TOKEN: 201,
            AUDIO_GEN_SPAN_TOKEN: 202,
            AUDIO_GEN_END_TOKEN: 203,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encode_calls.append(text)
        result = []
        for char in text:
            if char not in self._text:
                self._text[char] = self._next
                self._next += 1
            result.append(self._text[char])
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.get(token, -1)


def make_runtime() -> DotsTtsRuntime:
    runtime = object.__new__(DotsTtsRuntime)
    runtime.device = torch.device("cpu")
    runtime.sample_rate = 16
    runtime.precision = "fp32"
    runtime.max_generate_length = 5
    runtime.max_sequence_length = 2048
    runtime.model = SimpleNamespace(
        tokenizer=DummyTokenizer(),
        config=SimpleNamespace(patch_size=1, sampling=None),
        hop_size=8,
    )
    runtime._load_edit_source_audio = MethodType(
        lambda self, path: torch.zeros((1, 16), dtype=torch.float32),
        runtime,
    )
    return runtime


def test_inference_load_ignores_only_training_input_mask_embedding(
    tmp_path,
    monkeypatch,
) -> None:
    class DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    monkeypatch.setattr(model_module, "DotsTtsCore", DummyCore)
    artifact = tmp_path / "model.safetensors"
    save_file(
        {
            "weight": torch.ones(2),
            "input_mask_embedding": torch.full((2,), 3.0),
        },
        artifact,
    )
    core = DummyCore()
    DotsTtsModel._load_artifact_module(core, artifact)
    torch.testing.assert_close(core.weight, torch.ones(2))

    save_file(
        {"weight": torch.ones(2), "unrelated_training_key": torch.ones(1)},
        artifact,
    )
    with pytest.raises(RuntimeError, match="unrelated_training_key"):
        DotsTtsModel._load_artifact_module(DummyCore(), artifact)


def test_instruction_renders_both_transcripts() -> None:
    instruction = 'Hello <sub targ="small">brave</sub><ins> world</ins>!'
    assert render_source_text(instruction) == "Hello brave!"
    assert render_target_text(instruction) == "Hello small world!"


@pytest.mark.parametrize(
    ("instruction", "expected_tags", "expected_xvector"),
    [
        ("plain text", frozenset(), True),
        ('<emo type="happy">hello</emo>', frozenset({"emo"}), False),
        (
            '<enhance><bg desc="fan">hello</bg></enhance>',
            frozenset({"enhance", "bg"}),
            False,
        ),
        (
            '<enhance><sub targ="new">old</sub></enhance>',
            frozenset({"enhance", "sub"}),
            True,
        ),
        ("left<spk_transfer/>right", frozenset({"spk_transfer"}), True),
    ],
)
def test_edit_auto_xvector_uses_complete_operation_set(
    instruction: str,
    expected_tags: frozenset[str],
    expected_xvector: bool,
) -> None:
    assert instruction_operation_tags(instruction) == expected_tags
    assert resolve_edit_use_xvector("auto", instruction) is expected_xvector


def test_explicit_edit_xvector_mode_remains_boolean() -> None:
    assert resolve_edit_use_xvector(True, '<emo type="happy">hello</emo>') is True
    assert resolve_edit_use_xvector(False, '<sub targ="new">old</sub>') is False
    with pytest.raises(ValueError, match='boolean or "auto"'):
        normalize_edit_xvector_mode("off")


def test_optional_transcripts_override_or_fall_back_to_instruction() -> None:
    instruction = '<sub targ="target">source</sub>'
    assert DotsTtsRuntime._resolve_edit_transcripts(
        instruction=instruction,
        source_text=" explicit source ",
        target_text=" explicit target ",
    ) == ("explicit source", "explicit target", "request", "request")
    assert DotsTtsRuntime._resolve_edit_transcripts(
        instruction=instruction,
        source_text=" ",
        target_text=None,
    ) == ("source", "target", "instruction", "instruction")


@pytest.mark.parametrize(
    ("instruction", "message"),
    [
        ("<ins>only target</ins>", "empty source transcript"),
        ("<del>only source</del>", "empty target transcript"),
    ],
)
def test_empty_rendered_transcript_is_rejected(
    instruction: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTtsRuntime._resolve_edit_transcripts(
            instruction=instruction,
            source_text=None,
            target_text=None,
        )


def test_edit_schedule_has_complete_source_and_target_audio_boundaries() -> None:
    tokenizer = DummyTokenizer()
    schedule = build_edit_generation_schedule(
        source_text="source",
        target_text="target",
        instruction='<sub targ="target">source</sub>',
        tokenizer=tokenizer,
        source_text_prefix="[原文本]",
        source_audio_prefix="[原语音]",
        instruction_prefix="[编辑指令]",
        target_text_prefix="[编辑文本]",
        target_audio_prefix="[编辑后语音]",
        source_num_audio_tokens=2,
        target_max_audio_tokens=3,
    )["schedule_ids"]
    assert schedule.count(201) == 2
    assert schedule.count(202) == 5
    assert schedule.count(203) == 2
    assert schedule[-1] == 203


def test_edit_source_audio_normalizes_edges_and_aligns_token_boundary(
    tmp_path,
) -> None:
    source_path = tmp_path / "source.wav"
    content = torch.tensor(
        [0.25, -0.25, 0.5, -0.5, 0.75, -0.75],
        dtype=torch.float32,
    )
    waveform = torch.cat(
        [
            torch.zeros(8, dtype=torch.float32),
            content,
            torch.zeros(2, dtype=torch.float32),
        ]
    )
    sf.write(source_path, waveform.numpy(), 16, subtype="FLOAT")

    prepared = prepare_edit_source_audio(
        str(source_path),
        target_sample_rate=16,
        samples_per_llm_token=8,
    )
    expected = torch.cat(
        [
            torch.zeros(4, dtype=torch.float32),
            content,
            torch.zeros(6, dtype=torch.float32),
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(prepared, expected, rtol=0.0, atol=0.0)


def test_runtime_builds_edit_source_audio_fill() -> None:
    runtime = make_runtime()
    inputs = runtime._prepare_edit_inputs(
        source_audio_path="source.wav",
        instruction='Hello <sub targ="small">brave</sub>!',
    )
    assert inputs["source_text"] == "Hello brave!"
    assert inputs["text"] == "Hello small!"
    assert inputs["source_text_source"] == "instruction"
    assert inputs["target_text_source"] == "instruction"
    assert inputs["drop_num_gen_head_patch"] == 0
    fill = inputs["audio_fills"][0]
    assert fill == {
        "audio": fill["audio"],
        "span_count": 2,
        "fill_llm": True,
        "fill_fm_history": False,
        "use_xvector": False,
        "drop_tail_patch_count": 0,
    }


def test_runtime_maps_all_tts_prompt_modes_to_audio_fills() -> None:
    runtime = make_runtime()
    runtime._load_prompt_audio = MethodType(
        lambda self, path: torch.zeros((1, 16), dtype=torch.float32),
        runtime,
    )

    text_only = runtime._prepare_inputs(
        text="hello",
        prompt_audio_path=None,
        prompt_text=None,
        template_name="tts",
    )
    assert text_only["audio_fills"] == []
    assert text_only["drop_num_gen_head_patch"] == 0

    speaker_only = runtime._prepare_inputs(
        text="hello",
        prompt_audio_path="speaker.wav",
        prompt_text=None,
        template_name="tts",
    )
    speaker_fill = speaker_only["audio_fills"][0]
    assert speaker_fill["span_count"] == 0
    assert speaker_fill["fill_llm"] is False
    assert speaker_fill["use_xvector"] is True

    continuation = runtime._prepare_inputs(
        text="world",
        prompt_audio_path="speaker.wav",
        prompt_text="hello",
        template_name="tts",
    )
    continuation_fill = continuation["audio_fills"][0]
    assert continuation_fill["span_count"] == 2
    assert continuation_fill["fill_llm"] is True
    assert continuation_fill["fill_fm_history"] is True
    assert continuation_fill["drop_tail_patch_count"] == 1
    assert continuation["drop_num_gen_head_patch"] == 1


def test_edit_source_prefill_does_not_enter_fm_history() -> None:
    model = object.__new__(DotsTtsModel)
    torch.nn.Module.__init__(model)
    model._llm_max_sequence_length = 2048
    llm_hiddens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    model.core = SimpleNamespace()
    model._get_llm_inference = MethodType(
        lambda self: SimpleNamespace(
            step=lambda _state, **_kwargs: (None, llm_hiddens, None)
        ),
        model,
    )
    model._build_prefill_inputs_embeds = MethodType(
        lambda self, *_args, **_kwargs: torch.zeros((1, 3, 4)),
        model,
    )
    appended_hiddens: list[torch.Tensor] = []
    model._append_hidden_chunk = MethodType(
        lambda self, _state, hidden: appended_hiddens.append(hidden.clone()),
        model,
    )
    model._append_history_chunk = MethodType(
        lambda self, _state, _patch: pytest.fail(
            "Edit source patches must not enter FM history"
        ),
        model,
    )
    state = SimpleNamespace(llm_state=object(), llm_hiddens=None)

    position = model._prefill(
        torch.tensor([[1, 2, 2, 2]]),
        state=state,
        span_positions=torch.tensor([1, 2, 3]),
        audio_fills=[
            SimpleNamespace(
                patches=torch.zeros((1, 2, 1, 1)),
                span_count=2,
                fill_llm=True,
                fill_fm_history=False,
            )
        ],
        fill_patch_embeddings=[torch.zeros((1, 2, 4))],
        audio_placeholder_ids={2},
    )
    assert position == 3
    assert len(appended_hiddens) == 1
    torch.testing.assert_close(appended_hiddens[0], llm_hiddens[:, -1:])


def test_disabled_speaker_guidance_skips_xvector_extractor() -> None:
    class XVectorExtractor:
        def eval(self) -> None:
            raise AssertionError("speaker encoder should be skipped")

        def __call__(self, _audio: torch.Tensor) -> torch.Tensor:
            raise AssertionError("speaker encoder should be skipped")

    model = object.__new__(DotsTtsModel)
    torch.nn.Module.__init__(model)
    model.core = torch.nn.Linear(1, 1)
    model.xvector_extractor = XVectorExtractor()
    model.config = SimpleNamespace(patch_size=1)
    model.hop_size = 8
    model._prompt_feature_cache = OrderedDict()

    fill = model._prepare_audio_fill(
        torch.zeros((1, 16), dtype=torch.float32),
        span_count=0,
        fill_llm=False,
        fill_fm_history=False,
        use_xvector=False,
    )
    assert fill.g_cond is None


def test_edit_cli_keeps_transcripts_optional_and_guidance_disabled() -> None:
    args = parse_args(
        [
            "--model-name-or-path",
            "model",
            "--source-audio",
            "source.wav",
            "--instruction",
            '<sub targ="target">source</sub>',
            "--output",
            "edited.wav",
        ]
    )
    assert args.source_text is None
    assert args.target_text is None
    assert args.use_xvector is False
    assert args.ode_method is None
    assert args.num_steps is None
    assert args.guidance_scale is None
