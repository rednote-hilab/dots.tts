from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860
DEFAULT_MODEL_NAME_OR_PATH = "dots-studio/dots.tts.edit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apps" / "edit_playground" / "outputs"
DEFAULT_NOISE_PRESETS_DIR = REPO_ROOT / "apps" / "edit_playground" / "noise_presets"
DEFAULT_OUTPUT_RETENTION = 20
DEFAULT_PRECISION = "bfloat16"
DEFAULT_MAX_GENERATE_LENGTH = 512
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_ODE_METHOD = "euler"
SUPPORTED_ODE_METHODS = ("euler", "midpoint", "rk4")
DEFAULT_NUM_STEPS = 32
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SPEAKER_SCALE = 1.5
DEFAULT_SEED = 20260414
DEFAULT_TARGET_TEXT = "当然可以啦，你知道吗，我真的很喜欢你的声音。"
VOICE_PROMPT_PRESET_NAMES: tuple[str, ...] = ()
EDIT_SOURCE_PRESET_NAMES: tuple[str, ...] = ()
DEFAULT_PROMPT_PRESET_NAME = ""
DEFAULT_EDIT_SOURCE_PRESET_NAME = ""
