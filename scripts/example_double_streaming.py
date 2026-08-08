from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from dots_tts.runtime_double_streaming import (  # noqa: E402
    DotsTtsRuntimeDoubleStreaming,
)
from dots_tts.utils.logging import configure_logging  # noqa: E402
from dots_tts.utils.text import normalize_text  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Temporary example for dots.tts double streaming session API."
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local pretrained directory or Hugging Face repo id",
    )
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--output", default="double_streaming.wav", help="Output wav path")
    parser.add_argument(
        "--prompt-audio",
        default=None,
        help="Optional reference audio for ref_audio_only speaker conditioning",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache dir")
    parser.add_argument("--precision", default="bfloat16", help="Inference precision")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable inference optimization and warmup",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--ode-method",
        default=None,
        help="ODE solver method (default: artifact setting, otherwise euler)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Sampling steps (default: artifact setting, otherwise 10)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: artifact setting, otherwise 1.2)",
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=0.8,
        help="EOS stop threshold for finish_text() tail decode",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=500,
        help="Maximum number of decoded audio patches in double streaming",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Normalize text before tokenizer encode",
    )
    return parser.parse_args(argv)


def _prepare_text(text: str, *, normalize: bool) -> str:
    prepared = text.strip()
    if normalize:
        prepared = normalize_text(prepared)
    if not prepared:
        raise ValueError("Input text is empty after preprocessing.")
    return prepared


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    seed_everything(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        precision=args.precision,
        optimize=args.optimize,
        max_generate_length=args.max_generate_length,
    )
    prepared_text = _prepare_text(args.text, normalize=args.normalize_text)
    text_token_ids = runtime.model.tokenizer.encode(
        prepared_text,
        add_special_tokens=False,
    )
    if not text_token_ids:
        raise ValueError("Tokenizer produced no text tokens.")

    logger.info(
        "Double streaming example started: text_len={} text_token_count={} output={}",
        len(prepared_text),
        len(text_token_ids),
        output_path,
    )

    session = runtime.start_double_streaming(
        prompt_audio_path=args.prompt_audio,
        ode_method=args.ode_method,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        eos_threshold=args.eos_threshold,
    )

    chunks: list[torch.Tensor] = []
    for index, token_id in enumerate(text_token_ids, start=1):
        chunk = session.push_text_token(token_id)
        logger.info(
            "Double streaming step: token_index={} token_id={} emitted_audio={}",
            index,
            token_id,
            chunk is not None,
        )
        if chunk is not None:
            chunks.append(chunk.detach().cpu())

    for chunk in session.finish_text():
        chunks.append(chunk.detach().cpu())

    if not chunks:
        raise RuntimeError("Double streaming produced no audio chunks.")

    audio = torch.cat(chunks, dim=-1)
    sf.write(
        output_path,
        audio.float().squeeze().numpy(),
        runtime.sample_rate,
    )
    logger.info(
        "Double streaming example completed: output={} chunk_count={} samples={}",
        output_path,
        len(chunks),
        audio.shape[-1],
    )


if __name__ == "__main__":
    raise SystemExit(main())
