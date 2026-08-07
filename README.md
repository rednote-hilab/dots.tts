<p align="center">
  <img src="assets/logo.png" alt="dots.tts" width="280">
</p>

<p align="center">
  <a href="https://github.com/rednote-hilab/dots.tts"><img src="https://img.shields.io/badge/GitHub-rednote--hilab%2Fdots.tts-blue?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/collections/rednote-hilab/dotstts"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dots.tts%20collection-yellow" alt="Hugging Face"></a>
  <a href="https://arxiv.org/abs/2606.07080"><img src="https://img.shields.io/badge/arXiv-Report-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/spaces/rednote-hilab/dots.tts"><img src="https://img.shields.io/badge/Playground-Live-orange" alt="Playground"></a>
  <a href="https://rednote-hilab.github.io/dots.tts-demo/"><img src="https://img.shields.io/badge/Demo%20Page-Live-red" alt="Demo Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

**dots.tts** is a **2B-parameter fully continuous, end-to-end autoregressive (AR) text-to-speech system**. The backbone pairs a semantic encoder, an LLM, and an autoregressive flow-matching acoustic head over a **48 kHz** AudioVAE, with no discrete tokens anywhere in the pipeline.

dots.tts achieves the best average performance on **Seed-TTS-Eval**, with WERs of **0.94% / 1.30% / 6.60%** and SIM scores of **81.0 / 77.1 / 79.5** on the zh / en / zh-hard test sets, respectively. It further attains the **highest average speaker similarity (83.9)** on the 24-language **MiniMax multilingual** benchmark. Across other benchmarks, dots.tts also consistently demonstrates **open-source state-of-the-art performance**, exhibiting strong generation stability, voice cloning ability, and emotional expressiveness.

### News

* **[2026.08]** 🚀 [SGLang Omni](https://github.com/sgl-project/sglang-omni) now fully supports dots.tts, with voice cloning performance on Seed-TTS-Eval **EN** (N=1088, concurrency=18, warmup=4) reaching RTF mean / median / p95 = **0.247 / 0.223 / 0.465**; latency mean / median = **0.954 / 0.816** s (audio mean 4.01 s).

【TODO：Update it here】

* **[2026.07]** 🚀 Shipped a **high-performance inference path** — under `--optimize`, `dots.tts-soar` reaches RTF p50 **0.20 / 0.18** and first-chunk latency **225 ms / 69 ms** (voice cloning / text-only); `dots.tts-mf` reaches **0.15 / 0.13** and **204 ms / 68 ms** respectively. See the [Efficiency](#-efficiency) section for details.

* **[2026.06]** 🔥 We have released **dots.tts** — 2B fully continuous AR TTS, with pretrained / self-corrective-aligned / MeanFlow-distilled checkpoints and full inference & fine-tuning code under Apache-2.0.

---

## Contents

- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
  - [CLI](#cli)
  - [Python API](#python-api)
  - [Web Demo (Gradio)](#web-demo-gradio)
  - [Fine-tuning](#fine-tuning)
  - [MeanFlow Distillation](#meanflow-distillation)
- [Usage Tips](#-usage-tips)
- [Architecture](#-architecture)
- [Performance](#-performance)
  - [Seed-TTS-Eval](#seed-tts-eval)
  - [MiniMax Multilingual](#minimax-multilingual-24-languages)
  - [CV3-Eval](#cv3-eval)
  - [EmergentTTS-Eval](#emergenttts-eval)
- [Efficiency](#-efficiency)
- [Community Projects](#-community-projects)
- [Risks and Limitations](#%EF%B8%8F-risks-and-limitations)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Quick Start

### Installation

We recommend a fresh conda environment (Python 3.10–3.12):

```bash
conda create -n dots_tts python=3.10 -y
conda activate dots_tts
```

Install from PyPI:

```bash
pip install dots.tts
```

Or from source (for local development / editable install):

```bash
git clone https://github.com/rednote-hilab/dots.tts
cd dots.tts
pip install -e . -c constraints/recommended.txt
```

For training / linting extras:

```bash
pip install 'dots.tts[full]'
# or from source:
pip install -e .[full] -c constraints/recommended.txt
```

The `constraints/recommended.txt` file pins the reproducible versions;
`pyproject.toml` declares compatibility ranges.

To use SGLang Omni for high-performance high-concurrency voice cloning:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .
```

Detailed installation instructions can be found in this [guidance](https://sgl-project.github.io/sglang-omni/get_started/installation.html).

### Checkpoints

Three pretrained checkpoints are released on Hugging Face. All three share the same backbone — choose by the quality / inference-cost tradeoff:

| Model | Description | Recommended `--num-steps` |
|---|---|:---:|
| [`rednote-hilab/dots.tts-base`](https://huggingface.co/rednote-hilab/dots.tts-base) | Pretrained checkpoint. | `10`–`32` (default `10`) |
| [`rednote-hilab/dots.tts-soar`](https://huggingface.co/rednote-hilab/dots.tts-soar) | Self-corrective-aligned (SCA) checkpoint on top of `dots.tts-base`. Best voice cloning performance. | `10`–`32` (default `10`) |
| [`rednote-hilab/dots.tts-mf`](https://huggingface.co/rednote-hilab/dots.tts-mf) | MeanFlow-distilled student from `dots.tts-soar`. Recommended if you care about inference speed. | `4` |

Pass the repo id directly to `--model-name-or-path` (or `DotsTtsRuntime.from_pretrained`) — the snapshot is fetched on first use and cached locally.

### CLI

The package installs a `dots.tts` entry point:

```bash
# Continuation voice cloning (reference audio + transcript) — recommended, best SIM
dots.tts \
  --model-name-or-path rednote-hilab/dots.tts-soar \
  --text "Hello, this is a zero-shot voice cloning demonstration." \
  --prompt-audio /path/to/reference.wav \
  --prompt-text "The exact transcript of the reference audio." \
  --num-steps 10 \
  --output clone.wav

# X-vector-only voice cloning (reference audio only — timbre from speaker x-vector)
dots.tts \
  --model-name-or-path rednote-hilab/dots.tts-soar \
  --text "Hello, this is a zero-shot voice cloning demonstration." \
  --prompt-audio /path/to/reference.wav \
  --num-steps 10 \
  --output clone.wav

# Random-voice sampling (no reference) — only meaningful with a fine-tuned
# single-speaker checkpoint
dots.tts \
  --model-name-or-path rednote-hilab/dots.tts-soar \
  --text "Hello, this is a quick speech synthesis test." \
  --num-steps 10 \
  --output output.wav
```

Common flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--num-steps` | Flow-matching sampling steps (higher = better quality, lower = faster) | `10` |
| `--guidance-scale` | CFG scale (flow-matching only; MeanFlow has CFG fused into the student; values > 2 progressively amplify audio energy) | `1.2` |
| `--normalize-text` | Apply text normalization before inference (via [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)) | off |
| `--language` | Add an explicit language tag to the input text; accepts `none`, `auto_detect`, language codes such as `EN` / `ZH`, or names such as `english` / `chinese` | `none` |
| `--seed` | RNG seed (fixed seed → deterministic output) | `42` |

`dots.tts --help` lists the full set.

Notes:

- `--prompt-audio` selects the speaker voice — continuation cloning when paired with `--prompt-text`, x-vector-only cloning when used alone. Omitting `--prompt-audio` falls back to random-voice sampling, which is only meaningful on a fine-tuned single-speaker checkpoint.
- `--language` is useful for multilingual or code-switched text when you want to force the model-side language tag. For example, pass `--language EN` for English, `--language ZH` for Mandarin, `--language Cantonese` for Cantonese, or `--language auto_detect` to infer the tag from `--text`.
- Pass either a local model directory or a Hugging Face repo id.

### Python API

```python
from dots_tts.runtime import DotsTtsRuntime
import soundfile as sf

runtime = DotsTtsRuntime.from_pretrained(
    "rednote-hilab/dots.tts-soar",
    precision="bfloat16",
    optimize=True,  # torch.compile acceleration (warmup at load, faster steady-state)
)

result = runtime.generate(
    text="Hello, this is a quick speech synthesis test.",
    prompt_audio_path="/path/to/reference.wav",
    prompt_text="The exact transcript of the reference audio.",
    num_steps=10,
    guidance_scale=1.2,
)

sf.write("output.wav", result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])
```

For low-latency playback or streaming to a client, use `generate_stream` instead — it yields audio chunks (`torch.Tensor`, shape `(1, samples)`) as they are produced. Arguments are identical to `generate`:

```python
import torch

stream = runtime.generate_stream(
    text="Hello, this is a streaming speech synthesis test.",
    prompt_audio_path="/path/to/reference.wav",
    prompt_text="The exact transcript of the reference audio.",
    num_steps=10,
    guidance_scale=1.2,
)

chunks = []
for chunk in stream:
    chunks.append(chunk.detach().float().cpu())
    # handle_chunk(chunk)  # push to a player / websocket / etc.

audio = torch.cat(chunks, dim=-1).squeeze().numpy()
sf.write("output_stream.wav", audio, runtime.sample_rate)
```

### Web Demo (Gradio)

```bash
python apps/gradio/app.py \
  --model-name-or-path rednote-hilab/dots.tts-soar \
  --optimize
```

Defaults to `http://0.0.0.0:7860`. With `--optimize` the first launch runs warmup (slower startup, faster steady-state).

### Fine-tuning

This repo exposes fine-tuning and MeanFlow distillation entry points. Fine-tune from a released checkpoint with:

```bash
accelerate launch scripts/train_dots_tts.py --config configs/dots_tts.yaml
```

`configs/dots_tts.yaml` is a smoke configuration that verifies the pipeline runs end-to-end on commodity hardware. Replace `train.pretrained_model_path`, `train_data.sources` / `val_data.sources`, `train.output_dir`, and `train.max_train_steps` with your own values to use it.

A helper script downloads LJSpeech-1.1-48kHz and emits a train/valid JSONL manifest for the smoke run:

```bash
python scripts/prepare_train_jsonl_manifest.py --output-dir downloaded_data
```

Manifest format — one JSON per line, minimum three fields:

```json
{"fid": "sample-0001", "audio": "/abs/path/to/audio.wav", "text": "hello world"}
```

### MeanFlow Distillation

MeanFlow distillation trains a MeanFlow DiT student against a frozen flow-matching teacher. The teacher can be the released SOAR checkpoint or any compatible flow-matching dots.tts checkpoint you have fine-tuned yourself.

To use SOAR as the teacher, download it first:

```bash
huggingface-cli download rednote-hilab/dots.tts-soar \
  --local-dir pretrained_models/dots.tts-soar
```

Then launch distillation with the MeanFlow config:

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  scripts/train_dots_tts_meanflow.py \
  --config configs/dots_tts_meanflow.yaml \
  --teacher-model-path pretrained_models/dots.tts-soar
```

To distill from your own fine-tuned teacher, pass that checkpoint instead:

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  scripts/train_dots_tts_meanflow.py \
  --config configs/dots_tts_meanflow.yaml \
  --teacher-model-path /path/to/your_finetuned_teacher
```

`configs/dots_tts_meanflow.yaml` is a conservative smoke configuration that uses the same LJSpeech manifests produced by `scripts/prepare_train_jsonl_manifest.py`. Replace `train.pretrained_model_path`, `--teacher-model-path`, `train_data.sources` / `val_data.sources`, `train.output_dir`, and `train.max_train_steps` for your own distillation run.

By default, the script initializes the student from `train.pretrained_model_path`, adds the MeanFlow duration embedding, freezes the non-DiT modules, and trains `student.core.velocity_field_predictor`. MeanFlow does not run a separate CFG branch at inference time; the default `fused` mode distills the guided teacher target into the student. Training checkpoints save the MeanFlow student only; the frozen teacher is not written into the checkpoint model directory. Pass `--train-all-parameters` only if you want to update the full dots.tts model.

Common MeanFlow flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--teacher-model-path` | Frozen flow-matching teacher directory. Defaults to `train.pretrained_model_path` if omitted. | `train.pretrained_model_path` |
| `--teacher-steps` | Teacher rollout steps used to build the distillation target. Higher is slower and usually stronger. | `8` |
| `--teacher-solver` | Teacher ODE solver: `euler`, `midpoint`, or `rk4`. | `euler` |
| `--cfg-distill-mode` | `fused` distills a guided teacher target into the student; `natural` trains on sampled conditional/unconditional masks without fusing CFG. | `fused` |
| `--distill-cfg-scale` | Extra CFG coefficient used when `--cfg-distill-mode fused` is enabled. It matches inference `guidance_scale` semantics: `teacher_cond + scale * (teacher_cond - teacher_uncond)`. | `1.2` |
| `--anchor-prob` | Probability of using a zero-duration anchor sample in MeanFlow training. | `0.5` |
| `--debug` | Print the first few batch summaries and gradient diagnostics. | off |

---

### SGLang Omni Usage

【TODO：Update it here】

## 💡 Usage Tips

- **Keep the reference audio around 10s**. Longer audio won't yield better results.
- **`--prompt-text` should match what's actually spoken in the reference audio**. Mismatches degrade stability and may cause word-level errors.
- **Higher-quality references give better clones** — prefer a high sample rate, low background noise, no trailing noise, and natural-sounding speech.
- **Try different `--seed` values for prosody variation**. Each seed produces a different rhythm and intonation — resample a few times if the default doesn't feel right.
- **Increase `--num-steps` if quality isn't good enough**. More sampling steps trade compute for cleaner output and better expressiveness.
- **Force a pronunciation with Pinyin for polyphones.** Replace the character in the input text with its tone-marked pinyin — e.g. write `我生平不hào此道` to force `好` to be read as `hào`. Use tone-marked pinyin only (`hǎo`, `hào`, `bā`); numbered forms like `hao4` or `ha4o` are **not** recognized. Useful when reseeding doesn't fix a polyphone misread.

---

## 🏛 Architecture

A frozen **AudioVAE** encodes 48 kHz mono waveform into a continuous latent and decodes it back via a BigVGAN-style causal decoder. An **autoregressive backbone** predicts that latent one patch at a time, in three components:

- **Semantic encoder** — re-encodes each newly generated VAE patch into a compact embedding for the LLM, stripping high-variance acoustic detail.
- **LLM** — initialized from **Qwen2.5-1.5B-Base**, consumes BPE text directly (no phonemes), and emits one hidden state per audio step.
- **AR flow-matching head** — a DiT that conditions on the LLM hidden state and the AR prefix to denoise the next VAE patch, with a frozen CAM++ speaker x-vector as side input.

Two sequence layouts: *plain mode* places the full text as a prefix before the audio span (standard TTS); *[1T1A interleaved mode](scripts/example_double_streaming.py)* alternates one BPE token with one audio step, enabling low-latency streaming when driven by a duplex dialogue LLM. See the technical report for full architectural and training details.

---

## 📊 Performance

Baselines are taken from original publications or default-configuration open-source releases.

### Seed-TTS-Eval

Zero-shot, ~3 s reference prompt, scored by the benchmark's reference ASR and WavLM-SV similarity.

| Model | Params | test-en WER↓ / SIM↑ | test-zh WER↓ / SIM↑ | test-zh-hard WER↓ / SIM↑ | **Avg WER↓ / SIM↑** |
|---|---:|:---:|:---:|:---:|:---:|
| CosyVoice 3 | 1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / 75.8 | 3.06 / 75.3 |
| DiTAR | 0.6B | 1.69 / 73.5 | 1.02 / 75.3 | — | — |
| F5-TTS | 0.3B | 2.00 / 67.0 | 1.53 / 76.0 | 8.67 / 71.3 | 4.10 / 71.4 |
| FireRedTTS-2 | 1.5B | 1.95 / 66.5 | 1.14 / 73.6 | 8.98 / 70.3 | 4.02 / 70.1 |
| IndexTTS 2 | 1.5B | 2.23 / 70.6 | 1.03 / 76.5 | 7.12 / 75.5 | 3.46 / 74.2 |
| MegaTTS 3 | 0.5B | 2.79 / 77.1 | 1.52 / 79.0 | — | — |
| MiniMax-Speech | — | 1.65 / 69.2 | **0.83** / 78.3 | — | — |
| Qwen3-TTS | 1.7B | **1.23** / 71.7 | 1.22 / 77.0 | 6.76 / 74.8 | 3.07 / 74.5 |
| Seed-TTS | — | 2.25 / 76.2 | 1.12 / 79.6 | 7.59 / 77.6 | 3.65 / 77.8 |
| VibeVoice | 1.5B | 3.04 / 68.9 | 1.16 / 74.4 | — | — |
| VoxCPM 2 | 2B | 1.84 / 75.3 | 0.97 / 79.5 | 8.13 / 75.3 | 3.65 / 76.7 |
| **dots.tts (Pretrain)** | **2B** | 1.34 / 76.8 | 0.96 / 80.5 | 6.46 / 79.2 | **2.92** / 78.8 |
| **dots.tts (SCA)** | **2B** | 1.30 / **77.1** | 0.94 / **81.0** | 6.60 / **79.5** | 2.95 / **79.2** |
| **dots.tts (MF, NFE=4)** | **2B** | 1.29 / 76.2 | 0.94 / 80.0 | 6.60 / 78.5 | 2.94 / 78.2 |

### MiniMax Multilingual (24 languages)

Per-language WER / SIM on the MiniMax-Speech multilingual test set (100 utterances × 2 reference speakers per language). **Highest average SIM (83.9, SCA)**, with a dots.tts variant taking the per-language SIM lead outright on 19 of 24 languages and tying on 2 more. Content fidelity is on par with the strongest systems on high-resource / Western European splits, and trails on low-resource long-tail languages where SIM is still preserved.

<details>
<summary><b>Per-language WER / SIM (click to expand)</b></summary>

| Language | MiniMax | ElevenLabs | Fish-Audio S2 | VoxCPM 2 | **dots.tts (Pre.)** | **dots.tts (SCA)** | **dots.tts (MF$_4$)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Arabic | **1.67** / 73.6 | **1.67** / 70.6 | 3.50 / 75.0 | 13.05 / **79.1** | 37.91 / 77.5 | 36.19 / **79.1** | 39.65 / 77.6 |
| Cantonese* | 34.11 / 77.8 | 51.51 / 67.0 | 30.67 / 80.5 | 38.58 / 83.5 | 37.91 / 84.7 | 42.32 / **85.0** | 37.82 / 84.0 |
| Chinese | 2.25 / 78.0 | 16.03 / 67.7 | **0.73** / 81.6 | 1.14 / **82.5** | 1.08 / 82.3 | 0.77 / **82.5** | 1.01 / 81.8 |
| Czech | 3.88 / 79.6 | **2.11** / 68.5 | 2.84 / 79.8 | 24.13 / 78.3 | 5.05 / 83.8 | 4.25 / **84.2** | 5.67 / 83.9 |
| Dutch | 1.14 / 73.8 | **0.80** / 68.0 | 0.99 / 73.0 | 0.91 / 80.8 | 1.20 / 81.4 | 1.39 / **82.2** | 1.30 / 82.1 |
| English | 2.16 / 75.6 | 2.34 / 61.3 | 1.62 / 79.7 | 2.29 / 85.4 | 1.06 / 86.9 | **1.03** / **87.5** | 1.09 / 86.9 |
| Finnish | 4.67 / 83.5 | 2.96 / 75.9 | 3.33 / 81.9 | **2.63** / **89.0** | 3.44 / 88.0 | 4.08 / 88.3 | 3.61 / 88.3 |
| French | 4.10 / 62.8 | 5.22 / 53.5 | **3.05** / 69.8 | 4.53 / 73.5 | 3.82 / 78.2 | 3.56 / **78.6** | 3.26 / 78.5 |
| German | 1.91 / 73.3 | 0.57 / 61.4 | **0.55** / 76.7 | 0.68 / 80.3 | 1.03 / 79.5 | 1.70 / **80.6** | 0.91 / 79.5 |
| Greek | 2.02 / 82.6 | **0.99** / 73.3 | 5.74 / 79.5 | 2.84 / 86.0 | 2.97 / **87.6** | 3.00 / **87.6** | 3.19 / 87.3 |
| Hindi | 6.96 / 81.8 | **5.83** / 73.0 | 14.64 / 82.1 | 19.70 / **85.6** | 14.32 / 84.5 | 14.24 / 84.7 | 14.75 / 84.8 |
| Indonesian | 1.24 / 72.9 | **1.06** / 66.0 | 1.46 / 76.3 | 1.08 / 80.0 | 2.71 / 80.8 | 2.96 / 80.8 | 3.91 / **81.2** |
| Italian | 1.54 / 69.9 | 1.74 / 57.9 | **1.27** / 74.7 | 1.56 / 78.0 | 3.16 / 84.5 | 3.12 / **84.7** | 2.16 / 84.3 |
| Japanese | 3.52 / 77.6 | 10.65 / 73.8 | **2.76** / 79.6 | 4.63 / 82.8 | 7.16 / 83.1 | 5.28 / **83.7** | 5.17 / 83.1 |
| Korean | 1.75 / 77.6 | 1.87 / 70.0 | **1.18** / 81.7 | 1.96 / 83.3 | 5.30 / 84.3 | 5.66 / 83.6 | 3.93 / **84.9** |
| Polish | 1.42 / 80.2 | **0.77** / 72.9 | 1.26 / 81.9 | 1.14 / **88.4** | 2.72 / 87.3 | 3.59 / 87.8 | 3.42 / 87.5 |
| Portuguese | 1.88 / 80.5 | 1.33 / 71.1 | **1.14** / 78.1 | 1.94 / 83.7 | 1.64 / 83.1 | 2.00 / **84.3** | 2.40 / 83.1 |
| Romanian | 2.88 / 80.9 | **1.35** / 69.9 | 10.74 / 73.3 | 21.58 / 79.7 | 3.36 / 86.2 | 3.87 / **87.1** | 3.38 / 86.1 |
| Russian | 4.28 / 76.1 | 3.88 / 67.6 | **2.40** / 79.0 | 3.63 / 81.1 | 3.64 / 83.0 | 4.28 / **83.2** | 4.42 / **83.2** |
| Spanish | 1.03 / 76.2 | 1.08 / 61.5 | 0.91 / 77.6 | 1.44 / 83.1 | 0.96 / 83.9 | 1.27 / **84.0** | **0.80** / **84.0** |
| Thai | **2.70** / 80.0 | 73.94 / 58.8 | 4.23 / 78.6 | 2.96 / 84.0 | 7.45 / 83.8 | 7.86 / 83.9 | 8.03 / **84.2** |
| Turkish | 1.52 / 77.9 | **0.70** / 59.6 | 0.87 / 83.5 | 0.82 / 87.1 | 5.45 / **87.4** | 4.96 / 87.3 | 6.20 / 86.8 |
| Ukrainian | 1.08 / 73.0 | **1.00** / 64.7 | 2.30 / 74.7 | 6.32 / 79.8 | 1.61 / 80.5 | 1.27 / **81.2** | 1.66 / 80.0 |
| Vietnamese | **0.88** / 74.3 | 73.42 / 36.9 | 7.41 / 74.0 | 3.31 / 80.6 | 3.85 / 80.7 | 3.89 / **81.6** | 5.43 / 80.5 |
| **Average** | **2.8** / 76.6 | 7.5 / 65.5 | 3.7 / 78.0 | 5.7 / 82.3 | 6.6 / 83.5 | 6.8 / **83.9** | 6.8 / 83.5 |

</details>

<sub>*Cantonese WER reflects an ASR-faithfulness floor common to all systems; SIM remains comparable.</sub>

### CV3-Eval

Hard-subset Chinese/English plus a cross-lingual voice-cloning split. **Takes the table top on hard-en (MF$_4$ at 4.37) and leads both cross-lingual SIM subsets (SCA at 75.0 / 72.8)**, with the post-trained variants bracketing the prior leader on the hardest English subset.

| Model | zh W↓ | en W↓ | hard-zh W↓ | hard-en W↓ | en→zh W↓ / S↑ | zh→en W↓ / S↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CosyVoice 2 | 4.08 | 6.32 | 12.58 | 11.96 | 13.50 / 63.3 | 6.47 / 64.3 |
| CosyVoice 3 (1.5B) | 3.91 | 4.99 | 9.77 | 10.55 | **8.01** / 66.9 | **4.32** / 66.4 |
| Fish-Audio S2 | **2.65** | **2.43** | 9.10 | 4.40 | — | — |
| VoxCPM 2 | 3.65 | 5.00 | **8.55** | 8.48 | — | — |
| **dots.tts (Pretrain)** | 3.51 | 5.24 | 9.69 | 5.99 | 10.88 / 74.6 | 4.97 / 71.9 |
| **dots.tts (SCA)** | 3.71 | 4.50 | 9.22 | 4.49 | 10.75 / **75.0** | 5.66 / **72.8** |
| **dots.tts (MF, NFE=4)** | 3.95 | 4.05 | 9.10 | **4.37** | 10.73 / 73.8 | 5.24 / 70.9 |

### EmergentTTS-Eval

Win-rate judged head-to-head against `gpt-4o-mini-tts` by Gemini-2.5-Pro-0506 across six expressiveness-oriented scenarios. **SCA takes the top Syntactic Complexity score in the table (65.7%) — above every closed-source system** — and Pretrain posts the **best Emotions score among open-source systems (72.7%)**.

| Model | Voice | WER↓ | Overall↑ | Emotions↑ | Paraling.↑ | Foreign↑ | C. Pron.↑ | Quest.↑ | Syntax↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemini-2.5-Flash-TTS\* | Zephyr | 10.39 | **70.7%** | **95.9%** | **91.3%** | 58.5% | 55.7% | **63.0%** | 57.9% |
| Gemini-2.5-Pro-TTS\* | Zephyr | 11.79 | 69.3% | 86.9% | 82.3% | 58.2% | **64.8%** | 61.3% | 61.8% |
| gpt-4o-audio-preview\* | Ballad | 11.87 | 65.2% | 88.8% | 82.1% | **60.2%** | 40.4% | 57.0% | 59.5% |
| gpt-4o-mini-tts\* | Alloy | 10.76 | 56.3% | 59.2% | 58.8% | 57.3% | 52.4% | 52.7% | 57.1% |
| *baseline: gpt-4o-mini-tts* | Alloy | 10.61 | 50.0% | — | — | — | — | — | — |
| **dots.tts (Pretrain)** | basic\_ref\_en | 10.86 | 49.2% | 72.7% | 54.7% | 39.5% | 18.0% | 48.4% | 58.4% |
| **dots.tts (MF4)** | basic\_ref\_en | 11.75 | 47.9% | 59.8% | 55.2% | 36.3% | 16.7% | 50.5% | 64.8% |
| **dots.tts (SCA)** | basic\_ref\_en | 10.45 | 47.6% | 63.9% | 52.7% | 39.4% | 16.4% | 47.0% | **65.7%** |
| Qwen3-TTS | basic\_ref\_en | 17.32 | 42.8% | 39.8% | 50.7% | 25.4% | 30.0% | 48.9% | 60.4% |
| HumeAI\* | — | 12.85 | 42.7% | 61.6% | 36.9% | 34.6% | 34.3% | 43.2% | 44.6% |
| Qwen3-TTS | Ryan | 19.65 | 42.3% | 60.5% | 62.7% | 17.1% | 9.8% | 56.4% | 43.0% |
| VoxCPM 2 | basic\_ref\_en | 11.84 | 41.1% | 42.3% | 44.1% | 33.3% | 18.6% | 53.4% | 52.3% |
| MiniMax/speech-02-hd\* | EN-narr | **10.02** | 36.6% | 40.9% | 34.3% | 34.3% | 16.3% | 47.3% | 43.9% |
| 11Labs Multilingual v2\* | Brian | 11.19 | 33.9% | 30.4% | 45.5% | 35.5% | 14.5% | 39.5% | 35.5% |
| F5-TTS | basic\_ref\_en | 16.47 | 15.3% | 26.8% | 21.6% | 1.8% | 1.4% | 14.8% | 23.8% |

<sub>\* Closed-source / commercial. Table shows a selected subset for brevity — for the full leaderboard, see [EmergentTTS-Eval-public](https://github.com/boson-ai/EmergentTTS-Eval-public/blob/main/LEADERBOARD_gemini-2.5-pro-05-06.md).</sub>

---

## ⚡ Efficiency

Streaming-inference benchmarks under `--optimize` on a Seed-TTS-Eval mix (100 utterances across zh / en / zh-hard, first post-warmup request excluded, N=99 per group). `voice_cloning` uses reference audio + transcript; `text_only` uses text with no reference. Common config: `precision=bfloat16`, `guidance_scale=1.2`, `seed=42`; SOAR uses `num_steps=10`, MF uses `num_steps=4`. Hardware / stack: single H800, torch 2.8 + CUDA 12.8. The `--optimize` path also accelerates non-streaming `generate()` calls; numbers below are the streaming path.

> **Note:** `--optimize` triggers a one-shot `torch.compile` warmup that walks every DiT compile bucket + KvPrefill + vocoder chunk sizes. Cold start takes **~3 minutes** on H800; every subsequent request runs at the steady-state RTF above. Pass `warmup_on_optimize=False` to `DotsTtsRuntime` if you want to skip warmup and accept the first request paying the compile cost.

### Steady-State Latency

| Group | audio mean (s) | latency p50 / p90 (s) | first-chunk p50 / p90 (ms) | RTF mean / p50 / p90 | peak alloc (GB) |
|---|---:|---:|---:|---:|---:|
| SOAR / voice_cloning | 7.57 | 1.13 / 3.04 | 225 / 404 | 0.21 / 0.20 / 0.26 | 7.86 |
| SOAR / text_only     | 7.78 | 0.95 / 3.02 |  69 /  79 | 0.18 / 0.18 / 0.20 | 7.85 |
| MF   / voice_cloning | 7.46 | 0.88 / 1.73 | 204 / 381 | 0.16 / 0.15 / 0.21 | 5.74 |
| MF   / text_only     | 7.65 | 0.68 / 2.06 |  68 /  78 | 0.13 / 0.13 / 0.15 | 5.73 |


### Memory Footprint by Length Bucket

Bucket = total prompt + generated audio in latent patches (one patch ≈ 160 ms).

| Bucket | Total audio cap | SOAR / voice_cloning | SOAR / text_only | MF / voice_cloning | MF / text_only |
|---|---:|---:|---:|---:|---:|
| <64 patches  | <10.24s | 5.65 GB | 5.64 GB | 5.30 GB | 5.29 GB |
| <128 patches | <20.48s | 6.53 GB | 6.52 GB | 5.47 GB | 5.46 GB |
| <256 patches | <40.96s | 7.86 GB | 7.85 GB | 5.74 GB | 5.73 GB |
| <512 patches | <81.92s | 10.51 GB\* | 10.51 GB\* | 6.29 GB\* | N/A\*\* |

\* From explicit long-audio probes (actual spans within 256–512 patches).  
\*\* `mf / text_only` did not reach the 256–512 bucket under either synthetic (x4) or real long-text probes; longest observed 237 patches at 5.73 GB.

### SGLang Omni Efficiency

【TODO：Update it here】

## 🤝 Community Projects

Third-party ports and integrations of dots.tts, maintained by the community.

| Project | Description | Maintainer |
|---|---|---|
| [sglang-omni](https://github.com/sgl-project/sglang-omni) | SGLang Omni support for dots.tts | [@sgl-project](https://github.com/sgl-project) |
| [dots-tts-mlx](https://github.com/sb1992/dots-tts-mlx) | Pure-MLX inference port for Apple Silicon (Python) | [@sb1992](https://github.com/sb1992) |
| [mlx-swift-dots-tts](https://github.com/sammcj/mlx-swift-dots-tts) | Native MLX Swift port for Apple Silicon (no Python runtime) | [@sammcj](https://github.com/sammcj) |
| [Dots-TTS-ComfyUI](https://github.com/Saganaki22/Dots-TTS-ComfyUI) | ComfyUI custom nodes for TTS, voice cloning, and Whisper transcription | [@Saganaki22](https://github.com/Saganaki22) |

---

## ⚠️ Risks and Limitations

- **Misuse risk.** High-fidelity zero-shot voice cloning can produce highly realistic synthetic speech. The released checkpoints are intended for research and authorized deployment. Do **not** use dots.tts for impersonation, fraud, or disinformation. Combine downstream use with consent-aware reference-audio policies, robust synthetic-speech detection, and content watermarking. Clearly mark AI-generated audio.
- **Low-resource WER gap.** A BPE backbone inherits the text LLM's language coverage at the cost of a higher data appetite. On script-divergent and under-represented languages (Arabic, Hindi, Turkish, Vietnamese) the WER gap visible on the MiniMax benchmark reflects this, and the same long tail surfaces on the Foreign Words and Complex Pronunciation scenarios of EmergentTTS-Eval. Speaker similarity is preserved across these languages.
- **Speech-heavy training.** Although the AudioVAE is trained at 48 kHz and is modality-agnostic in principle, the backbone is trained on a speech-heavy mixture. Singing and unified speech + sound generation are not covered in this release.

---

## 📖 Citation

If you find dots.tts useful, please consider citing the technical report and starring the repository.

```bibtex
@article{dotstts2026,
  title         = {dots.tts Technical Report},
  author        = {dots.tts Team},
  year          = {2026},
  eprint        = {2606.07080},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
}
```

## 📄 License

dots.tts code and released checkpoints are licensed under [Apache-2.0](LICENSE).

## 🙏 Acknowledgments

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — LLM backbone initialization.
- [DiTAR](https://arxiv.org/abs/2502.03930) and [ARDiT](https://arxiv.org/abs/2406.05551) — for the continuous-AR + per-patch diffusion design.
- [HoliTok](https://github.com/bovod-sjtu/HoliTok) — for the AudioVAE design.
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) — for the vocoder design.
- [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker) — for speaker x-vector encoder.
