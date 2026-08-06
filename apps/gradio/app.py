from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for import_root in (REPO_ROOT, SRC_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from apps.gradio.constants import (  # noqa: E402
    DEFAULT_EXECUTION_MODE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HOST,
    DEFAULT_INPUT_TEXT,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PORT,
    DEFAULT_PRECISION,
    DEFAULT_PROMPT_NAME,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
)

if TYPE_CHECKING:
    import gradio as gr

DEBUG_GRADIO_ENABLED = os.environ.get("DEBUG_GRADIO", "0") == "1"


PLAYGROUND_CSS = """
.gradio-container {
    width: min(1600px, calc(100vw - 32px)) !important;
    max-width: none !important;
    margin: 0 auto !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

.gradio-container,
.gradio-container .gradio-container {
    --block-label-background-fill: #CCE5FF;
    --block-label-text-color: #6666FF;
    --block-label-border-color: #99c7ee;
    --block-label-text-weight: 600;
    --block-title-background-fill: #CCE5FF;
    --block-title-text-color: #6666FF;
    --block-title-border-color: #99c7ee;
    --block-title-border-width: var(--block-label-border-width);
    --block-title-radius: var(--block-label-radius);
    --block-title-padding: var(--block-label-padding);
    --block-title-text-size: var(--block-label-text-size);
    --block-title-text-weight: 600;
}

.gradio-container label[data-testid="block-label"],
.gradio-container label[data-testid="block-label"] *,
.gradio-container span[data-testid="block-info"],
.gradio-container span[data-testid="block-info"] * {
    background: #CCE5FF !important;
    border-color: #99c7ee !important;
    color: #6666FF !important;
    fill: #6666FF !important;
    font-family: Verdana, Geneva, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif !important;
    font-style: normal !important;
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
.gradio-container label[data-testid="block-label"],
.gradio-container span[data-testid="block-info"],
.gradio-container [data-testid="block-title"],
.gradio-container .block-title {
    border: var(--block-label-border-width) solid #99c7ee !important;
    border-top: none !important;
    border-left: none !important;
    border-radius: var(--block-label-radius) !important;
    box-shadow: var(--block-label-shadow) !important;
    padding: var(--block-label-padding) !important;
}
.gradio-container label[data-testid="block-label"],
.gradio-container label[data-testid="block-label"] *,
.gradio-container span[data-testid="block-info"],
.gradio-container span[data-testid="block-info"] *,
.gradio-container [data-testid="block-title"],
.gradio-container [data-testid="block-title"] *,
.gradio-container .block-title,
.gradio-container .block-title * {
    font-weight: 600 !important;
}
.gradio-container .block label > span,
.gradio-container .block label > span *,
.gradio-container .form label > span,
.gradio-container .form label > span *,
.gradio-container label > span:first-child,
.gradio-container label > span:first-child * {
    font-weight: 600 !important;
}
.strong-label [data-testid="block-label"],
.strong-label [data-testid="block-label"] *,
.strong-label span[data-testid="block-info"],
.strong-label span[data-testid="block-info"] *,
.strong-label [data-testid="block-title"],
.strong-label [data-testid="block-title"] *,
.strong-label .block-label,
.strong-label .block-label *,
.strong-label .block-title,
.strong-label .block-title *,
.strong-label label > span:first-child,
.strong-label label > span:first-child * {
    font-weight: 600 !important;
}
.gradio-container .info-text,
.gradio-container .info-text * {
    font-weight: 400 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="textbox"],
.gradio-container [contenteditable="true"] {
    font-weight: 400 !important;
}
.gradio-container label[data-testid="block-label"] > span:first-child {
    display: none !important;
}

.generate-button {
    background: #6666FF !important;
    color: #ffffff !important;
    border: 1px solid #5555ee !important;
    font-family: Verdana, Geneva, sans-serif !important;
}
.generate-button:hover {
    background: #5555ee !important;
}

#playground-banner {
    padding: 0;
    border-radius: 0;
    margin-bottom: 18px;
    background: transparent;
    border: 0;
}
#playground-banner h1 {
    margin: 0 0 4px 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0;
}
#playground-banner .subtitle {
    margin: 0;
    color: #1e293b;
    font-size: 0.9rem;
}

.info-card {
    padding: 14px 18px;
    border-radius: 8px;
    border: 1px solid #99c7ee;
    border-left: 4px solid #2563eb;
    background: transparent;
    font-size: 0.86rem;
    line-height: 1.55;
    margin-bottom: 16px;
    box-sizing: border-box;
    color: #0f172a;
}
.info-card .card-title,
.info-card .notice-title {
    display: block;
    font-weight: 600;
    font-size: 0.92rem;
    color: #0f172a;
}
.info-card .card-title {
    margin-bottom: 4px;
}
.info-card .notice-title {
    margin-top: 8px;
    margin-bottom: 4px;
}
.info-card ol,
.info-card ul {
    margin: 0;
    padding-left: 18px;
}
.info-card li {
    margin: 2px 0;
}

.main-workspace {
    gap: 18px !important;
    align-items: stretch !important;
}

.prompt-column,
.synthesis-column {
    gap: 14px !important;
}

.control-row,
.settings-slider-row {
    gap: 14px !important;
}

.settings-card {
    margin-top: 2px !important;
}

.generate-button {
    margin-top: 2px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    flex: 0 0 auto !important;
    min-height: 44px !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.output-audio {
    flex: 0 0 auto !important;
    min-height: 190px !important;
}
.output-audio audio {
    width: 100% !important;
}

@media (max-width: 768px) {
    .gradio-container {
        width: calc(100vw - 20px) !important;
    }

}

"""


def build_playground_theme(gr):
    return gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="md",
        text_size="md",
        spacing_size="md",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dots.tts Gradio app.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument(
        "--execution-mode",
        choices=("generate", "generate_stream"),
        default=DEFAULT_EXECUTION_MODE,
        help="Runtime execution mode fixed for the app",
    )
    parser.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        help="Inference precision fixed for the app runtime",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable runtime optimize acceleration",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Default model directory or Hugging Face repo id",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated wav outputs",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Path to the Gradio log file",
    )
    parser.add_argument(
        "--output-retention-count",
        type=int,
        default=DEFAULT_OUTPUT_RETENTION,
        help="Maximum number of generated wav files to keep",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=DEFAULT_MAX_GENERATE_LENGTH,
        help="Maximum generation schedule length fixed for the app runtime",
    )
    parser.add_argument(
        "--default-prompt-name",
        default=DEFAULT_PROMPT_NAME,
        help="Default built-in voice preset name",
    )
    parser.add_argument(
        "--default-precision",
        default=DEFAULT_PRECISION,
        choices=["bfloat16", "float32", "float16"],
        help="Default precision selected in the UI",
    )
    parser.add_argument(
        "--default-num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Default Num Steps selected in the UI",
    )
    parser.add_argument(
        "--default-guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Default Guidance Scale selected in the UI",
    )
    parser.add_argument(
        "--default-speaker-scale",
        type=float,
        default=DEFAULT_SPEAKER_SCALE,
        help="Default Speaker Scale selected in the UI",
    )
    parser.add_argument(
        "--default-max-generate-length",
        type=int,
        default=DEFAULT_MAX_GENERATE_LENGTH,
        help="Default Max Generate Length selected in the UI",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Start the Gradio server without running an initial synthesis warmup.",
    )
    return parser.parse_args(argv)


def build_startup_config_panel(gr, app_config) -> None:
    with gr.Accordion("启动固定参数", open=False):
        gr.Markdown("只读。修改这部分需要重启服务并传入新的启动参数。")
        gr.Textbox(
            label="Model",
            value=app_config.default_model_name_or_path,
            interactive=False,
        )
        with gr.Row():
            gr.Textbox(
                label="Execution Mode",
                value=app_config.execution_mode,
                interactive=False,
            )
            gr.Textbox(
                label="Precision",
                value=app_config.precision,
                interactive=False,
            )
        with gr.Row():
            gr.Number(
                label="Max Generate Length",
                value=app_config.max_generate_length,
                precision=0,
                interactive=False,
            )
            gr.Checkbox(
                label="Optimize",
                value=app_config.optimize,
                interactive=False,
            )


def build_demo(gr, app_config, app_service) -> "gr.Blocks":
    from apps.gradio.service import (
        GRADIO_SYNTHESIS_MODE_CHOICES,
        SynthesisRequest,
        build_prompt_choice_items,
        resolve_prompt_selection,
    )

    sampling_controls = app_service.sampling_controls()
    sampling_locked = bool(sampling_controls["locked"])

    def select_prompt_preset(prompt_name: str):
        audio_path, prompt_text = resolve_prompt_selection(
            prompt_name,
            app_config.prompt_presets,
        )
        return audio_path, prompt_text

    def run_synthesis(
        text: str,
        synthesis_mode: str,
        prompt_audio_path: str | None,
        prompt_text: str,
        ode_method: str,
        num_steps: float,
        guidance_scale: float,
        speaker_scale: float,
        normalize_text: bool,
        seed: float,
    ):
        resolved_synthesis_mode = synthesis_mode if DEBUG_GRADIO_ENABLED else "tts"
        request = SynthesisRequest(
            model_name_or_path=app_config.default_model_name_or_path,
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            execution_mode=app_config.execution_mode,
            template_name=resolved_synthesis_mode,
            ode_method=ode_method,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            speaker_scale=float(speaker_scale),
            normalize_text=normalize_text,
            seed=int(seed),
        )
        result = app_service.generate(request)
        return result.audio_path, result.metrics

    show_prompt_preset = bool(app_config.prompt_presets)

    with gr.Blocks(title="dots.tts") as demo:
        gr.HTML(
            "<style>\n"
            + PLAYGROUND_CSS
            + "\n</style>\n"
            + """
            <div id="playground-banner">
              <h1>dots.tts</h1>
              <p class="subtitle">Fully-continuous Autoregressive TTS · 48 kHz · Voice Cloning</p>
            </div>
            """,
        )

        gr.HTML(
            """
            <div class="info-card">
              <span class="card-title">使用说明 · Instructions</span>
              <ol>
                <li>上传参考音频并填写对应转写文本 · Upload prompt audio and fill in its transcript.</li>
                <li>在文本框中输入要合成的内容 · Enter the text to synthesize.</li>
                <li>点击 <b>Generate</b> 合成声音 · Click <b>Generate</b> to synthesize speech.</li>
              </ol>
            </div>
            """,
        )

        with gr.Row(equal_height=True, elem_classes="main-workspace"):
            with gr.Column(scale=1, min_width=480, elem_classes="prompt-column"):
                prompt_preset = gr.Dropdown(
                    label="音色 · Voice Preset",
                    choices=build_prompt_choice_items(app_config.prompt_presets),
                    value=app_config.default_prompt_name,
                    info="内置音色clone样本；选择后自动填入参考音频与转写。",
                    elem_id="voice-preset-dropdown",
                    elem_classes="strong-label",
                    visible=show_prompt_preset,
                )
                prompt_audio_path = gr.Audio(
                    label="参考音频 · Prompt Audio",
                    sources=["upload"],
                    type="filepath",
                    value=app_config.default_prompt_audio_path,
                    elem_classes="strong-label",
                )
                prompt_text = gr.Textbox(
                    label="参考音频转写 · Prompt Text",
                    lines=5,
                    value=app_config.default_prompt_text,
                    placeholder="Prompt audio 对应的文本转写（continuation cloning 必填）",
                    elem_classes="strong-label",
                )

            with gr.Column(scale=1, min_width=480, elem_classes="synthesis-column"):
                text = gr.Textbox(
                    label="待合成文本 · Text",
                    lines=5,
                    max_lines=8,
                    value=DEFAULT_INPUT_TEXT,
                    placeholder="输入待合成的文本",
                    elem_classes="strong-label",
                )
                with gr.Accordion("⚙️ Settings", open=False, elem_classes="settings-card"):
                    if sampling_locked:
                        gr.Textbox(
                            label="Solver",
                            value=str(sampling_controls["solver"]),
                            interactive=False,
                        )
                    with gr.Row(elem_classes="settings-slider-row"):
                        num_steps = gr.Slider(
                            label="Num Steps",
                            minimum=1,
                            maximum=32,
                            step=1,
                            value=sampling_controls["num_steps"],
                            interactive=not sampling_locked,
                        )
                    with gr.Row(elem_classes="settings-slider-row"):
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0 if sampling_locked else 1.0,
                            maximum=3.0,
                            step=0.1,
                            value=sampling_controls["guidance_scale"],
                            interactive=not sampling_locked,
                        )
                    with gr.Row(elem_classes="control-row"):
                        seed = gr.Number(
                            label="Seed",
                            value=DEFAULT_SEED,
                            precision=0,
                            scale=1,
                            min_width=180,
                        )
                        normalize_text = gr.Checkbox(
                            label="Normalize Text",
                            value=False,
                            scale=1,
                            min_width=180,
                        )
                generate = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg",
                    elem_classes="generate-button",
                )
                audio_out = gr.Audio(
                    label="生成音频 · Output",
                    type="filepath",
                    elem_classes="output-audio",
                )

        if DEBUG_GRADIO_ENABLED:
            with gr.Accordion("Debug", open=False):
                synthesis_mode = gr.Dropdown(
                    label="SynthesisMode",
                    choices=list(GRADIO_SYNTHESIS_MODE_CHOICES),
                    value="tts",
                    info="选择合成模式；界面显示名会自动映射到 runtime 对应模板。",
                )
                ode_method = gr.Textbox(
                    label="ODE Method",
                    value=sampling_controls["ode_method"],
                    lines=1,
                    interactive=not sampling_locked,
                )
                speaker_scale = gr.Slider(
                    label="Speaker Scale",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    value=app_config.default_speaker_scale,
                    info="说话人 x-vector 强度",
                )
                metrics = gr.JSON(label="Metrics", value=app_service.metadata())
                build_startup_config_panel(gr, app_config)
        else:
            synthesis_mode = gr.State(value="tts")
            ode_method = gr.State(value=sampling_controls["ode_method"])
            speaker_scale = gr.State(value=app_config.default_speaker_scale)
            metrics = gr.State(value={})

        generate.click(
            fn=run_synthesis,
            inputs=[
                text,
                synthesis_mode,
                prompt_audio_path,
                prompt_text,
                ode_method,
                num_steps,
                guidance_scale,
                speaker_scale,
                normalize_text,
                seed,
            ],
            outputs=[audio_out, metrics],
            concurrency_limit=1,
        )
        prompt_preset.change(
            fn=select_prompt_preset,
            inputs=[prompt_preset],
            outputs=[prompt_audio_path, prompt_text],
            concurrency_limit=1,
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


def main() -> None:
    args = parse_args()
    import gradio as gr
    from loguru import logger

    from apps.gradio.service import GradioAppService, build_gradio_app_config
    from dots_tts.utils.logging import configure_logging

    configure_logging(log_file=args.log_file)
    logger.info(
        "Gradio app starting: host={} port={} model_name_or_path={} output_dir={} "
        "log_file={} output_retention_count={} max_generate_length={} execution_mode={} precision={} optimize={} "
        "default_prompt_name={} skip_warmup={}",
        args.host,
        args.port,
        args.model_name_or_path,
        args.output_dir,
        args.log_file,
        args.output_retention_count,
        args.max_generate_length,
        args.execution_mode,
        args.precision,
        args.optimize,
        args.default_prompt_name,
        args.skip_warmup,
    )
    app_config = build_gradio_app_config(
        host=args.host,
        port=args.port,
        execution_mode=args.execution_mode,
        precision=args.precision,
        optimize=args.optimize,
        model_name_or_path=args.model_name_or_path,
        output_dir=Path(args.output_dir),
        output_retention_count=args.output_retention_count,
        max_generate_length=args.max_generate_length,
        default_prompt_name=args.default_prompt_name,
        default_precision=args.default_precision,
        default_num_steps=args.default_num_steps,
        default_guidance_scale=args.default_guidance_scale,
        default_speaker_scale=args.default_speaker_scale,
        default_max_generate_length=args.default_max_generate_length,
    )
    app_service = GradioAppService(app_config)
    if args.skip_warmup:
        logger.info("Gradio app warmup skipped by --skip-warmup.")
    else:
        warmup_metrics = app_service.warmup()
        logger.info("Gradio app warmup metrics: {}", warmup_metrics)
    demo = build_demo(gr, app_config, app_service)
    logger.info(
        "Gradio app ready: host={} port={} execution_mode={} precision={} optimize={} default_model_name_or_path={}",
        app_config.host,
        app_config.port,
        app_config.execution_mode,
        app_config.precision,
        app_config.optimize,
        app_config.default_model_name_or_path,
    )
    demo.launch(
        server_name=app_config.host,
        server_port=app_config.port,
        theme=build_playground_theme(gr),
        css=PLAYGROUND_CSS,
    )


if __name__ == "__main__":
    main()
