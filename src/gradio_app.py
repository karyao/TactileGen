"""Local Gradio application for inspecting one chart detection at a time."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.borealis import BOREALIS_CSS, borealis_theme
from src.detection import DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU, load_model, run_detection
from src.presentation import render_detection_overlay, summarize_detections


DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
EXAMPLE_IMAGE = Path(__file__).parents[1] / "tests/fixtures/contracts/val/example.png"

LOGGER = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """An upload problem that is safe to explain in the browser."""


@dataclass(frozen=True)
class AnalysisResult:
    summary: str
    overlay_rgb: Any
    detection_count: int


def ready_status() -> str:
    return '<div role="status" aria-live="polite">Ready for a chart image.</div>'


def analyzing_status() -> str:
    return '<div role="status" aria-live="polite">Analyzing chart.</div>'


def complete_status(detection_count: int) -> str:
    if detection_count == 0:
        message = "Analysis complete. No detections found."
    elif detection_count == 1:
        message = "Analysis complete. 1 detection found."
    else:
        message = f"Analysis complete. {detection_count} detections found."
    return f'<div role="status" aria-live="polite">{message}</div>'


def alert_message(message: str) -> str:
    return f'<div role="alert">{message}</div>'


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install the application dependencies: "
            "python -m pip install -r requirements-app.txt"
        ) from exc
    return gr


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or mps")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def _report_progress(progress, fraction: float, description: str) -> None:
    if progress is not None:
        progress(fraction, desc=description)


def decode_upload(image_path: str | Path):
    """Validate uploaded content and return its server path and BGR pixels."""
    import cv2
    from PIL import Image, UnidentifiedImageError

    if image_path is None:
        raise UploadValidationError("Choose a chart image before analyzing it.")
    path = Path(image_path)
    if not path.is_file():
        raise UploadValidationError("Choose a chart image before analyzing it.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The image must be 20 MB or smaller.")
    try:
        with Image.open(path) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise UploadValidationError("Upload a PNG, JPEG, or WebP image.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    "The decoded image must contain no more than 40 megapixels."
                )
            source.verify()
    except UploadValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise UploadValidationError(
            "The decoded image must contain no more than 40 megapixels."
        ) from exc
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise UploadValidationError(
            "The uploaded image could not be read. Try exporting it again."
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise UploadValidationError(
            "The uploaded image could not be read. Try exporting it again."
        )
    return path, image


def analyze_image(
    model,
    image_path: str | Path,
    confidence: float,
    nms_iou: float,
    *,
    device: str | None = None,
    progress: Callable[..., None] | None = None,
) -> AnalysisResult:
    """Validate, detect, summarize, and annotate one uploaded chart."""
    import cv2

    _report_progress(progress, 0.05, "Validating image")
    path, image = decode_upload(image_path)
    _report_progress(progress, 0.20, "Detecting chart elements")
    frame = run_detection(
        model, path, confidence=confidence, nms_iou=nms_iou, device=device
    )
    _report_progress(progress, 0.80, "Rendering annotations")
    overlay = render_detection_overlay(image, frame)
    rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    _report_progress(progress, 1.0, "Analysis complete")
    return AnalysisResult(
        summary=summarize_detections(frame),
        overlay_rgb=rgb_overlay,
        detection_count=len(frame.detections),
    )


def build_app(model, *, device: str | None = None):
    """Build the interface around an already-loaded model without launching it."""
    gr = _import_gradio()

    def ready_results():
        return (
            gr.update(value=ready_status()),
            gr.update(value="Upload a chart and choose Analyze chart.", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    def begin_analysis():
        return (
            *(gr.update(interactive=False) for _ in range(5)),
            gr.update(value=analyzing_status()),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    def analyze_event(image_path, confidence, nms_iou, progress=gr.Progress()):
        try:
            result = analyze_image(
                model,
                image_path,
                confidence,
                nms_iou,
                device=device,
                progress=progress,
            )
            return (
                gr.update(value=complete_status(result.detection_count)),
                gr.update(value="", visible=False),
                gr.update(value=result.summary, visible=True),
                gr.update(value=result.overlay_rgb, visible=True),
                gr.update(value="", visible=False),
            )
        except UploadValidationError as exc:
            LOGGER.info("Rejected upload: %s", exc)
            message = str(exc)
        except Exception:
            LOGGER.exception("Chart analysis failed")
            message = "Analysis failed. Check the terminal and try another image."
        return (
            gr.update(value=ready_status()),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=alert_message(message), visible=True),
        )

    def finish_analysis():
        return tuple(gr.update(interactive=True) for _ in range(5))

    def clear_all():
        return (
            gr.update(value=None, interactive=True),
            gr.update(value=DEFAULT_CONFIDENCE, interactive=True),
            gr.update(value=DEFAULT_NMS_IOU, interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            *ready_results(),
        )

    with gr.Blocks(
        title="TactileGen — Statistical Chart Element Detector",
        analytics_enabled=False,
        fill_width=False,
    ) as demo:
        with gr.Column(elem_classes="borealis-app"):
            with gr.Column(elem_classes="borealis-header"):
                gr.Markdown(
                    "# TactileGen\n\n"
                    "**Statistical Chart Element Detector**\n\n"
                    "Upload a vertical categorical bar chart to inspect the regions "
                    "detected by YOLO."
                )
                gr.HTML('<div class="borealis-divider" aria-hidden="true"></div>')

            gr.Markdown(
                "**Detection scope:** Validated for vertical categorical bar charts. "
                "Other chart types are unsupported. This prototype detects chart "
                "elements and bounding boxes; it does not read text, values, or trends.",
                elem_classes="borealis-scope",
            )

            with gr.Row(elem_classes="borealis-two-column"):
                with gr.Column(elem_classes="borealis-upload"):
                    image_input = gr.Image(
                        type="filepath",
                        sources=["upload"],
                        image_mode=None,
                        format="png",
                        label="Vertical bar chart image",
                        placeholder="Upload a chart image",
                        buttons=["fullscreen"],
                        alt_text="Preview of the uploaded chart",
                    )
                    gr.Markdown(
                        "Supported formats: PNG, JPEG, and WebP. Maximum file size: "
                        "20 MB. Maximum decoded size: 40 megapixels."
                    )
                    gr.Examples(
                        examples=[[str(EXAMPLE_IMAGE)]],
                        inputs=[image_input],
                        label="Example chart",
                        run_on_click=False,
                        cache_examples=False,
                    )
                    confidence = gr.Slider(
                        0.05,
                        1.0,
                        value=DEFAULT_CONFIDENCE,
                        step=0.05,
                        label="Minimum detection confidence",
                        info="Hide detections below this confidence score.",
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        nms_iou = gr.Slider(
                            0.10,
                            0.95,
                            value=DEFAULT_NMS_IOU,
                            step=0.05,
                            label="NMS IoU for duplicate suppression",
                            info=(
                                "Controls duplicate suppression during inference. "
                                "This is not evaluation or box-matching IoU."
                            ),
                        )
                    with gr.Row(elem_classes="borealis-nav"):
                        analyze_button = gr.Button("Analyze chart", variant="primary")
                        clear_button = gr.Button("Clear", variant="secondary")

                with gr.Column(elem_classes="borealis-results"):
                    status_output = gr.HTML(
                        ready_status(), elem_classes="borealis-status"
                    )
                    empty_output = gr.Markdown(
                        "Upload a chart and choose Analyze chart.",
                        visible=True,
                        elem_classes="borealis-empty",
                    )
                    summary_output = gr.Markdown(
                        "",
                        label="Detection summary",
                        visible=False,
                        elem_classes="borealis-summary",
                    )
                    annotated_output = gr.Image(
                        type="numpy",
                        format="png",
                        label="Annotated detections",
                        interactive=False,
                        visible=False,
                        buttons=["download", "fullscreen"],
                        alt_text="Uploaded chart with labeled detection boxes",
                    )
                    error_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )

        image_input.change(
            ready_results,
            outputs=[
                status_output,
                empty_output,
                summary_output,
                annotated_output,
                error_output,
            ],
            queue=False,
            api_visibility="private",
        )

        controls = [image_input, confidence, nms_iou, analyze_button, clear_button]
        result_components = [
            status_output,
            empty_output,
            summary_output,
            annotated_output,
            error_output,
        ]
        start = analyze_button.click(
            begin_analysis,
            outputs=[*controls, *result_components],
            queue=False,
            trigger_mode="once",
            api_visibility="private",
        )
        analysis = start.then(
            analyze_event,
            inputs=[image_input, confidence, nms_iou],
            outputs=result_components,
            show_progress="full",
            show_progress_on=annotated_output,
            concurrency_limit=1,
            concurrency_id="tactilegen-inference",
            api_name="analyze",
            api_description="Detect and annotate chart elements in one uploaded image.",
            api_visibility="private",
        )
        analysis.then(
            finish_analysis,
            outputs=controls,
            queue=False,
            api_visibility="private",
        )
        clear_button.click(
            clear_all,
            outputs=[*controls, *result_components],
            queue=False,
            api_visibility="private",
        )

    return demo.queue(default_concurrency_limit=1, max_size=4)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    try:
        _import_gradio()
        model = load_model(args.weights)
        demo = build_app(model, device=args.device)
        theme = borealis_theme()
    except Exception as exc:
        parser.error(f"Could not start TactileGen: {exc}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=False,
        max_file_size=MAX_UPLOAD_BYTES,
        theme=theme,
        css=BOREALIS_CSS,
        run_history=False,
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
