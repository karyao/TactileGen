"""Start TactileGen: parse options, load the model, build the page, and launch.

The implementation lives in src.gradio_ui. Imports below preserve the original
Python API so existing scripts can still import its helpers from this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.borealis import BOREALIS_CSS, borealis_theme
from src.detection import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_NMS_IOU,
    MAX_IMGSZ,
    load_model,
    validate_imgsz,
)
from src.gradio_ui.analysis import (
    UploadValidationError,
    AnalysisResult,
    decode_upload,
    analyze_image,
)
from src.gradio_ui.dependencies import import_gradio as _import_gradio, launch_app
from src.gradio_ui.formatting import (
    generation_report,
    ready_status,
    analyzing_status,
    complete_status,
    alert_message,
    completeness_summary,
)
from src.gradio_ui.layout import build_app
from src.gradio_ui.review import (
    ReviewFieldBinding,
    ReviewModel,
    ManualGenerationResult,
    ManualGenerationBlocked,
    build_review_model,
    review_text_rows,
    review_value_rows,
    update_review_draft,
    review_overrides,
    generate_review_submission,
)
from src.gradio_ui.settings import (
    DEFAULT_WEIGHTS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_UPLOAD_BYTES,
    MAX_IMAGE_PIXELS,
    SUPPORTED_FORMATS,
    EXAMPLE_IMAGE,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_OCR_ENGINE,
    OCR_WARNING,
    LEGEND_WARNING,
    RECONSTRUCTION_WARNING,
    LOGGER,
)
from src.ocr.config import CHART_OCR_ENGINES


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _imgsz(value: str) -> int:
    try:
        return validate_imgsz(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"imgsz must be a positive integer between 1 and {MAX_IMGSZ}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or mps")
    parser.add_argument(
        "--imgsz",
        type=_imgsz,
        default=DEFAULT_IMGSZ,
        help=(
            f"Initial YOLO inference size in pixels "
            f"(1–{MAX_IMGSZ}; default: {DEFAULT_IMGSZ})"
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    parser.add_argument(
        "--ocr",
        choices=CHART_OCR_ENGINES,
        default=DEFAULT_OCR_ENGINE,
        help="Initial OCR engine (default rapidocr; selectable in the browser)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    try:
        _import_gradio()
        model = load_model(args.weights)
        demo = build_app(
            model,
            ocr_engine=args.ocr,
            device=args.device,
            imgsz=args.imgsz,
        )
        theme = borealis_theme()
    except Exception as exc:
        parser.error(f"Could not start TactileGen: {exc}")
    launch_app(
        demo,
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=False,
        max_file_size=MAX_UPLOAD_BYTES,
        theme=theme,
        css=BOREALIS_CSS,
    )


if __name__ == "__main__":
    main()
