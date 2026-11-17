"""Defaults shared by the command-line launcher and the browser interface."""

from __future__ import annotations

import logging
from pathlib import Path


# Fine-tuned on vertical and horizontal bars (notebooks/kaggle_train_bars.ipynb).
DEFAULT_WEIGHTS = Path("models/yolo_bars_hv/best.pt")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
EXAMPLE_IMAGE = Path(__file__).parents[2] / "tests/fixtures/contracts/val/example.png"
DEFAULT_OCR_LANGUAGE = "eng"
OCR_WARNING = (
    "Text recognition was unavailable. Detection results are still shown. "
    "Check the terminal for details."
)
LEGEND_WARNING = (
    "Legend matching was unavailable. Detection and OCR results are still shown. "
    "Check the terminal for details."
)
RECONSTRUCTION_WARNING = (
    "Chart reconstruction was unavailable. Detection, OCR, and legend results "
    "that completed successfully are still shown. Check the terminal for details."
)

LOGGER = logging.getLogger("src.gradio_app")
