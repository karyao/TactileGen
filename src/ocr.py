"""Engine-neutral OCR orchestration for detected chart text regions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.contracts import (
    TEXT_ROLES,
    Detection,
    DetectionFrame,
    OCRFrame,
    RecognizedText,
)
from src.contract_io import load_detection_frame, write_ocr_frame


@dataclass(frozen=True)
class OCRBackendResult:
    """Text and confidence returned by an OCR backend for one detection."""

    text: str
    confidence: float


class OCRBackend(Protocol):
    """Recognition boundary implemented by concrete OCR engines.

    Backends receive the validated source image without engine-specific color
    conversion. Each adapter is responsible for converting crops into the
    channel layout and memory representation required by its OCR engine.
    """

    def recognize(
        self,
        image: np.ndarray,
        detection: Detection,
        *,
        language: str,
    ) -> OCRBackendResult | None:
        """Recognize one detection, or return ``None`` when it is unreadable."""
        ...


def _validate_image(image: np.ndarray, frame: DetectionFrame) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D multi-channel array")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or (image.ndim == 3 and image.shape[2] <= 0):
        raise ValueError("image dimensions must be positive")
    if (height, width) != (frame.height, frame.width):
        raise ValueError("image dimensions do not match the detection frame")


def run_ocr(
    image: np.ndarray,
    frame: DetectionFrame,
    backend: OCRBackend,
    language: str,
) -> OCRFrame:
    """Recognize all detected text roles in their original detection order.

    Backends receive a read-only view of the complete image and the authoritative
    detection. They must not return geometry; boxes remain in ``DetectionFrame``.
    ``None`` and blank text are treated as an attempted but unreadable region.
    """
    if not isinstance(frame, DetectionFrame):
        raise TypeError("frame must be a DetectionFrame")
    _validate_image(image, frame)
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a nonempty string")
    recognize = getattr(backend, "recognize", None)
    if not callable(recognize):
        raise TypeError("backend must provide a callable recognize method")

    readonly_image = image.view()
    readonly_image.setflags(write=False)
    text_detections = tuple(
        detection for detection in frame.detections if detection.role in TEXT_ROLES
    )
    recognize_many = getattr(backend, "recognize_many", None)
    if callable(recognize_many):
        results = recognize_many(readonly_image, text_detections, language=language)
        if not isinstance(results, (tuple, list)) or len(results) != len(text_detections):
            raise TypeError(
                "backend recognize_many must return one OCRBackendResult or None per detection"
            )
    else:
        results = [
            recognize(readonly_image, detection, language=language)
            for detection in text_detections
        ]

    regions = []
    for detection, result in zip(text_detections, results, strict=True):
        if result is None:
            regions.append(RecognizedText(detection.id, "", 0.0))
            continue
        if not isinstance(result, OCRBackendResult):
            raise TypeError("backend recognize must return OCRBackendResult or None")
        if not isinstance(result.text, str):
            raise TypeError("backend result text must be a string")
        if not result.text.strip():
            regions.append(RecognizedText(detection.id, "", 0.0))
            continue
        regions.append(RecognizedText(detection.id, result.text, result.confidence))

    return OCRFrame(frame.source_key, tuple(regions))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize text in regions from a YOLO detection JSON file."
    )
    parser.add_argument(
        "--engine", choices=("tesseract", "rapidocr"), default="tesseract"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="eng")
    return parser.parse_args(argv)


def _read_image(path: Path) -> np.ndarray:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_ocr_atomically(path: Path, frame: OCRFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        write_ocr_frame(temporary_path, frame)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_command(args: argparse.Namespace) -> OCRFrame:
    if not args.detections.is_file():
        raise FileNotFoundError(f"Missing detections: {args.detections}")
    image = _read_image(args.image)
    detections = load_detection_frame(args.detections)
    if args.engine == "tesseract":
        from src.tesseract_ocr import TesseractBackend

        backend = TesseractBackend()
    elif args.engine == "rapidocr":
        from src.rapidocr_ocr import RapidOCRBackend

        backend = RapidOCRBackend()
    else:  # argparse constrains this today; keep dispatch explicit for future engines.
        raise ValueError(f"Unsupported OCR engine: {args.engine}")
    result = run_ocr(image, detections, backend, args.language)
    _write_ocr_atomically(args.output, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        args = parse_args(argv)
        result = _run_command(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "source_key": result.source_key,
        "regions": len(result.regions),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    # Keep protocol/result class identity stable when the adapter imports
    # ``src.ocr`` while this file is executing under its ``-m`` name.
    sys.modules.setdefault("src.ocr", sys.modules[__name__])
    main()
