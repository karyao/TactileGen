"""Engine-neutral OCR orchestration for detected chart text regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.contracts import (
    TEXT_ROLES,
    Detection,
    DetectionFrame,
    OCRFrame,
    RecognizedText,
)


@dataclass(frozen=True)
class OCRBackendResult:
    """Text and confidence returned by an OCR backend for one detection."""

    text: str
    confidence: float


class OCRBackend(Protocol):
    """Recognition boundary implemented by concrete OCR engines."""

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
    regions = []
    for detection in frame.detections:
        if detection.role not in TEXT_ROLES:
            continue
        result = recognize(readonly_image, detection, language=language)
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
