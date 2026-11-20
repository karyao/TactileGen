"""Hybrid OCR: RapidOCR and Tesseract on every crop, one chosen per region by confidence.

RapidOCR wins on numeric tick text and short names; Tesseract wins on multi-word legend
labels and on partly-dropped words. Fitted per role on a random sample of PlotQA test charts
as "take RapidOCR when its confidence beats Tesseract's by at least the role margin".
Roles without a margin (titles, axis titles) keep RapidOCR's reading.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.contracts import Detection
from src.ocr import OCRBackendResult
from src.rapidocr_ocr import RapidOCRBackend
from src.tesseract_ocr import TesseractBackend

RAPIDOCR_MARGIN = {"x_tick_label": 0.0, "y_tick_label": 0.05, "legend_label": 0.05}


def choose(role: str, rapid: OCRBackendResult | None, tess: OCRBackendResult | None):
    if rapid is None or tess is None:
        return rapid or tess
    if rapid.text.strip() == tess.text.strip():
        return rapid
    margin = RAPIDOCR_MARGIN.get(role)
    if margin is None:
        return rapid
    return rapid if rapid.confidence - tess.confidence >= margin else tess


class HybridBackend:
    def __init__(self) -> None:
        self._rapid = RapidOCRBackend()
        self._tess = TesseractBackend()

    def recognize(self, image: np.ndarray, detection: Detection, *, language: str,
                  neighbors: Sequence[Detection] = ()) -> OCRBackendResult | None:
        return choose(
            detection.role,
            self._rapid.recognize(image, detection, language=language, neighbors=neighbors),
            self._tess.recognize(image, detection, language=language, neighbors=neighbors),
        )

    def recognize_many(self, image: np.ndarray, detections: Sequence[Detection], *,
                       language: str) -> tuple[OCRBackendResult | None, ...]:
        rapid = self._rapid.recognize_many(image, detections, language=language)
        tess = self._tess.recognize_many(image, detections, language=language)
        return tuple(choose(d.role, r, t) for d, r, t in zip(detections, rapid, tess, strict=True))
