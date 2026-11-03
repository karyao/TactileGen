"""Pytesseract adapter for YOLO-guided chart text recognition."""

from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np

from src.contracts import Detection
from src.ocr import OCRBackendResult


_LANGUAGE_EXPRESSION = re.compile(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*")
_MIN_PADDING = 2
_MAX_PADDING = 8
_TARGET_TEXT_HEIGHT = 32
_MAX_UPSCALE = 4
_TALL_ASPECT_RATIO = 1.5
_TESSERACT_CONFIG = "--psm 7"


def _load_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Install the OCR dependencies: python -m pip install -r requirements-ocr.txt"
        ) from exc
    return pytesseract


def _not_found_error(module, exc: BaseException) -> bool:
    error_type = getattr(module, "TesseractNotFoundError", None)
    return error_type is not None and isinstance(exc, error_type)


def _crop_with_padding(image: np.ndarray, detection: Detection) -> np.ndarray:
    height, width = image.shape[:2]
    box = detection.box
    padding = max(
        _MIN_PADDING,
        min(_MAX_PADDING, math.ceil(0.1 * min(box.x2 - box.x1, box.y2 - box.y1))),
    )
    left = max(0, math.floor(box.x1) - padding)
    top = max(0, math.floor(box.y1) - padding)
    right = min(width, math.ceil(box.x2) + padding)
    bottom = min(height, math.ceil(box.y2) + padding)
    return image[top:bottom, left:right].copy()


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Tesseract backend supports grayscale, BGR, or BGRA images")


def _orientations(crop: np.ndarray, detection: Detection) -> tuple[np.ndarray, ...]:
    box = detection.box
    if box.y2 - box.y1 <= _TALL_ASPECT_RATIO * (box.x2 - box.x1):
        return (crop,)
    return (
        crop,
        cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


def _recognition_candidates(
    image: np.ndarray, detection: Detection
) -> tuple[np.ndarray, ...]:
    crop = _crop_with_padding(image, detection)
    candidates = []
    for oriented in _orientations(crop, detection):
        gray = _grayscale(oriented)
        scale = min(_MAX_UPSCALE, max(1, math.ceil(_TARGET_TEXT_HEIGHT / gray.shape[0])))
        if scale > 1:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        _, thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        candidates.extend((gray, thresholded))
    return tuple(candidates)


def _candidate_result(data: dict[str, Any]) -> OCRBackendResult | None:
    raw_text = data.get("text", ())
    raw_confidence = data.get("conf", ())
    words = []
    confidences = []
    for text, confidence in zip(raw_text, raw_confidence):
        if not isinstance(text, str) or not text.strip():
            continue
        words.extend(text.split())
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_confidence) and numeric_confidence >= 0:
            confidences.append(min(numeric_confidence, 100.0))
    if not words:
        return None
    normalized_confidence = (
        sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
    )
    return OCRBackendResult(" ".join(words), normalized_confidence)


class TesseractBackend:
    """Recognize text inside authoritative YOLO detection boxes."""

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        if (isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a positive finite number")
        self.timeout_seconds = float(timeout_seconds)
        self._installed_languages: frozenset[str] | None = None

    def _languages(self, pytesseract) -> frozenset[str]:
        if self._installed_languages is None:
            try:
                self._installed_languages = frozenset(pytesseract.get_languages(config=""))
            except Exception as exc:
                if _not_found_error(pytesseract, exc):
                    raise RuntimeError(
                        "Tesseract is not installed or is not available on PATH"
                    ) from exc
                raise
        return self._installed_languages

    def _validate_language(self, pytesseract, language: str) -> None:
        if not isinstance(language, str) or _LANGUAGE_EXPRESSION.fullmatch(language) is None:
            raise ValueError(
                "language must contain Tesseract codes joined with '+', such as eng or eng+fra"
            )
        missing = [code for code in language.split("+")
                   if code not in self._languages(pytesseract)]
        if missing:
            raise RuntimeError(
                "Tesseract language pack is not installed: " + ", ".join(missing)
            )

    def recognize(
        self,
        image: np.ndarray,
        detection: Detection,
        *,
        language: str,
    ) -> OCRBackendResult | None:
        pytesseract = _load_pytesseract()
        self._validate_language(pytesseract, language)
        best = None
        for candidate in _recognition_candidates(image, detection):
            try:
                data = pytesseract.image_to_data(
                    candidate,
                    lang=language,
                    config=_TESSERACT_CONFIG,
                    output_type=pytesseract.Output.DICT,
                    timeout=self.timeout_seconds,
                )
            except RuntimeError as exc:
                raise RuntimeError("Tesseract recognition timed out") from exc
            except Exception as exc:
                if _not_found_error(pytesseract, exc):
                    raise RuntimeError(
                        "Tesseract is not installed or is not available on PATH"
                    ) from exc
                raise
            result = _candidate_result(data)
            if result is not None and (best is None or result.confidence > best.confidence):
                best = result
        return best
