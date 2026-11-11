"""Second OCR pass for category labels the first pass could not read.

Tesseract reads some rotated category labels as a lone symbol, nothing, or one
character, which makes the chart ask the user for the text. Re-reading the same
crop turned 90 and 270 degrees at page-segmentation mode 7 fixes most of them.
This module does that for exactly the labels the completeness check would
otherwise ask about, and leaves everything else alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import cv2
import numpy as np

from src.contracts import Detection, DetectionFrame, VerticalBarSpec
from src.ocr_crop import crop_text_region
from src.tactile.completeness import _suspicious_categories, _unreadable_categories
from src.tesseract_ocr import _MAX_UPSCALE, _TARGET_TEXT_HEIGHT, _grayscale, _load_pytesseract

PAGE_SEGMENTATION_MODE = 7   # one line of text
ROTATIONS = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
OCR_TIMEOUT_SECONDS = 5

Reader = Callable[[np.ndarray, Detection], "tuple[str, float] | None"]


def _candidates(crop: np.ndarray) -> list[np.ndarray]:
    """Greyscale and Otsu-thresholded versions of the crop turned 90 and 270 degrees."""
    found = []
    for rotation in ROTATIONS:
        gray = _grayscale(cv2.rotate(crop, rotation))
        scale = min(_MAX_UPSCALE, max(1, math.ceil(_TARGET_TEXT_HEIGHT / gray.shape[0])))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        found += [gray, thresholded]
    return found


def tesseract_rotated(image: np.ndarray, detection: Detection) -> tuple[str, float] | None:
    """Best (text, confidence) over the rotated crops at psm 7, or None if nothing was read."""
    pytesseract = _load_pytesseract()
    best = None
    for candidate in _candidates(crop_text_region(image, detection)):
        try:
            data = pytesseract.image_to_data(
                candidate, lang="eng", config=f"--psm {PAGE_SEGMENTATION_MODE}",
                output_type=pytesseract.Output.DICT, timeout=OCR_TIMEOUT_SECONDS)
        except RuntimeError:
            continue   # timed out on this candidate
        words = []
        for text, confidence in zip(data.get("text", ()), data.get("conf", ())):
            try:
                value = float(confidence)
            except (TypeError, ValueError):
                continue
            if isinstance(text, str) and text.strip() and value >= 0:
                words.append((text.strip(), min(value, 100.0)))
        if not words:
            continue
        result = (" ".join(w for w, _ in words), sum(c for _, c in words) / len(words) / 100.0)
        if best is None or result[1] > best[1]:
            best = result
    return best


def _x_labels(frame: DetectionFrame) -> list[Detection]:
    """The x tick labels in the order ``reconstruct_vbar`` assigned them to categories."""
    from src.reconstruct import MIN_CONFIDENCE, _best, _centre, _group_roles

    roles = _group_roles(frame, MIN_CONFIDENCE)
    plot = _best(roles.get("plot_area", []))
    margin = max(8.0, 0.05 * (plot.box.x2 - plot.box.x1)) if plot is not None else 0.0
    return sorted((d for d in roles.get("x_tick_label", [])
                   if plot is None or plot.box.x1 - margin <= _centre(d)[0] <= plot.box.x2 + margin),
                  key=lambda d: _centre(d)[0])


def _flagged(spec: VerticalBarSpec) -> list[int]:
    """Categories the completeness check would ask the user for: unreadable or a lone character."""
    single = [j for j, reason in _suspicious_categories(spec) if "single character" in reason]
    return sorted(set(_unreadable_categories(spec)) | set(single))


def reread_categories(spec: VerticalBarSpec, frame: DetectionFrame, image: np.ndarray,
                      *, reader: Reader = tesseract_rotated) -> VerticalBarSpec:
    """Return ``spec`` with flagged categories replaced by a re-read that passes the same rule.

    A re-read that is still unreadable or still a lone character is discarded, so the
    label stays a needs_input item. Accepted labels are marked ``category_reread:N`` and
    one ``category_reread_index:J`` flag each, so the confirm step can list them.
    """
    flagged = _flagged(spec)
    if not flagged:
        return spec
    labels = _x_labels(frame)
    if len(labels) != len(spec.categories):
        return spec
    categories = list(spec.categories)
    confidence = dict(spec.confidence)
    accepted: list[int] = []
    for j in flagged:
        result = reader(image, labels[j])
        if result is None:
            continue
        trial = replace(spec, categories=[*categories[:j], result[0], *categories[j + 1:]],
                        confidence={k: v for k, v in confidence.items() if k != f"category[{j}]"})
        if j in _flagged(trial):
            continue   # still fails the rule
        categories, confidence = trial.categories, trial.confidence
        accepted.append(j)
    if not accepted:
        return spec
    flags = list(spec.flags)
    if not any(key.startswith("category[") for key in confidence):
        flags = [f for f in flags if f != "category_unreadable"]
    flags += [f"category_reread:{len(accepted)}", *(f"category_reread_index:{j}" for j in accepted)]
    return replace(spec, categories=categories, confidence=confidence, flags=flags)
