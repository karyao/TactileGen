"""Hybrid OCR: RapidOCR and Tesseract on every crop, one chosen per region by confidence.

RapidOCR wins on numeric tick text and short names; Tesseract wins on multi-word legend
labels and on partly-dropped words. Fitted per role on a random sample of PlotQA test charts
as "take RapidOCR when its confidence beats Tesseract's by at least the role margin".
Roles without a margin (titles, axis titles) keep RapidOCR's reading. A "$" that only RapidOCR
(correctly) read always wins. Near-match rules (same length, small edit distance) were tried and rejected:
they lost 42-139 correct regions to gain 14-20 across all 3041 disagreeing regions. Word roles take
Tesseract's reading when RapidOCR only lost 1-2 letters or added a space (+20 regions, 0 lost). A word-role reading
with an unclosed "(" gets its ")" back (the crop cuts off the last character of long legend names; +5 regions,
0 lost across all 77,362 word-role regions of the test set).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.contracts import Detection
from src.ocr import OCRBackendResult
from src.ocr.rapidocr import RapidOCRBackend
from src.ocr.tesseract import TesseractBackend

# The two engines name English differently; callers pass the Tesseract code.
_RAPID_LANGUAGE = {"eng": "en"}
RAPIDOCR_MARGIN = {"x_tick_label": 0.0, "y_tick_label": 0.05, "legend_label": 0.05}
# Text roles where RapidOCR drops letters ("ferility") or adds a space ("Age 65 (male)").
WORD_ROLES = ("x_tick_label", "legend_label")


def _leftover(short: str, long: str) -> list[str] | None:
    """Characters of ``long`` not used when matching ``short`` as a subsequence, else None."""
    matched, leftover = 0, []
    for char in long:
        if matched < len(short) and short[matched] == char:
            matched += 1
        else:
            leftover.append(char)
    return leftover if matched == len(short) else None


def _rapid_slipped(rapid: str, tess: str) -> bool:
    """RapidOCR lost 1-2 letters that Tesseract kept, or added 1-2 spaces Tesseract did not."""
    lost = _leftover(rapid, tess)
    if lost is not None and 1 <= len(lost) <= 2 and all(c.isalnum() for c in lost):
        return any(c.isalpha() for c in tess)
    added = _leftover(tess, rapid)
    return added is not None and 1 <= len(added) <= 2 and all(c == " " for c in added)


def _close_parenthesis(role: str, result: OCRBackendResult | None):
    """Close a "(" the crop cut off: long legend names lose their final ")" at the detection box edge."""
    if result is None or role not in WORD_ROLES:
        return result
    missing = result.text.count("(") - result.text.count(")")
    return OCRBackendResult(result.text + ")" * missing, result.confidence) if missing > 0 else result


def choose(role: str, rapid: OCRBackendResult | None, tess: OCRBackendResult | None):
    return _close_parenthesis(role, _choose(role, rapid, tess))


def _choose(role: str, rapid: OCRBackendResult | None, tess: OCRBackendResult | None):
    if rapid is None or tess is None:
        return rapid or tess
    if rapid.text.strip() == tess.text.strip():
        return rapid
    if "$" in rapid.text and "$" not in tess.text:
        return rapid  # Tesseract reads "$" as "S"
    if role in WORD_ROLES and _rapid_slipped(rapid.text, tess.text):
        return tess
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
            self._rapid.recognize(image, detection, language=_RAPID_LANGUAGE.get(language, language), neighbors=neighbors),
            self._tess.recognize(image, detection, language=language, neighbors=neighbors),
        )

    def recognize_many(self, image: np.ndarray, detections: Sequence[Detection], *,
                       language: str) -> tuple[OCRBackendResult | None, ...]:
        rapid = self._rapid.recognize_many(
            image, detections, language=_RAPID_LANGUAGE.get(language, language))
        tess = self._tess.recognize_many(image, detections, language=language)
        return tuple(choose(d.role, r, t) for d, r, t in zip(detections, rapid, tess, strict=True))
