"""Colour filtering retains uncertain evidence and removes proven mismatches."""
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from src.contracts import OCRFrame, RecognizedText
from src.diagnostics.tracking import AnalysisDiagnostics
from src.reconstruction.horizontal import filter_hbars_by_legend
from src.reconstruction.legend_matcher import filter_bars_by_legend
from tests.test_legend_matcher import detection, frame


@pytest.fixture
def chart():
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    detections = frame(
        detection("swatch", "legend_swatch", (5, 5, 15, 15)),
        detection("label", "legend_label", (18, 5, 35, 15)),
        detection("good", "vertical_bar", (5, 25, 15, 55)),
        detection("bad", "vertical_bar", (40, 25, 50, 55)),
    )
    image[5:15, 5:15] = image[25:55, 5:15] = (0, 0, 255)
    image[25:55, 40:50] = (0, 255, 0)
    ocr = OCRFrame(detections.source_key, (RecognizedText("label", "North", 1),))
    return image, detections, ocr


@pytest.mark.parametrize("horizontal", [False, True])
def test_mixed_colours_filter_in_order_with_evidence(chart, horizontal):
    image, raw, ocr = chart
    if horizontal:
        raw = replace(raw, detections=tuple(
            replace(d, role="horizontal_bar") if d.role == "vertical_bar" else d
            for d in raw.detections))
    before = asdict(raw)
    diagnostics = AnalysisDiagnostics(settings={})
    result = (filter_hbars_by_legend if horizontal else filter_bars_by_legend)(
        image, raw, ocr, diagnostics=diagnostics)
    assert [d.id for d in result.frame.detections] == ["swatch", "label", "good"]
    assert [a.bar_id for a in result.mapping.assignments] == ["good"]
    assert result.rejected[0].detection is raw.detections[-1]
    assert [(d.bar_id, d.outcome) for d in result.decisions] == [("good", "kept"), ("bad", "rejected")]
    assert not result.blocked
    assert asdict(raw) == before
    events = diagnostics.record["events"]
    assert events[-1]["rejected_bar_ids"] == ["bad"]


@pytest.mark.parametrize("guard", ["ocr", "legend", "incomplete", "tiny", "crop"])
def test_unreliable_evidence_keeps_bars(chart, guard):
    image, raw, ocr = chart
    if guard == "ocr":
        ocr = None
    elif guard == "legend":
        raw = replace(raw, detections=raw.detections[2:])
    elif guard == "incomplete":
        raw = replace(raw, detections=(*raw.detections,
            detection("extra", "legend_label", (55, 5, 70, 15))))
    elif guard in ("tiny", "crop"):
        raw = replace(raw, detections=(*raw.detections[:-1],
            detection("bad", "vertical_bar", (40, 25, 45 if guard == "tiny" else 41, 55))))
    result = filter_bars_by_legend(image, raw, ocr)
    assert result.frame is raw
    assert not result.rejected
    assert not result.blocked
    assert [a.bar_id for a in result.mapping.assignments] == ["good", "bad"]
    expected_reason = {"ocr": "ocr_unavailable", "legend": "no_legend",
                       "incomplete": "incomplete_legend", "tiny": "unreliable_colour",
                       "crop": "unusable_crop"}[guard]
    assert next(d for d in result.decisions if d.bar_id == "bad").reason == expected_reason
    assert next(d for d in result.decisions if d.bar_id == "bad").outcome == "skipped"


def test_cutoff_is_inclusive_and_does_not_depend_on_label_text(chart):
    image, raw, ocr = chart
    ocr = OCRFrame(raw.source_key, (RecognizedText("label", "", 1),))
    with patch("src.reconstruction.legend_matcher._candidate_error", return_value=.03):
        result = filter_bars_by_legend(image, raw, ocr)
    assert not result.rejected
    assert all(d.outcome == "kept" for d in result.decisions)
    assert all(a.status != "matched" for a in result.mapping.assignments)
