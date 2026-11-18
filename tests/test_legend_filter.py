"""Legend colour checks report mismatches without discarding bars."""
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from src.contracts import OCRFrame, RecognizedText
from src.diagnostics.tracking import AnalysisDiagnostics
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
    result = filter_bars_by_legend(
        image, raw, ocr, bar_role="horizontal_bar" if horizontal else "vertical_bar",
        diagnostics=diagnostics)
    assert result.frame is raw
    assert [a.bar_id for a in result.mapping.assignments] == ["good", "bad"]
    assert result.mapping.assignments[-1].status == "weak_match"
    assert result.mismatched_bar_ids == ("bad",)
    assert [(d.bar_id, d.outcome) for d in result.decisions] == [("good", "kept"), ("bad", "mismatch")]
    assert asdict(raw) == before
    events = diagnostics.record["events"]
    assert events[-1]["mismatched_bar_ids"] == ["bad"]


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
    assert not result.mismatched_bar_ids
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
    assert not result.mismatched_bar_ids
    assert all(d.outcome == "kept" for d in result.decisions)
    assert all(a.status != "matched" for a in result.mapping.assignments)


@pytest.mark.parametrize("bar_role", ["vertical_bar", "horizontal_bar"])
@pytest.mark.parametrize("all_mismatch", [False, True])
def test_mixed_roles_only_score_selected_bars(chart, bar_role, all_mismatch):
    from src.reconstruction.legend_matcher import match_legend_to_bars

    image, raw, ocr = chart
    opposite_role = "horizontal_bar" if bar_role == "vertical_bar" else "vertical_bar"
    opposite = detection("opposite", opposite_role, (40, 25, 50, 55))
    raw = replace(raw, detections=(
        raw.detections[0], opposite, raw.detections[1],
        *(replace(d, role=bar_role) for d in raw.detections[2:]),
    ))
    if all_mismatch:
        image[25:55, 5:15] = (0, 255, 0)
    before = asdict(raw)
    mapping = match_legend_to_bars(image, raw, ocr, bar_role=bar_role)
    assert [a.bar_id for a in mapping.assignments] == ["good", "bad"]
    assert mapping.assignments[-1].status == "weak_match"
    assert mapping.assignments[0].status == ("weak_match" if all_mismatch else "matched")
    result = filter_bars_by_legend(image, raw, ocr, bar_role=bar_role)
    assert result.frame is raw
    assert result.mapping == mapping
    assert [d.bar_id for d in result.decisions] == ["good", "bad"]
    assert result.mismatched_bar_ids == (("good", "bad") if all_mismatch else ("bad",))
    assert asdict(raw) == before


@pytest.mark.parametrize("bar_role", ["vertical_bar", "horizontal_bar"])
def test_no_selected_bars_keeps_original_frame(chart, bar_role):
    from src.reconstruction.legend_matcher import match_legend_to_bars

    image, raw, ocr = chart
    opposite_role = "horizontal_bar" if bar_role == "vertical_bar" else "vertical_bar"
    raw = replace(raw, detections=tuple(
        replace(d, role=opposite_role) if d.role == "vertical_bar" else d
        for d in raw.detections))
    assert not match_legend_to_bars(image, raw, ocr, bar_role=bar_role).assignments
    result = filter_bars_by_legend(image, raw, ocr, bar_role=bar_role)
    assert result.frame is raw
    assert not result.mapping.assignments
    assert not result.decisions
    assert not result.mismatched_bar_ids


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("function", ["match_legend_to_bars", "filter_bars_by_legend",
                                     "_score_legend_to_bars"])
def test_invalid_bar_role_is_rejected(chart, empty, function):
    from src.reconstruction import legend_matcher

    image, raw, ocr = chart
    if empty:
        raw = replace(raw, detections=())
    with pytest.raises(ValueError, match="bar_role"):
        getattr(legend_matcher, function)(image, raw, ocr, bar_role="dot")
