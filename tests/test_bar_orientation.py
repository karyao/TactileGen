"""Orientation policy and filtering, independent of YOLO and OCR inference."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest

from src.bar_orientation import resolve_bar_orientation
from src.contract_io import load_detection_frame
from src.contracts import Box, Detection, DetectionFrame


def frame(vertical=(), horizontal=(), other=()):
    items = [("vertical_bar", c) for c in vertical]
    items += [("horizontal_bar", c) for c in horizontal]
    items += list(other)
    return DetectionFrame("test/chart.png", 200, 100, tuple(
        Detection(f"det-{i}", role, Box(10, 10, 20, 20), confidence)
        for i, (role, confidence) in enumerate(items)
    ))


def resolve(raw, preference="auto"):
    """Check the result contract on every policy-test outcome."""
    result = resolve_bar_orientation(raw, preference)
    assert result.preference == preference
    assert result.source == ("automatic" if preference == "auto" else "manual")
    assert isinstance(result.suppressed_ids, tuple)
    if result.status == "resolved":
        assert result.orientation in ("vertical", "horizontal")
        assert result.effective_frame is not None
        assert result.effective_frame is not raw
    else:
        assert result.status == "uncertain"
        assert result.orientation is None
        assert result.effective_frame is None
        assert result.suppressed_ids == ()
    return result


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_clear_orientation_and_minority_conflict(orientation):
    opposite = "horizontal" if orientation == "vertical" else "vertical"
    raw = frame(**{orientation: (0.9, 0.9), opposite: (0.3,)})
    result = resolve(raw)
    assert result.orientation == orientation
    assert result.reason == "automatic_selection"
    assert all(d.role != f"{opposite}_bar" for d in result.effective_frame.detections)


def test_confidence_weighting_can_overrule_bar_counts():
    result = resolve(frame(vertical=(1.0,) * 10, horizontal=(0.3,) * 11))
    assert result.orientation == "vertical"
    assert result.vertical_score == 10.0
    assert result.horizontal_score == math.fsum([0.3] * 11)


@pytest.mark.parametrize(("vertical", "horizontal", "reason"), [
    ((), (), "no_qualifying_bars"),
    ((0.29,), (0.29,), "no_qualifying_bars"),
    ((0.3,), (), "insufficient_evidence"),
    ((0.3,), (0.3,), "insufficient_evidence"),
    ((0.6,), (0.6,), "ambiguous_orientation"),
    ((0.9,), (0.8,), "ambiguous_orientation"),
])
def test_uncertainty_reason_precedence(vertical, horizontal, reason):
    result = resolve(frame(vertical, horizontal, other=(("chart_title", 1.0),)))
    assert result.status == "uncertain"
    assert result.reason == reason


def test_confidence_floor_is_inclusive():
    assert resolve(frame(vertical=(0.3, 0.3))).orientation == "vertical"
    below = math.nextafter(0.3, 0.0)
    result = resolve(frame(vertical=(below, below)))
    assert result.vertical_score == 0.0
    assert result.reason == "no_qualifying_bars"


def test_winning_score_floor_is_inclusive():
    assert resolve(frame(vertical=(0.6,))).orientation == "vertical"
    result = resolve(frame(vertical=(math.nextafter(0.6, 0.0),)))
    assert result.reason == "insufficient_evidence"


def test_share_floor_is_inclusive_without_epsilon():
    assert resolve(frame(vertical=(0.9,), horizontal=(0.3,))).orientation == "vertical"
    result = resolve(frame(vertical=(0.9,), horizontal=(0.300001,)))
    assert result.reason == "ambiguous_orientation"


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_manual_choice_bypasses_score_and_share_thresholds(orientation):
    opposite = "horizontal" if orientation == "vertical" else "vertical"
    raw = frame(**{orientation: (0.3,), opposite: (1.0, 1.0, 1.0)})
    assert resolve(raw).orientation == opposite
    result = resolve(raw, orientation)
    assert result.orientation == orientation
    assert result.reason == "manual_selection"
    assert getattr(result, f"{orientation}_score") == 0.3
    assert getattr(result, f"{opposite}_score") == 3.0


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("confidence", [None, 0.29])
def test_manual_choice_needs_qualifying_bar(orientation, confidence):
    raw = frame(**{orientation: () if confidence is None else (confidence,)})
    result = resolve(raw, orientation)
    assert result.reason == "requested_orientation_unavailable"
    assert result.status == "uncertain"


@pytest.mark.parametrize("preference", [None, [], {}, 1, True, "", "other", "Auto",
                                        "VERTICAL", "horizontal ", " vertical"])
def test_invalid_preferences_raise_value_error(preference):
    with pytest.raises(ValueError, match="preference"):
        resolve_bar_orientation(frame(), preference)


def test_filtering_preserves_raw_input_metadata_objects_and_order():
    raw = frame(vertical=(0.9, 0.1), horizontal=(0.3, 0.1),
                other=(("chart_title", 0.8), ("dot", 0.9), ("plot_area", 0.8)))
    # Interleave roles to exercise stable retained and suppressed ordering.
    raw = DetectionFrame(raw.source_key, raw.width, raw.height,
                         tuple(raw.detections[i] for i in (4, 2, 0, 3, 5, 1, 6)))
    original = raw.detections
    result = resolve(raw)
    effective = result.effective_frame
    assert result.suppressed_ids == ("det-2", "det-3")
    assert effective.source_key == raw.source_key
    assert (effective.width, effective.height) == (raw.width, raw.height)
    expected = tuple(d for d in original if d.role != "horizontal_bar")
    assert effective.detections == expected
    assert all(actual is retained for actual, retained in zip(effective.detections, expected))
    assert raw.detections is original
    assert any(d.confidence == 0.1 for d in effective.detections)


def test_resolved_without_removals_still_returns_new_frame():
    raw = frame(vertical=(0.9,))
    result = resolve(raw)
    assert result.effective_frame == raw
    assert result.suppressed_ids == ()


def test_result_is_frozen():
    result = resolve(frame(vertical=(0.9,)))
    with pytest.raises(FrozenInstanceError):
        result.orientation = "horizontal"


def test_duplicate_outside_plot_bars_still_contribute_to_raw_scores():
    box = Box(1, 1, 5, 5)
    raw = DetectionFrame("test/chart.png", 200, 100, (
        Detection("plot", "plot_area", Box(50, 20, 180, 90), 1.0),
        Detection("a", "vertical_bar", box, 0.3),
        Detection("b", "vertical_bar", box, 0.3),
    ))
    result = resolve(raw)
    assert result.vertical_score == 0.6
    assert result.orientation == "vertical"
    assert result.effective_frame.detections == raw.detections


def test_real_mixed_predictions_resolve_horizontal_and_remove_vertical_bars():
    fixture = Path(__file__).resolve().parent / "fixtures/orientation/mixed_predictions.detections.json"
    raw = load_detection_frame(fixture)
    original = raw.detections
    assert {d.role for d in original if d.role.endswith("_bar")
            and d.confidence >= 0.30} == {"vertical_bar", "horizontal_bar"}

    result = resolve(raw)
    assert result.orientation == "horizontal"
    assert result.reason == "automatic_selection"
    assert result.suppressed_ids == tuple(d.id for d in original if d.role == "vertical_bar")
    expected = tuple(d for d in original if d.role != "vertical_bar")
    assert result.effective_frame.detections == expected
    assert all(actual is retained for actual, retained in
               zip(result.effective_frame.detections, expected))
    assert raw.detections is original
