"""Dropping low-confidence bars that no legend swatch claims."""

from src.bar_filter import drop_unmatched_bars
from src.contracts import Box, Detection, DetectionFrame, LegendBarAssignment, LegendMappingFrame


def bar(identifier, x, confidence):
    return Detection(identifier, "vertical_bar", Box(x, 10, x + 10, 50), confidence)


def mapping(**status):
    return LegendMappingFrame(
        "input/c.png",
        tuple(LegendBarAssignment(k, "s", "l", "N" if v == "matched" else None, 0.9, v)
              for k, v in status.items()), ())


def frame(*bars):
    plot = Detection("p", "plot_area", Box(0, 0, 200, 60), 0.99)
    return DetectionFrame("input/c.png", 200, 60, (plot, *bars))


def test_a_weak_unmatched_bar_is_dropped_and_matched_or_confident_bars_stay():
    f = frame(bar("a", 10, 0.5), bar("b", 40, 0.5), bar("c", 70, 0.95))
    reduced = drop_unmatched_bars(f, mapping(a="weak_match", b="matched", c="weak_match"))
    assert {d.id for d in reduced.detections if d.role == "vertical_bar"} == {"b", "c"}


def test_nothing_to_drop_returns_none():
    f = frame(bar("a", 10, 0.5))
    assert drop_unmatched_bars(f, mapping(a="matched")) is None
