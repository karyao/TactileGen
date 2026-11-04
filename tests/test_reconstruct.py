"""Reconstruction from hand-built detections; no detector or OCR engine runs."""

import numpy as np
import pytest

from src.contracts import Box, ChartInterpretationError, Detection, DetectionFrame
from src.reconstruct import parse_tick_value, reconstruct_vbar

W, H = 600, 400
PLOT = (80, 40, 560, 340)
ZERO_Y, PX_PER_UNIT = 340.0, 1.5  # value v sits at y = 340 - 1.5 * v


def y_of(value):
    return ZERO_Y - PX_PER_UNIT * value


class Chart:
    """Collects detections, OCR text and pixels for one synthetic chart."""

    def __init__(self):
        self.detections, self.texts = [], {}
        self.image = np.full((H, W, 3), 255, np.uint8)
        self._n = 0

    def add(self, role, box, conf=0.9, text=None, ocr_conf=0.95):
        self._n += 1
        det_id = f"d{self._n}"
        self.detections.append(Detection(det_id, role, Box(*box), conf))
        if text is not None:
            self.texts[det_id] = (text, ocr_conf)
        return det_id

    def paint(self, box, bgr):
        x1, y1, x2, y2 = (round(v) for v in box)
        self.image[y1:y2, x1:x2] = bgr

    def frame(self):
        return DetectionFrame("val/1.png", W, H, tuple(self.detections))

    def run(self):
        return reconstruct_vbar(self.frame(), self.texts, self.image)


def base_chart(values=(120, 60, 180), titles=True, ticks=(0, 50, 100, 150, 200)):
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in ticks:
        yc = y_of(tick)
        chart.add("y_tick_label", (40, yc - 6, 75, yc + 6), text=str(tick))
    if titles:
        chart.add("chart_title", (150, 5, 450, 30), text="Sales by year")
        chart.add("x_axis_title", (280, 375, 360, 395), text="Year")
        chart.add("y_axis_title", (2, 150, 20, 250), text="Revenue")
    n = len(values)
    slot = (PLOT[2] - PLOT[0]) / n
    for i, value in enumerate(values):
        cx = PLOT[0] + slot * (i + 0.5)
        box = (cx - 30, y_of(value), cx + 30, ZERO_Y)
        chart.add("vertical_bar", box)
        chart.paint(box, (200, 100, 30))  # BGR
        chart.add("x_tick_label", (cx - 20, 345, cx + 20, 360), text=f"C{i}")
    return chart


def assert_values(spec, expected, series=0):
    got = spec.series[series].values
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=0.01)


def test_clean_three_bar_chart():
    spec = base_chart().run()
    assert_values(spec, [120, 60, 180])
    assert spec.categories == ["C0", "C1", "C2"]
    assert spec.title == "Sales by year"
    assert spec.x_axis_title == "Year" and spec.y_axis_title == "Revenue"
    assert spec.y_axis.min == pytest.approx(0, abs=1e-6)
    assert spec.y_axis.max == pytest.approx(200)
    assert spec.y_axis.tick_values == [0, 50, 100, 150, 200]
    assert spec.series[0].name is None
    assert spec.series[0].source_color == "#1e64c8"
    assert spec.bars_adjacent is False
    assert spec.flags == []
    assert spec.schema_version == "0.2"


def test_spurious_y_tick_is_dropped():
    chart = base_chart()
    chart.add("y_tick_label", (40, y_of(75) - 6, 75, y_of(75) + 6), text="900")
    spec = chart.run()
    assert_values(spec, [120, 60, 180])
    assert "y_ticks_dropped:1" in spec.flags
    assert 900 not in spec.y_axis.tick_values


def test_missing_title_is_flagged_not_raised():
    spec = base_chart(titles=False).run()
    assert spec.title is None and spec.x_axis_title is None and spec.y_axis_title is None
    assert {"title_missing", "x_axis_title_missing", "y_axis_title_missing"} <= set(spec.flags)
    assert_values(spec, [120, 60, 180])


def test_title_touching_right_edge_is_flagged():
    chart = base_chart()
    chart.detections = [d for d in chart.detections if d.role != "chart_title"]
    chart.add("chart_title", (300, 5, W, 30), text="A very long title")
    assert "title_touches_right_edge" in chart.run().flags


def test_two_series_grouped_bars():
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (0, 50, 100, 150, 200):
        chart.add("y_tick_label", (40, y_of(tick) - 6, 75, y_of(tick) + 6), text=str(tick))
    colours = {"red": (30, 40, 200), "green": (60, 180, 60)}
    slot = (PLOT[2] - PLOT[0]) / 3
    for i, (a, b) in enumerate([(100, 50), (150, 80), (40, 120)]):
        cx = PLOT[0] + slot * (i + 0.5)
        for dx, value, colour in ((-22, a, "red"), (22, b, "green")):
            box = (cx + dx - 20, y_of(value), cx + dx + 20, ZERO_Y)
            chart.add("vertical_bar", box)
            chart.paint(box, colours[colour])
        chart.add("x_tick_label", (cx - 20, 345, cx + 20, 360), text=f"G{i}")
    for row, (name, colour) in enumerate((("Alpha", "red"), ("Beta", "green"))):
        y = 365 + 18 * row
        chart.add("legend_swatch", (100, y, 114, y + 14))
        chart.paint((100, y, 114, y + 14), colours[colour])
        chart.add("legend_label", (120, y, 180, y + 14), text=name)
    spec = chart.run()
    assert [s.name for s in spec.series] == ["Alpha", "Beta"]
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [50, 80, 120], series=1)
    assert spec.categories == ["G0", "G1", "G2"]
    assert spec.flags == ["title_missing", "x_axis_title_missing", "y_axis_title_missing"]


def test_duplicate_overlapping_bar_boxes_are_merged():
    chart = base_chart()
    bar = next(d for d in chart.detections if d.role == "vertical_bar")
    b = bar.box
    chart.add("vertical_bar", (b.x1 + 2, b.y1 + 1, b.x2 + 2, b.y2), conf=0.5)
    spec = chart.run()
    assert len(spec.series[0].values) == 3
    assert_values(spec, [120, 60, 180])
    assert "category_count_mismatch" not in spec.flags
    assert "category_ambiguous" not in spec.flags


def test_negative_bar_uses_far_edge_from_zero():
    chart = Chart()
    chart.add("plot_area", (80, 40, 560, 340))
    # zero at y=190 here: value v -> y = 190 - v
    for tick in (-150, -100, -50, 0, 50, 100, 150):
        chart.add("y_tick_label", (40, 190 - tick - 6, 75, 190 - tick + 6), text=str(tick))
    for i, (box, value) in enumerate([((120, 90, 180, 190), 100), ((300, 190, 360, 250), -60)]):
        chart.add("vertical_bar", box)
        chart.paint(box, (200, 100, 30))
        chart.add("x_tick_label", (box[0], 345, box[2], 360), text=f"C{i}")
    assert_values(chart.run(), [100, -60])


def test_adjacent_bars():
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (0, 100, 200):
        chart.add("y_tick_label", (40, y_of(tick) - 6, 75, y_of(tick) + 6), text=str(tick))
    for i, value in enumerate((50, 100, 150)):
        box = (100 + 60 * i, y_of(value), 160 + 60 * i, ZERO_Y)
        chart.add("vertical_bar", box)
        chart.paint(box, (200, 100, 30))
        chart.add("x_tick_label", (box[0], 345, box[2], 360), text=f"C{i}")
    assert chart.run().bars_adjacent is True


def test_too_few_ticks_raises():
    chart = base_chart(ticks=(100,))
    with pytest.raises(ChartInterpretationError):
        chart.run()


@pytest.mark.parametrize("text, expected", [
    ("1,200", (1200.0, None)), ("-3.5", (-3.5, None)), ("−2", (-2.0, None)),
    ("1e6", (1e6, None)), ("1.5e+06", (1.5e6, None)), ("2K", (2000.0, None)),
    ("1.5M", (1.5e6, None)), ("3B", (3e9, None)), ("45%", (45.0, "%")),
])
def test_parse_tick_value(text, expected):
    assert parse_tick_value(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "1-2", "--3"])
def test_parse_tick_value_rejects_non_numbers(text):
    assert parse_tick_value(text) is None


def test_bar_clipped_at_plot_edge_is_flagged():
    clipped = base_chart(values=(120, 60, 200), ticks=(0, 50, 100, 150)).run()
    assert "value_clipped_at_axis" in clipped.flags
    assert clipped.y_axis.max == pytest.approx(200, rel=0.01)
    assert "value_clipped_at_axis" not in base_chart().run().flags
