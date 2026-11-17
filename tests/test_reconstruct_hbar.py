"""Horizontal-bar reconstruction from hand-built detections; no detector or OCR runs."""

import numpy as np
import pytest

from src.contracts import Box, ChartInterpretationError, Detection, DetectionFrame, HorizontalBarChart
from src.reconstruct_hbar import bar_orientation, reconstruct_hbar, transpose_frame
from src.tactile.completeness import READY, check_completeness, spec_to_chart

W, H = 600, 400
PLOT = (100, 40, 560, 340)
ZERO_X, PX_PER_UNIT = 100.0, 2.0  # value v sits at x = 100 + 2 * v


def x_of(value):
    return ZERO_X + PX_PER_UNIT * value


class Chart:
    """Detections, OCR text and pixels of one synthetic horizontal bar chart."""

    def __init__(self):
        self.detections, self.texts = [], {}
        self.image = np.full((H, W, 3), 255, np.uint8)

    def add(self, role, box, conf=0.9, text=None):
        det_id = f"d{len(self.detections) + 1}"
        self.detections.append(Detection(det_id, role, Box(*box), conf))
        if text is not None:
            self.texts[det_id] = (text, 0.95)
        return det_id

    def paint(self, box, bgr):
        x1, y1, x2, y2 = (round(v) for v in box)
        self.image[y1:y2, x1:x2] = bgr

    def frame(self):
        return DetectionFrame("test/1.png", W, H, tuple(self.detections))

    def run(self, **kwargs):
        return reconstruct_hbar(self.frame(), self.texts, self.image, **kwargs)


def hbar_chart(values=(120, 60, 180), names=("North", "South", "East"), titles=True,
               series_colours=((200, 100, 30),), legend=()):
    """Bars grow right from x=100; categories run top to bottom; values below."""
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (0, 50, 100, 150, 200):
        chart.add("x_tick_label", (x_of(tick) - 12, 345, x_of(tick) + 12, 358), text=str(tick))
    if titles:
        chart.add("chart_title", (150, 5, 450, 30), text="Sales by region")
        chart.add("x_axis_title", (280, 375, 380, 395), text="Revenue")
        chart.add("y_axis_title", (2, 150, 18, 250), text="Region")
    rows = values if isinstance(values[0], tuple) else tuple((v,) for v in values)
    slot = (PLOT[3] - PLOT[1]) / len(rows)
    thickness = 40 / len(series_colours)
    for i, (row, name) in enumerate(zip(rows, names)):
        cy = PLOT[1] + slot * (i + 0.5)
        top = cy - 20
        for s, (value, colour) in enumerate(zip(row, series_colours)):
            box = (ZERO_X, top + s * thickness, x_of(value), top + (s + 1) * thickness)
            chart.add("horizontal_bar", box)
            chart.paint(box, colour)
        chart.add("y_tick_label", (20, cy - 6, 95, cy + 6), text=name)
    for row, (name, colour) in enumerate(legend):
        y = 362 + 18 * row
        chart.add("legend_swatch", (450, y, 464, y + 14))
        chart.paint((450, y, 464, y + 14), colour)
        chart.add("legend_label", (470, y, 540, y + 14), text=name)
    return chart


def values_of(spec, series=0):
    return [pytest.approx(v, rel=0.01) for v in spec.series[series].values]


def test_orientation_follows_the_majority_of_bars():
    assert bar_orientation(hbar_chart().frame()) == "horizontal"
    chart = hbar_chart()
    chart.add("vertical_bar", (300, 100, 320, 200))
    assert bar_orientation(chart.frame()) == "horizontal"


def test_orientation_defaults_to_vertical_on_a_tie_or_no_bars():
    empty = DetectionFrame("test/1.png", W, H, ())
    assert bar_orientation(empty) == "vertical"
    chart = Chart()
    chart.add("horizontal_bar", (100, 50, 200, 70))
    chart.add("vertical_bar", (300, 100, 320, 200))
    assert bar_orientation(chart.frame()) == "vertical"


def test_low_confidence_bars_do_not_decide_the_orientation():
    chart = Chart()
    chart.add("vertical_bar", (300, 100, 320, 200), conf=0.9)
    for k in range(3):
        chart.add("horizontal_bar", (100, 50 + 30 * k, 200, 70 + 30 * k), conf=0.1)
    assert bar_orientation(chart.frame()) == "vertical"


def test_transpose_swaps_boxes_roles_and_size():
    original = hbar_chart().frame()
    frame = transpose_frame(original)
    assert (frame.width, frame.height) == (H, W)
    before = next(d for d in original.detections if d.role == "horizontal_bar")
    after = next(d for d in frame.detections if d.id == before.id)
    assert after.role == "vertical_bar"
    assert after.box == Box(before.box.y1, before.box.x1, before.box.y2, before.box.x2)
    assert {d.role for d in frame.detections} >= {"x_tick_label", "y_tick_label"}


def test_clean_horizontal_chart():
    spec = hbar_chart().run()
    assert spec.orientation == "horizontal"
    assert spec.categories == ["North", "South", "East"]   # top to bottom
    assert values_of(spec) == [120, 60, 180]
    assert spec.title == "Sales by region"
    assert spec.x_axis_title == "Revenue" and spec.y_axis_title == "Region"
    assert spec.y_axis.tick_values == [0, 50, 100, 150, 200]
    assert spec.y_axis.min == pytest.approx(0, abs=1e-6)
    assert spec.series[0].source_color == "#1e64c8"
    assert spec.flags == []


def test_missing_value_heading_is_reported_with_its_printed_axis():
    chart = hbar_chart()
    chart.detections = [d for d in chart.detections if d.role != "x_axis_title"]
    spec = chart.run()
    assert spec.x_axis_title is None and spec.y_axis_title == "Region"
    assert "x_axis_title_missing" in spec.flags
    assert "y_axis_title_missing" not in spec.flags
    assert "x_axis_title" not in spec.confidence and "y_axis_title" in spec.confidence


def test_title_edge_check_uses_the_image_width():
    chart = hbar_chart(titles=False)
    chart.add("chart_title", (150, 370, 450, 400), text="Bottom title")
    assert "title_touches_right_edge" not in chart.run().flags
    chart = hbar_chart(titles=False)
    chart.add("chart_title", (150, 5, 600, 30), text="Wide title")
    assert "title_touches_right_edge" in chart.run().flags


def test_no_bars_error_names_horizontal_bars():
    chart = hbar_chart()
    chart.detections = [d for d in chart.detections if d.role != "horizontal_bar"]
    with pytest.raises(ChartInterpretationError, match="no horizontal bars"):
        chart.run()


def test_missing_short_bar_is_rescued_from_pixels():
    chart = hbar_chart(values=(120, 60, 2))    # the 2-unit bar is 4 px long
    bars = [d for d in chart.detections if d.role == "horizontal_bar"]
    chart.detections.remove(bars[2])
    spec = chart.run()
    assert values_of(spec) == [120, 60, 2]
    assert spec.series[0].value_rescued == [False, False, True]
    assert "bars_rescued:1" in spec.flags


def test_two_series_with_legend_are_named_in_top_to_bottom_order():
    colours = ((30, 40, 200), (60, 180, 60))
    chart = hbar_chart(values=((100, 50), (150, 80), (40, 120)), series_colours=colours,
                       legend=(("Alpha", colours[0]), ("Beta", colours[1])))
    spec = chart.run()
    assert [s.name for s in spec.series] == ["Alpha", "Beta"]
    assert values_of(spec, 0) == [100, 150, 40]
    assert values_of(spec, 1) == [50, 80, 120]
    assert "series_ambiguous" not in spec.flags


def test_ready_spec_becomes_a_horizontal_chart():
    spec = hbar_chart().run()
    assert check_completeness(spec).status == READY
    chart = spec_to_chart(spec)
    assert isinstance(chart, HorizontalBarChart)
    assert chart.categories == ("North", "South", "East")
    assert chart.x_axis_title == "Revenue" and chart.y_axis_title == "Region"
