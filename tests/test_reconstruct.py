"""Reconstruction from hand-built detections; no detector or OCR engine runs."""

from unittest.mock import patch

import numpy as np
import pytest

from src.contracts import (
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
)
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


def _grouped(colours, legend=True):
    """Two-series grouped chart; colours are BGR for the two series."""
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (0, 50, 100, 150, 200):
        chart.add("y_tick_label", (40, y_of(tick) - 6, 75, y_of(tick) + 6), text=str(tick))
    slot = (PLOT[2] - PLOT[0]) / 3
    for i, pair in enumerate([(100, 50), (150, 80), (40, 120)]):
        cx = PLOT[0] + slot * (i + 0.5)
        for dx, value, colour in zip((-22, 22), pair, colours):
            box = (cx + dx - 20, y_of(value), cx + dx + 20, ZERO_Y)
            chart.add("vertical_bar", box)
            chart.paint(box, colour)
        chart.add("x_tick_label", (cx - 20, 345, cx + 20, 360), text=f"G{i}")
    for row, (name, colour) in enumerate(zip(("Alpha", "Beta"), colours)):
        if legend:
            y = 365 + 18 * row
            chart.add("legend_swatch", (100, y, 114, y + 14))
            chart.paint((100, y, 114, y + 14), colour)
            chart.add("legend_label", (120, y, 180, y + 14), text=name)
    return chart


# #FFA500 and #FF8C00 as BGR: about 25 apart in RGB, inside the clustering threshold.
SIMILAR = ((0, 165, 255), (0, 140, 255))


def _matched_mapping(chart, identities=((0, 0, "Alpha"), (1, 1, "Beta"))):
    frame = chart.frame()
    bars = [item for item in frame.detections if item.role == "vertical_bar"]
    swatches = [item for item in frame.detections if item.role == "legend_swatch"]
    labels = [item for item in frame.detections if item.role == "legend_label"]
    return frame, LegendMappingFrame(frame.source_key, tuple(
        LegendBarAssignment(
            bar.id,
            swatches[identities[index % len(identities)][0]].id,
            labels[identities[index % len(identities)][1]].id,
            identities[index % len(identities)][2],
            0.99,
            "matched",
        )
        for index, bar in enumerate(bars)
    ))


def test_similar_legend_colours_safely_remain_unnamed():
    spec = _grouped(SIMILAR).run()
    assert [s.name for s in spec.series] == [None, None]
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [50, 80, 120], series=1)
    assert "category_count_mismatch" not in spec.flags
    assert "series_ambiguous" in spec.flags


def test_supplied_legend_mapping_names_series_without_rerunning_matcher():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    frame, mapping = _matched_mapping(chart)

    with patch("src.reconstruct.match_legend_to_bars") as matcher:
        spec = reconstruct_vbar(
            frame, chart.texts, chart.image, legend_mapping=mapping
        )

    matcher.assert_not_called()
    assert [series.name for series in spec.series] == ["Alpha", "Beta"]
    assert "series_ambiguous" not in spec.flags


def test_mapping_keeps_nearly_identical_series_colours_separate():
    chart = _grouped(((30, 40, 200), (34, 44, 203)))
    frame, mapping = _matched_mapping(chart)

    spec = reconstruct_vbar(
        frame, chart.texts, chart.image, legend_mapping=mapping
    )

    assert [series.name for series in spec.series] == ["Alpha", "Beta"]
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [50, 80, 120], series=1)
    assert "category_count_mismatch" not in spec.flags
    assert "category_ambiguous" not in spec.flags
    assert "series_ambiguous" not in spec.flags


def test_mapping_remains_authoritative_for_low_confidence_swatches():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    chart.detections = [
        Detection(item.id, item.role, item.box, 0.29)
        if item.role == "legend_swatch" else item
        for item in chart.detections
    ]
    frame, mapping = _matched_mapping(chart)

    spec = reconstruct_vbar(
        frame, chart.texts, chart.image, legend_mapping=mapping
    )

    assert [series.name for series in spec.series] == ["Alpha", "Beta"]
    assert "series_ambiguous" not in spec.flags


def test_incomplete_mapping_keeps_affected_series_unnamed():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    frame = chart.frame()
    bars = [item for item in frame.detections if item.role == "vertical_bar"]
    swatches = [item for item in frame.detections if item.role == "legend_swatch"]
    labels = [item for item in frame.detections if item.role == "legend_label"]
    assignments = []
    for index, bar in enumerate(bars):
        series_index = index % 2
        if index == 2:
            assignments.append(LegendBarAssignment(
                bar.id, swatches[0].id, labels[0].id, None, 0.95, "ambiguous"
            ))
        else:
            assignments.append(LegendBarAssignment(
                bar.id,
                swatches[series_index].id,
                labels[series_index].id,
                ("Alpha", "Beta")[series_index],
                0.99,
                "matched",
            ))

    spec = reconstruct_vbar(
        frame,
        chart.texts,
        chart.image,
        legend_mapping=LegendMappingFrame(frame.source_key, tuple(assignments)),
    )

    assert [series.name for series in spec.series] == ["Alpha", "Beta", None]
    assert spec.series[0].values == pytest.approx([100, None, 40])
    assert spec.series[1].values == pytest.approx([50, 80, 120])
    assert spec.series[2].values == pytest.approx([None, 150, None])
    assert all(not any(series.value_rescued) for series in spec.series)
    assert "category_count_mismatch" in spec.flags
    assert "series_ambiguous" in spec.flags


def test_conflicting_identities_preserve_bars_but_suppress_names():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    frame = chart.frame()
    bars = [item for item in frame.detections if item.role == "vertical_bar"]
    swatches = [item for item in frame.detections if item.role == "legend_swatch"]
    labels = [item for item in frame.detections if item.role == "legend_label"]
    assignments = []
    for index, bar in enumerate(bars):
        if index % 2:
            identity = (swatches[1].id, labels[1].id, "Beta")
        elif index == 2:
            identity = (swatches[0].id, labels[1].id, "Other")
        else:
            identity = (swatches[0].id, labels[0].id, "Alpha")
        assignments.append(LegendBarAssignment(
            bar.id, *identity, 0.99, "matched"
        ))

    spec = reconstruct_vbar(
        frame,
        chart.texts,
        chart.image,
        legend_mapping=LegendMappingFrame(frame.source_key, tuple(assignments)),
    )

    assert [series.name for series in spec.series] == [None, "Beta", None]
    assert spec.series[0].values == pytest.approx([100, None, 40])
    assert spec.series[2].values == pytest.approx([None, 150, None])
    assert "series_ambiguous" in spec.flags


def test_similar_colours_without_legend_fall_back_to_one_cluster():
    spec = _grouped(SIMILAR, legend=False).run()
    assert len(spec.series) == 1
    assert {"category_count_mismatch", "category_ambiguous"} <= set(spec.flags)


# --- bar rescue ----------------------------------------------------------------

def _drop_detection(chart, index):
    """Remove the index-th vertical_bar detection, leaving its pixels on the image."""
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    chart.detections.remove(bars[index])


def test_missing_near_zero_bar_is_rescued_in_single_series_chart():
    chart = base_chart(values=(120, 60, 2))  # the 2-unit bar is 3 px tall
    _drop_detection(chart, 2)
    spec = chart.run()
    assert_values(spec, [120, 60, 2])
    assert spec.series[0].value_rescued == [False, False, True]
    assert spec.series[0].value_confidence[2] < spec.series[0].value_confidence[0]
    assert "bars_rescued:1" in spec.flags
    assert "category_count_mismatch" not in spec.flags
    assert "value_inferred_zero" not in spec.flags


def test_rescue_with_no_bar_pixels_infers_zero():
    chart = base_chart(values=(120, 60, 5))
    _drop_detection(chart, 2)
    chart.image[300:345, 400:580] = 255  # erase the third bar
    spec = chart.run()
    assert spec.series[0].values[2] == 0.0
    assert spec.series[0].value_rescued[2] is True
    assert "value_inferred_zero" in spec.flags and "bars_rescued:1" in spec.flags


def test_rescue_off_keeps_mismatch():
    chart = base_chart(values=(120, 60, 2))
    _drop_detection(chart, 2)
    spec = reconstruct_vbar(chart.frame(), chart.texts, chart.image, rescue_bars=False)
    assert "category_count_mismatch" in spec.flags
    assert spec.series[0].values[2] is None


def grouped_chart(colours, groups, missing=(), width=30, shift=0.0):
    """Adjacent bars of ``width`` px per group; ``shift`` moves a group off its tick."""
    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (0, 50, 100, 150, 200):
        chart.add("y_tick_label", (40, y_of(tick) - 6, 75, y_of(tick) + 6), text=str(tick))
    slot = (PLOT[2] - PLOT[0]) / len(groups)
    count = len(colours)
    for g, values in enumerate(groups):
        cx = PLOT[0] + slot * (g + 0.5)
        chart.add("x_tick_label", (cx - 20, 345, cx + 20, 360), text=f"G{g}")
        for s, (value, colour) in enumerate(zip(values, colours)):
            bx = cx + shift + (s - (count - 1) / 2) * width
            box = (bx - width / 2, y_of(value), bx + width / 2, ZERO_Y)
            chart.paint(box, colour)
            if (g, s) not in missing:
                chart.add("vertical_bar", box)
    for row, colour in enumerate(colours):
        y = 362 + 12 * row
        chart.add("legend_swatch", (100, y, 114, y + 10))
        chart.paint((100, y, 114, y + 10), colour)
        chart.add("legend_label", (120, y, 180, y + 10), text=f"S{row}")
    return chart


RED, GREEN, BLUE = (30, 40, 200), (60, 180, 60), (200, 100, 30)


def test_missing_near_zero_bar_is_rescued_in_grouped_chart():
    chart = grouped_chart((RED, GREEN), [(100, 50), (150, 2), (40, 120)], missing={(1, 1)})
    spec = chart.run()
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [50, 2, 120], series=1)
    assert spec.series[1].value_rescued == [False, True, False]
    assert spec.series[0].value_rescued == [False, False, False]
    assert "bars_rescued:1" in spec.flags
    assert "category_count_mismatch" not in spec.flags


def test_rescue_follows_group_offset_for_three_series():
    # PlotQA places groups of 3+ bars half a bar left of the tick, so slot
    # positions must come from the detected bars, not a centred formula.
    chart = grouped_chart((RED, GREEN, BLUE), [(100, 50, 80), (150, 60, 2), (40, 120, 90)],
                          missing={(1, 2)}, shift=-15.0)
    spec = chart.run()
    assert_values(spec, [80, 2, 90], series=2)
    assert spec.series[2].value_rescued == [False, True, False]
    assert "category_count_mismatch" not in spec.flags


def test_unrescuable_slot_keeps_mismatch():
    # The axis starts at 100, so the zero line is below the plot and nothing can be measured.
    def y_at(value):
        return PLOT[3] - 1.5 * (value - 100)

    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (100, 150, 200):
        chart.add("y_tick_label", (40, y_at(tick) - 6, 75, y_at(tick) + 6), text=str(tick))
    for i, value in enumerate((150, 120)):
        box = (100 + 200 * i, y_at(value), 160 + 200 * i, PLOT[3])
        chart.add("vertical_bar", box)
        chart.paint(box, (200, 100, 30))
    for i in range(3):
        chart.add("x_tick_label", (100 + 200 * i, 345, 160 + 200 * i, 360), text=f"C{i}")
    spec = chart.run()
    assert "baseline_outside_plot" in spec.flags
    assert "category_count_mismatch" in spec.flags


def test_rescue_counts_the_dark_bar_outline():
    chart = base_chart(values=(120, 60, 2))  # 3 px: rows 337..339
    _drop_detection(chart, 2)
    chart.image[337, 410:470] = (20, 40, 10)  # outline blended over the top row
    assert_values(chart.run(), [120, 60, 2])


# --- category guard ------------------------------------------------------------

def chart_with_label(text, ocr_conf=0.95):
    chart = base_chart()
    label = [d for d in chart.detections if d.role == "x_tick_label"][1]
    chart.texts[label.id] = (text, ocr_conf)
    return chart


@pytest.mark.parametrize("text", ["", "|", "-", "...", "  "])
def test_empty_or_symbol_only_category_is_unreadable(text):
    spec = chart_with_label(text).run()
    assert "category_unreadable" in spec.flags
    assert spec.confidence["category[1]"] < 0.2
    assert "category[0]" not in spec.confidence
    assert spec.confidence["categories"] < 0.9 * 0.95 * 0.9  # pulled down by the bad label


def test_low_ocr_confidence_alone_does_not_flag_a_category():
    spec = chart_with_label("Iran", ocr_conf=0.5).run()
    assert "category_unreadable" not in spec.flags


@pytest.mark.parametrize("text", ["Iran", "1999", "a", "3", "US$", "Latin America(developing only)"])
def test_readable_categories_are_not_flagged(text):
    spec = chart_with_label(text).run()
    assert "category_unreadable" not in spec.flags
    assert not any(key.startswith("category[") for key in spec.confidence)


def test_guard_off_only_flags_empty_text():
    def off(text, conf):
        chart = chart_with_label(text, conf)
        return reconstruct_vbar(chart.frame(), chart.texts, chart.image, category_guard=False)

    assert "category_unreadable" not in off("|", 0.95).flags
    assert "category_unreadable" in off("", 0.0).flags
