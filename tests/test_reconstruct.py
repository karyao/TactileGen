"""Vertical and horizontal reconstruction and parsing provenance."""

from dataclasses import asdict, replace
import json
from unittest.mock import patch
from zipfile import ZipFile

from unittest.mock import patch

import numpy as np
import pytest

from src.contracts import (
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    HorizontalBarChart,
    LegendBarAssignment,
    LegendEntry,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
)
from src.diagnostics.tracking import AnalysisDiagnostics
from src.reconstruction.vertical import _calibrate, _text_table, parse_tick_value, reconstruct_vbar
from src.reconstruction.horizontal import bar_orientation, reconstruct_hbar, transpose_frame
from src.tactile.completeness import READY, check_completeness, cut_off_flags, spec_to_chart

from tests.test_completeness import make_spec


# Reconstruct

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


def labelled_chart(texts):
    chart = base_chart(ticks=())
    for tick, text in zip((0, 50, 100, 150, 200), texts):
        chart.add("y_tick_label", (40, y_of(tick) - 6, 75, y_of(tick) + 6), text=text)
    return chart


def test_lost_decimal_point_in_tick_labels_is_repaired():
    spec = labelled_chart(("0", "02", "04", "06", "08")).run()   # 0, 0.2, ... 0.8
    assert spec.y_axis.tick_values == pytest.approx([0, 0.2, 0.4, 0.6, 0.8])
    assert_values(spec, [0.48, 0.24, 0.72])
    assert "y_tick_decimal_repaired:4" in spec.flags and "y_tick_leading_zero" not in spec.flags


def test_leading_zero_label_that_disagrees_with_the_axis_is_dropped_not_repaired():
    spec = labelled_chart(("0", "050", "100", "150", "200")).run()   # "050" would be 0.05
    assert "y_ticks_dropped:1" in spec.flags
    assert not any(f.startswith(("y_tick_decimal_repaired", "y_tick_leading_zero")) for f in spec.flags)
    assert_values(spec, [120, 60, 180])


def test_repair_that_cannot_be_trusted_keeps_the_blocker():
    # 0, 0.2, 0.8, 1.0 at equal pixel steps: several two-tick lines fit equally well.
    spec = labelled_chart(("0", "02", "08", "10", "")).run()
    assert "y_calibration_ambiguous" in spec.flags and "y_tick_leading_zero" in spec.flags


def test_ordinary_tick_labels_are_not_repaired():
    for texts in (("0", "50", "100", "150", "200"), ("0", "0.5", "1.0", "1.5", "2.0"),
                  ("-0", "50", "100", "150", "200")):
        flags = labelled_chart(texts).run().flags
        assert "y_tick_leading_zero" not in flags
        assert not any(f.startswith("y_tick_decimal_repaired") for f in flags)


def test_dropped_spurious_tick_with_leading_zero_is_not_repaired():
    chart = base_chart()
    chart.add("y_tick_label", (40, y_of(75) - 6, 75, y_of(75) + 6), text="0900")
    spec = chart.run()
    assert "y_ticks_dropped:1" in spec.flags
    assert "y_tick_leading_zero" not in spec.flags


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
    spec = chart.run()
    assert_values(spec, [100, -60])
    assert not any(f.startswith("bar_edge_corrected") for f in spec.flags)  # negative bar skipped


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

    with patch("src.reconstruction.vertical.filter_bars_by_legend") as matcher:
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


def test_mapped_close_colours_flow_to_tactile_svg():
    from src.tactile.completeness import (
        READY,
        Overrides,
        apply_overrides,
        check_completeness,
        spec_to_chart,
    )
    from src.tactile.layout import layout_vertical_bar_chart_pages
    from src.tactile.svg import render_svg

    chart = _grouped(((30, 40, 200), (34, 44, 203)))
    frame, mapping = _matched_mapping(chart)
    spec = reconstruct_vbar(
        frame, chart.texts, chart.image, legend_mapping=mapping
    )
    spec, _ = apply_overrides(spec, Overrides(
        title="Sales by group",
        x_axis_title="Group",
        y_axis_title="Sales",
    ))

    assert check_completeness(spec).status == READY
    pages = layout_vertical_bar_chart_pages(spec_to_chart(spec))
    assert len(pages) == 1
    svg = render_svg(pages[0])
    assert "<svg" in svg and "</svg>" in svg


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


def test_short_detected_bar_is_remeasured_from_pixels():
    chart = base_chart(values=(120, 60, 2))  # the 2-unit bar is 3 px tall
    spec = chart.run()
    assert_values(spec, [120, 60, 2])
    assert spec.series[0].value_rescued == [False, False, True]
    assert "short_bars_remeasured:1" in spec.flags and "bars_rescued:1" in spec.flags
    assert "category_count_mismatch" not in spec.flags


def test_short_box_that_misses_most_of_the_bar_gets_the_pixel_height():
    chart = base_chart(values=(120, 60, 4))
    third = [d for d in chart.detections if d.role == "vertical_bar"][2]
    box = third.box
    chart.paint((box.x1, y_of(40), box.x2, ZERO_Y), (200, 100, 30))  # really 40 units tall
    spec = chart.run()
    assert_values(spec, [120, 60, 40])
    assert spec.series[0].value_rescued[2] is True


def test_short_limit_is_six_px_or_three_percent_of_the_plot_height():
    # The plot is 300 px tall, so the limit is 9 px: 8 px (5 units) is short, 9 px (6 units) is not.
    short = base_chart(values=(120, 60, 5)).run()
    assert "short_bars_remeasured:1" in short.flags
    ok = base_chart(values=(120, 60, 6)).run()
    assert not any(f.startswith("short_bars_remeasured") for f in ok.flags)
    assert ok.series[0].value_rescued == [False, False, False]


def test_short_bars_are_kept_when_rescue_is_off_or_the_zero_line_is_missing():
    chart = base_chart(values=(120, 60, 2))
    spec = reconstruct_vbar(chart.frame(), chart.texts, chart.image, rescue_bars=False)
    assert not any(f.startswith("short_bars_remeasured") for f in spec.flags)
    assert spec.series[0].value_rescued == [False, False, False]
    assert_values(spec, [120, 60, 2])


def test_short_bar_colour_does_not_create_an_extra_series():
    chart = grouped_chart((RED, GREEN), [(100, 2), (150, 60), (40, 120)])
    third = [d for d in chart.detections if d.role == "vertical_bar"][1]
    chart.image[int(third.box.y1):int(third.box.y2), int(third.box.x1):int(third.box.x2)] = (90, 90, 90)
    spec = chart.run()
    assert len(spec.series) == 2 and "series_ambiguous" not in spec.flags
    assert_values(spec, [100, 150, 40], series=0)
    assert "short_bars_remeasured:1" in spec.flags


def test_series_made_only_of_short_bars_is_not_emitted_without_values():
    # Every bar of series 1 is 3-4 px tall. With nothing else in the series to measure the group
    # offset from, remeasuring them used to leave the series with all-None values.
    chart = grouped_chart((RED, GREEN), [(100, 2), (150, 2), (40, 2)])
    spec = chart.run()
    assert len(spec.series) == 2
    assert_values(spec, [100, 150, 40], series=0)
    assert all(v is not None for v in spec.series[1].values)
    assert_values(spec, [2, 2, 2], series=1)
    assert not any(f.startswith("short_bars_remeasured") for f in spec.flags)


def test_short_bars_are_still_remeasured_when_the_series_has_a_normal_bar():
    chart = grouped_chart((RED, GREEN), [(100, 2), (150, 60), (40, 2)])
    spec = chart.run()
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [2, 60, 2], series=1)
    assert spec.series[1].value_rescued == [True, False, True]
    assert "short_bars_remeasured:2" in spec.flags


def test_unmatched_group_of_short_bars_is_still_removed():
    # The outline colour of short boxes forms a "series" with no legend entry; leaving the boxes out of
    # grouping is what removes it (charts 22448, 24105, 29012, 30059, 30724 depend on this).
    chart = grouped_chart((RED, GREEN), [(100, 2), (150, 2), (40, 2)])
    chart.detections[:] = [d for d in chart.detections
                           if d.role not in ("legend_swatch", "legend_label") or d.box.y1 < 370]
    spec = chart.run()
    assert len(spec.series) == 1 and spec.series[0].name == "S0"
    assert_values(spec, [100, 150, 40])
    assert "short_bars_remeasured:3" in spec.flags


def _shift_top(chart, index, dy):
    """Move the detector's top edge of one bar down by ``dy`` px; the painted bar stays put."""
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    old = bars[index]
    chart.detections[chart.detections.index(old)] = Detection(
        old.id, old.role, Box(old.box.x1, old.box.y1 + dy, old.box.x2, old.box.y2), old.confidence)


def test_detector_top_edge_far_from_the_pixels_is_corrected():
    chart = base_chart(values=(120, 60, 180))
    _shift_top(chart, 1, 12)  # 12 px = 8 units too low
    spec = chart.run()
    assert_values(spec, [120, 60, 180])
    assert "bar_edge_corrected:1" in spec.flags
    assert spec.series[0].value_rescued == [False, False, False]


def test_small_edge_differences_keep_the_detector_value():
    chart = base_chart(values=(120, 60, 180))
    _shift_top(chart, 1, 4)
    spec = chart.run()
    assert not any(f.startswith("bar_edge_corrected") for f in spec.flags)
    assert spec.series[0].values[1] == pytest.approx(60 - 4 / 1.5, rel=0.01)


def test_edge_check_is_off_without_rescue():
    chart = base_chart(values=(120, 60, 180))
    _shift_top(chart, 1, 12)
    spec = reconstruct_vbar(chart.frame(), chart.texts, chart.image, rescue_bars=False)
    assert not any(f.startswith("bar_edge_corrected") for f in spec.flags)
    assert spec.series[0].values[1] == pytest.approx(52, rel=0.01)


def test_edge_check_ignores_bars_when_the_zero_line_is_outside_the_plot():
    def y_at(value):
        return PLOT[3] - 1.5 * (value - 100)

    chart = Chart()
    chart.add("plot_area", PLOT)
    for tick in (100, 150, 200):
        chart.add("y_tick_label", (40, y_at(tick) - 6, 75, y_at(tick) + 6), text=str(tick))
    for i, value in enumerate((150, 120)):
        box = (100 + 200 * i, y_at(value), 160 + 200 * i, PLOT[3])
        chart.add("vertical_bar", box)
        chart.paint((box[0], box[1] - 20, box[2], box[3]), (200, 100, 30))
        chart.add("x_tick_label", (box[0], 345, box[2], 360), text=f"C{i}")
    spec = chart.run()
    assert "baseline_outside_plot" in spec.flags
    assert not any(f.startswith("bar_edge_corrected") for f in spec.flags)


def _missed(spec):
    return [f for f in spec.flags if f.startswith("bars_mostly_missed")]


def test_chart_with_most_bars_undetected_is_flagged():
    chart = base_chart(values=(120, 60, 180))
    _drop_detection(chart, 2)
    _drop_detection(chart, 1)
    spec = chart.run()
    assert "bars_mostly_missed:67" in spec.flags  # 1 of 3 slots detected


def test_bars_mostly_missed_needs_fewer_than_sixty_percent():
    two_of_three = base_chart(values=(120, 60, 180))
    _drop_detection(two_of_three, 2)
    assert not _missed(two_of_three.run())
    three_of_five = base_chart(values=(120, 60, 90, 70, 100))
    _drop_detection(three_of_five, 4)
    _drop_detection(three_of_five, 3)
    assert not _missed(three_of_five.run())  # exactly 0.6
    two_of_five = base_chart(values=(120, 60, 90, 70, 100))
    for index in (4, 3, 2):
        _drop_detection(two_of_five, index)
    assert "bars_mostly_missed:60" in two_of_five.run().flags


def test_short_detected_bars_count_as_detected():
    chart = base_chart(values=(120, 60, 2, 2, 2))  # three 3 px bars, all detected
    spec = chart.run()
    assert "short_bars_remeasured:3" in spec.flags
    assert not _missed(spec)


def test_minus_sign_read_on_a_dropped_tick_label_is_flagged():
    chart = base_chart()
    chart.add("y_tick_label", (40, y_of(75) - 6, 75, y_of(75) + 6), text="-1")  # dropped from the fit
    spec = chart.run()
    assert "y_ticks_dropped:1" in spec.flags and "y_tick_negative_read" in spec.flags
    assert min(spec.y_axis.tick_values) == 0


def test_no_negative_read_flag_without_a_minus_sign():
    assert "y_tick_negative_read" not in base_chart().run().flags
    assert "y_tick_negative_read" not in labelled_chart(("0", "0.5", "1.0", "1.5", "2.0")).run().flags
    assert "y_tick_negative_read" in labelled_chart(("0", "- 50", "100", "150", "200")).run().flags

# --- legend entries whose series has no visible bars ---------------------------------

BAR_BGR = (200, 100, 30)          # the colour base_chart paints its bars
RED, GREEN = (30, 30, 200), (30, 200, 30)


def _legend_chart(entries, values=(120, 60, 180), legend_box=True):
    """base_chart plus a legend at the right edge. ``entries`` are
    ``(text, bgr, swatch_conf, ocr_conf)`` rows, listed top to bottom."""
    chart = base_chart(values)
    top = 50
    if legend_box:
        chart.add("legend", (563, top - 4, 600, top + 25 * len(entries)))
    for row, (text, colour, swatch_conf, ocr_conf) in enumerate(entries):
        y = top + 25 * row
        chart.add("legend_swatch", (566, y, 576, y + 10), conf=swatch_conf)
        chart.paint((566, y, 576, y + 10), colour)
        chart.add("legend_label", (578, y, 598, y + 10), text=text, ocr_conf=ocr_conf)
    return chart


def _row(text, colour, swatch_conf=0.9, ocr_conf=0.95):
    return (text, colour, swatch_conf, ocr_conf)


def test_unmatched_legend_entries_become_series_with_no_visible_bars():
    # Chart 24811 in miniature: three legend entries, bars only for the middle one.
    spec = _legend_chart([_row("Low", RED), _row("Kenya", BAR_BGR), _row("Vietnam", GREEN)]).run()

    assert [s.name for s in spec.series] == ["Low", "Kenya", "Vietnam"]      # legend order
    assert [s.no_visible_bars for s in spec.series] == [True, False, True]
    assert_values(spec, [120, 60, 180], series=1)
    for series in (spec.series[0], spec.series[2]):
        assert series.values == [None, None, None]          # nothing measured, and not zero
        assert series.value_rescued == [] and series.value_confidence == []
    assert [s.source_color for s in spec.series] == ["#c81e1e", "#1e64c8", "#1ec81e"]
    assert "series_ambiguous" not in spec.flags and spec.flags == []


def test_normal_multi_series_chart_has_no_extra_series():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    spec = chart.run()
    assert [s.name for s in spec.series] == ["Alpha", "Beta"]
    assert not any(s.no_visible_bars for s in spec.series)
    assert not [f for f in spec.flags if f.startswith(("legend_", "series_"))]


def test_single_series_without_a_legend_is_unchanged():
    spec = base_chart().run()
    assert len(spec.series) == 1 and spec.series[0].name is None
    assert not spec.series[0].no_visible_bars and spec.flags == []


def test_single_matched_entry_adds_nothing():
    spec = _legend_chart([_row("Kenya", BAR_BGR)]).run()
    assert [s.name for s in spec.series] == ["Kenya"] and spec.flags == []


def _skipped(spec):
    return [flag for flag in spec.flags if flag.startswith("legend_entry_skipped")]


def test_low_confidence_entries_are_reported_and_not_added():
    for swatch_conf, ocr_conf in ((0.4, 0.95), (0.9, 0.3)):
        spec = _legend_chart([_row("Kenya", BAR_BGR), _row("Ghost", GREEN, swatch_conf, ocr_conf)]).run()
        assert [s.name for s in spec.series] == ["Kenya"]
        assert _skipped(spec) == ["legend_entry_skipped:low_confidence:Ghost"]


def test_unreadable_legend_text_is_reported_and_not_added():
    spec = _legend_chart([_row("Kenya", BAR_BGR), _row("", GREEN)]).run()
    assert [s.name for s in spec.series] == ["Kenya"]
    assert len(_skipped(spec)) == 1 and _skipped(spec)[0].startswith("legend_entry_skipped:unreadable_text:")


def test_a_repeated_name_is_not_added_twice():
    spec = _legend_chart([_row("Kenya", BAR_BGR), _row("kenya", GREEN)]).run()
    assert [s.name for s in spec.series] == ["Kenya"]
    assert _skipped(spec) == ["legend_entry_skipped:duplicate_name:kenya"]


def test_a_swatch_outside_the_legend_box_is_not_added():
    chart = _legend_chart([_row("Kenya", BAR_BGR), _row("Ghost", GREEN)])
    legend = next(d for d in chart.detections if d.role == "legend")
    chart.detections[chart.detections.index(legend)] = type(legend)(
        legend.id, legend.role, Box(563, 46, 600, 70), legend.confidence)    # covers the first row only
    spec = chart.run()
    assert [s.name for s in spec.series] == ["Kenya"]
    assert _skipped(spec) == ["legend_entry_skipped:outside_legend:Ghost"]


def test_no_series_are_added_while_bars_are_not_all_matched():
    # A supplied unresolved mapping bypasses filtering. Its entries must not
    # create empty series while existing bars still have unsettled ownership.
    chart = _legend_chart([_row("Alpha", RED), _row("Beta", GREEN)])
    from src.reconstruction.legend_matcher import match_legend_to_bars
    from src.reconstruction.vertical import _legend_ocr_frame, _text_table

    frame = chart.frame()
    mapping = match_legend_to_bars(chart.image, frame, _legend_ocr_frame(frame, _text_table(chart.texts)))
    spec = reconstruct_vbar(frame, chart.texts, chart.image, legend_mapping=mapping)
    assert not any(s.no_visible_bars for s in spec.series)
    assert sorted(_skipped(spec)) == ["legend_entry_skipped:bars_not_settled:Alpha",
                                      "legend_entry_skipped:bars_not_settled:Beta"]


def _custom_entries(chart, order):
    """The matcher's assignments for ``chart`` with entries in ``order`` (indices into the legend)."""
    frame, mapping = _matched_mapping(chart)
    swatches = [d for d in frame.detections if d.role == "legend_swatch"]
    labels = [d for d in frame.detections if d.role == "legend_label"]
    names = ("Alpha", "Beta", "Gamma")
    entries = tuple(LegendEntry(swatches[i].id, labels[i].id, names[i]) for i in order)
    return frame, LegendMappingFrame(frame.source_key, mapping.assignments, entries)


def _grouped_with_third_entry():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    chart.add("legend_swatch", (200, 365, 214, 379))
    chart.paint((200, 365, 214, 379), (200, 100, 30))
    chart.add("legend_label", (220, 365, 280, 379), text="Gamma")
    return chart


def test_bar_less_series_slots_into_the_legend_order():
    chart = _grouped_with_third_entry()
    frame, mapping = _custom_entries(chart, order=(0, 2, 1))     # Alpha, Gamma (no bars), Beta
    spec = reconstruct_vbar(frame, chart.texts, chart.image, legend_mapping=mapping)
    assert [(s.name, s.no_visible_bars) for s in spec.series] == [
        ("Alpha", False), ("Gamma", True), ("Beta", False)]
    assert "legend_order_differs" not in spec.flags


def test_bar_order_wins_when_it_disagrees_with_the_legend_order():
    chart = _grouped_with_third_entry()
    frame, mapping = _custom_entries(chart, order=(1, 0, 2))     # legend lists Beta before Alpha
    spec = reconstruct_vbar(frame, chart.texts, chart.image, legend_mapping=mapping)
    assert [s.name for s in spec.series] == ["Alpha", "Beta", "Gamma"]
    assert spec.series[2].no_visible_bars and "legend_order_differs" in spec.flags


def test_a_hand_built_mapping_without_entries_adds_nothing():
    chart = _grouped(((30, 40, 200), (60, 180, 60)))
    frame, mapping = _matched_mapping(chart)
    assert mapping.entries == ()
    spec = reconstruct_vbar(frame, chart.texts, chart.image, legend_mapping=mapping)
    assert [s.name for s in spec.series] == ["Alpha", "Beta"]


# Reconstruct hbar

HBAR_PLOT = (100, 40, 560, 340)
ZERO_X, HBAR_PX_PER_UNIT = 100.0, 2.0  # value v sits at x = 100 + 2 * v


def x_of(value):
    return ZERO_X + HBAR_PX_PER_UNIT * value


class HorizontalChart:
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
    chart = HorizontalChart()
    chart.add("plot_area", HBAR_PLOT)
    for tick in (0, 50, 100, 150, 200):
        chart.add("x_tick_label", (x_of(tick) - 12, 345, x_of(tick) + 12, 358), text=str(tick))
    if titles:
        chart.add("chart_title", (150, 5, 450, 30), text="Sales by region")
        chart.add("x_axis_title", (280, 375, 380, 395), text="Revenue")
        chart.add("y_axis_title", (2, 150, 18, 250), text="Region")
    rows = values if isinstance(values[0], tuple) else tuple((v,) for v in values)
    slot = (HBAR_PLOT[3] - HBAR_PLOT[1]) / len(rows)
    thickness = 40 / len(series_colours)
    for i, (row, name) in enumerate(zip(rows, names)):
        cy = HBAR_PLOT[1] + slot * (i + 0.5)
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
    chart = HorizontalChart()
    chart.add("horizontal_bar", (100, 50, 200, 70))
    chart.add("vertical_bar", (300, 100, 320, 200))
    assert bar_orientation(chart.frame()) == "vertical"


def test_low_confidence_bars_do_not_decide_the_orientation():
    chart = HorizontalChart()
    chart.add("vertical_bar", (300, 100, 320, 200), conf=0.9)
    for k in range(3):
        chart.add("horizontal_bar", (100, 50 + 30 * k, 200, 70 + 30 * k), conf=0.1)
    assert bar_orientation(chart.frame()) == "vertical"


def _shorten_hbar(chart, index, dx):
    """Move the detector's right edge of one horizontal bar left by ``dx`` px."""
    bars = [d for d in chart.detections if d.role == "horizontal_bar"]
    old = bars[index]
    chart.detections[chart.detections.index(old)] = Detection(
        old.id, old.role, Box(old.box.x1, old.box.y1, old.box.x2 - dx, old.box.y2), old.confidence)


def test_horizontal_bar_end_far_from_the_pixels_is_corrected():
    chart = hbar_chart(values=(120, 60, 180))
    _shorten_hbar(chart, 2, 40)  # 40 px = 20 units short
    spec = chart.run()
    assert spec.series[0].values == pytest.approx([120, 60, 180], rel=0.01)
    assert "bar_edge_corrected:1" in spec.flags
    assert spec.series[0].value_rescued == [False, False, False]


def test_small_horizontal_edge_differences_keep_the_detector_value():
    chart = hbar_chart(values=(120, 60, 180))
    _shorten_hbar(chart, 2, 4)  # under BAR_EDGE_TOLERANCE_PX
    spec = chart.run()
    assert not any(f.startswith("bar_edge_corrected") for f in spec.flags)
    assert spec.series[0].values[2] == pytest.approx(180 - 4 / HBAR_PX_PER_UNIT, rel=0.01)


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


# Parsing provenance

def events(diagnostics, code):
    return [e for e in diagnostics.record['events'] if e['code'] == code]


def recorded(chart, **kwargs):
    diagnostics = AnalysisDiagnostics(settings={})
    diagnostics.source_images(chart.image, chart.frame())
    plain = reconstruct_vbar(chart.frame(), chart.texts, chart.image, **kwargs)
    traced = reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                             diagnostics=diagnostics, **kwargs)
    assert asdict(traced) == asdict(plain)
    assert check_completeness(traced, diagnostics=diagnostics) == check_completeness(plain)
    return traced, diagnostics


@pytest.mark.parametrize('kind', ['frame', 'regions', 'strings', 'tuples'])
def test_original_consumer_text_survives_all_legacy_formats(kind):
    raw = ' \t雪 e\u0301\n'
    region = RecognizedText('label', raw, .9)
    data = {'frame': OCRFrame('input/chart.png', (region,)), 'regions': [region],
            'strings': {'label': raw}, 'tuples': {'label': (raw, '.9')}}[kind]
    diagnostics = AnalysisDiagnostics(settings={})
    assert _text_table(data, diagnostics=diagnostics) == _text_table(data)
    event, = events(diagnostics, 'consumer_text')
    assert event['input_text'] == raw
    assert event['consumer_text'] == '雪 e\u0301'
    assert event['transformations'] == [{'operation': 'strip', 'before': raw, 'after': '雪 e\u0301'}]


@pytest.mark.parametrize('text, expected, reason', [
    (' \t$−1, 250K%\n', (-1250000., '%'), 'parsed'),
    ('–.5M', (-500000., None), 'parsed'),
    ('1e309', None, 'nonfinite'), ('1\u00a0000', None, 'invalid_number'),
    ('', None, 'invalid_number'), (' \t', None, 'invalid_number'),
    ('3.5B', (3500000000., None), 'parsed'),
])
def test_numeric_steps_are_actual_and_do_not_expand_normalization(text, expected, reason):
    diagnostics = AnalysisDiagnostics(settings={})
    assert parse_tick_value(text) == expected
    assert parse_tick_value(text, diagnostics=diagnostics, detection_id='tick') == expected
    event, = events(diagnostics, 'numeric_parse')
    assert event['reason'] == reason
    steps = event['transformations']
    assert steps[0]['before'] == text
    for before, after in zip(steps, steps[1:]):
        assert before['after'] == after['before']
    assert steps[-1]['after'] == event['numeric_text']
    json.dumps(diagnostics.record, allow_nan=False)


@pytest.mark.parametrize('state, expected_reason', [
    ('absent_detection', 'no_detection'), ('below_confidence', 'filtered_out'),
    ('absent_ocr', 'absent_ocr'), ('whitespace', 'empty_text'),
])
def test_missing_title_distinguishes_filtering_and_text_availability(state, expected_reason):
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'chart_title')
    if state == 'absent_detection':
        chart.detections.remove(title)
        del chart.texts[title.id]
    elif state == 'below_confidence':
        chart.detections[chart.detections.index(title)] = replace(title, confidence=.1)
    elif state == 'absent_ocr':
        del chart.texts[title.id]
    else:
        chart.texts[title.id] = (' \t\n', .8)
    spec, diagnostics = recorded(chart)
    assert spec.title is None
    decision = next(e for e in events(diagnostics, 'title_selection') if e['field'] == 'title')
    assert decision['reason'] == expected_reason
    diagnostics.warning('title_missing', 'reconstruction', flag='title_missing')
    warning, = diagnostics.record['warnings']
    assert decision['event_id'] in warning['evidence_ids']
    source = diagnostics.record['spec_sources']['title']
    assert decision['event_id'] in source['evidence_ids']
    finding = next(e for e in events(diagnostics, 'completeness_check')
                   if e['rule'] == 'missing_item' and e['field'] == 'title')
    assert finding['source'] == source


def test_duplicates_and_spatial_exclusions_have_measured_causes():
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'chart_title')
    duplicate = chart.add('chart_title', (150, 5, 450, 30), conf=.8, text='wrong')
    outside = chart.add('y_tick_label', (5, 380, 30, 395), text='999')
    _, diagnostics = recorded(chart)
    excluded = next(e for e in events(diagnostics, 'detection_deduplication')
                    if e['detection_id'] == duplicate)
    assert excluded['retained_detection_id'] == title.id
    assert excluded['measured_iou'] == 1.
    assert excluded['measured_iou'] > excluded['duplicate_iou']
    selection = next(e for e in events(diagnostics, 'tick_spatial_selection')
                     if e['role'] == 'y_tick_label')
    decision = next(e for e in selection['decisions'] if e['detection_id'] == outside)
    assert decision['outcome'] == 'outside_plot_band'
    assert decision['coordinate'] > selection['accepted_bounds'][1]


def test_unreadable_category_links_to_original_text_and_source_crop():
    chart = base_chart()
    category = next(d for d in chart.detections if d.role == 'x_tick_label')
    chart.texts[category.id] = (' \t|\n', .96)
    _, diagnostics = recorded(chart)
    decision = next(e for e in events(diagnostics, 'category_text_check') if e['unreadable'])
    assert decision['text_evidence']['input_text'] == ' \t|\n'
    assert decision['text'] == '|'
    assert diagnostics.record['spec_sources']['category[0]']['detection_ids'] == [category.id]
    assert any(s['detection_id'] == category.id for s in diagnostics.record['ocr_source_crops'])
    assert any(e['rule'] == 'unreadable_category' and e['field'] == 'category[0]'
               for e in events(diagnostics, 'completeness_check'))


@pytest.mark.parametrize('categories, rule, index', [
    (['Germany', 'Greece', 'i'], 'single_character_category', 2),
    (['2007', '2008', '6002'], 'year_category_outlier', 2),
])
def test_suspicious_categories_and_supplied_exemptions(categories, rule, index):
    spec = make_spec(categories=categories)
    diagnostics = AnalysisDiagnostics(settings={})
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    check = next(e for e in events(diagnostics, 'completeness_check')
                 if e['rule'] == rule and e['field'] == f'category[{index}]')
    assert check['triggered'] and not check['supplied_exemption']
    assert check['source']['availability'] == 'unavailable'
    supplied = replace(spec, flags=[f'category_supplied:{index}'])
    other = AnalysisDiagnostics(settings={})
    assert check_completeness(supplied, diagnostics=other) == check_completeness(supplied)
    check = next(e for e in events(other, 'completeness_check')
                 if e['rule'] == rule and e['field'] == f'category[{index}]')
    assert check['supplied_exemption']


def test_braille_uses_actual_whitespace_normalization_and_original_call_counts():
    from src.tactile.braille import to_braille
    spec = make_spec(title=' \tBad ]\n')
    with patch('src.tactile.completeness.to_braille', wraps=to_braille) as encoder:
        plain = check_completeness(spec)
        original_calls = encoder.call_count
    diagnostics = AnalysisDiagnostics(settings={})
    with patch('src.tactile.completeness.to_braille', wraps=to_braille) as encoder:
        traced = check_completeness(spec, diagnostics=diagnostics)
        assert encoder.call_count == original_calls
    assert traced == plain
    encoding = next(e for e in events(diagnostics, 'braille_input') if e['field'] == 'title')
    assert encoding['input_text'] == spec.title
    assert encoding['encoder_text'] == 'Bad ]'
    rejection = next(e for e in events(diagnostics, 'completeness_check')
                     if e['rule'] == 'braille_encoding' and e['field'] == 'title')
    assert rejection['rejected_characters'] == [']']
    assert rejection['evidence_ids'] == [encoding['event_id']]


def test_blockers_preserve_early_return_without_inventing_braille_checks():
    spec = make_spec(title='Bad ]', flags=['y_tick_leading_zero'])
    diagnostics = AnalysisDiagnostics(settings={})
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    assert not events(diagnostics, 'braille_input')
    assert any(e['rule'] == 'blocked_early_return' for e in events(diagnostics, 'completeness_check'))


def test_competing_calibration_sets_and_failure_before_refit_are_retained():
    ticks = [(0., 0.), (10., 10.), (20., 40.)]
    diagnostics = AnalysisDiagnostics(settings={})
    plain = _calibrate(ticks)
    assert _calibrate(ticks, diagnostics=diagnostics) == plain
    selected, = events(diagnostics, 'calibration_selection')
    assert selected['ambiguous']
    assert len(selected['competing_inlier_sets']) > 1
    candidates = events(diagnostics, 'calibration_candidate')
    for candidate in candidates:
        expected = [abs(v - (candidate['a'] * y + candidate['b'])) / abs(candidate['a'])
                    for y, v in ticks]
        assert candidate['residuals_pixels'] == expected
    failed = AnalysisDiagnostics(settings={})
    failure = RuntimeError('refit failed')
    with patch('src.reconstruction.vertical.np.polyfit', side_effect=failure), pytest.raises(RuntimeError) as caught:
        _calibrate(ticks, diagnostics=failed)
    assert caught.value is failure
    assert events(failed, 'calibration_selection')
    assert not events(failed, 'y_calibration')


def test_reconstruction_failure_preserves_parse_evidence_and_original_exception():
    chart = base_chart()
    ticks = [d for d in chart.detections if d.role == 'y_tick_label']
    for tick in ticks:
        chart.texts[tick.id] = ('?', .9)
    diagnostics = AnalysisDiagnostics(settings={})
    with pytest.raises(ChartInterpretationError) as plain:
        chart.run()
    with pytest.raises(ChartInterpretationError) as traced:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image, diagnostics=diagnostics)
    assert str(plain.value) == str(traced.value)
    assert len(events(diagnostics, 'numeric_parse')) == len(ticks)
    assert events(diagnostics, 'reconstruction_blocked')[-1]['required_count'] == 2
    assert not events(diagnostics, 'calibration_inputs')


def test_title_cutoff_measurements_and_series_sources_follow_final_order():
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    spec, diagnostics = recorded(chart, legend_mapping=mapping)
    labels = {entry.name: entry.label_id for entry in mapping.entries}
    for index, series in enumerate(spec.series):
        assert diagnostics.record['spec_sources'][f'series_name[{index}]']['detection_ids'] == [labels[series.name]]
    chart.add("chart_title", (150, 5, 450, 30), text="Sales")
    title = next(d for d in chart.detections if d.role == 'chart_title')
    chart.image[int(title.box.y1):int(title.box.y2), -3:] = 0
    assert cut_off_flags(chart.frame(), chart.image, diagnostics=diagnostics) == cut_off_flags(chart.frame(), chart.image)
    cutoff = next(e for e in events(diagnostics, 'title_cutoff_check') if e['field'] == 'title')
    assert cutoff['dark_pixel_count'] >= cutoff['required_ink_pixels']
    assert cutoff['flag'] == 'title_cut_off'


def test_event_links_warning_deduplication_and_json_zip_roundtrip(tmp_path):
    chart = base_chart(titles=False)
    _, diagnostics = recorded(chart)
    for flag in ['title_missing', 'title_missing', 'x_axis_title_missing']:
        diagnostics.warning(flag, 'reconstruction', flag=flag)
    assert len(diagnostics.record['warnings']) == 2
    all_ids = {e['event_id'] for e in diagnostics.record['events']}
    assert len(all_ids) == len(diagnostics.record['events'])
    for warning in diagnostics.record['warnings']:
        assert warning['evidence_ids']
        assert set(warning['evidence_ids']) <= all_ids
    diagnostics.finish()
    persisted = diagnostics.persist(tmp_path)
    archive_path = diagnostics.export_zip(tmp_path)
    with ZipFile(archive_path) as archive:
        assert json.loads(archive.read('diagnostics.json')) == json.loads(persisted.read_text())
    other = AnalysisDiagnostics(settings={})
    new_id = other.event('test', 'event')
    assert new_id not in all_ids
    assert other.record.get('spec_sources', {}) == {}


@pytest.mark.parametrize('label, confidence, expected', [
    (' \t', .9, 'unreadable_text'), ('Ghost', .1, 'low_confidence'),
    (' KENYA ', .9, 'duplicate_name'),
])
def test_legend_skip_keeps_exact_flag_guard_and_text(label, confidence, expected):
    chart = _legend_chart([_row('Kenya', BAR_BGR), _row(label, GREEN, ocr_conf=confidence)])
    spec, diagnostics = recorded(chart)
    decision = next(e for e in events(diagnostics, 'legend_entry_decision') if e['outcome'] == 'skipped')
    assert decision['first_failing_guard'] == expected
    assert decision['text_evidence']['input_text'] == label
    assert decision['text'] == label.strip()
    assert decision['flag'] in spec.flags
    diagnostics.warning('legend_entry_skipped', 'reconstruction', flag=decision['flag'])
    assert decision['event_id'] in diagnostics.record['warnings'][0]['evidence_ids']
    if expected == 'duplicate_name':
        assert decision['duplicate_name_key'] == 'kenya'
        assert 'kenya' in decision['existing_name_keys']
    if expected == 'low_confidence':
        assert decision['ocr_confidence'] < decision['confidence_threshold']


def test_calibration_rejects_flat_ticks_with_original_error_and_evidence():
    ticks = [(0., 10.), (20., 10.)]
    diagnostics = AnalysisDiagnostics(settings={})
    with pytest.raises(ChartInterpretationError) as plain:
        _calibrate(ticks)
    with pytest.raises(ChartInterpretationError) as traced:
        _calibrate(ticks, diagnostics=diagnostics)
    assert str(plain.value) == str(traced.value)
    selected, = events(diagnostics, 'calibration_selection')
    assert selected['outcome'] == 'rejected'
    assert selected['flag'] == 'chart_interpretation_blocked'
    assert all(e['reason'] == 'equal_value' for e in events(diagnostics, 'calibration_candidate'))


def test_series_field_relationships_survive_later_reconstruction_failure():
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    diagnostics = AnalysisDiagnostics(settings={})
    failure = RuntimeError('legend ordering failed')
    with patch('src.reconstruction.vertical._in_legend_order', side_effect=failure), pytest.raises(RuntimeError) as caught:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                         legend_mapping=mapping, diagnostics=diagnostics)
    assert caught.value is failure
    assert diagnostics.record['spec_sources']['series_name[0]']['detection_ids'] == [mapping.entries[1].label_id]
    assert diagnostics.record['spec_sources']['series_name[1]']['detection_ids'] == [mapping.entries[2].label_id]
    assert 'category[0]' in diagnostics.record['spec_sources']


def test_cutoff_finding_links_to_the_actual_cutoff_flag_not_an_earlier_edge_note():
    spec = make_spec(flags=['title_touches_right_edge', 'title_cut_off'])
    diagnostics = AnalysisDiagnostics(settings={})
    edge_id = diagnostics.event('reconstruction', 'title_edge_check', flag='title_touches_right_edge')
    cutoff_id = diagnostics.event('reconstruction', 'title_cutoff_check', flag='title_cut_off')
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    finding = next(e for e in events(diagnostics, 'completeness_check')
                   if e['rule'] == 'missing_or_cutoff_title')
    assert finding['flag'] == 'title_cut_off'
    assert cutoff_id in finding['evidence_ids']
    assert edge_id not in finding['evidence_ids']


def test_multiple_legend_skips_keep_distinct_evidence_under_one_warning_code():
    chart = _legend_chart([_row('Kenya', BAR_BGR), _row('Ghost', GREEN, ocr_conf=.1),
                           _row(' KENYA ', (20, 120, 70))])
    spec, diagnostics = recorded(chart)
    decisions = [e for e in events(diagnostics, 'legend_entry_decision') if e['outcome'] == 'skipped']
    assert {e['first_failing_guard'] for e in decisions} == {'low_confidence', 'duplicate_name'}
    for flag in spec.flags:
        if flag.startswith('legend_entry_skipped:'):
            diagnostics.warning('legend_entry_skipped', 'reconstruction', flag=flag)
    warning, = diagnostics.record['warnings']
    assert {e['event_id'] for e in decisions} <= set(warning['evidence_ids'])


def test_numeric_normalization_is_not_repeated_for_diagnostics():
    from src.reconstruction.vertical import _normalized_tick_text
    chart = base_chart()
    with patch('src.reconstruction.vertical._normalized_tick_text', wraps=_normalized_tick_text) as normalize:
        chart.run()
        plain_calls = normalize.call_count
    diagnostics = AnalysisDiagnostics(settings={})
    with patch('src.reconstruction.vertical._normalized_tick_text', wraps=_normalized_tick_text) as normalize:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image, diagnostics=diagnostics)
        assert normalize.call_count == plain_calls


def test_all_bars_with_unmatched_colours_reach_reconstruction():
    chart = _legend_chart([_row("Alpha", RED), _row("Beta", GREEN)])
    spec = chart.run()
    assert "bar_colour_mismatch:3" in spec.flags
    assert any(series.values == pytest.approx([120, 60, 180]) for series in spec.series)
    assert any("Review the series assignments" in note for note in check_completeness(spec).notes)


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("loose_swatch", [False, True])
def test_legend_mismatches_preserve_values_or_block_output(horizontal, loose_swatch):
    from src.reconstruction.horizontal import transpose_image
    from src.reconstruction.legend_matcher import filter_bars_by_legend
    from src.reconstruction.vertical import _legend_ocr_frame, _text_table

    legend_rows = [_row("Kenya", BAR_BGR)]
    if loose_swatch:
        legend_rows.append(_row("Other", GREEN))
    chart = _legend_chart(legend_rows)
    bar = [d for d in chart.detections if d.role == "vertical_bar"][1]
    chart.paint((bar.box.x1, bar.box.y1, bar.box.x2, bar.box.y2), GREEN)
    frame, image = chart.frame(), chart.image
    if loose_swatch:
        swatches = [d for d in frame.detections if d.role == "legend_swatch"]
        swatch = swatches[-1]
        # Retain the swatch detection and label while diluting the crop with white.
        chart.paint((swatch.box.x1, swatch.box.y1,
                     swatch.box.x2, swatch.box.y2), (255, 255, 255))
        x = int((swatch.box.x1 + swatch.box.x2) / 2)
        y = int((swatch.box.y1 + swatch.box.y2) / 2)
        chart.paint((x - 1, y - 1, x + 1, y + 1), GREEN)
    filtered = filter_bars_by_legend(image, frame, _legend_ocr_frame(frame, _text_table(chart.texts)))
    assert filtered.frame is frame
    assert filtered.mismatched_bar_ids == (bar.id,)
    if horizontal:
        frame, image = transpose_frame(frame), transpose_image(image)
    reconstruct = reconstruct_hbar if horizontal else reconstruct_vbar
    spec = reconstruct(frame, chart.texts, image, legend_mapping=filtered.mapping,
                       mismatched_bar_ids=filtered.mismatched_bar_ids)
    assert "bar_colour_mismatch:1" in spec.flags
    assert any(value == pytest.approx(60) for series in spec.series for value in series.values)
    assert any("Review the series assignments" in note for note in check_completeness(spec).notes)


def _reconstruct_with_mismatch_evidence(chart, mismatch_ids, horizontal):
    """Exercise supplied mismatch evidence independently of colour sampling."""
    from functools import partial
    from src.reconstruction.horizontal import transpose_image
    from src.reconstruction.legend_matcher import filter_bars_by_legend
    from src.reconstruction.vertical import _legend_ocr_frame

    frame = chart.frame()
    mapping = filter_bars_by_legend(
        chart.image, frame, _legend_ocr_frame(frame, _text_table(chart.texts))).mapping
    # All bars in these fixtures belong to the same series. Supply its semantic
    # identity explicitly so collision tests exercise slots rather than grouping.
    identity = next(a for a in mapping.assignments if a.status == "matched")
    mapping = replace(mapping, assignments=tuple(
        replace(identity, bar_id=d.id) for d in frame.detections if d.role == "vertical_bar"))
    diagnostics = AnalysisDiagnostics(settings={})
    options = dict(legend_mapping=mapping, mismatched_bar_ids=mismatch_ids, rescue_bars=False)
    if horizontal:
        # The horizontal wrapper has no diagnostics argument; capture the same
        # vertical reconstruction events without changing that public interface.
        with patch("src.reconstruction.horizontal.reconstruct_vbar",
                   side_effect=partial(reconstruct_vbar, diagnostics=diagnostics)):
            spec = reconstruct_hbar(transpose_frame(frame), chart.texts,
                                    transpose_image(chart.image), **options)
    else:
        spec = reconstruct_vbar(frame, chart.texts, chart.image,
                                diagnostics=diagnostics, **options)
    return spec, diagnostics


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("exclusion", ["below_confidence", "duplicate", "outside_plot"])
@pytest.mark.parametrize("eligible_mismatch", [False, True])
def test_excluded_colour_mismatches_do_not_block(horizontal, exclusion, eligible_mismatch):
    chart = _legend_chart([_row("Kenya", BAR_BGR)])
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    original = bars[0]
    box = original.box
    if exclusion == "outside_plot":
        box = Box(570, 100, 590, 180)
    candidate_id = chart.add("vertical_bar", (box.x1, box.y1, box.x2, box.y2),
                             conf=0.1 if exclusion == "below_confidence" else 0.8)
    mismatch_ids = (candidate_id, bars[1].id) if eligible_mismatch else (candidate_id,)
    spec, diagnostics = _reconstruct_with_mismatch_evidence(chart, mismatch_ids, horizontal)

    assert spec.series[0].values == pytest.approx([120, 60, 180])
    assert not any(f.startswith("bar_colour_unaccounted:") for f in spec.flags)
    assert [f for f in spec.flags if f.startswith("bar_colour_mismatch:")] == (
        ["bar_colour_mismatch:1"] if eligible_mismatch else [])
    assessment = check_completeness(spec)
    assert not assessment.blockers
    notes = [note for note in assessment.notes if "Review the series assignments" in note]
    assert notes == (["1 detected bar(s) could not be matched to the legend. "
                      "Review the series assignments."] if eligible_mismatch else [])
    assert not events(diagnostics, "bar_colour_unaccounted")
    if exclusion == "below_confidence":
        decision = next(e for e in events(diagnostics, "detection_confidence")
                        if e["detection_id"] == candidate_id)
        assert decision["outcome"] == "below_confidence"
    elif exclusion == "duplicate":
        decision = next(e for e in events(diagnostics, "detection_deduplication")
                        if e["detection_id"] == candidate_id)
        assert decision["outcome"] == "duplicate"
        assert decision["retained_detection_id"] == original.id
        assert decision["measured_iou"] > decision["duplicate_iou"]
    else:
        selection, = events(diagnostics, "plot_selection")
        assert candidate_id in selection["excluded_bar_ids"]


def test_excluded_mismatch_retains_original_legend_evidence():
    chart = _legend_chart([_row("Kenya", BAR_BGR)])
    box = (570, 100, 590, 180)
    candidate_id = chart.add("vertical_bar", box)
    chart.paint(box, GREEN)
    spec, diagnostics = recorded(chart)

    decision = next(e for e in events(diagnostics, "bar_colour_validation")
                    if e["bar_id"] == candidate_id)
    assert decision["outcome"] == "mismatch"
    assert decision["reason"] == "colour_mismatch"
    matching, = events(diagnostics, "legend_colour_filter")
    assert matching["mismatched_bar_ids"] == [candidate_id]
    selection, = events(diagnostics, "plot_selection")
    assert candidate_id in selection["excluded_bar_ids"]
    assert not any(f.startswith("bar_colour_") for f in spec.flags)
    assert not check_completeness(spec).blockers


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("loser_first", [False, True])
def test_eligible_colour_mismatch_losing_slot_is_a_non_forceable_blocker(horizontal, loser_first):
    chart = _legend_chart([_row("Kenya", BAR_BGR)])
    winner = next(d for d in chart.detections if d.role == "vertical_bar")
    centre = (winner.box.x1 + winner.box.x2) / 2
    # Non-overlapping boxes survive deduplication but claim the same category.
    loser_x = centre + (-40 if loser_first else 40)
    loser_id = chart.add("vertical_bar", (loser_x - 10, y_of(90), loser_x + 10, ZERO_Y))
    spec, diagnostics = _reconstruct_with_mismatch_evidence(chart, (loser_id,), horizontal)
    assert spec.series[0].values == pytest.approx([120, 60, 180])
    assert "bar_colour_mismatch:1" in spec.flags
    assert "bar_colour_unaccounted:1" in spec.flags
    collision, = events(diagnostics, "category_ambiguous")
    assert collision["selected_bar_id"] == winner.id
    assert set(collision["competing_bar_ids"]) == {winner.id, loser_id}
    unaccounted, = events(diagnostics, "bar_colour_unaccounted")
    assert unaccounted["bar_ids"] == [loser_id]
    assessment = check_completeness(spec)
    assert assessment.status == "cannot_build"
    assert not assessment.forceable


def _record_short_rescue(chart, mismatch_ids, horizontal, **options):
    from functools import partial
    from src.reconstruction.horizontal import transpose_image
    from src.reconstruction.legend_matcher import filter_bars_by_legend
    from src.reconstruction.vertical import _legend_ocr_frame

    frame, image = chart.frame(), chart.image
    if "legend_mapping" not in options:
        options["legend_mapping"] = filter_bars_by_legend(image, frame,
            _legend_ocr_frame(frame, _text_table(chart.texts))).mapping
    reconstruct = reconstruct_vbar
    if horizontal:
        frame, image = transpose_frame(frame), transpose_image(image)
        reconstruct = reconstruct_hbar
    options = dict(options, mismatched_bar_ids=mismatch_ids)
    plain = reconstruct(frame, chart.texts, image, **options)
    diagnostics = AnalysisDiagnostics(settings={})
    if horizontal:
        with patch("src.reconstruction.horizontal.reconstruct_vbar",
                   side_effect=partial(reconstruct_vbar, diagnostics=diagnostics)):
            traced = reconstruct(frame, chart.texts, image, **options)
    else:
        traced = reconstruct(frame, chart.texts, image, diagnostics=diagnostics, **options)
    assert asdict(plain) == asdict(traced)
    return traced, diagnostics


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("matched_identity", [False, True])
def test_short_mismatch_links_original_id_to_final_reordered_slot(horizontal, matched_identity):
    from src.reconstruction.legend_matcher import filter_bars_by_legend
    from src.reconstruction.vertical import _legend_ocr_frame

    chart = _legend_chart([_row("Low", RED), _row("Kenya", BAR_BGR),
                           _row("Other", GREEN)], values=(120, 60, 4))
    short = [d for d in chart.detections if d.role == "vertical_bar"][-1]
    mapping = filter_bars_by_legend(chart.image, chart.frame(),
        _legend_ocr_frame(chart.frame(), _text_table(chart.texts))).mapping
    if not matched_identity:
        mapping = replace(mapping, assignments=tuple(
            LegendBarAssignment(a.bar_id) if a.bar_id == short.id else a
            for a in mapping.assignments))
    spec, diagnostics = _record_short_rescue(chart, (short.id,), horizontal, legend_mapping=mapping)
    assert spec.series[1].values == pytest.approx([120, 60, 4])
    assert spec.series[1].value_rescued == [False, False, True]
    assert "bar_colour_mismatch:1" in spec.flags
    assert not any(f.startswith("bar_colour_unaccounted:") for f in spec.flags)
    rescue, = events(diagnostics, "pixel_rescue")
    assert (rescue["bar_id"], rescue["series_index"], rescue["category_index"]) == (short.id, 1, 2)
    assert not check_completeness(spec).blockers


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("failure", ["zero", "unrelated", "occupied", "competitor", "identity"])
def test_unverified_short_mismatch_keeps_blocker(horizontal, failure):
    entries = [_row("Kenya", BAR_BGR)]
    if failure == "identity":
        entries.append(_row("Other", GREEN))
    chart = _legend_chart(entries, values=(120, 60, 4))
    short = [d for d in chart.detections if d.role == "vertical_bar"][-1]
    options = {}
    if failure == "zero":
        chart.paint((short.box.x1, 320, short.box.x2, ZERO_Y), (255, 255, 255))
    elif failure == "unrelated":
        # Rescue still measures C2, but the detection does not cover its probe.
        chart.detections[chart.detections.index(short)] = replace(
            short, box=Box(short.box.x1 + 40, short.box.y1, short.box.x2 + 40, short.box.y2))
    elif failure == "occupied":
        chart.add("vertical_bar", (short.box.x1, y_of(30), short.box.x2, ZERO_Y))
        chart.paint((short.box.x1, y_of(30), short.box.x2, ZERO_Y), BAR_BGR)
    elif failure == "competitor":
        # A non-mismatched short detection also claims the rescue; boxes do not overlap.
        chart.add("vertical_bar", (short.box.x1, 326, short.box.x2, 330))
    else:
        from src.reconstruction.legend_matcher import filter_bars_by_legend
        from src.reconstruction.vertical import _legend_ocr_frame
        mapping = filter_bars_by_legend(chart.image, chart.frame(),
            _legend_ocr_frame(chart.frame(), _text_table(chart.texts))).mapping
        identity = next(a for a in mapping.assignments if a.status == "matched")
        other = mapping.entries[1]
        options["legend_mapping"] = replace(mapping, assignments=tuple(
            a for a in mapping.assignments if a.bar_id != short.id
        ) + (replace(identity, bar_id=short.id, swatch_id=other.swatch_id,
                     label_id=other.label_id, series_name=other.name),))
    spec, diagnostics = _record_short_rescue(chart, (short.id,), horizontal, **options)
    assert "bar_colour_unaccounted:1" in spec.flags
    unaccounted, = events(diagnostics, "bar_colour_unaccounted")
    assert unaccounted["bar_ids"] == [short.id]
    assert all(e["bar_id"] is None for e in events(diagnostics, "pixel_rescue"))
    assessment = check_completeness(spec)
    assert assessment.status == "cannot_build" and not assessment.forceable


@pytest.mark.parametrize("horizontal", [False, True])
def test_short_rescue_clears_only_verified_mismatch(horizontal):
    chart = _legend_chart([_row("Kenya", BAR_BGR)], values=(120, 4, 4))
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    chart.paint((bars[2].box.x1, 320, bars[2].box.x2, ZERO_Y), (255, 255, 255))
    spec, diagnostics = _record_short_rescue(chart, (bars[1].id, bars[2].id), horizontal)
    assert "bar_colour_mismatch:2" in spec.flags
    assert "bar_colour_unaccounted:1" in spec.flags
    unaccounted, = events(diagnostics, "bar_colour_unaccounted")
    assert unaccounted["bar_ids"] == [bars[2].id]
    assert [e["bar_id"] for e in events(diagnostics, "pixel_rescue")] == [bars[1].id, None]


@pytest.mark.parametrize("horizontal", [False, True])
def test_short_detection_claiming_two_rescued_series_stays_unaccounted(horizontal):
    from src.reconstruction.legend_matcher import filter_bars_by_legend
    from src.reconstruction.vertical import _legend_ocr_frame

    chart = grouped_chart((RED, GREEN), [(4, 4), (150, 60), (40, 120)])
    first, second = [d for d in chart.detections if d.role == "vertical_bar"][:2]
    chart.detections[chart.detections.index(first)] = replace(first,
        box=Box(first.box.x1, first.box.y1, second.box.x2, first.box.y2))
    chart.detections.remove(second)
    mapping = filter_bars_by_legend(chart.image, chart.frame(),
        _legend_ocr_frame(chart.frame(), _text_table(chart.texts))).mapping
    mapping = replace(mapping, assignments=tuple(
        LegendBarAssignment(a.bar_id) if a.bar_id == first.id else a
        for a in mapping.assignments))
    spec, diagnostics = _record_short_rescue(chart, (first.id,), horizontal, legend_mapping=mapping)
    rescues = events(diagnostics, "pixel_rescue")
    assert len(rescues) == 2 and all(e["value"] != 0 for e in rescues)
    assert all(e["bar_id"] is None for e in rescues)
    assert "bar_colour_unaccounted:1" in spec.flags
    assert not check_completeness(spec).forceable


@pytest.mark.parametrize("missed_entry", [True, False])
def test_complete_mismatched_series_can_be_named_in_review(missed_entry):
    chart = _grouped((RED, GREEN))
    swatch = chart.detections[-2]
    if missed_entry:
        del chart.detections[-2:]
    else:
        box = swatch.box
        chart.paint((box.x1, box.y1, box.x2, box.y2), (255, 255, 255))
        chart.paint((box.x1 + 6, box.y1 + 6, box.x1 + 8, box.y1 + 8), GREEN)
    spec = chart.run()
    assert "bar_colour_mismatch:3" in spec.flags
    assert [series.name for series in spec.series] == ["Alpha", None]
    assert spec.series[0].values == pytest.approx([100, 150, 40])
    assert spec.series[1].values == pytest.approx([50, 80, 120])
    assessment = check_completeness(spec)
    assert assessment.status == "needs_input"
    assert any(item.field == "series_name[1]" for item in assessment.missing)


@pytest.mark.parametrize("horizontal", [False, True])
def test_complete_fallback_grid_can_be_named_in_review(horizontal):
    from src.tactile.completeness import Overrides, apply_overrides

    chart = _grouped((RED, GREEN))
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    # No semantic identities are available; every bar has mismatch evidence.
    mapping = LegendMappingFrame(chart.frame().source_key, tuple(
        LegendBarAssignment(bar.id) for bar in bars))
    spec, _ = _record_short_rescue(chart, [bar.id for bar in bars], horizontal,
                                 legend_mapping=mapping)

    assert [series.name for series in spec.series] == [None, None]
    assert_values(spec, [100, 150, 40], series=0)
    assert_values(spec, [50, 80, 120], series=1)
    assert "bar_colour_mismatch:6" in spec.flags
    assessment = check_completeness(spec)
    assert assessment.status == "needs_input"
    assert not assessment.blockers
    assert {item.field for item in assessment.missing} >= {
        "series_name[0]", "series_name[1]"}

    corrected, _ = apply_overrides(spec, Overrides(
        title="Sales", x_axis_title="Group", y_axis_title="Revenue",
        series_names={0: "Alpha", 1: "Beta"}))
    assert check_completeness(corrected).status == "ready"
    assert [series.values for series in corrected.series] == [
        series.values for series in spec.series]


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("failure", ["incomplete", "collision", "order", "missing_evidence"])
def test_unsafe_fallback_grid_keeps_review_blocked(horizontal, failure):
    chart = _grouped((RED, GREEN))
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    if failure == "incomplete":
        # Leave pixels in place: pixel rescue must not establish review eligibility.
        chart.detections.remove(bars[2])
    elif failure == "collision":
        box = bars[0].box
        moved = Box(box.x1 - 50, bars[2].box.y1, box.x2 - 50, ZERO_Y)
        chart.detections[chart.detections.index(bars[2])] = replace(bars[2], box=moved)
        chart.paint((moved.x1, moved.y1, moved.x2, moved.y2), RED)
    elif failure == "order":
        for bar, colour in zip(bars[2:4], (GREEN, RED)):
            box = bar.box
            chart.paint((box.x1, box.y1, box.x2, box.y2), colour)
    remaining = [d for d in chart.detections if d.role == "vertical_bar"]
    mapping = LegendMappingFrame(chart.frame().source_key, tuple(
        LegendBarAssignment(bar.id) for bar in remaining))
    mismatch_ids = [bar.id for bar in remaining]
    if failure == "missing_evidence":
        mismatch_ids.pop()
    spec, _ = _record_short_rescue(chart, mismatch_ids, horizontal,
                                 legend_mapping=mapping)

    assert "series_ambiguous" in spec.flags
    assert check_completeness(spec).status == "cannot_build"
    if failure == "collision":
        assert "category_ambiguous" in spec.flags
