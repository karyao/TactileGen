"""Horizontal bar layout and the BANA rules that depend on bar direction."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from src.contract_io import load_bar_chart, load_vertical_bar_chart, write_bar_chart
from src.contracts import BarSeries, HorizontalBarChart
from src.tactile.__main__ import main as tactile_main
from src.tactile.braille import BLANK, GUIDE_DOT
from src.tactile.constants import BAR_MIN_WIDTH, DOT_DIAMETER, DOT_PITCH, GUIDE_DOTS_MIN
from src.tactile.layout_hbar import (
    layout_horizontal_bar_chart,
    layout_horizontal_bar_chart_pages,
)
from src.tactile.rules import check_layout
from src.tactile.svg import render_svg

FIXTURE = Path(__file__).parent / "fixtures" / "tactile" / "rainfall_hbar.json"
RAIN = HorizontalBarChart(
    "demo/1.png", "Rainfall by city", "Rainfall (mm)", "City",
    ("Vancouver", "Calgary", "Toronto", "Halifax"),
    (BarSeries(None, (1150.0, 420.0, 830.0, 1450.0)),))
BOOKS = HorizontalBarChart(
    "demo/2.png", "Books read by grade", "Books", "Grade",
    ("Grade 3", "Grade 4", "Grade 5"),
    (BarSeries("Fiction", (12.0, 18.0, 9.0)), BarSeries("Non-fiction", (5.0, 7.0, 11.0))))
LONG = HorizontalBarChart(
    "demo/3.png", "Exports", "Value (billion USD)", "Country",
    ("Central African Republic", "Bosnia and Herzegovina", "Chad"),
    (BarSeries(None, (2.1, 5.4, 1.2)),))


def rule_ids(layout):
    return [v.rule.id for v in check_layout(layout)]


def by_role(layout, role):
    return [l for l in layout.labels if l.role == role]


class HorizontalLayoutTest(unittest.TestCase):
    def test_sample_charts_pass_every_rule(self):
        for chart in (RAIN, BOOKS, LONG):
            with self.subTest(chart.title):
                layout = layout_horizontal_bar_chart(chart)
                self.assertEqual(layout.orientation, "horizontal")
                self.assertEqual(layout.x_label_mode, "full")
                self.assertEqual(rule_ids(layout), [])
                self.assertEqual(layout.needs_review, ())

    def test_bars_run_right_from_the_axis_in_proportion(self):
        layout = layout_horizontal_bar_chart(RAIN)
        axis = layout.by_id("axis-y")
        # Values are measured from the centre of the axis line, where 0 is.
        lengths = {b.category_index: b.x + b.width - axis.x1 for b in layout.bars}
        for bar in layout.bars:
            self.assertGreater(bar.x, axis.x1)
            self.assertGreaterEqual(bar.height, BAR_MIN_WIDTH)
        self.assertAlmostEqual(lengths[3] / lengths[1], 1450 / 420, places=6)
        # Categories keep their printed order, top to bottom.
        tops = [min(b.y for b in layout.bars if b.category_index == j) for j in range(4)]
        self.assertEqual(tops, sorted(tops))

    def test_labels_are_left_aligned_with_guide_dots_and_centred_on_their_bars(self):
        layout = layout_horizontal_bar_chart(RAIN)
        labels = by_role(layout, "category")
        self.assertEqual(len({l.x for l in labels}), 1)
        axis = layout.by_id("axis-y")
        for label in labels:
            dots = 0
            while label.cells[-1 - dots] == GUIDE_DOT:
                dots += 1
            self.assertGreaterEqual(dots, GUIDE_DOTS_MIN)
            self.assertEqual(label.cells[-1 - dots], BLANK)
            self.assertLess(label.bounds.x2, axis.x1)
            bar = next(b for b in layout.bars if b.category_index == label.category_index)
            self.assertAlmostEqual(label.bounds.y1 + label.bounds.y2, 2 * bar.y + bar.height,
                                   places=6)

    def test_runovers_are_left_justified_and_only_the_last_line_has_guide_dots(self):
        layout = layout_horizontal_bar_chart(LONG)
        lines = [l for l in by_role(layout, "category") if l.category_index == 0]
        self.assertGreater(len(lines), 1)
        self.assertEqual(len({l.x for l in lines}), 1)
        self.assertNotIn(GUIDE_DOT, lines[0].cells)
        self.assertEqual(lines[-1].cells[-1], GUIDE_DOT)
        self.assertEqual([l.owner_id for l in lines], [None] * (len(lines) - 1) + ["axis-y"])

    def test_values_sit_on_grid_lines_with_dots_456_of_the_first_cell(self):
        layout = layout_horizontal_bar_chart(RAIN)
        columns = sorted(l.x1 for l in layout.lines
                         if l.kind in ("grid", "axis") and l.x1 == l.x2)
        values = by_role(layout, "x_value")
        self.assertEqual([v.text for v in values], ["0", "500", "1000", "1500"])
        offset = DOT_DIAMETER / 2 + DOT_PITCH
        for value, column in zip(values, columns):
            self.assertAlmostEqual(value.x + offset, column, places=6)
            self.assertGreater(value.y, layout.plot.y2)

    def test_headings_follow_bana_placement(self):
        layout = layout_horizontal_bar_chart(RAIN)
        values = by_role(layout, "x_value")
        x_heading = by_role(layout, "x_heading")[0]
        self.assertAlmostEqual(x_heading.x, min(v.x for v in values))
        self.assertGreater(x_heading.y, values[0].y)
        y_heading = by_role(layout, "y_heading")[0]
        self.assertAlmostEqual(y_heading.x, by_role(layout, "category")[0].x)
        self.assertLess(y_heading.bounds.y2, layout.plot.y1)

    def test_key_lists_series_top_to_bottom(self):
        layout = layout_horizontal_bar_chart(BOOKS)
        self.assertEqual([s.series_index for s in layout.key_samples], [0, 1])
        first_set = sorted((b for b in layout.bars if b.category_index == 0), key=lambda b: b.y)
        self.assertEqual([b.series_index for b in first_set], [0, 1])

    def test_many_long_categories_switch_to_keys(self):
        names = tuple(f"Very long region name number {k}" for k in range(7))
        chart = HorizontalBarChart("demo/4.png", "Regions", "Value", "Region", names,
                                   (BarSeries(None, tuple(float(k + 1) for k in range(7))),))
        layout = layout_horizontal_bar_chart(chart)
        self.assertNotEqual(layout.x_label_mode, "full")
        self.assertEqual(rule_ids(layout), [])
        self.assertEqual(len([l for l in layout.labels if l.role == "key"
                              and l.category_index is not None and l.id.count("-") == 2]), 7)

    def test_pages_and_svg(self):
        pages = layout_horizontal_bar_chart_pages(BOOKS)
        self.assertEqual(len(pages), 1)
        svg = render_svg(pages[0], show_print=True)
        self.assertIn("bar-1-2", svg)
        self.assertIn("Grade 5", svg)


class ChartFileTest(unittest.TestCase):
    def test_fixture_loads_as_a_horizontal_chart_and_round_trips(self):
        chart = load_bar_chart(FIXTURE)
        self.assertIsInstance(chart, HorizontalBarChart)
        self.assertEqual(chart.categories[0], "Vancouver")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chart.json"
            write_bar_chart(path, chart)
            self.assertEqual(load_bar_chart(path), chart)
        with self.assertRaises(ValueError):
            load_vertical_bar_chart(FIXTURE)

    def test_cli_renders_the_fixture_and_passes_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rain.svg"
            with redirect_stdout(io.StringIO()) as printed:
                code = tactile_main(["--chart", str(FIXTURE), "--output", str(out), "--strict"])
            self.assertEqual(code, 0)
            self.assertIn("x-value-0", out.read_text())
        self.assertIn("all passed", printed.getvalue())


class DirectionRulesTest(unittest.TestCase):
    def test_missing_guide_dots_are_reported(self):
        layout = layout_horizontal_bar_chart(RAIN)
        label = layout.by_id("category-0")
        bare = replace(label, cells=label.cells[:-3])
        broken = replace(layout, labels=tuple(bare if l is label else l for l in layout.labels))
        self.assertIn("guide_dots", rule_ids(broken))

    def test_value_off_its_grid_line_is_reported(self):
        layout = layout_horizontal_bar_chart(RAIN)
        label = layout.by_id("x-value-1")
        moved = replace(label, x=label.x + 2)
        broken = replace(layout, labels=tuple(moved if l is label else l for l in layout.labels))
        self.assertIn("value_alignment", rule_ids(broken))

    def test_thin_horizontal_bars_are_reported_by_their_thickness(self):
        layout = layout_horizontal_bar_chart(RAIN)
        thin = tuple(replace(b, height=5.0) for b in layout.bars)
        self.assertIn("bar_width", rule_ids(replace(layout, bars=thin)))
        # Long bars are fine: length is not limited for horizontal bars.
        self.assertNotIn("bar_width", rule_ids(layout))


if __name__ == "__main__":
    unittest.main()
