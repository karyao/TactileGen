"""Tactile layout, braille, BANA rule checks and SVG output."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

from src.contract_io import load_vertical_bar_chart
from src.contracts import BarSeries, VerticalBarChart
from src.tactile import check_layout, layout_vertical_bar_chart, render_svg
from src.tactile.__main__ import main
from src.tactile.braille import to_braille, to_unicode
from src.tactile.layout import (
    BAR_MAX_WIDTH,
    BAR_MIN_WIDTH,
    GRID_MIN_SPACING,
    LABEL_MAX_GAP,
    LABEL_MIN_GAP,
    Line,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "tactile"


def load(name: str) -> VerticalBarChart:
    return load_vertical_bar_chart(FIXTURES / f"{name}.json")


def rule_ids(layout) -> set[str]:
    return {v.rule.id for v in check_layout(layout)}


class BrailleTest(unittest.TestCase):
    def check(self, text, expected):
        self.assertEqual(to_unicode(to_braille(text)), expected)

    def test_letters_and_capitals(self):
        self.check("abc", "⠁⠃⠉")
        self.check("Cat", "⠠⠉⠁⠞")
        self.check("USA map", "⠠⠠⠥⠎⠁⠀⠍⠁⠏")

    def test_numbers(self):
        self.check("12", "⠼⠁⠃")
        self.check("2.5", "⠼⠃⠲⠑")
        self.check("1,000", "⠼⠁⠂⠚⠚⠚")
        self.check("-5", "⠐⠤⠼⠑")

    def test_grade1_indicator_after_digit(self):
        self.check("3a", "⠼⠉⠰⠁")
        self.check("3k", "⠼⠉⠅")   # k-z cannot be read as digits

    def test_punctuation(self):
        self.check("Rainfall (mm)", "⠠⠗⠁⠊⠝⠋⠁⠇⠇⠀⠐⠣⠍⠍⠐⠜")
        self.check("50%", "⠼⠑⠚⠨⠴")

    def test_matches_liblouis_on_demo_labels(self):
        """Cross-check against liblouis UEB grade 1; skipped when liblouis is absent."""
        try:
            import louis
        except ImportError:
            self.skipTest("liblouis Python bindings (louis) are not installed")
        texts = {label.text for name in ("rainfall", "books_by_grade", "crowded")
                 for label in layout_vertical_bar_chart(load(name)).labels}
        for text in sorted(texts):
            with self.subTest(text=text):
                expected = louis.translateString(["unicode.dis", "en-ueb-g1.ctb"], text)
                self.assertEqual(to_unicode(to_braille(text)), expected.replace(" ", "⠀"))

    def test_unsupported_text_is_rejected(self):
        for text in ("café", "", "   ", "a_b"):
            with self.assertRaises(ValueError):
                to_braille(text)


class LayoutTest(unittest.TestCase):
    def test_demo_charts_pass_every_rule(self):
        for name in ("rainfall", "books_by_grade"):
            with self.subTest(name=name):
                self.assertEqual(check_layout(layout_vertical_bar_chart(load(name))), [])

    def test_contract_fixture_passes_every_rule(self):
        chart = load_vertical_bar_chart(ROOT / "tests/fixtures/contracts/vertical_bar_chart.json")
        self.assertEqual(check_layout(layout_vertical_bar_chart(chart)), [])

    def test_bar_heights_follow_values(self):
        layout = layout_vertical_bar_chart(load("rainfall"))
        heights = [bar.height for bar in layout.bars]
        values = [bar.value for bar in layout.bars]
        ratios = [h / v for h, v in zip(heights, values)]
        for ratio in ratios:
            self.assertAlmostEqual(ratio, ratios[0], delta=0.02)

    def test_bars_within_bana_widths(self):
        for bar in layout_vertical_bar_chart(load("books_by_grade")).bars:
            self.assertTrue(BAR_MIN_WIDTH <= bar.width <= BAR_MAX_WIDTH)

    def test_value_labels_numeric_and_aligned_with_grid(self):
        layout = layout_vertical_bar_chart(load("rainfall"))
        values = [l for l in layout.labels if l.role == "y_value"]
        self.assertEqual(values[0].text, "0")
        self.assertTrue(all(l.numeric for l in values))
        grid_ys = sorted(l.y1 for l in layout.lines if l.kind == "grid")
        # Dots 2-5 (the middle row) sit on each grid line.
        middles = sorted(l.bounds.y1 + (l.bounds.y2 - l.bounds.y1) / 2 for l in values[1:])
        for middle, grid_y in zip(middles, grid_ys):
            self.assertAlmostEqual(middle, grid_y, places=6)

    def test_key_only_for_several_series(self):
        self.assertEqual(layout_vertical_bar_chart(load("rainfall")).key_samples, ())
        layout = layout_vertical_bar_chart(load("books_by_grade"))
        self.assertEqual([s.series_index for s in layout.key_samples], [0, 1])
        self.assertTrue(all(s.bounds.y2 < layout.plot.y1 for s in layout.key_samples))

    def test_long_category_labels_are_keyed(self):
        chart = VerticalBarChart(
            "demo/long.png", "Long Names", "Region", "Count",
            ("Northern Territory", "South Australia", "Western Australia"),
            (BarSeries(None, (3, 5, 4)),))
        layout = layout_vertical_bar_chart(chart)
        self.assertTrue(layout.categories_keyed)
        self.assertEqual([l.text for l in layout.labels if l.role == "category"], ["a", "b", "c"])
        self.assertEqual(check_layout(layout), [])

    def test_crowded_chart_is_reported_not_dropped(self):
        layout = layout_vertical_bar_chart(load("crowded"))
        self.assertEqual(len(layout.bars), 36)
        self.assertTrue({"bar_width", "label_clearance", "fits_page"} <= rule_ids(layout))

    def test_negative_values_are_rejected(self):
        chart = VerticalBarChart("demo/n.png", "T", "X", "Y", ("a",), (BarSeries(None, (-1,)),))
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(chart)


class RuleTest(unittest.TestCase):
    """Each rule fires on a deliberately broken copy of a passing layout."""

    def setUp(self):
        self.layout = layout_vertical_bar_chart(load("books_by_grade"))

    def test_bar_width(self):
        bars = (replace(self.layout.bars[0], width=BAR_MIN_WIDTH - 1),) + self.layout.bars[1:]
        self.assertIn("bar_width", rule_ids(replace(self.layout, bars=bars)))

    def test_label_clearance(self):
        title = self.layout.labels[0]
        moved = replace(title, x=self.layout.bars[0].x, y=self.layout.bars[0].y)
        self.assertIn("label_clearance",
                      rule_ids(replace(self.layout, labels=(moved,) + self.layout.labels[1:])))

    def test_label_owner_distance(self):
        labels = tuple(replace(l, x=l.x - LABEL_MAX_GAP) if l.role == "y_value" else l
                       for l in self.layout.labels)
        self.assertIn("label_owner_distance", rule_ids(replace(self.layout, labels=labels)))

    def test_fits_page(self):
        labels = (replace(self.layout.labels[0], x=self.layout.usable.x2),) + self.layout.labels[1:]
        self.assertIn("fits_page", rule_ids(replace(self.layout, labels=labels)))

    def test_grid_spacing(self):
        grid = next(l for l in self.layout.lines if l.kind == "grid")
        extra = replace(grid, id="grid-extra", y1=grid.y1 + GRID_MIN_SPACING / 2,
                        y2=grid.y2 + GRID_MIN_SPACING / 2)
        self.assertIn("grid_spacing", rule_ids(replace(self.layout,
                                                       lines=self.layout.lines + (extra,))))

    def test_line_hierarchy(self):
        lines = tuple(replace(l, strength=5) if l.kind == "grid" else l for l in self.layout.lines)
        self.assertIn("line_hierarchy", rule_ids(replace(self.layout, lines=lines)))

    def test_primary_line_length(self):
        lines = tuple(replace(l, x2=l.x1 + 5) if l.id == "axis-x" else l
                      for l in self.layout.lines)
        self.assertIn("primary_line_length", rule_ids(replace(self.layout, lines=lines)))

    def test_tick_length(self):
        tick = Line("tick-1", "tick", 50, 100, 50, 105, 0.5, 1)
        self.assertIn("tick_length", rule_ids(replace(self.layout,
                                                      lines=self.layout.lines + (tick,))))

    def test_numeric_indicator(self):
        labels = tuple(replace(l, cells=to_braille("ten")) if l.id == "y-value-1" else l
                       for l in self.layout.labels)
        self.assertIn("numeric_indicator", rule_ids(replace(self.layout, labels=labels)))

    def test_all_bars_labeled(self):
        labels = tuple(l for l in self.layout.labels if l.id != "category-1")
        self.assertIn("all_bars_labeled", rule_ids(replace(self.layout, labels=labels)))

    def test_key_order_and_textures(self):
        swapped = tuple(replace(s, y=-s.y) for s in self.layout.key_samples)
        self.assertIn("key_order", rule_ids(replace(self.layout, key_samples=swapped)))
        same = replace(self.layout, series_textures=("dots", "dots"))
        self.assertIn("distinct_textures", rule_ids(same))

    def test_key_before_graph(self):
        samples = tuple(replace(s, y=self.layout.plot.y2) for s in self.layout.key_samples)
        self.assertIn("key_before_graph", rule_ids(replace(self.layout, key_samples=samples)))

    def test_key_sample_size(self):
        samples = (replace(self.layout.key_samples[0], height=5),) + self.layout.key_samples[1:]
        self.assertIn("key_sample_size", rule_ids(replace(self.layout, key_samples=samples)))

    def test_y_heading_length(self):
        labels = tuple(replace(l, cells=to_braille("a" * 21)) if l.role == "y_heading" else l
                       for l in self.layout.labels)
        self.assertIn("y_heading_length", rule_ids(replace(self.layout, labels=labels)))

    def test_texture_count(self):
        many = replace(self.layout, series_textures=tuple(f"t{k}" for k in range(6)))
        self.assertIn("texture_count", rule_ids(many))

    def test_key_format(self):
        labels = tuple(replace(l, y=l.y + 3) if l.id == "key-0" else l for l in self.layout.labels)
        self.assertIn("key_format", rule_ids(replace(self.layout, labels=labels)))

    def test_layout_refuses_more_than_five_series(self):
        chart = VerticalBarChart("demo/six.png", "T", "X", "Y", ("a",),
                                 tuple(BarSeries(f"s{k}", (1,)) for k in range(6)))
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(chart)

    def test_label_gap_constants_match_bana(self):
        self.assertAlmostEqual(LABEL_MIN_GAP, 3.175)
        self.assertAlmostEqual(LABEL_MAX_GAP, 6.35)


class SvgTest(unittest.TestCase):
    def test_svg_draws_every_braille_dot(self):
        layout = layout_vertical_bar_chart(load("books_by_grade"))
        root = ElementTree.fromstring(render_svg(layout).split("\n", 1)[1])
        ns = {"svg": "http://www.w3.org/2000/svg"}
        braille = root.find(".//svg:g[@id='braille']", ns)
        dots = braille.findall(".//svg:circle", ns)
        self.assertEqual(len(dots), sum(len(cell) for l in layout.labels for cell in l.cells))
        self.assertTrue(root.get("width").endswith("mm"))

    def test_cli_writes_svg_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out = Path(tmp) / "chart.svg"
            self.assertEqual(main(["--chart", str(FIXTURES / "rainfall.json"),
                                   "--output", str(out), "--show-print", "--strict"]), 0)
            self.assertIn("<svg", out.read_text(encoding="utf-8"))
            crowded = Path(tmp) / "crowded.svg"
            self.assertEqual(main(["--chart", str(FIXTURES / "crowded.json"), "--output",
                                   str(crowded), "--show-violations", "--strict"]), 1)
            self.assertIn('id="violations"', crowded.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
