"""Tactile layout, braille, rules, SVG, generation, and pipeline."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch
from xml.etree import ElementTree
import xml.etree.ElementTree as ElementTree
from zipfile import ZipFile

from tests.legend_filter_helpers import passthrough_filter
from src.reconstruction.legend_matcher import LegendFilterResult
from src.contract_io import load_bar_chart, load_vertical_bar_chart, write_bar_chart
from src.contracts import (
    BarSeries,
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    HorizontalBarChart,
    SeriesSpec,
    VerticalBarChart,
    VerticalBarSpec,
    YAxisSpec,
)
from src.tactile import (
    RULES,
    Violation,
    check_layout,
    constants,
    layout_vertical_bar_chart,
    layout_vertical_bar_chart_pages,
    pipeline,
    render_svg,
)
from src.tactile.__main__ import main, main as tactile_main
from src.tactile.braille import (
    BLANK,
    GUIDE_DOT,
    NUMERIC_INDICATOR,
    TN_CLOSE,
    TN_OPEN,
    to_braille,
    to_unicode,
)
from src.tactile.constants import BAR_MIN_WIDTH, DOT_DIAMETER, DOT_PITCH, GUIDE_DOTS_MIN
from src.tactile.generation import GenerationError, cleanup_generation, generate_tactile_artifacts
from src.tactile.keys import abbreviation_keys, is_valid_abbreviation_key, numeric_keys
from src.tactile.layout import (
    BAR_MAX_WIDTH,
    BAR_MIN_WIDTH,
    GRID_MIN_SPACING,
    LABEL_MAX_GAP,
    LABEL_MIN_GAP,
    Line,
)
from src.tactile.layout_hbar import layout_horizontal_bar_chart, layout_horizontal_bar_chart_pages
from src.tactile.rules import RULES, Violation, check_layout
from src.tactile.svg import render_svg


# Tactile

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
        grid_ys = sorted({l.y1 for l in layout.lines if l.kind == "grid"})   # lines may be cut in pieces
        # Dots 2-5 (the middle row) sit on each grid line.
        middles = sorted(l.bounds.y1 + (l.bounds.y2 - l.bounds.y1) / 2 for l in values[1:])
        for middle, grid_y in zip(middles, grid_ys):
            self.assertAlmostEqual(middle, grid_y, places=6)

    def test_key_only_for_several_series(self):
        self.assertEqual(layout_vertical_bar_chart(load("rainfall")).key_samples, ())
        layout = layout_vertical_bar_chart(load("books_by_grade"))
        self.assertEqual([s.series_index for s in layout.key_samples], [0, 1])
        self.assertTrue(all(s.bounds.y2 < layout.plot.y1 for s in layout.key_samples))

    def test_single_series_with_a_name_gets_a_key(self):
        chart = load("rainfall")
        named = replace(chart, series=(replace(chart.series[0], name="Kenya"),))
        layout = layout_vertical_bar_chart(named)
        self.assertEqual([s.series_index for s in layout.key_samples], [0])
        self.assertTrue(all(s.bounds.y2 < layout.plot.y1 for s in layout.key_samples))
        self.assertEqual(layout.key_samples[0].texture, layout.bars[0].texture)
        self.assertEqual([l.text for l in layout.labels if l.role == "key"], ["Kenya"])
        self.assertEqual(rule_ids(layout), set())

    def test_single_series_without_a_name_gets_no_key(self):
        layout = layout_vertical_bar_chart(load("rainfall"))
        self.assertEqual(layout.key_samples, ())
        self.assertEqual([l for l in layout.labels if l.role == "key"], [])

    def test_crowded_chart_is_reported_not_dropped(self):
        layout = layout_vertical_bar_chart(load("crowded"))
        self.assertEqual(len(layout.bars), 36)
        self.assertTrue({"bar_width", "label_clearance", "fits_page"} <= rule_ids(layout))

    def test_negative_values_are_rejected(self):
        chart = VerticalBarChart("demo/n.png", "T", "X", "Y", ("a",), (BarSeries(None, (-1,)),))
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(chart)


class BarSpacingTest(unittest.TestCase):
    """Fix 1: bar width depends on the series count, with gaps between bars and sets."""

    def chart(self, n_categories, n_series):
        return VerticalBarChart(
            "demo/spacing.png", "Spacing", "X", "Y",
            tuple(f"c{j}" for j in range(n_categories)),
            tuple(BarSeries(f"s{s}", tuple(float(j + s + 1) for j in range(n_categories)))
                  for s in range(n_series)))

    def gaps(self, layout):
        bars = sorted(layout.bars, key=lambda b: b.x)
        return [(a.category_index == b.category_index, b.bounds.x1 - a.bounds.x2)
                for a, b in zip(bars, bars[1:])]

    def test_gaps_between_bars_and_sets_use_the_constants(self):
        layout = layout_vertical_bar_chart(self.chart(4, 3))
        for same_set, gap in self.gaps(layout):
            expected = constants.BAR_GAP if same_set else constants.GROUP_GAP
            self.assertAlmostEqual(gap, expected, places=6)
        first = min(layout.bars, key=lambda b: b.x)
        axis = next(l for l in layout.lines if l.id == "axis-y")
        self.assertAlmostEqual(first.bounds.x1 - axis.bounds.x2, constants.GROUP_GAP, places=6)

    def test_sets_are_distinguishable(self):
        """Bars in a set are joined (§6.6.6.7); sets are separated by a larger gap."""
        self.assertEqual(constants.BAR_GAP, 0.0)
        self.assertGreater(constants.GROUP_GAP, constants.BAR_GAP)
        layout = layout_vertical_bar_chart(load("books_by_grade"))
        inside = [gap for same, gap in self.gaps(layout) if same]
        between = [gap for same, gap in self.gaps(layout) if not same]
        self.assertGreater(min(between), max(inside))

    def test_width_shrinks_as_series_are_added(self):
        widths = [layout_vertical_bar_chart(self.chart(6, n)).bars[0].width for n in (1, 2, 3)]
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertLess(widths[2], widths[1])

    def test_width_fills_the_plot_when_not_capped(self):
        layout = layout_vertical_bar_chart(self.chart(6, 2))
        width = layout.bars[0].width
        self.assertLess(width, constants.BAR_MAX_WIDTH)
        right = max(b.bounds.x2 for b in layout.bars)
        axis = next(l for l in layout.lines if l.id == "axis-x")
        self.assertAlmostEqual(axis.bounds.x2, layout.usable.x2, places=6)
        self.assertAlmostEqual(axis.x2 - right, constants.GROUP_GAP, places=6)

    def test_width_formula(self):
        from src.tactile.layout import bar_width_for, set_width
        width = bar_width_for(200.0, 4, 3)
        total = 4 * set_width(width, 3) + 5 * constants.GROUP_GAP
        self.assertAlmostEqual(total, 200.0)

    def test_width_is_capped_at_the_bana_maximum(self):
        layout = layout_vertical_bar_chart(self.chart(2, 1))
        self.assertAlmostEqual(layout.bars[0].width, constants.BAR_MAX_WIDTH)

    def test_touching_bars_are_reported(self):
        layout = layout_vertical_bar_chart(self.chart(3, 2))
        first, second = sorted((b for b in layout.bars if b.category_index == 0),
                               key=lambda b: b.x)
        moved = replace(second, x=first.x + first.width)
        bars = tuple(moved if b.id == second.id else b for b in layout.bars)
        self.assertIn("bar_spacing", rule_ids(replace(layout, bars=bars)))


def one_series(names, title="Names"):
    return VerticalBarChart("demo/names.png", title, "Thing", "Count", tuple(names),
                            (BarSeries(None, tuple(float(k + 1) for k in range(len(names)))),))


def categories(layout):
    return [l.text for l in layout.labels if l.role == "category"]


class KeySymbolTest(unittest.TestCase):
    """BANA §5.8.1 alphabetic and numeric key symbols."""

    def test_abbreviation_keys_follow_bana(self):
        months = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
        keys = abbreviation_keys(months)
        self.assertEqual(len(set(keys)), len(keys))
        for key, month in zip(keys, months):
            with self.subTest(month=month):
                self.assertTrue(is_valid_abbreviation_key(key))
                self.assertEqual(key[0], month[0].lower())   # §5.8.1.1 recommendation
        self.assertEqual(keys[:3], ["jn", "fr", "ma"])

    def test_invalid_keys(self):
        for key in ("a", "aa", "ac", "cc", "ca", "abc", "ab", "abcd", "St", "st", "the", "s1"):
            with self.subTest(key=key):
                self.assertFalse(is_valid_abbreviation_key(key))
        for key in ("no", "jn", "se", "wtr"):
            with self.subTest(key=key):
                self.assertTrue(is_valid_abbreviation_key(key))

    def test_no_abbreviation_without_a_dot_3_or_6_letter(self):
        with self.assertRaises(ValueError):
            abbreviation_keys(["Abaca baba"])
        with self.assertRaises(ValueError):
            abbreviation_keys(["2019"])

    def test_numeric_keys_have_indicator_and_no_period(self):
        for key in numeric_keys(12):
            cells = to_braille(key)
            self.assertEqual(cells[0], NUMERIC_INDICATOR)
            self.assertNotIn(".", key)


class OverflowOrderTest(unittest.TestCase):
    """Fix 2: full labels -> abbreviation keys -> numeric keys -> review."""

    def test_labels_that_fit_stay_full(self):
        layout = layout_vertical_bar_chart(one_series(["Jan", "Feb", "Mar"]))
        self.assertEqual(layout.x_label_mode, "full")
        self.assertEqual(categories(layout), ["Jan", "Feb", "Mar"])
        self.assertEqual(layout.needs_review, ())

    def test_long_labels_become_abbreviation_keys(self):
        names = ["Northern Territory", "South Australia", "Western Australia"]
        layout = layout_vertical_bar_chart(one_series(names))
        self.assertEqual(layout.x_label_mode, "abbreviation")
        self.assertEqual(categories(layout), ["no", "so", "we"])
        key = [l.text for l in layout.labels if l.role == "key"]
        self.assertEqual(key, ["no Northern Territory", "so South Australia", "we Western Australia"])
        self.assertEqual(check_layout(layout), [])

    def test_key_entry_format(self):
        layout = layout_vertical_bar_chart(one_series(["Northern Territory", "South Australia",
                                                       "Western Australia"]))
        entry = next(l for l in layout.labels if l.id == "key-category-0")
        self.assertAlmostEqual(entry.x, layout.usable.x1)   # symbol in cell 1, §5.8.4.5
        self.assertEqual(entry.cells[2], frozenset())        # one blank cell after it

    def test_unabbreviatable_labels_become_numbers(self):
        layout = layout_vertical_bar_chart(one_series(["Abaca baba", "Cafe dad", "Bead cage"]))
        self.assertEqual(layout.x_label_mode, "number")
        self.assertEqual(categories(layout), ["1", "2", "3"])
        self.assertEqual(check_layout(layout), [])

    def test_nothing_fits_is_flagged_for_review(self):
        layout = layout_vertical_bar_chart(load("crowded"))
        self.assertTrue(layout.needs_review)
        self.assertIn("x_label_overflow", rule_ids(layout))

    def test_mode_can_be_forced(self):
        layout = layout_vertical_bar_chart(one_series(["Jan", "Feb"]), x_labels="number")
        self.assertEqual(categories(layout), ["1", "2"])
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(one_series(["Jan"]), x_labels="letters")

    def test_single_letter_keys_are_reported(self):
        layout = layout_vertical_bar_chart(one_series(["Northern Territory", "South Australia",
                                                       "Western Australia"]))
        labels = tuple(replace(l, text="a", cells=to_braille("a")) if l.id == "category-0" else l
                       for l in layout.labels)
        self.assertIn("key_symbols", rule_ids(replace(layout, labels=labels)))


class HeadingPlacementTest(unittest.TestCase):
    """Fix 3: headings stay in their BANA §6.6.6.6 positions, not centred."""

    def heading(self, layout, role):
        return min((l for l in layout.labels if l.role == role), key=lambda l: l.y)

    def test_demo_headings_follow_bana(self):
        for name in ("rainfall", "books_by_grade", "crowded"):
            with self.subTest(name=name):
                layout = layout_vertical_bar_chart(load(name))
                first_value = min(l.x for l in layout.labels if l.role == "category")
                self.assertAlmostEqual(self.heading(layout, "x_heading").x, first_value)
                first_cell = min(l.x for l in layout.labels if l.role == "y_value")
                self.assertAlmostEqual(self.heading(layout, "y_heading").x, first_cell)
                self.assertNotIn("heading_placement", rule_ids(layout))

    def test_centred_headings_are_reported(self):
        layout = layout_vertical_bar_chart(load("rainfall"))
        plot_centre = (layout.plot.x1 + layout.plot.x2) / 2
        labels = tuple(replace(l, x=plot_centre - l.width / 2)
                       if l.role in ("x_heading", "y_heading") else l for l in layout.labels)
        violations = [v for v in check_layout(replace(layout, labels=labels))
                      if v.rule.id == "heading_placement"]
        self.assertEqual(len(violations), 2)


class YHeadingClearanceTest(unittest.TestCase):
    """Fix 4: the vertical heading clears the top value label by Y_HEADING_CLEARANCE."""

    def gap(self, layout):
        heading = max((l for l in layout.labels if l.role == "y_heading"), key=lambda l: l.y)
        top = min((l for l in layout.labels if l.role == "y_value"), key=lambda l: l.y)
        return top.bounds.y1 - heading.bounds.y2

    def test_clearance_for_one_and_two_line_headings(self):
        for name in ("rainfall", "crowded"):   # crowded has a two-line heading
            with self.subTest(name=name):
                layout = layout_vertical_bar_chart(load(name))
                self.assertGreaterEqual(self.gap(layout), constants.Y_HEADING_CLEARANCE - 1e-9)

    def test_clearance_constant_is_applied(self):
        larger = constants.LINE_PITCH
        with patch("src.tactile.layout.Y_HEADING_CLEARANCE", larger):
            layout = layout_vertical_bar_chart(load("rainfall"), value_labels=False)
        self.assertAlmostEqual(self.gap(layout), larger)

    def test_clearance_is_at_least_the_bana_minimum(self):
        self.assertGreaterEqual(constants.Y_HEADING_CLEARANCE, constants.LABEL_MIN_GAP)


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


class ConstantsTest(unittest.TestCase):
    """Spacing constants equal the inch values BANA states."""

    def test_bana_measurements(self):
        inch = 25.4
        expected = {
            "LABEL_MIN_GAP": inch / 8, "LABEL_MAX_GAP": inch / 4,
            "BAR_MIN_WIDTH": inch * 3 / 8, "BAR_MAX_WIDTH": inch,
            "GRID_MIN_SPACING": inch * 3 / 8, "PRIMARY_LINE_MIN_LENGTH": inch / 2,
            "TICK_HALF_LENGTH": inch / 4, "KEY_SAMPLE_WIDTH": inch,
            "KEY_SAMPLE_HEIGHT": inch / 2, "PAGE_WIDTH": inch * 11.5, "PAGE_HEIGHT": inch * 11,
            # BANA "Size and Spacing of Braille Characters" (Spec 800 §3.2).
            "DOT_DIAMETER": inch * 0.057, "DOT_PITCH": inch * 0.092,
            "CELL_PITCH": inch * 0.245, "LINE_PITCH": inch * 0.400,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(getattr(constants, name), value)
        self.assertEqual((constants.CELLS_PER_LINE, constants.LINES_PER_PAGE), (40, 25))

    def test_labels_are_placed_at_the_bana_minimum(self):
        self.assertEqual(constants.LABEL_GAP, constants.LABEL_MIN_GAP)

    def test_y_axis_ends_at_the_top_grid_line(self):
        layout = layout_vertical_bar_chart(load("rainfall"))
        axis = next(l for l in layout.lines if l.id == "axis-y")
        top_grid = min(l.y1 for l in layout.lines if l.kind == "grid")
        self.assertAlmostEqual(axis.y1, top_grid)


class SectionTest(unittest.TestCase):
    """Item 4: graphs too dense for one page are divided into sections (§3.9),
    with the series key on a key page before the graph pages (§6.6.6.4)."""

    @classmethod
    def setUpClass(cls):
        cls.pages = layout_vertical_bar_chart_pages(load("crowded"))
        cls.key = cls.pages[0]
        cls.graphs = [p for p in cls.pages if p.page_kind == "graph"]

    def labels(self, page, role):
        return sorted((l for l in page.labels if l.role == role), key=lambda l: (l.y, l.x))

    def test_charts_that_fit_stay_on_one_page(self):
        for name in ("rainfall", "books_by_grade"):
            with self.subTest(name=name):
                pages = layout_vertical_bar_chart_pages(load(name))
                self.assertEqual(len(pages), 1)
                self.assertIsNone(pages[0].section)

    def test_key_page_comes_first(self):
        self.assertEqual(self.key.page_kind, "key")
        self.assertEqual([s.series_index for s in self.key.key_samples], [0, 1, 2])
        self.assertTrue(self.labels(self.key, "title"))
        self.assertEqual(check_layout(self.key), [])
        notes = " ".join(l.text for l in self.labels(self.key, "note"))
        self.assertIn(f"divided by category into {len(self.graphs)} parts", notes)

    def test_graph_pages_have_no_series_key_or_title(self):
        for page in self.graphs:
            self.assertEqual(page.key_samples, ())
            self.assertTrue(page.series_key_elsewhere)
            self.assertEqual(self.labels(page, "title"), [])   # §5.3.3

    def test_crowded_chart_is_divided_and_every_bar_kept(self):
        self.assertGreater(len(self.graphs), 1)
        self.assertEqual([p.section for p in self.graphs],
                         [(k, len(self.graphs)) for k in range(1, len(self.graphs) + 1)])
        self.assertEqual(sum(len(p.bars) for p in self.graphs), 36)
        self.assertEqual(sum(p.category_count for p in self.graphs), 12)

    def test_only_the_missing_overview_is_reported(self):
        self.assertEqual({v.rule.id for v in check_layout(self.graphs[0])}, {"overview_first"})
        for page in self.graphs[1:]:
            self.assertEqual(check_layout(page), [])

    def test_plots_keep_their_height(self):
        """With the key moved off, each plot has several grid intervals, not one."""
        for page in self.graphs:
            self.assertGreaterEqual(len([l for l in page.lines if l.kind == "grid"]), 4)

    def test_every_section_has_the_same_scale(self):
        def scale(page):
            values = self.labels(page, "y_value")
            return page.plot.y1, page.plot.y2, [(l.text, round(l.y, 6)) for l in values]
        first = scale(self.graphs[0])
        for page in self.graphs[1:]:
            self.assertEqual(scale(page), first)

    def test_section_titles_on_every_graph_page(self):
        for k, page in enumerate(self.graphs, 1):
            notes = " ".join(l.text for l in self.labels(page, "note"))
            self.assertIn(f"Part {k} of {len(self.graphs)}:", notes)

    def test_notes_use_transcriber_note_indicators_and_cells(self):
        page = self.key
        note = self.labels(page, "note")
        self.assertEqual(note[0].cells[:3], TN_OPEN)
        self.assertEqual(note[-1].cells[-3:], TN_CLOSE)
        cell = constants.CELL_PITCH
        self.assertAlmostEqual(note[0].x, page.usable.x1 + 6 * cell)   # cell 7, §5.6.2
        self.assertGreater(len(note), 1)
        for runover in note[1:]:
            self.assertAlmostEqual(runover.x, page.usable.x1 + 4 * cell)   # cell 5

    def test_keys_are_consistent_across_sections(self):
        symbols = [l.text for p in self.graphs for l in sorted(
            (l for l in p.labels if l.role == "category"), key=lambda l: l.x)]
        months = load("crowded").categories
        mode = self.graphs[0].x_label_mode
        expected = (list(months) if mode == "full" else
                    abbreviation_keys(months) if mode == "abbreviation" else numeric_keys(12))
        self.assertEqual(symbols, expected)

    def test_single_series_split_has_no_key_page(self):
        names = tuple(f"Item {k}" for k in range(1, 25))
        chart = VerticalBarChart("demo/many.png", "Many Items", "Item", "Count", names,
                                 (BarSeries(None, tuple(float(k) for k in range(1, 25))),))
        pages = layout_vertical_bar_chart_pages(chart)
        self.assertGreater(len(pages), 1)
        self.assertTrue(all(p.page_kind == "graph" for p in pages))
        self.assertTrue(self.labels(pages[0], "title"))
        self.assertIn("divided by category", " ".join(l.text for l in self.labels(pages[0], "note")))

    def test_one_page_is_kept_when_no_split_works(self):
        # Make every layout break a rule that splitting cannot fix.
        broken = [Violation(RULES["bar_width"], "forced", ())]
        with patch("src.tactile.rules.check_layout", return_value=broken):
            pages = layout_vertical_bar_chart_pages(load("books_by_grade"))
        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0].section)


def all_label_texts() -> list[str]:
    """Every label text the layout produces for the demo and overflow charts."""
    layouts = [layout_vertical_bar_chart(load(name))
               for name in ("rainfall", "books_by_grade", "crowded")]
    layouts.append(layout_vertical_bar_chart(one_series(
        ["Northern Territory", "South Australia", "Western Australia"])))
    layouts.append(layout_vertical_bar_chart(one_series(["Abaca baba", "Cafe dad", "Bead cage"])))
    return sorted({label.text for layout in layouts for label in layout.labels})


class LiblouisRoundTripTest(unittest.TestCase):
    """Fix 5: back-translating each label's braille with liblouis gives the label.

    Uses the UEB grade 1 table (en-ueb-g1.ctb) with Unicode braille input.
    Skipped when the liblouis Python bindings (``louis``) are not installed.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import louis
        except ImportError:
            raise unittest.SkipTest("liblouis Python bindings (louis) are not installed")
        cls.louis = louis

    def test_every_label_round_trips(self):
        texts = all_label_texts()
        self.assertGreater(len(texts), 50)
        for text in texts:
            with self.subTest(text=text):
                braille = to_unicode(to_braille(text)).replace("⠀", " ")
                back = self.louis.backTranslateString(["unicode.dis", "en-ueb-g1.ctb"], braille)
                self.assertEqual(" ".join(back.split()), text)


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
                                   str(crowded), "--one-page", "--show-violations",
                                   "--strict"]), 1)
            self.assertIn('id="violations"', crowded.read_text(encoding="utf-8"))
            parts = Path(tmp) / "parts.svg"
            main(["--chart", str(FIXTURES / "crowded.json"), "--output", str(parts)])
            written = sorted(p.name for p in Path(tmp).glob("parts-*.svg"))
            self.assertEqual(written, ["parts-1.svg", "parts-2.svg", "parts-3.svg"])  # key page + 2 parts


class NoVisibleBarsTest(unittest.TestCase):
    """Legend entries with no visible bars are keyed, noted, and draw no bars."""

    def chart(self, n_categories=4):
        categories = tuple(str(2000 + j) for j in range(n_categories))
        return VerticalBarChart(
            "demo/legend.png", "Net energy imports", "Years", "Imports",
            categories,
            (BarSeries("Low & middle income", (), has_bars=False),
             BarSeries("Kenya", tuple(13.0 + j for j in range(n_categories))),
             BarSeries("Vietnam", (), has_bars=False)))

    def note_text(self, layout):
        return " ".join(label.text for label in layout.labels if label.role == "note")

    def test_every_legend_entry_has_a_key_sample_and_braille_name_in_order(self):
        layout = layout_vertical_bar_chart(self.chart())
        samples = sorted(layout.key_samples, key=lambda s: s.y)
        self.assertEqual([s.series_index for s in samples], [0, 1, 2])
        series_keys = [l for l in layout.labels if l.role == "key" and str(l.owner_id).startswith("key-sample")]
        self.assertEqual([l.text for l in series_keys], ["Low & middle income", "Kenya", "Vietnam"])
        self.assertEqual(len(set(layout.series_textures)), 3)
        braille = [to_unicode(label.cells) for label in layout.labels if label.id == "key-0"]
        self.assertEqual(braille, [to_unicode(to_braille("Low & middle income"))])

    def test_only_series_with_bars_draw_bars_and_take_bar_slots(self):
        layout = layout_vertical_bar_chart(self.chart())
        self.assertEqual({bar.series_index for bar in layout.bars}, {1})
        self.assertEqual(len(layout.bars), 4)
        solo = VerticalBarChart("demo/solo.png", "Net energy imports", "Years", "Imports",
                                self.chart().categories, (self.chart().series[1],))
        self.assertAlmostEqual(layout.bars[0].width, layout_vertical_bar_chart(solo).bars[0].width)
        self.assertEqual(layout.bars[1].texture, layout.series_textures[1])

    def test_the_transcribers_note_names_the_entries_without_saying_zero(self):
        layout = layout_vertical_bar_chart(self.chart())
        note = self.note_text(layout)
        self.assertEqual(note, "No bars are visible for: Low & middle income, Vietnam.")
        self.assertNotIn("zero", note.lower())
        note_labels = [l for l in layout.labels if l.role == "note"]
        self.assertEqual(to_unicode(note_labels[0].cells)[:1], to_unicode(TN_OPEN)[:1])
        first_key_y = min(l.y for l in layout.labels if str(l.owner_id).startswith("key-sample"))
        self.assertTrue(all(l.y < first_key_y for l in note_labels))      # notes come before the key

    def test_the_layout_passes_every_rule(self):
        violations = [v for v in check_layout(layout_vertical_bar_chart(self.chart()))
                      if v.rule.id != "overview_first"]
        self.assertEqual(violations, [])

    def test_a_chart_with_bars_for_every_series_is_laid_out_as_before(self):
        plain = VerticalBarChart("demo/p.png", "T", "X", "Y", ("a", "b", "c"),
                                 (BarSeries("A", (1.0, 2.0, 3.0)), BarSeries("B", (3.0, 2.0, 1.0))))
        layout = layout_vertical_bar_chart(plain)
        self.assertEqual(len(layout.bars), 6)
        self.assertNotIn("note", {l.role for l in layout.labels})
        self.assertEqual(layout, layout_vertical_bar_chart(replace(
            plain, series=tuple(replace(s, has_bars=True) for s in plain.series))))

    def test_a_single_unnamed_series_still_gets_no_key_and_no_note(self):
        unnamed = VerticalBarChart("demo/u.png", "T", "X", "Y", ("a", "b"), (BarSeries(None, (1.0, 2.0)),))
        layout = layout_vertical_bar_chart(unnamed)
        self.assertEqual(layout.key_samples, ())
        self.assertNotIn("note", {l.role for l in layout.labels})
        self.assertFalse([l for l in layout.labels if str(l.owner_id).startswith("key-sample")])

    def test_a_divided_graph_puts_the_note_and_every_key_on_the_key_page(self):
        pages = layout_vertical_bar_chart_pages(self.chart(n_categories=30))
        self.assertGreater(len(pages), 2)
        key_page = pages[0]
        self.assertEqual(key_page.page_kind, "key")
        self.assertIn("No bars are visible for: Low & middle income, Vietnam.", self.note_text(key_page))
        self.assertEqual(len(key_page.key_samples), 3)
        for page in pages[1:]:
            self.assertNotIn("No bars are visible", self.note_text(page))
            self.assertEqual(page.key_samples, ())
            self.assertEqual({bar.series_index for bar in page.bars}, {1})
        for page in pages:
            self.assertEqual([v for v in check_layout(page) if v.rule.id != "overview_first"], [])

    def test_more_than_five_legend_entries_are_still_refused(self):
        series = tuple(BarSeries(f"s{k}", (), has_bars=False) for k in range(5)) + (
            BarSeries("with bars", (1.0, 2.0)),)
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(VerticalBarChart("demo/m.png", "T", "X", "Y", ("a", "b"), series))


# Tactile hbar

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


def horizontal_rule_ids(layout):
    return [v.rule.id for v in check_layout(layout)]


def by_role(layout, role):
    return [l for l in layout.labels if l.role == role]


class HorizontalLayoutTest(unittest.TestCase):
    def test_grid_spacing_matches_tick_increments(self):
        for chart in (RAIN, BOOKS, LONG):
            with self.subTest(chart=chart.title):
                pages = layout_horizontal_bar_chart_pages(chart)
                # Horizontal pagination is not supported yet.
                self.assertEqual(len(pages), 1)
                page = pages[0]
                ticks = [float(label.text) for label in sorted(
                    by_role(page, "x_value"), key=lambda label: label.x)]
                coordinates = sorted({page.plot.x1, *(line.x1 for line in page.lines
                                                     if line.kind == "grid")})
                self.assertGreaterEqual(len(ticks), 2)
                self.assertEqual(ticks[0], 0)
                self.assertEqual(len(coordinates), len(ticks))
                mm_per_unit = (page.plot.x2 - page.plot.x1) / (ticks[-1] - ticks[0])
                for index in range(1, len(ticks)):
                    increment = ticks[index] - ticks[index - 1]
                    self.assertGreater(increment, 0)
                    spacing = coordinates[index] - coordinates[index - 1]
                    self.assertAlmostEqual(spacing / increment, mm_per_unit, delta=1e-6)

    def test_sample_charts_pass_every_rule(self):
        for chart in (RAIN, BOOKS, LONG):
            with self.subTest(chart.title):
                layout = layout_horizontal_bar_chart(chart)
                self.assertEqual(layout.orientation, "horizontal")
                self.assertEqual(layout.x_label_mode, "full")
                self.assertEqual(horizontal_rule_ids(layout), [])
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
        self.assertEqual(horizontal_rule_ids(layout), [])
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
        self.assertIn("guide_dots", horizontal_rule_ids(broken))

    def test_value_off_its_grid_line_is_reported(self):
        layout = layout_horizontal_bar_chart(RAIN)
        label = layout.by_id("x-value-1")
        moved = replace(label, x=label.x + 2)
        broken = replace(layout, labels=tuple(moved if l is label else l for l in layout.labels))
        self.assertIn("value_alignment", horizontal_rule_ids(broken))

    def test_thin_horizontal_bars_are_reported_by_their_thickness(self):
        layout = layout_horizontal_bar_chart(RAIN)
        thin = tuple(replace(b, height=5.0) for b in layout.bars)
        self.assertIn("bar_width", horizontal_rule_ids(replace(layout, bars=thin)))
        # Long bars are fine: length is not limited for horizontal bars.
        self.assertNotIn("bar_width", horizontal_rule_ids(layout))


# Tactile generation

def make_spec(**changes):
    fields = dict(
        source_key="input/chart.png",
        title="Sales",
        x_axis_title="Year",
        y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]),
        categories=["2001", "2002", "2003"],
        series=[SeriesSpec(None, "#112233", [10.0, 20.0, 30.0])],
        bars_adjacent=False,
        flags=[],
        confidence={},
    )
    fields.update(changes)
    return VerticalBarSpec(**fields)


def divided_spec():
    categories = [f"Category number {index}" for index in range(12)]
    return make_spec(
        categories=categories,
        series=[
            SeriesSpec("Alpha", "#112233", [float(i + 1) for i in range(12)]),
            SeriesSpec("Beta", "#445566", [float(i + 2) for i in range(12)]),
            SeriesSpec("Gamma", "#778899", [float(i + 3) for i in range(12)]),
        ],
    )


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_single_page_artifacts_and_zip(self):
        result = generate_tactile_artifacts(make_spec(), self.root)

        self.assertEqual(len(result.pages), 1)
        page = result.pages[0]
        self.assertEqual(
            (page.page_number, page.page_count, page.page_kind, page.section, page.caption),
            (1, 1, "graph", None, "Tactile graphic"),
        )
        self.assertEqual(page.review_svg.name, "chart-review.svg")
        self.assertEqual(page.clean_svg.name, "chart.svg")
        self.assertIn("PRINT LABELS FOR REVIEW ONLY", page.review_svg.read_text())
        self.assertNotIn('id="print"', page.clean_svg.read_text())
        self.assertNotIn('id="violations"', page.clean_svg.read_text())
        with ZipFile(result.zip_path) as archive:
            self.assertEqual(archive.namelist(), ["chart.svg"])
            self.assertEqual(archive.read("chart.svg"), page.clean_svg.read_bytes())

    def test_horizontal_spec_uses_the_horizontal_layout(self):
        result = generate_tactile_artifacts(make_spec(orientation="horizontal"), self.root)

        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].violations, ())
        svg = result.pages[0].clean_svg.read_text()
        # Horizontal values sit under the axis as x values, not y values.
        self.assertIn('id="x-value-0"', svg)
        self.assertNotIn('id="y-value-0"', svg)

    def test_multi_page_metadata_order_and_overview_finding(self):
        result = generate_tactile_artifacts(divided_spec(), self.root)

        self.assertGreater(len(result.pages), 2)
        self.assertEqual(result.pages[0].page_kind, "key")
        self.assertEqual(result.pages[0].caption, f"Page 1 of {len(result.pages)} — key")
        graph_pages = [page for page in result.pages if page.page_kind == "graph"]
        self.assertEqual(
            [page.section for page in graph_pages],
            [(i, len(graph_pages)) for i in range(1, len(graph_pages) + 1)],
        )
        self.assertIn("overview_first", {v.rule.id for v in graph_pages[0].violations})
        self.assertEqual(
            graph_pages[0].caption,
            f"Page 2 of {len(result.pages)} — part 1 of {len(graph_pages)}",
        )
        with ZipFile(result.zip_path) as archive:
            self.assertEqual(archive.namelist(), [p.clean_svg.name for p in result.pages])
            self.assertFalse(any("review" in name for name in archive.namelist()))

    def test_review_overlay_only_boxes_addressable_violations(self):
        addressable = Violation(RULES["bar_width"], "too narrow", ("bar-0-0",))
        with mock.patch("src.tactile.generation.check_layout", return_value=[addressable]):
            result = generate_tactile_artifacts(make_spec(), self.root)
        review = result.pages[0].review_svg.read_text()
        self.assertIn('id="violations"', review)
        self.assertIn("too narrow", review)
        self.assertNotIn('id="violations"', result.pages[0].clean_svg.read_text())

        report_only = Violation(RULES["overview_first"], "overview needed", ())
        with mock.patch("src.tactile.generation.check_layout", return_value=[report_only]):
            result = generate_tactile_artifacts(make_spec(), self.root)
        svg = ElementTree.fromstring(result.pages[0].review_svg.read_text().split("\n", 1)[1])
        ns = {"svg": "http://www.w3.org/2000/svg"}
        group = svg.find(".//svg:g[@id='violations']", ns)
        self.assertIsNotNone(group)
        self.assertEqual(list(group), [])

    def test_global_notes_are_separate_from_page_review_information(self):
        spec = make_spec(
            flags=["y_ticks_not_arithmetic"],
            series=[SeriesSpec(None, "#112233", [-0.1, 20.0, 30.0])],
        )
        result = generate_tactile_artifacts(spec, self.root)
        joined = " ".join(result.notes)
        self.assertIn("not evenly spaced", joined)
        self.assertIn("Values below the axis minimum were set to 0", joined)
        self.assertIsInstance(result.pages[0].notes, tuple)
        self.assertIsInstance(result.pages[0].needs_review, tuple)

    def test_non_ready_and_invalid_root_are_validation_errors(self):
        before = set(self.root.iterdir())
        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(make_spec(title=None), self.root)
        self.assertEqual(raised.exception.stage, "validation")
        self.assertEqual(set(self.root.iterdir()), before)

        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(make_spec(), self.root / "missing")
        self.assertEqual(raised.exception.stage, "validation")

    def test_real_layout_failure_removes_job_directory(self):
        spec = make_spec(y_axis=YAxisSpec(-50, 50, [-50, 0, 50]),
                         series=[SeriesSpec(None, "#112233", [-10.0, 20.0, 30.0])])
        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(spec, self.root)
        self.assertEqual(raised.exception.stage, "layout")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_render_write_and_zip_failures_remove_partial_job(self):
        cases = (
            ("render", mock.patch("src.tactile.generation.render_svg", side_effect=ValueError("x"))),
            ("write", mock.patch("pathlib.Path.write_text", side_effect=OSError("x"))),
            ("zip", mock.patch("src.tactile.generation.ZipFile", side_effect=OSError("x"))),
        )
        for expected_stage, patcher in cases:
            with self.subTest(expected_stage), patcher:
                with self.assertRaises(GenerationError) as raised:
                    generate_tactile_artifacts(make_spec(), self.root)
                self.assertEqual(raised.exception.stage, expected_stage)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_later_page_write_failure_and_cleanup_failure_keep_primary_error(self):
        with mock.patch(
            "pathlib.Path.write_text",
            side_effect=[None, None, OSError("second page failed")],
        ):
            with self.assertRaises(GenerationError) as raised:
                generate_tactile_artifacts(divided_spec(), self.root)
        self.assertEqual(raised.exception.stage, "write")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(list(self.root.iterdir()), [])

        with (
            mock.patch("src.tactile.generation.render_svg", side_effect=ValueError("render")),
            mock.patch(
                "src.tactile.generation._remove_job_dir",
                side_effect=OSError("cleanup"),
            ),
        ):
            with self.assertRaises(GenerationError) as raised:
                generate_tactile_artifacts(make_spec(), self.root)
        self.assertEqual(raised.exception.stage, "render")
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIsInstance(raised.exception.cleanup_error, OSError)

    def test_names_are_sanitized_limited_and_isolated(self):
        names = {
            "input/a name.png": "a_name.svg",
            "input/ééé.png": "chart.svg",
            "input/---.png": "chart.svg",
            f"input/{'a' * 300}.png": f"{'a' * 120}.svg",
        }
        for source_key, expected in names.items():
            with self.subTest(source_key):
                result = generate_tactile_artifacts(make_spec(source_key=source_key), self.root)
                self.assertEqual(result.pages[0].clean_svg.name, expected)

        first = generate_tactile_artifacts(make_spec(), self.root)
        second = generate_tactile_artifacts(make_spec(), self.root)
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertTrue(first.pages[0].clean_svg.exists())
        self.assertTrue(second.pages[0].clean_svg.exists())

    def test_cleanup_is_idempotent_and_preserves_siblings(self):
        first = generate_tactile_artifacts(make_spec(), self.root)
        second = generate_tactile_artifacts(make_spec(), self.root)
        first.pages[0].review_svg.unlink()
        cleanup_generation(first, self.root)
        cleanup_generation(first, self.root)
        self.assertFalse(first.output_dir.exists())
        self.assertTrue(second.output_dir.exists())

        invalid = replace(second, output_dir=self.root / "not-a-generation")
        with self.assertRaises(ValueError):
            cleanup_generation(invalid, self.root)
        self.assertTrue(second.output_dir.exists())


# Tactile pipeline

def pipeline_spec(**changes):
    fields = dict(
        source_key="input/a.png", title="Sales", x_axis_title="Year", y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]), categories=["2001", "2002", "2003"],
        series=[SeriesSpec(None, "#112233", [10.0, 20.0, 30.0])],
        bars_adjacent=False, flags=[], confidence={})
    fields.update(changes)
    return VerticalBarSpec(**fields)


def detection_frame(vertical=(), horizontal=(), other=()):
    """A real detection frame for the shared orientation resolver."""
    items = [("vertical_bar", c) for c in vertical]
    items += [("horizontal_bar", c) for c in horizontal]
    items += list(other)
    return DetectionFrame("test/chart.png", 200, 100, tuple(
        Detection(f"det-{i}", role, Box(10, 10, 20, 20), confidence)
        for i, (role, confidence) in enumerate(items)))


class Pipeline(unittest.TestCase):
    def go(self, spec=None, error=None, argv=(), tty=False, answers=()):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(out_path), *argv])
            replies = iter(answers)
            out, err = io.StringIO(), io.StringIO()
            code = pipeline.run(
                args, analyse_fn=lambda *_, **__: (spec, error, None),
                input_fn=lambda prompt: next(replies), stdin_is_tty=tty, out=out, err=err)
            return code, out.getvalue(), err.getvalue(), out_path.exists()

    def test_ready_chart_writes_svg(self):
        code, out, err, written = self.go(pipeline_spec(), argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("0: 2001", out)

    def test_analyse_passes_legend_mapping_to_reconstruction(self):
        image = mock.sentinel.image
        raw = detection_frame(vertical=(0.9, 0.9, 0.9), horizontal=(0.3,))
        ocr = mock.sentinel.ocr
        mapping = mock.sentinel.mapping
        reconstructed = pipeline_spec()
        backend = mock.sentinel.backend

        with (
            mock.patch("cv2.imread", return_value=image),
            mock.patch("src.detection.load_model", return_value=mock.sentinel.model),
            mock.patch("src.detection.run_detection", return_value=raw),
            mock.patch("src.ocr.run_ocr", return_value=ocr) as ocr_call,
            mock.patch("src.ocr.tesseract.TesseractBackend", return_value=backend),
            mock.patch(
                "src.reconstruction.legend_matcher.filter_bars_by_legend",
                side_effect=lambda image, frame, ocr: LegendFilterResult(frame, mapping)
            ) as matcher,
            mock.patch(
                "src.reconstruction.vertical.reconstruct_vbar", return_value=reconstructed
            ) as reconstruct,
            mock.patch.object(pipeline, "cut_off_flags", return_value=[]) as cut_off,
        ):
            result = pipeline.analyse(Path("chart.png"), Path("weights.pt"))

        effective = matcher.call_args.args[1]
        self.assertEqual([d.role for d in effective.detections], ["vertical_bar"] * 3)
        ocr_call.assert_called_once_with(image, effective, backend, "eng")
        matcher.assert_called_once_with(image, effective, ocr)
        reconstruct.assert_called_once_with(
            effective, ocr, image, legend_mapping=mapping
        )
        cut_off.assert_called_once_with(effective, image)
        self.assertEqual(result, (reconstructed, None, effective))
        self.assertEqual(len(raw.detections), 4)  # raw detections untouched

    def test_unconfirmed_non_tty_warns_and_continues(self):
        code, _, err, written = self.go(pipeline_spec())
        self.assertEqual((code, written), (0, True))
        self.assertIn("not confirmed", err)

    def test_tty_decline_writes_nothing(self):
        code, _, _, written = self.go(pipeline_spec(), tty=True, answers=["n"])
        self.assertEqual((code, written), (4, False))

    def test_missing_text_non_tty_exits_2_with_flags(self):
        code, out, _, written = self.go(
            pipeline_spec(title=None, flags=["title_missing"]), argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)

    def test_flags_supply_missing_text(self):
        code, out, _, written = self.go(
            pipeline_spec(title=None, flags=["title_missing"]), argv=["--yes", "--title", "T"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("Supplied by user: chart title supplied by user", out)

    def test_tty_prompts_and_skip_stops(self):
        spec = pipeline_spec(title=None, x_axis_title=None,
                         flags=["title_missing", "x_axis_title_missing"])
        code, out, _, written = self.go(spec, tty=True, answers=["My title", ""])
        self.assertEqual((code, written), (2, False))
        self.assertIn("Still missing: x_axis_title", out)

    def test_tty_prompt_answers_build(self):
        code, out, _, written = self.go(
            pipeline_spec(title=None, flags=["title_missing"]), tty=True, answers=["My title", "y"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("Supplied by user: chart title", out)

    def test_cannot_build_and_force(self):
        spec = pipeline_spec(flags=["category_ambiguous"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (3, False))
        self.assertIn("--force", out)
        code, _, err, written = self.go(spec, argv=["--yes", "--force"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("WARNING: --force", err)

    def test_force_never_skips_non_ambiguity_flags(self):
        code, _, _, written = self.go(pipeline_spec(flags=["baseline_outside_plot"]),
                                      argv=["--yes", "--force"])
        self.assertEqual((code, written), (3, False))

    def test_interpretation_error_exits_3(self):
        code, out, _, _ = self.go(error=ChartInterpretationError("no x tick labels detected"))
        self.assertEqual(code, 3)
        self.assertIn("No x tick labels", out)

    def test_unencodable_name_exits_2_and_flag_fixes_it(self):
        spec = pipeline_spec(series=[SeriesSpec("Chile", "#112233", [1.0, 2.0, 3.0]),
                                 SeriesSpec("E] Salvador", "#445566", [3.0, 2.0, 1.0])])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--series 1=NAME", out)
        code, _, _, written = self.go(spec, argv=["--yes", "--series", "1=El Salvador"])
        self.assertEqual((code, written), (0, True))

    def test_tty_answer_that_cannot_be_encoded_stops(self):
        code, out, _, written = self.go(
            pipeline_spec(title=None, flags=["title_missing"]), tty=True, answers=["Bad]"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)

    def many(self):
        n = 30
        return pipeline_spec(categories=[f"Category number {i}" for i in range(n)],
                         series=[SeriesSpec(None, "#112233", [float(i + 1) for i in range(n)])])

    def test_overview_page_is_reported_for_a_transcriber_and_never_fails_strict(self):
        code, out, err, _ = self.go(self.many(), argv=["--yes", "--strict"])
        self.assertEqual(code, 0)  # divided into several files, so no single out.svg
        self.assertEqual(out.count("Needs a transcriber: overview page (BANA 3.9.1) not generated"), 1)
        self.assertNotIn("overview_first", out)

    def test_strict_exits_1_on_a_real_violation_default_exits_0(self):
        found = [Violation(RULES["bar_width"], "too narrow", ("bar-0-0",))]
        with mock.patch.object(pipeline, "check_layout", return_value=found):
            code, _, err, written = self.go(pipeline_spec(), argv=["--yes", "--strict"])
            self.assertEqual((code, written), (1, True))
            self.assertIn("--strict: 1 BANA violation", err)
            code, out, _, _ = self.go(pipeline_spec(), argv=["--yes"])
            self.assertEqual(code, 0)
            self.assertIn("too narrow", out)

    def test_negative_values_are_zeroed_on_an_axis_from_zero_but_rejected_on_a_negative_axis(self):
        zero_axis = pipeline_spec(series=[SeriesSpec(None, "#112233", [-0.6, 20.0, 30.0])])
        code, out, _, written = self.go(zero_axis, argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("Values below the axis minimum were set to 0", out)
        negative_axis = pipeline_spec(y_axis=YAxisSpec(-50, 50, [-50, 0, 50]),
                                  series=[SeriesSpec(None, "#112233", [-10.0, 20.0, 30.0])])
        code, _, err, written = self.go(negative_axis, argv=["--yes"])
        self.assertEqual((code, written), (1, False))
        self.assertIn("negative bar values are not supported", err)

    def test_clipped_bar_builds_with_a_note_unless_the_axis_is_negative(self):
        code, out, _, written = self.go(pipeline_spec(flags=["value_clipped_at_axis"]), argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("lower bound", out)
        spec = pipeline_spec(y_axis=YAxisSpec(-50, 50, [-50, 0, 50]), flags=["value_clipped_at_axis"])
        for extra in ([], ["--force"]):
            code, _, _, written = self.go(spec, argv=["--yes", *extra])
            self.assertEqual((code, written), (3, False))

    def test_lost_minus_signs_keep_a_clipped_chart_blocked_even_with_force(self):
        spec = pipeline_spec(flags=["value_clipped_at_axis", "y_tick_negative_read", "y_ticks_dropped:2"])
        for extra in ([], ["--force"]):
            code, out, _, written = self.go(spec, argv=["--yes", *extra])
            self.assertEqual((code, written), (3, False))
            self.assertIn("runs past the end of the y axis", out)

    def test_bars_mostly_missed_blocks_until_forced(self):
        spec = pipeline_spec(flags=["bars_mostly_missed:67"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (3, False))
        self.assertIn("67% of bars were not detected and would be drawn as 0", out)
        self.assertIn("Re-run with --force to build anyway.", out)
        code, out, err, written = self.go(spec, argv=["--yes", "--force"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("WARNING: --force", err)
        self.assertIn("bars_mostly_missed:67", out)

    def test_suspicious_category_asks_and_flag_fixes_it(self):
        spec = pipeline_spec(categories=["Germany", "i", "Greece"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--category 1=TEXT", out)
        code, out, _, written = self.go(spec, argv=["--yes", "--category", "1=Hungary"])
        self.assertEqual((code, written), (0, True))

    def test_cut_off_title_asks_and_confirming_it_builds(self):
        spec = pipeline_spec(flags=["title_cut_off"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)
        self.assertIn("cut off at the image edge", out)
        code, out, _, written = self.go(spec, argv=["--yes", "--title", "Sales"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("chart title supplied by user", out)

    def test_tty_confirming_a_cut_off_y_title(self):
        spec = pipeline_spec(flags=["y_axis_title_cut_off"])
        code, _, _, written = self.go(spec, tty=True, answers=["Revenue", "y"])
        self.assertEqual((code, written), (0, True))

    def test_uneven_y_ticks_build_with_a_note(self):
        code, out, _, written = self.go(pipeline_spec(flags=["y_ticks_not_arithmetic"]), argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("not evenly spaced", out)

    def test_leading_zero_tick_exits_3_even_with_force(self):
        spec = pipeline_spec(flags=["y_tick_leading_zero"])
        for extra in ([], ["--force"]):
            code, out, _, written = self.go(spec, argv=["--yes", *extra])
            self.assertEqual((code, written), (3, False))
            self.assertIn("decimal point may have been lost", out)

    def test_single_series_key_follows_the_legend(self):
        def svg(series):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "out.svg"
                args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
                code = pipeline.run(args, analyse_fn=lambda *_, **__: (pipeline_spec(series=series), None, None),
                                    input_fn=input, stdin_is_tty=False,
                                    out=io.StringIO(), err=io.StringIO())
                self.assertEqual(code, 0)
                return path.read_text()

        values = [10.0, 20.0, 30.0]
        named = svg([SeriesSpec("Kenya", "#112233", values)])
        unnamed = svg([SeriesSpec(None, "#112233", values)])
        two = svg([SeriesSpec("A", "#112233", values), SeriesSpec("B", "#445566", values)])
        self.assertIn("key-sample-0", named)
        self.assertNotIn("key-sample", unnamed)
        self.assertIn("key-sample-0", two)
        self.assertIn("key-sample-1", two)

    def test_analyse_sends_horizontal_charts_to_the_horizontal_reconstruction(self):
        image = mock.sentinel.image
        raw = detection_frame(horizontal=(0.9, 0.9, 0.9), vertical=(0.3,))
        ocr = mock.sentinel.ocr
        reconstructed = pipeline_spec(orientation="horizontal")
        with (
            mock.patch("cv2.imread", return_value=image),
            mock.patch("src.detection.load_model", return_value=mock.sentinel.model),
            mock.patch("src.detection.run_detection", return_value=raw),
            mock.patch("src.ocr.run_ocr", return_value=ocr) as ocr_call,
            mock.patch("src.ocr.tesseract.TesseractBackend", return_value=mock.sentinel.backend),
            mock.patch("src.reconstruction.legend_matcher.filter_bars_by_legend") as vertical_matcher,
            mock.patch("src.reconstruction.horizontal.filter_hbars_by_legend",
                       side_effect=passthrough_filter) as horizontal_filter,
            mock.patch("src.reconstruction.horizontal.reconstruct_hbar",
                       return_value=reconstructed) as reconstruct,
            mock.patch.object(pipeline, "cut_off_flags", return_value=[]) as cut_off,
        ):
            result = pipeline.analyse(Path("chart.png"), Path("weights.pt"))
        effective = reconstruct.call_args.args[0]
        self.assertEqual([d.role for d in effective.detections], ["horizontal_bar"] * 3)
        ocr_call.assert_called_once_with(image, effective, mock.sentinel.backend, "eng")
        reconstruct.assert_called_once_with(effective, ocr, image,
                                            legend_mapping=passthrough_filter(image, effective, ocr).mapping)
        horizontal_filter.assert_called_once_with(image, effective, ocr)
        vertical_matcher.assert_not_called()
        cut_off.assert_called_once_with(effective, image)
        self.assertEqual(result, (reconstructed, None, effective))
        self.assertEqual(len(raw.detections), 4)  # raw detections untouched

    def test_horizontal_spec_gets_the_horizontal_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
            out = io.StringIO()
            code = pipeline.run(
                args, analyse_fn=lambda *_, **__: (pipeline_spec(orientation="horizontal"), None, None),
                input_fn=input, stdin_is_tty=False, out=out, err=io.StringIO())
            svg = path.read_text()
        self.assertEqual(code, 0)
        self.assertIn("x-value-0", svg)
        self.assertNotIn("y-value-0", svg)
        self.assertIn("all passed", out.getvalue())

    def test_ocr_engine_defaults_to_tesseract(self):
        args = pipeline.parse_args(["--image", "a.png", "--output", "o.svg"])
        self.assertEqual(args.ocr, "tesseract")
        args = pipeline.parse_args(["--image", "a.png", "--output", "o.svg", "--ocr", "rapidocr"])
        self.assertEqual(args.ocr, "rapidocr")

    def test_bar_orientation_flag_defaults_and_accepts_manual_choices(self):
        args = pipeline.parse_args(["--image", "a.png", "--output", "o.svg"])
        self.assertEqual(args.bar_orientation, "auto")
        for choice in ("vertical", "horizontal"):
            args = pipeline.parse_args(
                ["--image", "a.png", "--output", "o.svg", "--bar-orientation", choice])
            self.assertEqual(args.bar_orientation, choice)

    def test_bar_orientation_flag_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit):
            pipeline.parse_args(
                ["--image", "a.png", "--output", "o.svg", "--bar-orientation", "diagonal"])

    def test_run_forwards_bar_orientation_to_analysis(self):
        seen = []

        def fake_analyse(image, weights, *, bar_orientation="auto"):
            seen.append(bar_orientation)
            return pipeline_spec(), None, None

        for extra in ([], ["--bar-orientation", "horizontal"],
                      ["--bar-orientation", "vertical"]):
            with tempfile.TemporaryDirectory() as tmp:
                args = pipeline.parse_args(
                    ["--image", "a.png", "--output", str(Path(tmp) / "out.svg"),
                     "--yes", *extra])
                code = pipeline.run(
                    args, analyse_fn=fake_analyse, input_fn=input, stdin_is_tty=False,
                    out=io.StringIO(), err=io.StringIO())
            self.assertEqual(code, 0)
        self.assertEqual(seen, ["auto", "horizontal", "vertical"])

    def test_main_binds_ocr_engine_and_forwards_orientation(self):
        with (
            mock.patch.object(pipeline, "run", return_value=0) as run_mock,
            mock.patch.object(pipeline, "analyse",
                              return_value=(None, None, None)) as analyse_mock,
        ):
            code = pipeline.main([
                "--image", "a.png", "--output", "o.svg",
                "--ocr", "rapidocr", "--bar-orientation", "horizontal",
            ])
        self.assertEqual(code, 0)
        analyse_fn = run_mock.call_args.kwargs["analyse_fn"]
        analyse_fn("chart.png", "weights.pt", bar_orientation="horizontal")
        analyse_mock.assert_called_once_with(
            "chart.png", "weights.pt",
            ocr_engine="rapidocr", bar_orientation="horizontal",
        )

    def test_manual_bar_orientation_overrides_automatic_selection(self):
        image = mock.sentinel.image
        ocr = mock.sentinel.ocr
        mapping = mock.sentinel.mapping
        reconstructed = pipeline_spec()
        # Horizontal-dominant detections forced to the vertical pipeline.
        raw = detection_frame(horizontal=(0.9, 0.9, 0.9), vertical=(0.3,))
        with (
            mock.patch("cv2.imread", return_value=image),
            mock.patch("src.detection.load_model", return_value=mock.sentinel.model),
            mock.patch("src.detection.run_detection", return_value=raw),
            mock.patch("src.ocr.run_ocr", return_value=ocr),
            mock.patch("src.ocr.tesseract.TesseractBackend",
                       return_value=mock.sentinel.backend),
            mock.patch("src.reconstruction.legend_matcher.filter_bars_by_legend",
                       side_effect=lambda image, frame, ocr: LegendFilterResult(frame, mapping)) as matcher,
            mock.patch("src.reconstruction.vertical.reconstruct_vbar",
                       return_value=reconstructed) as reconstruct,
            mock.patch("src.reconstruction.horizontal.reconstruct_hbar") as h_reconstruct,
            mock.patch.object(pipeline, "cut_off_flags", return_value=[]),
        ):
            result = pipeline.analyse(Path("chart.png"), Path("weights.pt"),
                                      bar_orientation="vertical")
        effective = matcher.call_args.args[1]
        self.assertEqual([d.role for d in effective.detections], ["vertical_bar"])
        reconstruct.assert_called_once_with(effective, ocr, image,
                                            legend_mapping=mapping)
        h_reconstruct.assert_not_called()
        self.assertEqual(result, (reconstructed, None, effective))
        self.assertEqual(len(raw.detections), 4)  # raw detections untouched

    def test_uncertain_orientation_never_starts_ocr_or_reconstruction(self):
        image = mock.sentinel.image
        cases = [
            # (frame, preference, reason fragment)
            (detection_frame(), "auto", "No usable bars were detected."),
            (detection_frame(vertical=(0.5,)), "auto", "too little evidence"),
            (detection_frame(vertical=(0.9,), horizontal=(0.9,)), "auto",
             "conflicting orientation evidence"),
            (detection_frame(horizontal=(0.9, 0.9)), "vertical",
             "No usable vertical bars were detected."),
        ]
        for raw, preference, fragment in cases:
            with self.subTest(fragment=fragment):
                with (
                    mock.patch("cv2.imread", return_value=image),
                    mock.patch("src.detection.load_model",
                               return_value=mock.sentinel.model),
                    mock.patch("src.detection.run_detection", return_value=raw),
                    mock.patch("src.ocr.tesseract.TesseractBackend") as backend,
                    mock.patch("src.ocr.run_ocr") as ocr,
                    mock.patch("src.reconstruction.legend_matcher.filter_bars_by_legend") as matcher,
                    mock.patch("src.reconstruction.vertical.reconstruct_vbar") as v_reconstruct,
                    mock.patch("src.reconstruction.horizontal.reconstruct_hbar") as h_reconstruct,
                    mock.patch.object(pipeline, "cut_off_flags") as cut_off,
                ):
                    spec, error, frame = pipeline.analyse(
                        Path("chart.png"), Path("weights.pt"),
                        bar_orientation=preference)
                self.assertIsNone(spec)
                self.assertIsInstance(error, ChartInterpretationError)
                self.assertIn(fragment, str(error))
                self.assertIn("--bar-orientation vertical", str(error))
                self.assertIn("--bar-orientation horizontal", str(error))
                self.assertIs(frame, raw)
                for never_called in (backend, ocr, matcher, v_reconstruct,
                                     h_reconstruct, cut_off):
                    never_called.assert_not_called()

    def test_unresolved_orientation_exits_3_with_guidance_and_no_output(self):
        error = ChartInterpretationError(
            "Bar orientation is unclear. The detected bars provide conflicting "
            "orientation evidence. Pass --bar-orientation vertical or "
            "--bar-orientation horizontal to choose.")
        for extra in ([], ["--force"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "nested" / "out.svg"
                args = pipeline.parse_args(
                    ["--image", "a.png", "--output", str(out_path),
                     "--yes", *extra])
                out, err = io.StringIO(), io.StringIO()
                code = pipeline.run(
                    args, analyse_fn=lambda *_, **__: (None, error, None),
                    input_fn=input, stdin_is_tty=False, out=out, err=err)
                self.assertEqual(code, 3)
                self.assertFalse(out_path.exists())
                self.assertFalse(out_path.parent.exists())
                self.assertIn("--bar-orientation vertical", out.getvalue())
                self.assertIn("--bar-orientation horizontal", out.getvalue())
                self.assertNotIn("Wrote", out.getvalue())
                self.assertNotIn("WARNING: --force", err.getvalue())

    def test_bad_flag_syntax(self):
        code, _, err, _ = self.go(pipeline_spec(), argv=["--category", "oops"])
        self.assertEqual(code, 1)
        self.assertIn("INDEX=TEXT", err)

    def test_series_with_no_visible_bars_are_keyed_noted_and_do_not_stop_the_build(self):
        series = [SeriesSpec("Low & middle income", "#aa0000", [None, None, None], no_visible_bars=True),
                  SeriesSpec("Kenya", "#112233", [10.0, 20.0, 30.0]),
                  SeriesSpec("Vietnam", "#00aa00", [None, None, None], no_visible_bars=True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
            out, err = io.StringIO(), io.StringIO()
            code = pipeline.run(args, analyse_fn=lambda *_, **__: (pipeline_spec(series=series), None, None),
                                input_fn=input, stdin_is_tty=False, out=out, err=err)
            # Value labels can push the key onto its own page (out-1.svg, out-2.svg, ...).
            svg = "".join(page.read_text() for page in sorted(path.parent.glob("out*.svg")))
        self.assertEqual(code, 0)
        shown = out.getvalue()
        self.assertIn("0: Low & middle income (no visible bars)", shown)
        self.assertIn("1: Kenya\n", shown)
        self.assertIn("2: Vietnam (no visible bars)", shown)
        self.assertIn("Note: No bars are visible for: Low & middle income, Vietnam", shown)
        for sample in ("key-sample-0", "key-sample-1", "key-sample-2"):
            self.assertIn(sample, svg)
        self.assertEqual(svg.count('id="bar-'), 3)           # Kenya's three bars only


if __name__ == "__main__":
    unittest.main()


class NoVisibleBarsTest(unittest.TestCase):
    """Legend entries with no visible bars are keyed, noted, and draw no bars."""

    def chart(self, n_categories=4):
        categories = tuple(str(2000 + j) for j in range(n_categories))
        return VerticalBarChart(
            "demo/legend.png", "Net energy imports", "Years", "Imports",
            categories,
            (BarSeries("Low & middle income", (), has_bars=False),
             BarSeries("Kenya", tuple(13.0 + j for j in range(n_categories))),
             BarSeries("Vietnam", (), has_bars=False)))

    def note_text(self, layout):
        return " ".join(label.text for label in layout.labels if label.role == "note")

    def test_every_legend_entry_has_a_key_sample_and_braille_name_in_order(self):
        layout = layout_vertical_bar_chart(self.chart())
        samples = sorted(layout.key_samples, key=lambda s: s.y)
        self.assertEqual([s.series_index for s in samples], [0, 1, 2])
        series_keys = [l for l in layout.labels if l.role == "key" and str(l.owner_id).startswith("key-sample")]
        self.assertEqual([l.text for l in series_keys], ["Low & middle income", "Kenya", "Vietnam"])
        self.assertEqual(len(set(layout.series_textures)), 3)
        braille = [to_unicode(label.cells) for label in layout.labels if label.id == "key-0"]
        self.assertEqual(braille, [to_unicode(to_braille("Low & middle income"))])

    def test_only_series_with_bars_draw_bars_and_take_bar_slots(self):
        layout = layout_vertical_bar_chart(self.chart())
        self.assertEqual({bar.series_index for bar in layout.bars}, {1})
        self.assertEqual(len(layout.bars), 4)
        solo = VerticalBarChart("demo/solo.png", "Net energy imports", "Years", "Imports",
                                self.chart().categories, (self.chart().series[1],))
        self.assertAlmostEqual(layout.bars[0].width, layout_vertical_bar_chart(solo).bars[0].width)
        self.assertEqual(layout.bars[1].texture, layout.series_textures[1])

    def test_the_transcribers_note_names_the_entries_without_saying_zero(self):
        layout = layout_vertical_bar_chart(self.chart(), value_labels=False)
        note = self.note_text(layout)
        self.assertEqual(note, "No bars are visible for: Low & middle income, Vietnam.")
        self.assertNotIn("zero", note.lower())
        note_labels = [l for l in layout.labels if l.role == "note"]
        self.assertEqual(to_unicode(note_labels[0].cells)[:1], to_unicode(TN_OPEN)[:1])
        first_key_y = min(l.y for l in layout.labels if str(l.owner_id).startswith("key-sample"))
        self.assertTrue(all(l.y < first_key_y for l in note_labels))      # notes come before the key

    def test_the_layout_passes_every_rule(self):
        violations = [v for v in check_layout(layout_vertical_bar_chart(self.chart()))
                      if v.rule.id != "overview_first"]
        self.assertEqual(violations, [])

    def test_a_chart_with_bars_for_every_series_is_laid_out_as_before(self):
        plain = VerticalBarChart("demo/p.png", "T", "X", "Y", ("a", "b", "c"),
                                 (BarSeries("A", (1.0, 2.0, 3.0)), BarSeries("B", (3.0, 2.0, 1.0))))
        layout = layout_vertical_bar_chart(plain, value_labels=False)
        self.assertEqual(len(layout.bars), 6)
        self.assertNotIn("note", {l.role for l in layout.labels})
        self.assertEqual(layout, layout_vertical_bar_chart(replace(
            plain, series=tuple(replace(s, has_bars=True) for s in plain.series)),
            value_labels=False))

    def test_a_single_unnamed_series_still_gets_no_key_and_no_note(self):
        unnamed = VerticalBarChart("demo/u.png", "T", "X", "Y", ("a", "b"), (BarSeries(None, (1.0, 2.0)),))
        layout = layout_vertical_bar_chart(unnamed, value_labels=False)
        self.assertEqual(layout.key_samples, ())
        self.assertNotIn("note", {l.role for l in layout.labels})
        self.assertFalse([l for l in layout.labels if str(l.owner_id).startswith("key-sample")])

    def test_a_divided_graph_puts_the_note_and_every_key_on_the_key_page(self):
        pages = layout_vertical_bar_chart_pages(self.chart(n_categories=30))
        self.assertGreater(len(pages), 2)
        key_page = pages[0]
        self.assertEqual(key_page.page_kind, "key")
        self.assertIn("No bars are visible for: Low & middle income, Vietnam.", self.note_text(key_page))
        self.assertEqual(len(key_page.key_samples), 3)
        for page in pages[1:]:
            self.assertNotIn("No bars are visible", self.note_text(page))
            self.assertEqual(page.key_samples, ())
            self.assertEqual({bar.series_index for bar in page.bars}, {1})
        for page in pages:
            self.assertEqual([v for v in check_layout(page) if v.rule.id != "overview_first"], [])

    def test_more_than_five_legend_entries_are_still_refused(self):
        series = tuple(BarSeries(f"s{k}", (), has_bars=False) for k in range(5)) + (
            BarSeries("with bars", (1.0, 2.0)),)
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(VerticalBarChart("demo/m.png", "T", "X", "Y", ("a", "b"), series))


class MinimumPlotHeightTest(unittest.TestCase):
    """A key that leaves the plot too short goes on a key page before the graph page."""

    def short(self):
        """24811-like: three keyed series, numbered categories and a long y title."""
        return VerticalBarChart(
            "demo/short.png", "Net energy imports of a country", "Years",
            "Net energy imports % of energy use", ("2003", "2004", "2005", "2006"),
            (BarSeries("Low & middle income", (), has_bars=False),
             BarSeries("Kenya", (13.0, 14.0, 15.0, 16.0)),
             BarSeries("Vietnam", (), has_bars=False)))

    def roomy(self):
        return VerticalBarChart("demo/roomy.png", "T", "X", "Y", ("a", "b", "c"),
                                (BarSeries("A", (1.0, 2.0, 3.0)), BarSeries("B", (3.0, 2.0, 1.0))))

    def height(self, layout):
        return layout.plot.y2 - layout.plot.y1

    def test_the_minimum_holds_five_grid_intervals(self):
        self.assertAlmostEqual(constants.MIN_PLOT_HEIGHT, 5 * GRID_MIN_SPACING)

    def test_the_example_really_is_too_short_on_one_page(self):
        self.assertLess(self.height(layout_vertical_bar_chart(self.short())),
                        constants.MIN_PLOT_HEIGHT)

    def test_a_short_plot_moves_the_key_and_note_to_a_key_page(self):
        pages = layout_vertical_bar_chart_pages(self.short())
        self.assertEqual([p.page_kind for p in pages], ["key", "graph"])
        key_page, graph = pages
        self.assertEqual(len(key_page.key_samples), 3)
        self.assertIn("No bars are visible for: Low & middle income, Vietnam.",
                      " ".join(l.text for l in key_page.labels if l.role == "note"))
        self.assertEqual(graph.key_samples, ())
        self.assertNotIn("No bars are visible", " ".join(l.text for l in graph.labels))
        self.assertGreaterEqual(self.height(graph), constants.MIN_PLOT_HEIGHT)
        self.assertEqual({bar.series_index for bar in graph.bars}, {1})
        self.assertEqual(len(graph.bars), 4)

    def test_both_pages_pass_every_rule(self):
        for page in layout_vertical_bar_chart_pages(self.short()):
            self.assertEqual(check_layout(page), [])

    def test_a_chart_with_enough_plot_stays_on_one_page(self):
        pages = layout_vertical_bar_chart_pages(self.roomy())
        self.assertEqual(len(pages), 1)
        self.assertGreaterEqual(self.height(pages[0]), constants.MIN_PLOT_HEIGHT)
        self.assertEqual(pages[0], layout_vertical_bar_chart(self.roomy()))
        self.assertEqual(len(pages[0].key_samples), 2)

    def test_a_single_unnamed_series_has_no_key_to_move(self):
        unnamed = VerticalBarChart("demo/u.png", "T", "X", "A very long y axis title " * 3,
                                   tuple(str(j) for j in range(4)), (BarSeries(None, (1.0, 2.0, 3.0, 4.0)),))
        self.assertEqual(len(layout_vertical_bar_chart_pages(unnamed)), 1)
