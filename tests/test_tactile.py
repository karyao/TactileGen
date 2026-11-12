"""Tactile layout, braille, BANA rule checks and SVG output."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.contract_io import load_vertical_bar_chart
from src.contracts import BarSeries, VerticalBarChart
from src.tactile import (
    RULES,
    Violation,
    check_layout,
    constants,
    layout_vertical_bar_chart,
    layout_vertical_bar_chart_pages,
    render_svg,
)
from src.tactile.__main__ import main
from src.tactile.braille import NUMERIC_INDICATOR, TN_CLOSE, TN_OPEN, to_braille, to_unicode
from src.tactile.keys import abbreviation_keys, is_valid_abbreviation_key, numeric_keys
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
            layout = layout_vertical_bar_chart(load("rainfall"))
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


if __name__ == "__main__":
    unittest.main()
