"""Per-bar braille value labels (src/tactile/value_labels.py and the layout/rules around it)."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.contract_io import load_vertical_bar_chart, write_vertical_bar_chart
from src.contracts import BarSeries, SeriesSpec, VerticalBarChart, VerticalBarSpec, YAxisSpec
from src.tactile import check_layout, constants, layout_vertical_bar_chart
from src.tactile.braille import has_numeric_indicator, to_braille
from src.tactile.completeness import spec_to_chart_with_notes
from src.tactile.layout import BrailleLabel, layout_vertical_bar_chart_pages
from src.tactile.value_labels import significant_cap, value_text

CATS = ("2003", "2004", "2005", "2006")
ESTIMATE = "Bar values are estimated from the print."


def chart(series, cats=CATS, title="Imports"):
    return VerticalBarChart("demo/values.png", title, "Years", "Imports", cats, tuple(series))


def values(layout):
    return [l for l in layout.labels if l.role == "value"]


def note_text(layout):
    return " ".join(l.text for l in layout.labels if l.role == "note")


def bar_for(layout, label):
    return layout.by_id("bar-" + label.id[len("value-"):])


def rules(layout):
    return [v for v in check_layout(layout) if v.rule.id != "overview_first"]


class ValueTextTest(unittest.TestCase):
    def test_rounds_to_the_tick_decimals(self):
        self.assertEqual(value_text(14.5, 0, 3), "15")
        self.assertEqual(value_text(0.314, 1, 3), "0.3")
        self.assertEqual(value_text(0.314, 2, 3), "0.31")

    def test_caps_significant_digits_unless_the_ticks_need_more(self):
        self.assertEqual(value_text(123456.789, 0, 3), "123000")
        self.assertEqual(significant_cap(["0", "500", "1000"]), 3)
        self.assertEqual(significant_cap(["1234", "2468"]), 4)
        self.assertEqual(value_text(1234.5, 0, significant_cap(["1234", "2468"])), "1235")

    def test_a_nonzero_value_is_never_shown_as_zero(self):
        self.assertEqual(value_text(0.04, 1, 3), "0.04")
        self.assertEqual(value_text(0.0, 2, 3), "0")

    def test_negative_values_keep_their_sign(self):
        self.assertEqual(value_text(-12.5, 0, 3), "-13")
        self.assertTrue(has_numeric_indicator(to_braille("-13")))


class SingleSeriesTest(unittest.TestCase):
    def layout(self):
        return layout_vertical_bar_chart(chart([BarSeries("Kenya", (13.0, 14.5, 15.25, 16.0))]))

    def test_every_bar_has_a_centred_numeric_label_just_above_it(self):
        layout = self.layout()
        labels = values(layout)
        self.assertEqual([l.text for l in labels], ["13", "15", "15", "16"])
        for label in labels:
            bar = bar_for(layout, label)
            self.assertAlmostEqual((label.bounds.x1 + label.bounds.x2) / 2,
                                   (bar.bounds.x1 + bar.bounds.x2) / 2, places=6)
            self.assertGreaterEqual(bar.bounds.y1 - label.bounds.y2, constants.LABEL_MIN_GAP)
            self.assertLessEqual(bar.bounds.y1 - label.bounds.y2, constants.LABEL_MAX_GAP)
            self.assertTrue(label.numeric and has_numeric_indicator(label.cells))
            self.assertEqual(label.owner_id, bar.id)

    def test_decimals_follow_the_tick_step(self):
        layout = layout_vertical_bar_chart(chart([BarSeries(None, (0.31, 0.45, 0.52, 0.68))]))
        self.assertEqual([l.text for l in values(layout)], ["0.3", "0.5", "0.5", "0.7"])

    def test_fewer_decimals_when_a_label_is_wider_than_its_bar(self):
        three = ("a", "b", "c")
        layout = layout_vertical_bar_chart(chart([BarSeries(None, (0.31, 0.34, 0.38))], three))
        self.assertEqual([l.text for l in values(layout)], ["0.3", "0.3", "0.4"])
        self.assertEqual(rules(layout), [])

    def test_the_estimate_note_comes_before_the_graph_and_labels_pass_every_rule(self):
        layout = self.layout()
        self.assertIn(ESTIMATE, note_text(layout))
        notes = [l for l in layout.labels if l.role == "note"]
        self.assertTrue(all(n.bounds.y2 < layout.plot.y1 for n in notes))
        self.assertEqual(rules(layout), [])

    def test_grid_lines_are_cut_around_labels(self):
        layout = self.layout()
        grid = [l for l in layout.lines if l.kind == "grid"]
        self.assertGreater(len(grid), len({round(l.y1, 3) for l in grid}))   # a line in pieces
        self.assertEqual([v for v in rules(layout) if v.rule.id == "label_clearance"], [])

    def test_value_labels_can_be_turned_off(self):
        plain = layout_vertical_bar_chart(chart([BarSeries("Kenya", (13.0, 14.5, 15.25, 16.0))]),
                                          value_labels=False)
        self.assertEqual(values(plain), [])
        self.assertNotIn(ESTIMATE, note_text(plain))


class MultiSeriesTest(unittest.TestCase):
    def test_each_bar_of_each_series_is_labelled_with_its_own_value(self):
        series = (BarSeries("A", (10.0, 20.0, 30.0)), BarSeries("B", (30.0, 25.0, 5.0)),
                  BarSeries("C", (7.0, 8.0, 9.0)))
        layout = layout_vertical_bar_chart(chart(series, ("a", "b", "c")))
        shown = {l.id: l.text for l in values(layout)}
        self.assertEqual(shown, {f"value-{s}-{j}": str(int(v)) for s, ser in enumerate(series)
                                 for j, v in enumerate(ser.values)})
        self.assertEqual(rules(layout), [])

    def test_labels_never_touch_neighbouring_bars_or_labels(self):
        layout = layout_vertical_bar_chart(chart(
            (BarSeries("A", (10.0, 20.0, 30.0)), BarSeries("B", (30.0, 25.0, 5.0))), ("a", "b", "c")))
        for label in values(layout):
            for other in (*layout.bars, *(l for l in layout.labels if l is not label)):
                self.assertGreaterEqual(label.bounds.gap(other.bounds) + 1e-6,
                                        constants.LABEL_MIN_GAP, (label.id, other.id))


class NoRoomTest(unittest.TestCase):
    def long_numbers(self):
        return layout_vertical_bar_chart(chart(
            [BarSeries(None, (123456.789, 234567.891, 345678.912))], ("a", "b", "c")))

    def test_a_label_wider_than_its_bar_goes_on_a_lead_line(self):
        layout = self.long_numbers()
        raised = [l for l in layout.lines if l.kind == "lead"]
        self.assertEqual([l.id for l in raised], ["lead-0-2"])
        self.assertAlmostEqual(raised[0].length, constants.LEAD_LINE_LENGTH)
        label = values(layout)[0]
        self.assertEqual((label.id, label.text, label.owner_id), ("value-0-2", "346000", "lead-0-2"))
        self.assertEqual(rules(layout), [])

    def test_bars_with_no_room_are_omitted_and_named_in_a_note(self):
        layout = self.long_numbers()
        self.assertEqual([l.id for l in values(layout)], ["value-0-2"])
        self.assertIn("No value is shown for bars too narrow for it: a, b.", note_text(layout))
        self.assertIn(ESTIMATE, note_text(layout))

    def test_a_chart_with_no_room_for_labels_keeps_its_old_layout(self):
        many = chart([BarSeries(None, tuple(float(10 + i) for i in range(12)))],
                     tuple(str(i) for i in range(12)))
        layout = layout_vertical_bar_chart(many)
        self.assertEqual(values(layout), [])
        self.assertIn("bar values are not labelled: no room on the page for them", layout.notes)
        self.assertEqual(layout, replace(layout_vertical_bar_chart(many, value_labels=False),
                                         notes=layout.notes))


class SkippedBarsTest(unittest.TestCase):
    def test_rescued_bars_get_no_label_and_are_named(self):
        layout = layout_vertical_bar_chart(chart([BarSeries(
            "Kenya", (13.0, 0.0, 15.0, 16.0), rescued=(False, True, False, False))]))
        self.assertEqual([l.id for l in values(layout)], ["value-0-0", "value-0-2", "value-0-3"])
        self.assertIn("No value is shown for bars whose height was measured without a detected "
                      "bar: 2004.", note_text(layout))

    def test_series_with_no_visible_bars_get_no_label(self):
        layout = layout_vertical_bar_chart(chart([
            BarSeries("Low", (), has_bars=False), BarSeries("Kenya", (13.0, 14.0, 15.0, 16.0)),
            BarSeries("Vietnam", (), has_bars=False)]))
        self.assertEqual({l.id.split("-")[1] for l in values(layout)}, {"1"})
        self.assertEqual(len(values(layout)), 4)
        self.assertEqual(rules(layout), [])

    def test_negative_bars_are_still_rejected_by_the_layout(self):
        with self.assertRaises(ValueError):
            layout_vertical_bar_chart(chart([BarSeries(None, (1.0, -2.0, 3.0, 4.0))]))

    def test_the_rescued_flags_reach_the_chart_from_the_spec(self):
        spec = VerticalBarSpec(
            "input/a.png", "T", "X", "Y", YAxisSpec(0, 100, [0, 50, 100]), ["a", "b", "c"],
            [SeriesSpec("Kenya", "#112233", [1.0, 0.0, 3.0],
                        value_rescued=[False, True, False], value_confidence=[0.9, 0.2, 0.9])],
            bars_adjacent=False, flags=["bars_rescued:1", "value_inferred_zero"], confidence={})
        built, _ = spec_to_chart_with_notes(spec)
        self.assertEqual(built.series[0].rescued, (False, True, False))
        plain, _ = spec_to_chart_with_notes(replace(spec, series=[SeriesSpec(
            "Kenya", "#112233", [1.0, 2.0, 3.0])], flags=[]))
        self.assertEqual(plain.series[0].rescued, ())

    def test_rescued_must_match_the_values_and_survives_json(self):
        with self.assertRaises(ValueError):
            BarSeries("A", (1.0, 2.0), rescued=(True,))
        original = chart([BarSeries("A", (1.0, 2.0, 3.0, 4.0), rescued=(False, True, False, False)),
                          BarSeries("B", (1.0, 2.0, 3.0, 4.0))])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chart.json"
            write_vertical_bar_chart(path, original)
            self.assertNotIn('"rescued": []', path.read_text())
            self.assertEqual(load_vertical_bar_chart(path), original)


class RuleTest(unittest.TestCase):
    def setUp(self):
        self.layout = layout_vertical_bar_chart(chart([BarSeries("Kenya", (13.0, 14.5, 15.25, 16.0))]))

    def moved(self, **change):
        label = values(self.layout)[0]
        labels = tuple(replace(l, **change) if l is label else l for l in self.layout.labels)
        return replace(self.layout, labels=labels), label

    def ids(self, layout):
        return {v.rule.id for v in rules(layout)}

    def test_a_label_too_far_above_its_bar_breaks_clearance(self):
        moved, label = self.moved(y=values(self.layout)[0].y - 10)
        self.assertIn("value_label_clearance", self.ids(moved))

    def test_a_label_too_close_to_its_bar_breaks_clearance(self):
        moved, label = self.moved(y=values(self.layout)[0].y + 2.0)
        self.assertTrue({"value_label_clearance", "label_clearance"} <= self.ids(moved))

    def test_an_off_centre_label_is_reported(self):
        moved, _ = self.moved(x=values(self.layout)[0].x + 2.0)
        self.assertIn("value_label_clearance", self.ids(moved))

    def test_a_label_wider_than_its_bar_without_a_lead_line_breaks_fit(self):
        label = values(self.layout)[0]
        wide = to_braille("123456789012")
        moved, _ = self.moved(text="123456789012", cells=wide, x=label.x - 20)
        self.assertIn("value_label_fit", self.ids(moved))

    def test_the_new_rules_are_registered_with_their_sources(self):
        from src.tactile import RULES
        self.assertEqual(RULES["value_label_clearance"].section, "§3.4.3.12")
        self.assertEqual(RULES["value_label_fit"].section, "not from BANA")


class DividedGraphTest(unittest.TestCase):
    def test_every_graph_page_of_a_divided_chart_passes_every_rule(self):
        cats = tuple(f"Month {i}" for i in range(1, 25))
        pages = layout_vertical_bar_chart_pages(chart(
            [BarSeries("A", tuple(float(i) for i in range(1, 25))),
             BarSeries("B", tuple(float(30 - i) for i in range(1, 25)))], cats))
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertEqual(rules(page), [])


if __name__ == "__main__":
    unittest.main()
