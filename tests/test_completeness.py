"""Completeness checks and overrides on hand-built specs; no detector or OCR runs."""

import unittest

from src.contracts import (
    Box, ChartInterpretationError, Detection, DetectionFrame, SeriesSpec,
    VerticalBarSpec, YAxisSpec,
)
from src.tactile.completeness import (
    BLOCKING_FLAGS, CANNOT_BUILD, FORCEABLE_FLAGS, NEEDS_INPUT, READY, Overrides,
    apply_overrides, check_completeness, completeness_from_error, spec_to_chart,
)


def make_spec(**changes):
    fields = dict(
        source_key="val/1.png", title="Sales", x_axis_title="Year", y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]), categories=["2001", "2002", "2003"],
        series=[SeriesSpec(None, "#112233", [10.0, 20.0, 30.0])],
        bars_adjacent=False, flags=[], confidence={"title": 0.9})
    fields.update(changes)
    return VerticalBarSpec(**fields)


def two_series(names=("A", "B")):
    return [SeriesSpec(names[0], "#112233", [1.0, 2.0, 3.0]),
            SeriesSpec(names[1], "#445566", [3.0, 2.0, 1.0])]


class Status(unittest.TestCase):
    def test_ready(self):
        result = check_completeness(make_spec())
        self.assertEqual((result.status, result.missing, result.blockers), (READY, [], []))

    def test_missing_titles(self):
        spec = make_spec(title=None, x_axis_title=None, y_axis_title=None,
                         flags=["title_missing", "x_axis_title_missing", "y_axis_title_missing"])
        result = check_completeness(spec)
        self.assertEqual(result.status, NEEDS_INPUT)
        self.assertEqual([(m.field, m.cli_flag) for m in result.missing],
                         [("title", "--title"), ("x_axis_title", "--x-title"),
                          ("y_axis_title", "--y-title")])

    def test_none_title_without_flag_is_still_missing(self):
        result = check_completeness(make_spec(title=None))
        self.assertEqual([m.field for m in result.missing], ["title"])

    def test_unreadable_category(self):
        spec = make_spec(categories=["2001", "?", "2003"], flags=["category_unreadable"],
                         confidence={"category[1]": 0.1})
        result = check_completeness(spec)
        self.assertEqual(result.status, NEEDS_INPUT)
        self.assertEqual([(m.field, m.cli_flag) for m in result.missing],
                         [("category[1]", "--category 1=TEXT")])

    def test_unnamed_series_on_multi_series_chart(self):
        spec = make_spec(series=two_series(("A", None)))
        result = check_completeness(spec)
        self.assertEqual(result.status, NEEDS_INPUT)
        self.assertEqual([(m.field, m.cli_flag) for m in result.missing],
                         [("series_name[1]", "--series 1=NAME")])

    def test_unnamed_single_series_is_ready(self):
        self.assertEqual(check_completeness(make_spec()).status, READY)

    def test_each_blocking_flag(self):
        for flag in BLOCKING_FLAGS:
            with self.subTest(flag=flag):
                result = check_completeness(make_spec(flags=[flag]))
                self.assertEqual(result.status, CANNOT_BUILD)
                self.assertIn("rainfall.json", result.blockers[0])
                self.assertEqual(result.forceable, flag in FORCEABLE_FLAGS)

    def test_none_value_blocks(self):
        spec = make_spec(series=[SeriesSpec("A", None, [1.0, None, 3.0])])
        result = check_completeness(spec)
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertFalse(result.forceable)

    def test_empty_categories_block(self):
        spec = make_spec(categories=[], series=[SeriesSpec("A", None, [])])
        result = check_completeness(spec)
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertFalse(result.forceable)

    def test_notes(self):
        spec = make_spec(flags=["bars_rescued:2", "value_inferred_zero", "y_ticks_dropped:1",
                                "title_touches_right_edge"])
        result = check_completeness(spec)
        self.assertEqual(result.status, READY)
        self.assertEqual(len(result.notes), 4)
        self.assertIn("2 bar value", result.notes[0])
        self.assertIn("1 y-axis tick", result.notes[2])

    def test_value_inferred_zero_note_counts_values_and_stays_a_note(self):
        spec = make_spec(
            series=[SeriesSpec(None, "#112233", [0.0, 5.0, 0.0], value_rescued=[True, False, True])],
            flags=["bars_rescued:2", "value_inferred_zero"])
        result = check_completeness(spec)
        self.assertEqual((result.status, result.blockers), (READY, []))
        note, = [n for n in result.notes if "visible" in n]
        self.assertIn("No bar was visible for 2 value(s)", note)
        self.assertIn("below the axis that are not drawn", note)

    def test_blocker_beats_missing_text(self):
        spec = make_spec(title=None, flags=["title_missing", "series_ambiguous"])
        result = check_completeness(spec)
        self.assertEqual((result.status, result.missing), (CANNOT_BUILD, []))


class Encodable(unittest.TestCase):
    def test_salvador_misread_needs_input(self):
        spec = make_spec(series=two_series(("Chile", "E] Salvador")))
        result = check_completeness(spec)
        self.assertEqual(result.status, NEEDS_INPUT)
        item, = result.missing
        self.assertEqual((item.field, item.cli_flag), ("series_name[1]", "--series 1=NAME"))
        self.assertIn("']'", item.reason)

    def test_each_field_names_its_flag(self):
        spec = make_spec(title="A]", x_axis_title="B[", y_axis_title="C}",
                         categories=["2001", "2{2", "2003"])
        flags = {m.field: m.cli_flag for m in check_completeness(spec).missing}
        self.assertEqual(flags, {"title": "--title", "x_axis_title": "--x-title",
                                 "y_axis_title": "--y-title", "category[1]": "--category 1=TEXT"})

    def test_override_with_bad_characters_stays_needs_input(self):
        fixed, _ = apply_overrides(make_spec(), Overrides(title="Bad]"))
        self.assertEqual([m.field for m in check_completeness(fixed).missing], ["title"])

    def test_override_with_good_text_clears_it(self):
        spec = make_spec(series=two_series(("Chile", "E] Salvador")))
        fixed, _ = apply_overrides(spec, Overrides(series_names={1: "El Salvador"}))
        self.assertEqual(check_completeness(fixed).status, READY)

    def test_no_duplicate_when_already_missing(self):
        spec = make_spec(categories=["2001", "]", "2003"], flags=["category_unreadable"],
                         confidence={"category[1]": 0.1})
        self.assertEqual([m.field for m in check_completeness(spec).missing], ["category[1]"])

    def test_normal_text_is_ready(self):
        spec = make_spec(title="Goods (2007-2012), 50%", categories=["East Asia", "-5", "3.5"])
        self.assertEqual(check_completeness(spec).status, READY)


class Errors(unittest.TestCase):
    def frame(self, *roles):
        detections = tuple(Detection(f"d{i}", role, Box(0, 0, 10, 10), 0.9)
                           for i, role in enumerate(roles))
        return DetectionFrame("val/1.png", 100, 100, detections)

    def test_x_ticks_detected_as_axis_titles(self):
        error = ChartInterpretationError("no x tick labels detected")
        result = completeness_from_error(error, self.frame("x_axis_title", "x_axis_title", "x_axis_title"))
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertIn("3 label(s) may have been detected as axis titles", result.blockers[0])
        self.assertIn("rainfall.json", result.blockers[0])

    def test_x_ticks_without_frame(self):
        result = completeness_from_error(ChartInterpretationError("no x tick labels detected"))
        self.assertNotIn("axis titles", result.blockers[0])

    def test_y_ticks_and_no_bars(self):
        y = completeness_from_error(ChartInterpretationError("fewer than 2 readable y tick labels"))
        self.assertIn("y-axis tick labels", y.blockers[0])
        bars = completeness_from_error(
            ChartInterpretationError("no vertical bars detected inside the plot area"),
            self.frame("horizontal_bar"))
        self.assertIn("horizontal bars", bars.blockers[0])


class Overrides_(unittest.TestCase):
    def test_overrides_clear_flags_and_add_notes(self):
        spec = make_spec(
            title=None, y_axis_title=None, categories=["2001", "?", "2003"],
            series=two_series(("A", None)),
            flags=["title_missing", "y_axis_title_missing", "category_unreadable",
                   "title_touches_right_edge", "bars_rescued:1"],
            confidence={"category[1]": 0.1})
        fixed, notes = apply_overrides(spec, Overrides(
            title="T", y_axis_title="Y", categories={1: "2002"}, series_names={1: "B"}))
        self.assertEqual(fixed.flags, ["bars_rescued:1"])
        self.assertEqual((fixed.title, fixed.y_axis_title, fixed.categories[1],
                          fixed.series[1].name), ("T", "Y", "2002", "B"))
        self.assertNotIn("category[1]", fixed.confidence)
        self.assertIn("chart title supplied by user", notes)
        self.assertEqual(len(notes), 4)
        self.assertEqual(check_completeness(fixed).status, READY)
        self.assertEqual(spec.title, None)  # input is not mutated

    def test_partial_category_override_keeps_flag(self):
        spec = make_spec(categories=["?", "?", "2003"], flags=["category_unreadable"],
                         confidence={"category[0]": 0.1, "category[1]": 0.1})
        fixed, _ = apply_overrides(spec, Overrides(categories={0: "2001"}))
        self.assertIn("category_unreadable", fixed.flags)
        self.assertEqual([m.field for m in check_completeness(fixed).missing], ["category[1]"])

    def test_override_does_not_fix_cannot_build(self):
        spec = make_spec(title=None, flags=["title_missing", "category_ambiguous"])
        fixed, _ = apply_overrides(spec, Overrides(title="T"))
        self.assertEqual(fixed.flags, ["category_ambiguous"])
        self.assertEqual(check_completeness(fixed).status, CANNOT_BUILD)

    def test_bad_overrides_rejected(self):
        for overrides in (Overrides(categories={9: "x"}), Overrides(series_names={5: "x"}),
                          Overrides(title="  ")):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                apply_overrides(make_spec(), overrides)


class Build(unittest.TestCase):
    def test_ready_spec_builds(self):
        chart = spec_to_chart(make_spec())
        self.assertEqual((chart.title, chart.categories), ("Sales", ("2001", "2002", "2003")))
        self.assertEqual(chart.series[0].values, (10.0, 20.0, 30.0))

    def test_unready_spec_raises(self):
        for spec in (make_spec(title=None), make_spec(flags=["series_ambiguous"])):
            with self.subTest(flags=spec.flags), self.assertRaisesRegex(ValueError, "not ready"):
                spec_to_chart(spec)


if __name__ == "__main__":
    unittest.main()
