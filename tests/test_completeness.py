"""Completeness checks and overrides on hand-built specs; no detector or OCR runs."""

import unittest

import numpy as np

from src.contracts import (
    Box, ChartInterpretationError, Detection, DetectionFrame, SeriesSpec,
    VerticalBarSpec, YAxisSpec,
)
from src.tactile.completeness import (
    BLOCKING_FLAGS, CANNOT_BUILD, FORCEABLE_FLAGS, NEEDS_INPUT, READY, SELF_EXPLAINING_FLAGS,
    Overrides,
    apply_overrides, check_completeness, completeness_from_error, spec_to_chart,
    spec_to_chart_with_notes, cut_off_flags,
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

    def test_bars_mostly_missed_message_and_forceable(self):
        result = check_completeness(make_spec(flags=["bars_mostly_missed:67"]))
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertEqual(result.blockers, [
            "67% of bars were not detected and would be drawn as 0 (the data may include values "
            "below the axis, or the detector may have missed bars). Re-run with --force to build anyway."])
        self.assertTrue(result.forceable)

    def test_each_blocking_flag(self):
        for flag in BLOCKING_FLAGS:
            with self.subTest(flag=flag):
                # a negative tick keeps value_clipped_at_axis a blocker
                result = check_completeness(make_spec(
                    flags=[flag], y_axis=YAxisSpec(-50, 50, [-50, 0, 50])))
                self.assertEqual(result.status, CANNOT_BUILD)
                if flag not in SELF_EXPLAINING_FLAGS:
                    self.assertIn("rainfall.json", result.blockers[0])
                self.assertEqual(result.forceable, flag in FORCEABLE_FLAGS)

    def test_uneven_y_ticks_are_only_a_note(self):
        result = check_completeness(make_spec(flags=["y_ticks_not_arithmetic", "y_ticks_dropped:2"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertTrue(any("not evenly spaced" in n for n in result.notes))
        self.assertTrue(any("2 y-axis tick" in n for n in result.notes))

    def test_series_from_swatch_is_a_note(self):
        result = check_completeness(make_spec(flags=["series_from_swatch:2"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertTrue(any("2 series had no detected bars" in n for n in result.notes))

    def test_short_bars_remeasured_is_a_note(self):
        result = check_completeness(make_spec(flags=["short_bars_remeasured:3"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertTrue(any("3 very short bar(s)" in n for n in result.notes))

    def test_bar_edge_corrected_is_a_note(self):
        result = check_completeness(make_spec(flags=["bar_edge_corrected:2"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertTrue(any("2 bar(s) had their top edge corrected" in n for n in result.notes))

    def test_clipped_bar_is_a_note_on_an_axis_from_zero(self):
        result = check_completeness(make_spec(flags=["value_clipped_at_axis"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertEqual([n for n in result.notes if "past the end" in n],
                         ["At least one bar runs past the end of the y axis, so its value is a lower bound."])

    def test_clipped_bar_still_blocks_with_a_negative_tick_and_cannot_be_forced(self):
        spec = make_spec(y_axis=YAxisSpec(-50, 50, [-50, 0, 50]), flags=["value_clipped_at_axis"])
        result = check_completeness(spec)
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertIn("runs past the end of the y axis", result.blockers[0])
        self.assertFalse(result.forceable)
        self.assertFalse(any("lower bound" in n for n in result.notes))

    def test_decimal_repair_is_a_note(self):
        result = check_completeness(make_spec(flags=["y_tick_decimal_repaired:3"]))
        self.assertEqual((result.status, result.blockers), (READY, []))
        self.assertTrue(any("3 y-axis tick label(s) had lost their decimal point" in n for n in result.notes))

    def test_leading_zero_tick_blocks_and_cannot_be_forced(self):
        result = check_completeness(make_spec(flags=["y_tick_leading_zero"]))
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertIn("decimal point may have been lost", result.blockers[0])
        self.assertIn("rainfall.json", result.blockers[0])
        self.assertFalse(result.forceable)

    def test_leading_zero_with_uneven_ticks_still_blocks(self):
        result = check_completeness(make_spec(flags=["y_ticks_not_arithmetic", "y_tick_leading_zero"]))
        self.assertEqual(result.status, CANNOT_BUILD)
        self.assertEqual(len(result.blockers), 1)

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


class SuspiciousCategories(unittest.TestCase):
    def spec(self, cats, **kw):
        return make_spec(categories=cats, series=[SeriesSpec(None, "#112233", [1.0] * len(cats))], **kw)

    def test_single_character_among_long_names(self):
        result = check_completeness(self.spec(["Germany", "Greece", "i", "India"]))
        self.assertEqual(result.status, NEEDS_INPUT)
        item, = result.missing
        self.assertEqual((item.field, item.cli_flag), ("category[2]", "--category 2=TEXT"))
        self.assertIn("single character", item.reason)

    def test_single_character_is_fine_when_categories_are_short(self):
        self.assertEqual(check_completeness(self.spec(["A", "B", "C"])).status, READY)
        self.assertEqual(check_completeness(self.spec(["US", "UK", "x"])).status, READY)

    def test_year_far_from_median(self):
        result = check_completeness(self.spec(["2007", "2008", "6002", "2010", "2011"]))
        item, = result.missing
        self.assertEqual(item.field, "category[2]")
        self.assertIn("median year", item.reason)

    def test_two_years_that_disagree_are_both_asked_about(self):
        result = check_completeness(self.spec(["2000", "9002"]))
        self.assertEqual([m.field for m in result.missing], ["category[0]", "category[1]"])

    def test_ordinary_and_long_spans_pass(self):
        self.assertEqual(check_completeness(self.spec(["1960", "1980", "2000", "2012"])).status, READY)
        self.assertEqual(check_completeness(self.spec(["2013", "2012", "2011"])).status, READY)
        self.assertEqual(check_completeness(self.spec(["1996", "2005", "2012"])).status, READY)

    def test_mixed_categories_do_not_trigger_the_year_rule(self):
        self.assertEqual(check_completeness(self.spec(["2000", "Total", "9002"])).status, READY)

    def test_supplying_text_clears_it_and_is_not_asked_again(self):
        spec = self.spec(["Germany", "Greece", "i", "India"])
        fixed, _ = apply_overrides(spec, Overrides(categories={2: "Hungary"}))
        self.assertEqual(check_completeness(fixed).status, READY)
        # the user may confirm a real one-character label
        fixed, _ = apply_overrides(spec, Overrides(categories={2: "i"}))
        self.assertEqual(check_completeness(fixed).status, READY)

    def test_no_duplicate_with_unreadable_category(self):
        spec = self.spec(["Germany", "Greece", "?", "India"], flags=["category_unreadable"],
                         confidence={"category[2]": 0.1})
        self.assertEqual([m.field for m in check_completeness(spec).missing], ["category[2]"])

    def test_cannot_build_still_wins(self):
        spec = self.spec(["Germany", "i", "Greece"], flags=["series_ambiguous"])
        self.assertEqual(check_completeness(spec).status, CANNOT_BUILD)


class CutOffTitles(unittest.TestCase):
    W, H = 200, 120

    def frame(self, *detections):
        items = tuple(Detection(f"d{i}", role, Box(*box), 0.9)
                      for i, (role, box) in enumerate(detections))
        return DetectionFrame("val/1.png", self.W, self.H, items)

    def image(self, ink=None):
        image = np.full((self.H, self.W, 3), 255, np.uint8)
        if ink:
            x1, y1, x2, y2 = ink
            image[y1:y2, x1:x2] = 0
        return image

    def test_ink_in_last_columns_of_the_title_rows_flags_the_chart_title(self):
        frame = self.frame(("chart_title", (20, 5, 190, 25)))
        self.assertEqual(cut_off_flags(frame, self.image((196, 8, 200, 20))), ["title_cut_off"])

    def test_ink_elsewhere_or_outside_the_title_rows_does_not(self):
        frame = self.frame(("chart_title", (20, 5, 190, 25)))
        self.assertEqual(cut_off_flags(frame, self.image((100, 8, 140, 20))), [])
        self.assertEqual(cut_off_flags(frame, self.image((196, 60, 200, 100))), [])
        self.assertEqual(cut_off_flags(frame, self.image()), [])

    def test_single_stray_pixel_is_not_ink(self):
        frame = self.frame(("chart_title", (20, 5, 190, 25)))
        image = self.image()
        image[10, 199] = 0
        self.assertEqual(cut_off_flags(frame, image), [])

    def test_y_axis_title_box_touching_the_top_edge(self):
        for top, expected in ((0.0, ["y_axis_title_cut_off"]), (2.0, ["y_axis_title_cut_off"]),
                              (3.0, [])):
            with self.subTest(top=top):
                frame = self.frame(("y_axis_title", (2, top, 20, 100)))
                self.assertEqual(cut_off_flags(frame, self.image()), expected)

    def test_no_title_detections_means_no_flags(self):
        self.assertEqual(cut_off_flags(self.frame(), self.image()), [])

    def test_cut_off_titles_become_needs_input_with_a_clear_message(self):
        spec = make_spec(flags=["title_cut_off", "y_axis_title_cut_off"])
        result = check_completeness(spec)
        self.assertEqual(result.status, NEEDS_INPUT)
        self.assertEqual([(m.field, m.cli_flag) for m in result.missing],
                         [("title", "--title"), ("y_axis_title", "--y-title")])
        for item in result.missing:
            self.assertIn("cut off at the image edge", item.reason)
            self.assertIn("same text to confirm", item.reason)

    def test_supplied_or_confirmed_title_is_not_asked_again(self):
        spec = make_spec(flags=["title_cut_off", "y_axis_title_cut_off"])
        fixed, notes = apply_overrides(spec, Overrides(title="Sales", y_axis_title="Revenue"))
        self.assertEqual(fixed.flags, [])
        self.assertEqual(check_completeness(fixed).status, READY)
        self.assertIn("chart title supplied by user", notes)

    def test_other_title_text_is_still_asked_for_separately(self):
        spec = make_spec(flags=["title_cut_off", "y_axis_title_cut_off"])
        fixed, _ = apply_overrides(spec, Overrides(title="Sales"))
        self.assertEqual([m.field for m in check_completeness(fixed).missing], ["y_axis_title"])

    def test_missing_title_is_not_listed_twice(self):
        spec = make_spec(title=None, flags=["title_missing", "title_cut_off"])
        self.assertEqual([m.field for m in check_completeness(spec).missing], ["title"])


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
        self.assertEqual(fixed.flags, ["bars_rescued:1", "category_supplied:1"])
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

    def test_negatives_become_zero_when_the_axis_starts_at_zero(self):
        spec = make_spec(series=[SeriesSpec(None, "#112233", [-0.4, 100.0, -60.0])])
        chart, notes = spec_to_chart_with_notes(spec)
        self.assertEqual(chart.series[0].values, (0.0, 100.0, 0.0))
        self.assertEqual(notes, ["Values below the axis minimum were set to 0 (2 value(s))."])

    def test_negative_axis_leaves_negative_values_for_the_layout_to_reject(self):
        spec = make_spec(y_axis=YAxisSpec(-50, 50, [-50, 0, 50]),
                         series=[SeriesSpec(None, "#112233", [-5.0, 40.0, 20.0])])
        chart, notes = spec_to_chart_with_notes(spec)
        self.assertEqual((chart.series[0].values, notes), ((-5.0, 40.0, 20.0), []))

    def test_no_adjustment_when_nothing_is_negative(self):
        self.assertEqual(spec_to_chart_with_notes(make_spec())[1], [])

    def test_unready_spec_raises(self):
        for spec in (make_spec(title=None), make_spec(flags=["series_ambiguous"])):
            with self.subTest(flags=spec.flags), self.assertRaisesRegex(ValueError, "not ready"):
                spec_to_chart(spec)


if __name__ == "__main__":
    unittest.main()
