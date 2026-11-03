"""Scoring helpers of the Stage 2 end-to-end run."""

import unittest

from src.baseline.e2e_stage2 import compare, flag_scores
from src.contracts import BarSeries, InterpretationResult, ReviewItem, VerticalBarChart


def chart(categories, values, ticks=(0.0, 50.0, 100.0)):
    return VerticalBarChart("val/a.png", None, None, None, tuple(categories),
                            (BarSeries(None, tuple(values)),), y_tick_values=ticks)


def result(c, status="accepted"):
    items = (ReviewItem("bar", 0, flagged=True, reason="x"),) if status == "needs_review" else ()
    return InterpretationResult("val/a.png", status, c, items)


class CompareTest(unittest.TestCase):
    def test_identical_charts_have_no_mistakes(self):
        ref = result(chart("AB", (10, 20)))
        self.assertEqual(compare(ref, result(chart("AB", (10, 20)))),
                         {"built": True, "categories_wrong": 0, "values_wrong": 0, "decade": 0})

    def test_category_names_compare_after_normalization(self):
        ref = result(chart(["Speciﬁc", "B"], (10, 20)))
        self.assertEqual(compare(ref, result(chart(["Specific", "C"], (10, 20))))["categories_wrong"], 1)

    def test_values_off_by_more_than_one_percent_of_span_count(self):
        ref = result(chart("AB", (10, 20)))                  # span 100, so 1 unit is the limit
        got = compare(ref, result(chart("AB", (10.5, 22))))
        self.assertEqual(got["values_wrong"], 1)

    def test_factor_of_ten_is_counted_separately(self):
        ref = result(chart("AB", (10, 20)))
        self.assertEqual(compare(ref, result(chart("AB", (100, 20))))["decade"], 1)

    def test_no_chart_counts_everything_wrong(self):
        ref = result(chart("AB", (10, 20)))
        none = InterpretationResult("val/a.png", "needs_review", None, (), ("a category label has no readable text",))
        self.assertEqual(compare(ref, none), {"built": False, "categories_wrong": 2, "values_wrong": 2, "decade": 0})


class FlagScoresTest(unittest.TestCase):
    def rec(self, wrong, flagged, reference_ok=True):
        return {"reference_ok": reference_ok, "wrong": wrong, "wrong_values": wrong, "flagged": flagged}

    def test_precision_recall_silent_and_false_flags(self):
        records = [self.rec(True, True), self.rec(True, False), self.rec(False, True), self.rec(False, False),
                   self.rec(True, False, reference_ok=False)]   # charts without a trusted reference are ignored
        s = flag_scores(records)
        self.assertEqual((s["reference_charts"], s["wrong"], s["flagged"], s["caught"]), (4, 2, 2, 1))
        self.assertEqual((s["silent_errors"], s["false_flags"]), (1, 1))
        self.assertEqual((s["precision"], s["recall"]), (0.5, 0.5))

    def test_nothing_wrong_and_nothing_flagged_is_perfect_recall(self):
        s = flag_scores([self.rec(False, False)])
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
