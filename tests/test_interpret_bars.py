"""Stage 1 interpreter on small synthetic charts (ground-truth style frames)."""

import unittest

from src.contracts import Box, Detection, DetectionFrame, OCRFrame, RecognizedText
from src.interpret_bars import interpret_vertical_bars

W, H = 400, 300
BOTTOM = 250.0
PX_PER_UNIT = 2.2   # value axis 0..100 spans 220 px


def y_of(value, zero=BOTTOM, px=PX_PER_UNIT):
    return zero - px * value


class Chart:
    """Builds a frame and its OCR reads; tests tweak the lists before interpreting."""

    def __init__(self, values=(30, 60, 90, 45), ticks=(0, 20, 40, 60, 80, 100), zero=BOTTOM):
        self.dets, self.reads, self.n = [], {}, 0
        self.zero = zero
        self.pitch = 300 / max(len(values), 1)
        for i, v in enumerate(values):
            cx = 80 + self.pitch * (i + 0.5)
            half = 0.25 * self.pitch
            top, bottom = sorted((y_of(v, zero), zero))
            self.add("vertical_bar", cx - half, top, cx + half, bottom)
            self.add("x_tick", cx - 1, 252, cx + 1, 256)
            self.add("x_tick_label", cx - 0.45 * self.pitch, 258, cx + 0.45 * self.pitch, 272, text="ABCDEFGHIJK"[i])
        for t in ticks:
            y = y_of(t, zero)
            self.add("y_tick", 45, y - 1, 49, y + 1)
            self.add("y_tick_label", 10, y - 7, 40, y + 7, text=str(t))
        self.add("plot_area", 50, y_of(max(ticks), zero) - 10, 390, zero + 0.0)

    def add(self, role, x1, y1, x2, y2, text=None, confidence=1.0):
        self.n += 1
        det = Detection(f"d{self.n}", role, Box(x1, y1, x2, y2), 1.0)
        self.dets.append(det)
        if text is not None:
            self.reads[det.id] = RecognizedText(det.id, text, confidence)
        return det

    def frames(self):
        return (DetectionFrame("val/s.png", W, H, tuple(self.dets)),
                OCRFrame("val/s.png", tuple(self.reads.values())))

    def run(self, **kwargs):
        return interpret_vertical_bars(*self.frames(), **kwargs)

    def set_text(self, role, index, text, confidence=1.0):
        det = [d for d in self.dets if d.role == role][index]
        self.reads[det.id] = RecognizedText(det.id, text, confidence)


class InterpretVerticalBarsTest(unittest.TestCase):
    def test_clean_chart_is_accepted_with_values_and_axis(self):
        result = Chart().run()
        self.assertEqual(result.status, "accepted", result.reasons)
        chart = result.chart
        self.assertEqual(chart.categories, ("A", "B", "C", "D"))
        self.assertEqual(chart.series[0].values, (30.0, 60.0, 90.0, 45.0))
        self.assertEqual(chart.y_tick_values, (0.0, 20.0, 40.0, 60.0, 80.0, 100.0))
        self.assertEqual((chart.y_min, chart.y_max), (0.0, 104.5))  # plot area reaches past the top tick
        self.assertEqual((chart.title, chart.x_axis_title, chart.y_axis_title), (None, None, None))
        self.assertIsNone(chart.series[0].name)
        bar_items = [i for i in result.items if i.kind == "bar"]
        self.assertEqual([i.value for i in bar_items], [30.0, 60.0, 90.0, 45.0])
        self.assertAlmostEqual(bar_items[0].uncertainty, 0.5 / PX_PER_UNIT)

    def test_titles_come_from_reads_and_missing_ones_stay_none(self):
        c = Chart()
        c.add("chart_title", 100, 2, 300, 18, text="Cases by\\nstage")
        c.add("y_axis_title", 0, 100, 8, 200, text="Cases")
        chart = c.run().chart
        self.assertEqual(chart.title, "Cases by stage")
        self.assertEqual(chart.y_axis_title, "Cases")
        self.assertIsNone(chart.x_axis_title)

    def test_bar_and_label_count_mismatch_needs_review_without_a_chart(self):
        c = Chart()
        gone = [x for x in c.dets if x.role == "x_tick_label"][0].id
        c.dets = [d for d in c.dets if d.id != gone]
        del c.reads[gone]
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.chart)
        self.assertIn("4 bars but 3 category labels", result.reasons)

    def test_out_of_scope_charts_are_rejected(self):
        many = Chart(values=tuple([10] * 11)).run()
        self.assertEqual(many.status, "rejected")
        self.assertTrue(many.reasons[0].startswith("out of scope"))
        c = Chart()
        c.add("legend_swatch", 300, 10, 310, 20)
        c.add("legend_swatch", 300, 30, 310, 40)
        self.assertEqual(c.run().status, "rejected")
        empty = Chart(values=())
        self.assertEqual(empty.run().status, "rejected")

    def test_one_wrong_tick_read_is_flagged_but_the_chart_is_still_right(self):
        c = Chart()
        c.set_text("y_tick_label", 3, "600")  # 60 misread
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.chart.series[0].values, (30.0, 60.0, 90.0, 45.0))
        flagged = [i for i in result.items if i.flagged]
        self.assertEqual([(i.kind, i.text) for i in flagged], [("y_tick_label", "600")])

    def test_unreadable_tick_text_is_flagged_not_guessed(self):
        c = Chart()
        c.set_text("y_tick_label", 1, "2o")
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertIn(("y_tick_label", "not a readable number"), [(i.kind, i.reason) for i in result.items if i.flagged])

    def test_too_few_parsed_ticks_leaves_no_chart(self):
        c = Chart(ticks=(0, 50, 100))
        for i in (0, 1):
            c.set_text("y_tick_label", i, "?")
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.chart)

    def test_mixed_percent_suffix_is_flagged(self):
        c = Chart()
        for i, t in enumerate((0, 20, 40, 60, 80, 100)):
            c.set_text("y_tick_label", i, f"{t}%" if i % 2 == 0 else str(t))
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertTrue(any("%" in r for r in result.reasons))
        self.assertIsNotNone(result.chart)

    def test_consistent_percent_suffix_is_fine(self):
        c = Chart()
        for i, t in enumerate((0, 20, 40, 60, 80, 100)):
            c.set_text("y_tick_label", i, f"{t}%")
        self.assertEqual(c.run().status, "accepted")

    def test_empty_category_read_needs_review_without_a_chart(self):
        c = Chart()
        c.set_text("x_tick_label", 2, "  ")
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertIsNone(result.chart)
        self.assertTrue(any(i.kind == "category" and i.flagged and i.index == 2 for i in result.items))

    def test_low_confidence_is_flagged_only_when_a_threshold_is_given(self):
        c = Chart()
        c.set_text("x_tick_label", 1, "B", confidence=0.4)
        self.assertEqual(c.run().status, "accepted")
        result = c.run(min_confidence=0.5)
        self.assertEqual(result.status, "needs_review")
        self.assertIsNotNone(result.chart)

    def test_value_labels_that_disagree_with_the_axis_are_flagged(self):
        agree, disagree = Chart(), Chart()
        for chart, texts in ((agree, ("30", "60", "90", "45")), (disagree, ("300", "600", "900", "450"))):
            for i, v in enumerate((30, 60, 90, 45)):
                cx = 80 + chart.pitch * (i + 0.5)
                chart.add("value_label", cx - 15, y_of(v) - 14, cx + 15, y_of(v) - 2, text=texts[i])
        self.assertEqual(agree.run().status, "accepted")
        result = disagree.run()
        self.assertEqual(result.status, "needs_review")
        self.assertTrue(any("printed on the bars" in r for r in result.reasons))

    def test_negative_bars_read_from_the_bottom_edge(self):
        zero = 150.0
        c = Chart(values=(40, -30, 20), ticks=(-50, 0, 50), zero=zero)
        result = c.run()
        self.assertEqual(result.status, "accepted", result.reasons)
        self.assertEqual(result.chart.series[0].values, (40.0, -30.0, 20.0))

    def test_missing_tick_marks_fall_back_to_label_centres_and_say_so(self):
        c = Chart()
        c.dets = [d for d in c.dets if d.role != "y_tick"]
        result = c.run()
        self.assertEqual(result.status, "needs_review")
        self.assertIn("tick marks missing or not one per label; using label centres", result.reasons)
        self.assertEqual(result.chart.series[0].values, (30.0, 60.0, 90.0, 45.0))

    def test_bars_without_a_shared_baseline_are_flagged(self):
        c = Chart()
        bar = [d for d in c.dets if d.role == "vertical_bar"][1]
        c.dets[c.dets.index(bar)] = Detection(bar.id, "vertical_bar", Box(bar.box.x1, 100, bar.box.x2, 160), 1.0)
        result = c.run()
        self.assertIn("bars do not share a baseline (stacked, floating or mixed bars)", result.reasons)

    def test_single_legend_entry_names_the_series(self):
        c = Chart()
        c.add("legend_swatch", 300, 10, 310, 20)
        c.add("legend_label", 315, 10, 380, 20, text="Cases")
        self.assertEqual(c.run().chart.series[0].name, "Cases")

    def test_mismatched_frames_raise(self):
        det, ocr = Chart().frames()
        with self.assertRaisesRegex(ValueError, "different source images"):
            interpret_vertical_bars(det, OCRFrame("test/other.png", ocr.regions))


if __name__ == "__main__":
    unittest.main()
