"""Category re-read fallback with mocked OCR and synthetic crops."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.contracts import Box, Detection, DetectionFrame, SeriesSpec, VerticalBarSpec, YAxisSpec
from src.tactile import category_reread, pipeline
from src.tactile.category_reread import reread_categories, tesseract_rotated
from src.tactile.completeness import (
    NEEDS_INPUT, READY, Overrides, apply_overrides, check_completeness,
)

W, H = 400, 200
NAMES = ["Germany", "Greece", "|", "India"]


def make_spec(categories=None, **changes):
    cats = list(categories or NAMES)
    fields = dict(
        source_key="input/a.png", title="Sales", x_axis_title="Year", y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]), categories=cats,
        series=[SeriesSpec(None, "#112233", [1.0] * len(cats))], bars_adjacent=False,
        flags=[], confidence={})
    fields.update(changes)
    return VerticalBarSpec(**fields)


def unreadable_spec(categories=None, bad=(2,)):
    return make_spec(categories, flags=["category_unreadable"],
                     confidence={f"category[{j}]": 0.1 for j in bad})


def make_frame(count=4):
    items = [Detection("plot", "plot_area", Box(20, 10, 390, 150), 0.9)]
    for j in range(count):
        x = 40 + 90 * j
        items.append(Detection(f"x{j}", "x_tick_label", Box(x, 160, x + 40, 195), 0.9))
    return DetectionFrame("input/a.png", W, H, tuple(items))


IMAGE = np.full((H, W, 3), 255, np.uint8)


class Reread(unittest.TestCase):
    def test_lone_symbol_is_replaced_and_marked(self):
        spec = reread_categories(unreadable_spec(), make_frame(), IMAGE,
                                 reader=lambda image, detection: ("Fiji", 0.9))
        self.assertEqual(spec.categories, ["Germany", "Greece", "Fiji", "India"])
        self.assertIn("category_reread:1", spec.flags)
        self.assertIn("category_reread_index:2", spec.flags)
        self.assertNotIn("category_unreadable", spec.flags)
        self.assertNotIn("category[2]", spec.confidence)
        self.assertEqual(check_completeness(spec).status, READY)
        self.assertTrue(any("1 category label(s) were re-read" in n for n in check_completeness(spec).notes))

    def test_the_reader_gets_the_detection_for_that_category(self):
        seen = []
        reread_categories(unreadable_spec(), make_frame(), IMAGE,
                          reader=lambda image, detection: seen.append(detection.id))
        self.assertEqual(seen, ["x2"])

    def test_single_character_among_long_names_is_replaced(self):
        spec = make_spec(["Germany", "Greece", "i", "India"])
        fixed = reread_categories(spec, make_frame(), IMAGE, reader=lambda *_: ("Hungary", 0.8))
        self.assertEqual(fixed.categories[2], "Hungary")
        self.assertEqual(check_completeness(fixed).status, READY)

    def test_a_result_that_still_fails_the_rule_is_discarded(self):
        for bad in ("|", "i", " ", "§"):
            with self.subTest(bad=bad):
                spec = reread_categories(unreadable_spec(), make_frame(), IMAGE,
                                         reader=lambda *_, bad=bad: (bad, 0.9))
                self.assertEqual(spec.categories, NAMES)
                self.assertFalse(any(f.startswith("category_reread") for f in spec.flags))
                self.assertEqual(check_completeness(spec).status, NEEDS_INPUT)

    def test_no_result_leaves_the_label_for_the_user(self):
        spec = reread_categories(unreadable_spec(), make_frame(), IMAGE, reader=lambda *_: None)
        self.assertEqual(check_completeness(spec).status, NEEDS_INPUT)

    def test_partial_success_keeps_the_other_label_unreadable(self):
        cats = ["Germany", "|", "|", "India"]
        answers = {"x1": ("Greece", 0.9), "x2": None}
        spec = reread_categories(unreadable_spec(cats, bad=(1, 2)), make_frame(), IMAGE,
                                 reader=lambda image, detection: answers[detection.id])
        self.assertEqual(spec.categories, ["Germany", "Greece", "|", "India"])
        self.assertIn("category_reread:1", spec.flags)
        self.assertIn("category_unreadable", spec.flags)
        self.assertEqual([m.field for m in check_completeness(spec).missing], ["category[2]"])

    def test_nothing_flagged_means_the_reader_is_never_called(self):
        spec = make_spec(["Germany", "Greece", "Spain", "India"])
        called = mock.Mock()
        self.assertIs(reread_categories(spec, make_frame(), IMAGE, reader=called), spec)
        called.assert_not_called()

    def test_a_correct_single_character_category_in_a_short_chart_is_not_touched(self):
        spec = make_spec(["A", "B", "C", "D"])
        self.assertIs(reread_categories(spec, make_frame(), IMAGE, reader=mock.Mock()), spec)

    def test_label_count_mismatch_is_left_alone(self):
        spec = unreadable_spec()
        self.assertIs(reread_categories(spec, make_frame(3), IMAGE, reader=mock.Mock()), spec)

    def test_supplying_the_text_drops_the_reread_marker(self):
        spec = reread_categories(unreadable_spec(), make_frame(), IMAGE, reader=lambda *_: ("Fiji", 0.9))
        fixed, _ = apply_overrides(spec, Overrides(categories={2: "Fiji Islands"}))
        self.assertFalse(any(f.startswith("category_reread") for f in fixed.flags))


class DefaultReader(unittest.TestCase):
    """``tesseract_rotated`` with a fake pytesseract: it takes the higher-confidence candidate."""

    def fake_tesseract(self, replies):
        calls = []

        def image_to_data(candidate, **kwargs):
            calls.append((candidate.shape, kwargs["config"]))
            text, conf = replies[len(calls) - 1]
            return {"text": [text], "conf": [conf]}

        module = mock.Mock()
        module.image_to_data = image_to_data
        module.Output.DICT = "dict"
        return module, calls

    def test_takes_the_highest_confidence_of_four_rotated_candidates(self):
        module, calls = self.fake_tesseract([("a", 40), ("Fiji", 91), ("b", 55), ("", -1)])
        detection = Detection("x0", "x_tick_label", Box(10, 10, 60, 50), 0.9)
        with mock.patch.object(category_reread, "_load_pytesseract", return_value=module):
            result = tesseract_rotated(IMAGE, detection)
        self.assertEqual(result, ("Fiji", 0.91))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(config == "--psm 7" for _, config in calls))
        self.assertTrue(all(h > w for (h, w), _ in calls))   # a wide crop is turned portrait

    def test_nothing_read_returns_none(self):
        module, _ = self.fake_tesseract([("", -1)] * 4)
        detection = Detection("x0", "x_tick_label", Box(10, 10, 60, 50), 0.9)
        with mock.patch.object(category_reread, "_load_pytesseract", return_value=module):
            self.assertIsNone(tesseract_rotated(IMAGE, detection))


class PipelineSummary(unittest.TestCase):
    def test_confirm_step_lists_the_reread_labels(self):
        spec = reread_categories(unreadable_spec(), make_frame(), IMAGE, reader=lambda *_: ("Fiji", 0.9))
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--image", "a.png", "--output", str(Path(tmp) / "o.svg"), "--yes"])
            out, err = io.StringIO(), io.StringIO()
            code = pipeline.run(args, analyse_fn=lambda *_: (spec, None, None),
                                stdin_is_tty=False, out=out, err=err)
        self.assertEqual(code, 0)
        self.assertIn("re-read by fallback: 2: 'Fiji'", out.getvalue())
        self.assertIn("1 category label(s) were re-read", out.getvalue())

    def test_no_line_when_nothing_was_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--image", "a.png", "--output", str(Path(tmp) / "o.svg"), "--yes"])
            out = io.StringIO()
            pipeline.run(args, analyse_fn=lambda *_: (make_spec(["Germany", "Greece", "Spain", "India"]), None, None),
                         stdin_is_tty=False, out=out, err=io.StringIO())
        self.assertNotIn("re-read by fallback", out.getvalue())


if __name__ == "__main__":
    unittest.main()
