"""Pipeline control flow with a fake analysis step; no detector or OCR runs."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.contracts import ChartInterpretationError, SeriesSpec, VerticalBarSpec, YAxisSpec
from src.tactile import pipeline
from src.tactile.rules import RULES, Violation


def make_spec(**changes):
    fields = dict(
        source_key="input/a.png", title="Sales", x_axis_title="Year", y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]), categories=["2001", "2002", "2003"],
        series=[SeriesSpec(None, "#112233", [10.0, 20.0, 30.0])],
        bars_adjacent=False, flags=[], confidence={})
    fields.update(changes)
    return VerticalBarSpec(**fields)


class Pipeline(unittest.TestCase):
    def go(self, spec=None, error=None, argv=(), tty=False, answers=()):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(out_path), *argv])
            replies = iter(answers)
            out, err = io.StringIO(), io.StringIO()
            code = pipeline.run(
                args, analyse_fn=lambda *_: (spec, error, None),
                input_fn=lambda prompt: next(replies), stdin_is_tty=tty, out=out, err=err)
            return code, out.getvalue(), err.getvalue(), out_path.exists()

    def test_ready_chart_writes_svg(self):
        code, out, err, written = self.go(make_spec(), argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("0: 2001", out)

    def test_analyse_passes_legend_mapping_to_reconstruction(self):
        image = mock.sentinel.image
        frame = mock.sentinel.frame
        ocr = mock.sentinel.ocr
        mapping = mock.sentinel.mapping
        reconstructed = make_spec()
        backend = mock.sentinel.backend

        with (
            mock.patch("cv2.imread", return_value=image),
            mock.patch("src.detection.load_model", return_value=mock.sentinel.model),
            mock.patch("src.detection.run_detection", return_value=frame),
            mock.patch("src.ocr.run_ocr", return_value=ocr),
            mock.patch("src.tesseract_ocr.TesseractBackend", return_value=backend),
            mock.patch(
                "src.legend_matcher.match_legend_to_bars", return_value=mapping
            ) as matcher,
            mock.patch(
                "src.reconstruct.reconstruct_vbar", return_value=reconstructed
            ) as reconstruct,
            mock.patch.object(pipeline, "cut_off_flags", return_value=[]),
            mock.patch("src.reconstruct_hbar.bar_orientation", return_value="vertical"),
        ):
            result = pipeline.analyse(Path("chart.png"), Path("weights.pt"))

        matcher.assert_called_once_with(image, frame, ocr)
        reconstruct.assert_called_once_with(
            frame, ocr, image, legend_mapping=mapping
        )
        self.assertEqual(result, (reconstructed, None, frame))

    def test_unconfirmed_non_tty_warns_and_continues(self):
        code, _, err, written = self.go(make_spec())
        self.assertEqual((code, written), (0, True))
        self.assertIn("not confirmed", err)

    def test_tty_decline_writes_nothing(self):
        code, _, _, written = self.go(make_spec(), tty=True, answers=["n"])
        self.assertEqual((code, written), (4, False))

    def test_missing_text_non_tty_exits_2_with_flags(self):
        code, out, _, written = self.go(
            make_spec(title=None, flags=["title_missing"]), argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)

    def test_flags_supply_missing_text(self):
        code, out, _, written = self.go(
            make_spec(title=None, flags=["title_missing"]), argv=["--yes", "--title", "T"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("Supplied by user: chart title supplied by user", out)

    def test_tty_prompts_and_skip_stops(self):
        spec = make_spec(title=None, x_axis_title=None,
                         flags=["title_missing", "x_axis_title_missing"])
        code, out, _, written = self.go(spec, tty=True, answers=["My title", ""])
        self.assertEqual((code, written), (2, False))
        self.assertIn("Still missing: x_axis_title", out)

    def test_tty_prompt_answers_build(self):
        code, out, _, written = self.go(
            make_spec(title=None, flags=["title_missing"]), tty=True, answers=["My title", "y"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("Supplied by user: chart title", out)

    def test_cannot_build_and_force(self):
        spec = make_spec(flags=["category_ambiguous"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (3, False))
        self.assertIn("--force", out)
        code, _, err, written = self.go(spec, argv=["--yes", "--force"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("WARNING: --force", err)

    def test_force_never_skips_non_ambiguity_flags(self):
        code, _, _, written = self.go(make_spec(flags=["baseline_outside_plot"]),
                                      argv=["--yes", "--force"])
        self.assertEqual((code, written), (3, False))

    def test_interpretation_error_exits_3(self):
        code, out, _, _ = self.go(error=ChartInterpretationError("no x tick labels detected"))
        self.assertEqual(code, 3)
        self.assertIn("No x tick labels", out)

    def test_unencodable_name_exits_2_and_flag_fixes_it(self):
        spec = make_spec(series=[SeriesSpec("Chile", "#112233", [1.0, 2.0, 3.0]),
                                 SeriesSpec("E] Salvador", "#445566", [3.0, 2.0, 1.0])])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--series 1=NAME", out)
        code, _, _, written = self.go(spec, argv=["--yes", "--series", "1=El Salvador"])
        self.assertEqual((code, written), (0, True))

    def test_tty_answer_that_cannot_be_encoded_stops(self):
        code, out, _, written = self.go(
            make_spec(title=None, flags=["title_missing"]), tty=True, answers=["Bad]"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)

    def many(self):
        n = 30
        return make_spec(categories=[f"Category number {i}" for i in range(n)],
                         series=[SeriesSpec(None, "#112233", [float(i + 1) for i in range(n)])])

    def test_overview_page_is_reported_for_a_transcriber_and_never_fails_strict(self):
        code, out, err, _ = self.go(self.many(), argv=["--yes", "--strict"])
        self.assertEqual(code, 0)  # divided into several files, so no single out.svg
        self.assertEqual(out.count("Needs a transcriber: overview page (BANA 3.9.1) not generated"), 1)
        self.assertNotIn("overview_first", out)

    def test_strict_exits_1_on_a_real_violation_default_exits_0(self):
        found = [Violation(RULES["bar_width"], "too narrow", ("bar-0-0",))]
        with mock.patch.object(pipeline, "check_layout", return_value=found):
            code, _, err, written = self.go(make_spec(), argv=["--yes", "--strict"])
            self.assertEqual((code, written), (1, True))
            self.assertIn("--strict: 1 BANA violation", err)
            code, out, _, _ = self.go(make_spec(), argv=["--yes"])
            self.assertEqual(code, 0)
            self.assertIn("too narrow", out)

    def test_tiny_negative_is_zeroed_with_a_note_but_a_real_one_is_rejected(self):
        tiny = make_spec(series=[SeriesSpec(None, "#112233", [-0.1, 20.0, 30.0])])
        code, out, _, written = self.go(tiny, argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("treated as 0", out)
        real = make_spec(series=[SeriesSpec(None, "#112233", [-10.0, 20.0, 30.0])])
        code, _, err, written = self.go(real, argv=["--yes"])
        self.assertEqual((code, written), (1, False))
        self.assertIn("negative bar values are not supported", err)

    def test_suspicious_category_asks_and_flag_fixes_it(self):
        spec = make_spec(categories=["Germany", "i", "Greece"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--category 1=TEXT", out)
        code, out, _, written = self.go(spec, argv=["--yes", "--category", "1=Hungary"])
        self.assertEqual((code, written), (0, True))

    def test_cut_off_title_asks_and_confirming_it_builds(self):
        spec = make_spec(flags=["title_cut_off"])
        code, out, _, written = self.go(spec, argv=["--yes"])
        self.assertEqual((code, written), (2, False))
        self.assertIn("--title", out)
        self.assertIn("cut off at the image edge", out)
        code, out, _, written = self.go(spec, argv=["--yes", "--title", "Sales"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("chart title supplied by user", out)

    def test_tty_confirming_a_cut_off_y_title(self):
        spec = make_spec(flags=["y_axis_title_cut_off"])
        code, _, _, written = self.go(spec, tty=True, answers=["Revenue", "y"])
        self.assertEqual((code, written), (0, True))

    def test_uneven_y_ticks_build_with_a_note(self):
        code, out, _, written = self.go(make_spec(flags=["y_ticks_not_arithmetic"]), argv=["--yes"])
        self.assertEqual((code, written), (0, True))
        self.assertIn("not evenly spaced", out)

    def test_leading_zero_tick_exits_3_even_with_force(self):
        spec = make_spec(flags=["y_tick_leading_zero"])
        for extra in ([], ["--force"]):
            code, out, _, written = self.go(spec, argv=["--yes", *extra])
            self.assertEqual((code, written), (3, False))
            self.assertIn("decimal point may have been lost", out)

    def test_single_series_key_follows_the_legend(self):
        def svg(series):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "out.svg"
                args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
                code = pipeline.run(args, analyse_fn=lambda *_: (make_spec(series=series), None, None),
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
        image, frame, ocr = mock.sentinel.image, mock.sentinel.frame, mock.sentinel.ocr
        reconstructed = make_spec(orientation="horizontal")
        with (
            mock.patch("cv2.imread", return_value=image),
            mock.patch("src.detection.load_model", return_value=mock.sentinel.model),
            mock.patch("src.detection.run_detection", return_value=frame),
            mock.patch("src.ocr.run_ocr", return_value=ocr),
            mock.patch("src.tesseract_ocr.TesseractBackend"),
            mock.patch("src.legend_matcher.match_legend_to_bars") as vertical_matcher,
            mock.patch("src.reconstruct_hbar.bar_orientation", return_value="horizontal"),
            mock.patch("src.reconstruct_hbar.reconstruct_hbar",
                       return_value=reconstructed) as reconstruct,
            mock.patch.object(pipeline, "cut_off_flags", return_value=[]),
        ):
            result = pipeline.analyse(Path("chart.png"), Path("weights.pt"))
        reconstruct.assert_called_once_with(frame, ocr, image)
        vertical_matcher.assert_not_called()
        self.assertEqual(result, (reconstructed, None, frame))

    def test_horizontal_spec_gets_the_horizontal_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
            out = io.StringIO()
            code = pipeline.run(
                args, analyse_fn=lambda *_: (make_spec(orientation="horizontal"), None, None),
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

    def test_bad_flag_syntax(self):
        code, _, err, _ = self.go(make_spec(), argv=["--category", "oops"])
        self.assertEqual(code, 1)
        self.assertIn("INDEX=TEXT", err)


if __name__ == "__main__":
    unittest.main()

    def test_series_with_no_visible_bars_are_keyed_noted_and_do_not_stop_the_build(self):
        series = [SeriesSpec("Low & middle income", "#aa0000", [None, None, None], no_visible_bars=True),
                  SeriesSpec("Kenya", "#112233", [10.0, 20.0, 30.0]),
                  SeriesSpec("Vietnam", "#00aa00", [None, None, None], no_visible_bars=True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.svg"
            args = pipeline.parse_args(["--image", "a.png", "--output", str(path), "--yes"])
            out, err = io.StringIO(), io.StringIO()
            code = pipeline.run(args, analyse_fn=lambda *_: (make_spec(series=series), None, None),
                                input_fn=input, stdin_is_tty=False, out=out, err=err)
            svg = path.read_text()
        self.assertEqual(code, 0)
        shown = out.getvalue()
        self.assertIn("0: Low & middle income (no visible bars)", shown)
        self.assertIn("1: Kenya\n", shown)
        self.assertIn("2: Vietnam (no visible bars)", shown)
        self.assertIn("Note: No bars are visible for: Low & middle income, Vietnam", shown)
        for sample in ("key-sample-0", "key-sample-1", "key-sample-2"):
            self.assertIn(sample, svg)
        self.assertEqual(svg.count('id="bar-'), 3)           # Kenya's three bars only

