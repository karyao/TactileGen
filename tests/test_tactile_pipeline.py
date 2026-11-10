"""Pipeline control flow with a fake analysis step; no detector or OCR runs."""

import io
import tempfile
import unittest
from pathlib import Path

from src.contracts import ChartInterpretationError, SeriesSpec, VerticalBarSpec, YAxisSpec
from src.tactile import pipeline


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

    def test_bad_flag_syntax(self):
        code, _, err, _ = self.go(make_spec(), argv=["--category", "oops"])
        self.assertEqual(code, 1)
        self.assertIn("INDEX=TEXT", err)


if __name__ == "__main__":
    unittest.main()
