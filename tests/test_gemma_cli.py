"""Offline CLI artifact and failure tests; no inference packages are needed."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree
from unittest.mock import patch

from src.baseline import gemma
from src.baseline.vlm_response import parse_chart_response
from src.contract_io import load_bar_chart
from src.contracts import HorizontalBarChart, VerticalBarChart


FIXTURES = Path(__file__).parent / "fixtures" / "vlm"


class GemmaCLITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "new" / "run"
        self.image = self.root / "chart.png"
        self.raw = " \n" + (FIXTURES / "vertical_response.json").read_text() + "\r\n\t"
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.real_extract = gemma.GemmaChartExtractor.extract_result
        self.dependencies = patch.object(gemma, "_load_dependencies", side_effect=AssertionError("unexpected model load")).start()
        self.extract = patch.object(gemma.GemmaChartExtractor, "extract_result", autospec=True,
                                    side_effect=self.extract_fixture).start()
        self.addCleanup(patch.stopall)

    def extract_fixture(self, extractor, image, *, source_key=None):
        initial = json.loads((self.output / "run_metadata.json").read_text())
        self.assertEqual(initial["completion_status"], "started")
        parsed = parse_chart_response(self.raw, image=image, source_key=source_key)
        metadata = {**extractor._metadata(), "completion_status": "complete",
                    "resolved_device": "cpu", "resolved_dtype": "float32",
                    "resolved_revision": "commit", "inference_seconds": 0.1,
                    "placeholder_fields": list(parsed.placeholder_fields)}
        return gemma.GemmaExtractionResult(parsed.chart, self.raw, metadata)

    def run_cli(self, extra=()):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return gemma.main(["--image", str(self.image), "--output-dir", str(self.output), *extra])

    def metadata(self):
        return json.loads((self.output / "run_metadata.json").read_text())

    def test_success_both_orientations_and_flag_forms(self):
        for filename, kind in (("vertical_response.json", VerticalBarChart),
                               ("horizontal_response.json", HorizontalBarChart)):
            for flag in ([], ["--extract-only"]):
                with self.subTest(filename=filename, flag=flag):
                    self.output = self.root / (filename + str(bool(flag)))
                    self.raw = " \n" + (FIXTURES / filename).read_text() + "\r\n\t"
                    self.assertEqual(self.run_cli(flag), 0)
                    expected = {"chart.json", "raw_response.txt", "run_metadata.json"}
                    if not flag:
                        expected.add("chart.svg")
                        ElementTree.parse(self.output / "chart.svg")
                    self.assertEqual({p.name for p in self.output.iterdir()}, expected)
                    self.assertEqual((self.output / "raw_response.txt").read_bytes(), self.raw.encode())
                    chart = load_bar_chart(self.output / "chart.json")
                    self.assertIs(type(chart), kind)
                    self.assertEqual(chart.source_key, "vlm/chart.png")
                    metadata = self.metadata()
                    self.assertEqual(metadata["completion_status"], "complete")
                    self.assertEqual(metadata["rendering"]["status"], "skipped" if flag else "complete")
                    self.assertEqual(metadata["rendering"]["options"], {
                        "show_print": False, "show_violations": False,
                        "x_labels": "auto", "one_page": False, "strict": False})
                    if not flag:
                        page, = metadata["rendering"]["pages"]
                        self.assertTrue(page["saved"])
                        self.assertEqual(page["filename"], "chart.svg")
                        self.assertEqual(page["violations"], [])
                    self.assertEqual(metadata["image"], str(self.image))
                    self.assertEqual(metadata["source_key"], chart.source_key)
                    self.assertEqual(metadata["resolved_revision"], "commit")
                    self.assertEqual(metadata["inference_seconds"], 0.1)
                    self.assertEqual(metadata["placeholder_fields"],
                                     [] if kind is VerticalBarChart else ["title", "y_axis_title"])
        self.dependencies.assert_not_called()

    def test_defaults_and_all_overrides_forwarded(self):
        original = gemma.GemmaChartExtractor
        with patch.object(gemma, "GemmaChartExtractor", wraps=original) as constructor:
            self.assertEqual(self.run_cli(), 0)
            constructor.assert_called_once_with(model_id=gemma.DEFAULT_MODEL_ID, revision=None,
                                                device="auto", dtype="auto", max_new_tokens=4096)
        self.output = self.root / "overridden"
        with patch.object(gemma, "GemmaChartExtractor", wraps=original) as constructor:
            self.assertEqual(self.run_cli(["--model-id", "local/model", "--revision", "abc",
                                          "--device", "cuda", "--dtype", "bfloat16",
                                          "--max-new-tokens", "32", "--source-key", "eval/custom.png"]), 0)
            constructor.assert_called_once_with(model_id="local/model", revision="abc", device="cuda",
                                                dtype="bfloat16", max_new_tokens=32)
        self.assertEqual(self.extract.call_args.args[1], self.image)
        self.assertEqual(self.extract.call_args.kwargs, {"source_key": "eval/custom.png"})
        self.assertEqual(load_bar_chart(self.output / "chart.json").source_key, "eval/custom.png")

    def test_argument_errors_do_not_touch_output(self):
        for extra in (["--device", "gpu"], ["--dtype", "int4"], ["--max-new-tokens", "0"],
                      ["--max-new-tokens", "no"], ["--source-key", "invalid"],
                      ["--model-id", ""], ["--revision", " "], ["--x-labels", "invalid"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as caught:
                self.run_cli(extra)
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(self.output.exists())
        for argv in ([], ["--image", str(self.image)], ["--output-dir", str(self.output)]):
            with self.subTest(argv=argv), redirect_stderr(self.stderr), self.assertRaises(SystemExit) as caught:
                gemma.main(argv)
            self.assertEqual(caught.exception.code, 2)
        self.extract.assert_not_called()

    def test_existing_empty_directory(self):
        self.output.mkdir(parents=True)
        self.assertEqual(self.run_cli(), 0)

    def test_real_input_and_dependency_failures_write_diagnostics(self):
        self.extract.side_effect = self.real_extract
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.metadata()["error_stage"], "input")
        self.dependencies.assert_not_called()
        self.output = self.root / "dependencies"
        self.image.write_bytes(b"placeholder image")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.metadata()["error_stage"], "dependencies")
        self.assertFalse((self.output / "chart.json").exists())

    def test_existing_contents_untouched_and_extraction_not_started(self):
        for entry in ("previous.txt", ".hidden", "subdirectory"):
            with self.subTest(entry=entry):
                self.output = self.root / entry
                self.output.mkdir()
                sentinel = self.output / entry
                if entry == "subdirectory":
                    sentinel.mkdir()
                else:
                    sentinel.write_bytes(b"preserve me")
                self.assertEqual(self.run_cli(), 1)
                self.assertEqual(list(self.output.iterdir()), [sentinel])
                if sentinel.is_file():
                    self.assertEqual(sentinel.read_bytes(), b"preserve me")
        self.output = self.root / "file"
        self.output.write_bytes(b"preserve me")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.output.read_bytes(), b"preserve me")
        self.extract.assert_not_called()

    def test_directory_creation_failure_is_safe(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("secret-token")):
            self.assertEqual(self.run_cli(), 1)
        self.assertNotIn("secret-token", self.stderr.getvalue())
        self.extract.assert_not_called()

    def test_each_extraction_error_saves_available_diagnostics(self):
        renderer = patch.object(gemma, "_render_chart").start()
        for stage in ("input", "dependencies", "device", "loading", "inference", "parsing"):
            with self.subTest(stage=stage):
                self.output = self.root / stage
                raw = self.raw if stage == "parsing" else None
                self.extract.side_effect = gemma.GemmaExtractionError(
                    stage, "Safe failure message", gemma.GemmaChartExtractor()._metadata(), raw)
                self.assertEqual(self.run_cli(), 1)
                self.assertFalse((self.output / "chart.json").exists())
                self.assertEqual((self.output / "raw_response.txt").exists(), raw is not None)
                metadata = self.metadata()
                self.assertEqual(metadata["completion_status"], "failed")
                self.assertEqual(metadata["error_stage"], stage)
                self.assertEqual(metadata["source_key"], "vlm/chart.png")
                self.assertIsNone(metadata["inference_seconds"])
                self.assertIn(stage, self.stderr.getvalue())
        renderer.assert_not_called()

    def test_real_adapter_rejects_malformed_and_truncated_responses(self):
        def fail_from_adapter(extractor, image, *, source_key=None):
            try:
                parse_chart_response(self.raw, image=image, source_key=source_key)
            except ValueError:
                raise gemma.GemmaExtractionError("parsing", "Invalid response", extractor._metadata(), self.raw) from None
            self.fail("Expected invalid response")

        self.extract.side_effect = fail_from_adapter
        for index, response in enumerate(("", "prose " + self.raw, self.raw.rstrip()[:-3])):
            with self.subTest(response=response):
                self.output = self.root / str(index)
                self.raw = response
                self.assertEqual(self.run_cli(), 1)
                self.assertEqual((self.output / "raw_response.txt").read_bytes(), response.encode())
                self.assertFalse((self.output / "chart.json").exists())
                self.assertEqual(self.metadata()["error_stage"], "parsing")

    def test_unexpected_exception_does_not_leak_secret(self):
        self.extract.side_effect = RuntimeError("secret-token")
        self.assertEqual(self.run_cli(), 1)
        self.assertNotIn("secret-token", self.stderr.getvalue())
        self.assertNotIn("secret-token", json.dumps(self.metadata()))
        self.assertEqual(self.metadata()["completion_status"], "failed")

    def test_partial_chart_write_is_cleaned_up_and_marked_failed(self):
        def fail_chart(path, chart):
            path.write_text("partial data")
            raise OSError("secret-token")

        with patch.object(gemma, "write_bar_chart", side_effect=fail_chart):
            with patch.object(gemma, "_render_chart") as renderer:
                self.assertEqual(self.run_cli(), 1)
                renderer.assert_not_called()
        self.assertEqual({p.name for p in self.output.iterdir()}, {"raw_response.txt", "run_metadata.json"})
        self.assertEqual(self.metadata()["error_stage"], "output")
        self.assertNotIn("secret-token", self.stderr.getvalue())
        self.assertNotIn("completed", self.stdout.getvalue())

    def test_raw_write_failure_still_saves_failure_metadata(self):
        original = gemma._atomic_write

        def fail_raw(path, writer):
            if path.name == "raw_response.txt":
                raise OSError("secret-token")
            original(path, writer)

        with patch.object(gemma, "_atomic_write", side_effect=fail_raw):
            self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.metadata()["error_stage"], "output")
        self.assertFalse((self.output / "chart.json").exists())

    def test_initial_metadata_failure_stops_before_extraction(self):
        with patch.object(gemma, "_write_metadata", side_effect=OSError("secret-token")) as write:
            self.assertEqual(self.run_cli(), 1)
        write.assert_called_once()
        self.extract.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_final_metadata_failure_keeps_chart_and_started_record(self):
        original = gemma._write_metadata

        def fail_final(path, metadata):
            if metadata["completion_status"] != "started":
                raise OSError("secret-token")
            original(path, metadata)

        with patch.object(gemma, "_write_metadata", side_effect=fail_final) as write:
            self.assertEqual(self.run_cli(), 1)
        self.assertEqual(write.call_count, 2)
        self.assertEqual(self.metadata()["completion_status"], "started")
        self.assertIsInstance(load_bar_chart(self.output / "chart.json"), VerticalBarChart)
        self.assertTrue((self.output / "chart.svg").exists())
        self.assertIn("diagnostics are incomplete", self.stderr.getvalue())
        self.assertNotIn("secret-token", self.stderr.getvalue())

    def test_replace_failure_cleans_temporary_file(self):
        self.output.mkdir(parents=True)
        with patch.object(Path, "replace", side_effect=OSError("cannot replace")):
            with self.assertRaises(OSError):
                gemma._atomic_write(self.output / "raw_response.txt", lambda path: path.write_text("data"))
        self.assertEqual(list(self.output.iterdir()), [])

    def test_help_without_optional_dependencies(self):
        result = subprocess.run([sys.executable, "-S", "-m", "src.baseline.gemma", "--help"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--extract-only", result.stdout)
        self.assertIn("--show-print", result.stdout)
        self.assertIn("--strict", result.stdout)

    def test_extract_only_import_isolation(self):
        script = '''
import sys
from unittest.mock import patch
from src.baseline import gemma
from src.baseline.vlm_response import parse_chart_response
raw = open("tests/fixtures/vlm/vertical_response.json").read()
chart = parse_chart_response(raw, image="chart.png").chart
result = gemma.GemmaExtractionResult(chart, raw, {"completion_status": "complete"})
with patch.object(gemma.GemmaChartExtractor, "extract_result", return_value=result):
    assert gemma.main(["--image", "chart.png", "--output-dir", sys.argv[1],
                       "--extract-only", "--strict", "--show-print"]) == 0
assert not any(name == "src.tactile" or name.startswith("src.tactile.") for name in sys.modules)
'''
        subprocess.run([sys.executable, "-S", "-c", script, str(self.output)],
                       cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
        self.assertEqual(self.metadata()["rendering"]["status"], "skipped")

    def use_crowded_chart(self):
        self.raw = (FIXTURES.parent / "tactile" / "crowded.json").read_text()

    def test_multi_page_order_and_review_notes(self):
        self.use_crowded_chart()
        self.assertEqual(self.run_cli(), 0)
        pages = self.metadata()["rendering"]["pages"]
        self.assertGreater(len(pages), 1)
        self.assertEqual(pages[0]["page_kind"], "key")
        self.assertTrue(any(p["notes"] or p["needs_review"] for p in pages))
        for number, page in enumerate(pages, 1):
            self.assertEqual(page["page_number"], number)
            self.assertEqual(page["filename"], f"chart-{number}.svg")
            self.assertTrue(page["saved"])
            ElementTree.parse(self.output / page["filename"])

    def test_rule_findings_warn_or_fail_strict_after_saving(self):
        from src.tactile.rules import RULES, Violation
        violation = Violation(RULES["bar_width"], "Test width violation", ("bar-1",))
        for strict in (False, True):
            with self.subTest(strict=strict):
                self.output = self.root / f"strict-{strict}"
                with patch("src.tactile.rules.check_layout", return_value=[violation]):
                    self.assertEqual(self.run_cli(["--strict"] if strict else []), int(strict))
                metadata = self.metadata()
                self.assertEqual(metadata["completion_status"], "failed" if strict else "complete")
                self.assertEqual(metadata["rendering"]["status"], "complete")
                page, = metadata["rendering"]["pages"]
                self.assertTrue(page["saved"])
                self.assertEqual(page["violations"][0]["rule_id"], "bar_width")
                self.assertEqual(page["violations"][0]["element_ids"], ["bar-1"])
                if strict:
                    self.assertEqual(metadata["error_stage"], "validation")

    def test_renderer_flags_and_orientation_dispatch(self):
        from src.tactile import layout, layout_hbar, svg
        for filename, module, single, multiple, label_key in (
            ("vertical_response.json", layout, "layout_vertical_bar_chart", "layout_vertical_bar_chart_pages", "x_labels"),
            ("horizontal_response.json", layout_hbar, "layout_horizontal_bar_chart", "layout_horizontal_bar_chart_pages", "labels"),
        ):
            for one_page in (False, True):
                self.output = self.root / f"{filename}-{one_page}"
                self.raw = (FIXTURES / filename).read_text()
                target = single if one_page else multiple
                with patch.object(module, target, wraps=getattr(module, target)) as build, \
                        patch.object(svg, "render_svg", wraps=svg.render_svg) as render:
                    flags = ["--show-print", "--show-violations", "--x-labels", "number"]
                    if one_page:
                        flags.append("--one-page")
                    self.assertEqual(self.run_cli(flags), 0)
                self.assertEqual(build.call_args.kwargs, {label_key: "number"})
                self.assertTrue(render.call_args.kwargs["show_print"])
                self.assertIsNotNone(render.call_args.kwargs["violations"])

    def test_notes_and_review_alone_do_not_fail_strict(self):
        from src.tactile import layout
        original = layout.layout_vertical_bar_chart_pages

        def reviewed(chart, **kwargs):
            return [replace(p, notes=("Review note",), needs_review=("Review reason",))
                    for p in original(chart, **kwargs)]

        # Isolate the CLI policy: the real checker normally translates review
        # reasons into violations, which strict mode must still respect.
        with patch.object(layout, "layout_vertical_bar_chart_pages", side_effect=reviewed), \
                patch("src.tactile.rules.check_layout", return_value=[]):
            self.assertEqual(self.run_cli(["--strict"]), 0)
        page, = self.metadata()["rendering"]["pages"]
        self.assertEqual(page["notes"], ["Review note"])
        self.assertEqual(page["needs_review"], ["Review reason"])
        self.assertIn("NEEDS REVIEW: Review reason", self.stdout.getvalue())

    def test_unsupported_charts_keep_extraction(self):
        base = json.loads(self.raw)
        negative = {**base, "series": [{"name": None, "values": [-1] * len(base["categories"])}]}
        many = {**base, "series": [{"name": f"Series {i}", "values": [1] * len(base["categories"])} for i in range(6)]}
        for index, data in enumerate((negative, many, {**base, "title": "Unsupported \u2603"})):
            self.output = self.root / f"unsupported-{index}"
            self.raw = json.dumps(data)
            self.assertEqual(self.run_cli(), 1)
            self.assertEqual(self.metadata()["error_stage"], "rendering")
            self.assertEqual(self.metadata()["rendering"]["status"], "failed")
            self.assertEqual(json.loads((self.output / "chart.json").read_text())["series"], data["series"])
            self.assertTrue((self.output / "raw_response.txt").exists())
            self.assertFalse(list(self.output.glob("*.svg")))

    def test_rendering_failure_is_sanitized(self):
        for target in ("src.tactile.layout.layout_vertical_bar_chart_pages",
                       "src.tactile.rules.check_layout", "src.tactile.svg.render_svg"):
            self.output = self.root / target
            with patch(target, side_effect=RuntimeError("secret-token")):
                self.assertEqual(self.run_cli(), 1)
            self.assertEqual(self.metadata()["error_stage"], "rendering")
            self.assertTrue((self.output / "chart.json").exists())
            self.assertNotIn("secret-token", json.dumps(self.metadata()))
            self.assertNotIn("secret-token", self.stderr.getvalue())

    def test_later_page_serialization_failure_publishes_no_svgs(self):
        from src.tactile import svg
        self.use_crowded_chart()
        original = svg.render_svg
        calls = 0

        def fail_second(layout, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("secret-token")
            return original(layout, **kwargs)

        with patch.object(svg, "render_svg", side_effect=fail_second):
            self.assertEqual(self.run_cli(), 1)
        self.assertEqual(calls, 2)
        self.assertFalse(list(self.output.glob("*.svg")))
        self.assertEqual(self.metadata()["error_stage"], "rendering")
        self.assertEqual(self.metadata()["rendering"]["pages"], [])

    def test_renderer_import_failure_retains_extraction(self):
        import builtins
        original = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if name.startswith("src.tactile"):
                raise ImportError("secret-token")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_import):
            self.assertEqual(self.run_cli(), 1)
        self.assertTrue((self.output / "chart.json").exists())
        self.assertEqual(self.metadata()["error_stage"], "rendering")
        self.assertNotIn("secret-token", self.stderr.getvalue())

    def test_page_write_failures_track_saved_pages_and_override_strict(self):
        self.use_crowded_chart()
        original = gemma._atomic_write
        for failed_number in (1, 2):
            self.output = self.root / f"write-{failed_number}"

            def fail_page(path, writer):
                if path.name == f"chart-{failed_number}.svg":
                    def partial(pending):
                        pending.write_text("partial")
                        raise OSError("secret-token")
                    return original(path, partial)
                return original(path, writer)

            with patch.object(gemma, "_atomic_write", side_effect=fail_page):
                self.assertEqual(self.run_cli(["--strict"]), 1)
            metadata = self.metadata()
            self.assertEqual(metadata["error_stage"], "output")
            self.assertEqual(metadata["rendering"]["status"], "failed")
            for page in metadata["rendering"]["pages"]:
                self.assertEqual(page["saved"], page["page_number"] < failed_number)
                self.assertEqual((self.output / page["filename"]).exists(), page["saved"])
            self.assertFalse(list(self.output.glob("*.tmp")))
            self.assertNotIn("secret-token", self.stderr.getvalue())
