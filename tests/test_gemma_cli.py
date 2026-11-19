"""Offline CLI artifact and failure tests; no inference packages are needed."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
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
                    self.assertEqual({p.name for p in self.output.iterdir()},
                                     {"chart.json", "raw_response.txt", "run_metadata.json"})
                    self.assertEqual((self.output / "raw_response.txt").read_bytes(), self.raw.encode())
                    chart = load_bar_chart(self.output / "chart.json")
                    self.assertIs(type(chart), kind)
                    self.assertEqual(chart.source_key, "vlm/chart.png")
                    metadata = self.metadata()
                    self.assertEqual(metadata["completion_status"], "complete")
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
                      ["--model-id", ""], ["--revision", " "]):
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
            self.assertEqual(self.run_cli(), 1)
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
        self.assertIn("SVG rendering is not connected", " ".join(result.stdout.split()))
