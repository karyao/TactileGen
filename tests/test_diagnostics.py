"""Analysis records, including event-handler failures and partial recovery."""

import contextlib
import io
import json
import logging
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid


from src.diagnostics import AnalysisDiagnostics, configure_analysis_logging
from src.rapidocr_ocr import RapidOCRBackend
from src.tesseract_ocr import TesseractBackend


class DiagnosticsTest(unittest.TestCase):
    def test_stage_start_is_logged_before_execution_and_completion(self):
        for outcome in ("succeeded", "failed", "blocked"):
            with self.subTest(outcome=outcome):
                collector = AnalysisDiagnostics(settings={})
                blocked_errors = (ValueError,) if outcome == "blocked" else ()
                with self.assertLogs("src.gradio_app", level="INFO") as logs:
                    with (contextlib.nullcontext() if outcome == "succeeded"
                          else self.assertRaisesRegex(ValueError, "OCR unavailable")):
                        with collector.stage("ocr", blocked_errors=blocked_errors):
                            self.assertEqual(len(logs.records), 1)
                            self.assertEqual(logs.records[0].levelno, logging.INFO)
                            self.assertEqual(logs.records[0].getMessage(),
                                             f"[run {collector.run_id}] stage=ocr outcome=started")
                            stage = collector.record["stages"]["ocr"]
                            self.assertIsNotNone(stage["started_at"])
                            self.assertIsNone(stage["outcome"])
                            self.assertIsNone(stage["ended_at"])
                            if outcome != "succeeded":
                                raise ValueError("OCR unavailable")
                self.assertEqual(len(logs.records), 2)
                self.assertIn(f"stage=ocr outcome={outcome} duration=", logs.output[1])
                self.assertEqual(logs.records[1].levelno,
                                 logging.ERROR if outcome == "failed" else logging.INFO)
                self.assertEqual(collector.record["stages"]["ocr"]["outcome"], outcome)
                self.assertTrue(all(f"[run {collector.run_id}]" in line for line in logs.output))
                self.assertIn("stage=ocr outcome=started", collector._log_entries[1])

    def test_skipped_stage_has_no_start_message(self):
        collector = AnalysisDiagnostics(settings={})
        with self.assertLogs("src.gradio_app", level="INFO") as logs:
            collector.skip("ocr", "no text regions")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("stage=ocr outcome=skipped reason=no text regions", logs.output[0])
        self.assertIsNone(collector.record["stages"]["ocr"]["started_at"])

    def test_event_normalizes_nested_non_finite_evidence(self):
        collector = AnalysisDiagnostics(settings={})
        nested = {"values": [float("inf"), {"negative": float("-inf")}],
                  "tuple": (float("nan"), 1.7e308, 0.0, -2, True, None, "text")}
        collector.event("reconstruction", "numeric_evidence", nested=nested)
        event = collector.record["events"][0]
        self.assertEqual(event, {
            "stage": "reconstruction", "code": "numeric_evidence",
            "event_id": f"{collector.run_id}:event-00001",
            "nested": {"values": ["inf", {"negative": "-inf"}],
                       "tuple": ("nan", 1.7e308, 0.0, -2, True, None, "text")},
        })
        json.dumps(collector.record, allow_nan=False)
        self.assertEqual(nested["values"][0], float("inf"))

    def test_interleaved_collectors_and_chained_exceptions_round_trip(self):
        first = AnalysisDiagnostics(settings={"language": "eng"})
        second = AnalysisDiagnostics(settings={"language": "fra"})
        with first.stage("validation"):
            with second.stage("validation"):
                second.warning("sample_warning", "validation")
        try:
            with second.stage("detection"):
                try:
                    raise ValueError("underlying cause")
                except ValueError as cause:
                    raise RuntimeError("detection failed") from cause
        except RuntimeError:
            pass
        first.finish()
        second.finish(failed=True)
        with tempfile.TemporaryDirectory() as temporary:
            one = first.persist(Path(temporary))
            two = second.persist(Path(temporary))
            record = json.loads(two.read_text())
            self.assertEqual(json.loads(one.read_text()), first.record)
            self.assertEqual(record, second.record)
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])
        self.assertNotEqual(uuid.UUID(first.run_id), uuid.UUID(second.run_id))
        self.assertEqual(first.record["warnings"], [])
        self.assertEqual(first.record["exceptions"], [])
        self.assertEqual(record["outcome"], "failed")
        self.assertIn("underlying cause", record["exceptions"][0]["traceback"])
        self.assertIn("direct cause", record["exceptions"][0]["traceback"])
        self.assertNotIn("locals", record["exceptions"][0])
        self.assertIn("detection", record["stages"]["ocr"]["reason"])
        self.assertGreaterEqual(record["duration_seconds"], 0)
        self.assertTrue(record["ended_at"].endswith("+00:00"))

    def test_metadata_is_best_effort_and_does_not_initialize_engines(self):
        collector = AnalysisDiagnostics(settings={})
        model = types.SimpleNamespace(
            names={0: "chart_title"}, ckpt_path=Path("weights.pt"), task="detect",
            overrides={"imgsz": 640, "unrelated": object()},
        )
        with patch("src.diagnostics.metadata.version", side_effect=RuntimeError("missing metadata")):
            collector.capture_metadata(model, TesseractBackend(2), "eng+fra")
        self.assertEqual(collector.record["metadata"]["model"]["checkpoint"], "weights.pt")
        self.assertEqual(collector.record["metadata"]["model"]["configured_prediction_options"],
                         {"imgsz": 640})
        self.assertTrue(all(value is None for value in collector.record["metadata"]["packages"].values()))
        self.assertEqual(collector.record["metadata"]["ocr"]["effective_language"], "eng+fra")
        # Bypass construction: metadata must not require an initialized engine.
        collector.capture_metadata(model, object.__new__(RapidOCRBackend), "eng")
        self.assertEqual(collector.record["metadata"]["ocr"]["effective_language"], "multilingual")
        self.assertIn("not applied", collector.record["metadata"]["ocr"]["language_behavior"])
        collector.capture_metadata(object(), object(), "eng")
        self.assertEqual(collector.record["metadata"]["ocr"]["language_behavior"], "unknown")
        class BrokenMetadata:
            def diagnostic_metadata(self, **kwargs):
                raise RuntimeError("metadata unavailable")
        collector.capture_metadata(object(), BrokenMetadata(), "eng")
        self.assertIn("metadata unavailable", collector.record["metadata"]["ocr"]["metadata_error"])


    def test_logging_setup_is_idempotent_and_terminal_summaries_are_visible(self):
        logger = logging.getLogger("src.gradio_app")
        original = logger.handlers[:], logger.level, logger.propagate
        third_party_level = logging.getLogger("ultralytics").level
        try:
            logger.handlers = []
            terminal = io.StringIO()
            with contextlib.redirect_stderr(terminal):
                configure_analysis_logging()
                configure_analysis_logging()
                collector = AnalysisDiagnostics(settings={})
                with collector.stage("validation"):
                    pass
                collector.finish()
                with tempfile.TemporaryDirectory() as temporary:
                    path = collector.persist(Path(temporary))
                    archive_path = collector.export_zip(Path(temporary))
                    collector.log_summary(record_path=path, archive_path=archive_path)
            self.assertEqual(len(logger.handlers), 1)
            output = terminal.getvalue()
            self.assertEqual(output.count("analysis started"), 1)
            self.assertEqual(output.count("analysis outcome=succeeded"), 1)
            self.assertIn(str(path), output)
            self.assertTrue(all(f"[run {collector.run_id}]" in line for line in output.splitlines()))
            self.assertEqual(logging.getLogger("ultralytics").level, third_party_level)
        finally:
            logger.handlers, logger.level, logger.propagate = original

    def test_storage_error_retains_record_and_cleans_up(self):
        collector = AnalysisDiagnostics(settings={})
        collector.finish()
        with tempfile.TemporaryDirectory() as temporary, \
             patch("src.diagnostics.os.replace", side_effect=OSError("disk full")), \
             self.assertLogs("src.gradio_app", level="INFO") as logs:
            self.assertIsNone(collector.persist(Path(temporary)))
            collector.log_summary(record_path=None, archive_path=None)
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])
        self.assertEqual(collector.record["outcome"], "succeeded")
        self.assertIn("disk full", "\n".join(logs.output))
        self.assertIn("record=unavailable", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
