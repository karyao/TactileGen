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

from PIL import Image

from src.contracts import Box, ChartInterpretationError, Detection, DetectionFrame, OCRFrame, RecognizedText
from src.diagnostics import AnalysisDiagnostics, configure_analysis_logging
from src.gradio_app import UploadValidationError, analyze_image, build_app
from src.rapidocr_ocr import RapidOCRBackend
from src.tesseract_ocr import TesseractBackend
from tests.gradio_helpers import make_spec


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

    def test_optional_collector_preserves_exceptions_and_invalid_settings(self):
        collector = AnalysisDiagnostics(settings={"confidence": float("nan")})
        with self.assertRaises(UploadValidationError), \
             patch("src.gradio_ui.analysis.run_detection") as detect:
            analyze_image(object(), None, .25, .7, diagnostics=collector)
        detect.assert_not_called()
        collector.finish(failed=True)
        self.assertEqual(collector.record["stages"]["validation"]["outcome"], "failed")
        with tempfile.TemporaryDirectory() as temporary:
            path = collector.persist(Path(temporary))
            self.assertEqual(json.loads(path.read_text())["settings"]["confidence"], "nan")

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

    def test_event_records_success_failures_and_repeated_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "chart.png"
            Image.new("RGB", (100, 80), "white").save(image)
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("title", "chart_title", Box(5, 5, 50, 15), .9),
                Detection("tick", "y_tick_label", Box(5, 20, 15, 30), .9),
            ))
            ocr = OCRFrame(frame.source_key, (
                RecognizedText("title", "Sales", .8), RecognizedText("tick", "", 0),
            ))
            # Use real record persistence while retaining a copy beyond app shutdown.
            records = []
            paths = []
            persist = AnalysisDiagnostics.persist
            def save(collector, artifact_root):
                path = persist(collector, artifact_root)
                paths.append(path)
                records.append(json.loads(path.read_text()))
                return path
            demo = build_app(object(), ocr_backend=object(), device="cpu")
            event_binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            event = event_binding.fn
            error_index = max(
                index for index, output in enumerate(event_binding.outputs)
                if type(output).__name__ == "HTML" and "borealis-error" in output.elem_classes
            )
            try:
                with patch.object(AnalysisDiagnostics, "persist", save), \
                     patch("src.gradio_ui.analysis.run_detection", return_value=frame) as detect, \
                     patch("src.gradio_ui.analysis.run_ocr", return_value=ocr) as recognize, \
                     patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=object()), \
                     patch("src.gradio_ui.analysis.render_detection_overlay", side_effect=lambda image, *a, **k: image), \
                     patch("src.gradio_ui.analysis.summarize_detections", return_value="summary"), \
                     patch("src.gradio_ui.callbacks.legend_rows", return_value=[]), \
                     patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=make_spec()) as reconstruct, \
                     patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]), \
                     self.assertLogs("src.gradio_app", level="INFO") as logs:
                    success = event(str(image), .25, .7, "eng")
                    event(None, .25, .7, "eng")
                    detect.side_effect = RuntimeError("detector unavailable")
                    event(str(image), .25, .7, "eng")
                    detect.side_effect = None
                    recognize.side_effect = RuntimeError("OCR unavailable")
                    partial = event(str(image), .25, .7, "eng")
                    recognize.side_effect = None
                    reconstruct.side_effect = ChartInterpretationError("fewer than 2 readable y tick labels")
                    event(str(image), .25, .7, "eng")
                    reconstruct.side_effect = None
                    with patch("src.gradio_ui.callbacks.write_ocr_download", side_effect=OSError("export unavailable")):
                        event(str(image), .25, .7, "eng")
                    with patch("src.diagnostics.os.replace", side_effect=OSError("disk full")), \
                         patch.object(AnalysisDiagnostics, "persist", persist):
                        storage_failure = event(str(image), .25, .7, "eng")
                    reconstruct.return_value = make_spec(flags=["y_ticks_dropped:1"])
                    event(str(image), .25, .7, "eng")
                self.assertFalse(success[error_index]["visible"])
                self.assertFalse(partial[error_index]["visible"])
                self.assertFalse(storage_failure[error_index]["visible"])
                self.assertEqual(len(records), 7)
                self.assertEqual(len(set(r["run_id"] for r in records)), 7)
                self.assertEqual(records[0]["outcome"], "succeeded")
                self.assertTrue(all(s["outcome"] == "succeeded" for s in records[0]["stages"].values()))
                self.assertEqual(records[0]["counts"], {
                    "detections": 2, "text_regions": 2, "ocr_returned_regions": 2,
                    "ocr_readable": 1, "ocr_empty": 1,
                })
                for record, stage in ((records[1], "validation"), (records[2], "detection"),
                                      (records[5], "result_preparation")):
                    self.assertEqual(record["outcome"], "failed")
                    self.assertEqual(record["stages"][stage]["outcome"], "failed")
                    self.assertTrue(record["exceptions"])
                self.assertEqual(records[1]["stages"]["detection"]["outcome"], "skipped")
                self.assertEqual(records[2]["stages"]["ocr"]["outcome"], "skipped")
                self.assertEqual(records[3]["outcome"], "completed_with_warnings")
                self.assertIsNone(records[3]["counts"]["ocr_readable"])
                self.assertEqual(records[3]["stages"]["reconstruction"]["outcome"], "skipped")
                self.assertEqual(records[3]["stages"]["presentation"]["outcome"], "succeeded")
                self.assertEqual(records[4]["stages"]["reconstruction"]["outcome"], "blocked")
                self.assertEqual(records[4]["completeness_status"], "cannot_build")
                self.assertEqual(records[0]["exceptions"], [])
                self.assertEqual(records[6]["spec_flags"], ["y_ticks_dropped:1"])
                self.assertIn({"code": "y_ticks_dropped", "stage": "reconstruction"},
                              records[6]["warnings"])
                self.assertEqual(records[6]["outcome"], "completed_with_warnings")
                terminal = "\n".join(logs.output)
                self.assertEqual(terminal.count("analysis outcome="), 8)
                for record in records:
                    run_logs = [line for line in logs.output if f"[run {record['run_id']}]" in line]
                    self.assertEqual(sum("analysis outcome=" in line for line in run_logs), 1)
                    self.assertIn("analysis outcome=", run_logs[-1])
                    self.assertIn(f"warnings={len(record['warnings'])}", run_logs[-1])
                    self.assertIn("diagnostic ZIP created", run_logs[-2])
                for record, path in zip(records, paths):
                    self.assertIn(f"[run {record['run_id']}]", terminal)
                    self.assertIn(str(path), terminal)
                self.assertIn("detector unavailable", terminal)
                self.assertNotIn("Traceback (most recent call last)", terminal)
            finally:
                demo.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
