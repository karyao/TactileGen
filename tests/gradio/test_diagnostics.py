"""Gradio analysis diagnostics, OCR orchestration, and download integration."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from PIL import Image

from src.contracts import Box, ChartInterpretationError, Detection, DetectionFrame, OCRFrame, RecognizedText
from src.diagnostics import AnalysisDiagnostics
from src.gradio_app import UploadValidationError, analyze_image, build_app
from src.rapidocr_ocr import RapidOCRBackend
from tests.gradio.helpers import make_spec
from tests.test_ocr_trace import IMAGE, SHORT, collector
from tests.test_rapidocr_ocr import fake_rapidocr, output
from tests.test_reconstruct import base_chart


class GradioDiagnosticsTest(unittest.TestCase):
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


    def test_app_passes_collector_to_real_ocr_orchestration(self):
        module, _, _ = fake_rapidocr([output(" label ", .8)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        frame = DetectionFrame("input/chart.png", 60, 40, (SHORT,))
        diagnostics = collector()
        with patch("src.gradio_ui.analysis.decode_upload", return_value=(Path("chart.png"), IMAGE)), \
             patch("src.gradio_ui.analysis.run_detection", return_value=frame):
            result = analyze_image(object(), "chart.png", .25, .7,
                                   ocr_backend=backend, diagnostics=diagnostics)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(result.ocr_frame.regions[0].text, "label")
        self.assertEqual(trace["attempts"][0]["raw_output"]["text"], " label ")
        self.assertEqual(trace["results"][0]["application_result"]["text"], "label")



@pytest.mark.parametrize('failure', [None, 'detection', 'ocr', 'legend_matching', 'reconstruction', 'presentation'])
def test_zip_preserves_partial_stage_outputs(tmp_path, failure):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    diagnostic = AnalysisDiagnostics(settings={'confidence': .25})
    functions = {'detection': 'run_detection', 'ocr': 'run_ocr',
                 'legend_matching': 'match_legend_to_bars', 'reconstruction': 'reconstruct_vbar',
                 'presentation': 'render_detection_overlay'}
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch('src.gradio_ui.analysis.run_detection', return_value=frame))
        stack.enter_context(patch('src.gradio_ui.analysis.run_ocr', return_value=ocr))
        if failure:
            stack.enter_context(patch('src.gradio_ui.analysis.' + functions[failure], side_effect=RuntimeError('injected failure')))
        try:
            result = analyze_image(object(), image_path, .25, .7, diagnostics=diagnostic)
            assert result.diagnostics is diagnostic
        except RuntimeError:
            assert failure in ('detection', 'presentation')
        diagnostic.finish(failed=failure in ('detection', 'presentation'))
    path = diagnostic.export_zip(tmp_path)
    with ZipFile(path) as archive:
        report = json.loads(archive.read('diagnostics.json'))
        assert 'injected failure' in archive.read('report.txt').decode() if failure else not report['exceptions']
        assert 'input.png' not in archive.namelist()
        if failure != 'detection':
            assert 'detections.json' in archive.namelist()
            crops = [e for e in report['events'] if e['code'] == 'source_crop']
            assert len(crops) == len(ocr.regions)
            assert all(e['filename'] in archive.namelist() for e in crops)
            assert cv2.imdecode(np.frombuffer(archive.read(crops[0]['filename']), dtype=np.uint8), 1) is not None
        if failure not in ('detection', 'ocr'):
            assert 'ocr.json' in archive.namelist()
        if failure in ('reconstruction', 'presentation', None):
            assert 'legend.json' in archive.namelist()
        if failure == 'presentation':
            assert 'spec.json' in archive.namelist()



def test_gradio_download_reset_and_export_failure(tmp_path):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    event = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(event.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_ui.analysis.run_detection', return_value=frame), \
             patch('src.gradio_ui.analysis.run_ocr', return_value=ocr):
            success = event.fn(str(image_path), .25, .7, 'eng')
            assert success[index]['visible']
            with ZipFile(success[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert 'annotated.png' in archive.namelist()
                assert record['outputs']['spec']['categories'] == ['C0', 'C1', 'C2']
            rejected = event.fn(None, .25, .7, 'eng')
            assert rejected[index]['visible']
            assert rejected[index]['value'] != success[index]['value']
            with ZipFile(rejected[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert record['outcome'] == 'failed'
                assert 'input.png' not in archive.namelist()
            with patch('src.diagnostics.os.replace', side_effect=OSError('disk full')):
                failed_export = event.fn(str(image_path), .25, .7, 'eng')
            assert not failed_export[index]['visible']
            assert 'Diagnostic export failed' in failed_export[index+1]['value']
            assert failed_export[8] is not None  # Review remains available.
        for name in ('ready_results', 'begin_analysis', 'clear_all'):
            callback = next(fn for fn in demo.fns.values() if fn.api_name == name)
            updates = callback.fn()
            offset = len(callback.outputs) - len(event.outputs)
            assert not updates[index+offset]['visible']
            assert updates[index+offset]['value'] is None
    finally:
        demo.close(verbose=False)



def test_report_failure_keeps_app_analysis_and_shows_specific_error(tmp_path):
    chart = base_chart()
    image = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(key, text, confidence)
                   for key, (text, confidence) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    binding = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(binding.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_ui.analysis.run_detection', return_value=frame), \
             patch('src.gradio_ui.analysis.run_ocr', return_value=ocr), \
             patch('src.diagnostics_report.DiagnosticReport.render', side_effect=RuntimeError('formatting failed')):
            updates = binding.fn(str(image), .25, .7, 'eng')
        assert updates[8] is not None  # Analysis review state survived packaging failure.
        assert not updates[index]['visible'] and updates[index]['value'] is None
        assert 'Diagnostic export failed' in updates[index + 1]['value']
        assert 'report generation' in updates[index + 1]['value']
        assert 'formatting failed' in updates[index + 1]['value']
    finally:
        demo.close(verbose=False)

