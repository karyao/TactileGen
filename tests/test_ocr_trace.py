"""Engine-response evidence and parity with diagnostics disabled."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np

from src.contracts import Box, Detection, DetectionFrame
from src.diagnostics import AnalysisDiagnostics
from src.ocr import OCRBackendResult, run_ocr
from src.ocr_trace import adapter_evidence, snapshot
from src.rapidocr_ocr import RapidOCRBackend
from src.tesseract_ocr import TesseractBackend
from tests.test_rapidocr_ocr import fake_rapidocr, output
from tests.test_tesseract_ocr import fake_pytesseract


IMAGE = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
SHORT = Detection("short", "x_tick_label", Box(2, 2, 20, 10), .9)
TALL = Detection("tall", "y_axis_title", Box(25, 2, 29, 30), .9)


def collector():
    return AnalysisDiagnostics(settings={})


def recognize(backend, items, diagnostics=None):
    frame = DetectionFrame("input/chart.png", 60, 40, tuple(items))
    return run_ocr(IMAGE, frame, backend, "eng", diagnostics=diagnostics)


def tesseract_data(pages):
    data = {"text": [], "conf": [], "page_num": []}
    for page, rows in enumerate(pages, 1):
        for text, confidence in rows:
            data["text"].append(text)
            data["conf"].append(confidence)
            data["page_num"].append(page)
    return data


class OCRTraceTest(unittest.TestCase):
    def rapid(self, raw, items=(SHORT,), *, diagnostics=None, cls_error=None):
        module, engine, _ = fake_rapidocr([])
        engine.recognize_txt = Mock(side_effect=raw if isinstance(raw, Exception) else None,
                                    return_value=raw)
        if cls_error:
            engine.cls_and_rotate = Mock(side_effect=cls_error)
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        result = recognize(backend, items, diagnostics)
        return result, engine

    def tess(self, raw, items=(SHORT,), *, diagnostics=None):
        module = fake_pytesseract(results=[raw])
        module.submitted_pages = []
        if not isinstance(raw, Exception):
            def response(path, **kwargs):
                from PIL import Image, ImageSequence
                with Image.open(path) as image:
                    module.submitted_pages.extend(np.array(page) for page in ImageSequence.Iterator(image))
                return raw
            module.image_to_data.side_effect = response
        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            result = recognize(TesseractBackend(), items, diagnostics)
        return result, module

    def test_rapid_exact_strings_all_attempts_and_first_tie(self):
        raw = types.SimpleNamespace(txts=(" \t雪 e\u0301\n", "tie", " \n", ""),
                                    scores=(np.float32(.75), .75, float("nan"), .9))
        diagnostics = collector()
        ordinary, plain_engine = self.rapid(raw, (TALL, SHORT))
        traced, engine = self.rapid(raw, (TALL, SHORT), diagnostics=diagnostics)
        self.assertEqual(ordinary, traced)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(trace["batches"][0]["raw_output"]["txts"]["value"], list(raw.txts))
        self.assertEqual([a["raw_output"]["text_state"] for a in trace["attempts"]],
                         ["text", "text", "whitespace", "empty"])
        self.assertEqual([a["selection"] for a in trace["attempts"]],
                         ["selected", "not_selected", "unreadable", "unreadable"])
        self.assertEqual(trace["attempts"][0]["raw_output"]["text"], raw.txts[0])
        self.assertEqual(trace["attempts"][0]["adapter_result"]["text"], "雪 e\u0301")
        self.assertEqual(trace["results"][0]["selected_attempt_id"], trace["attempts"][0]["attempt_id"])
        self.assertEqual(trace["results"][1]["application_result"], {"text": "", "confidence": 0.0})
        self.assertEqual([a["preprocessing"]["rotation_degrees_clockwise"] for a in trace["attempts"]],
                         [0, 90, -90, 0])
        self.assertEqual(engine.recognize_txt.call_count, 1)
        self.assertEqual(len(engine.cls_calls), 1)
        for left, right in zip(plain_engine.recognize_txt.call_args.args[0], engine.recognize_txt.call_args.args[0]):
            np.testing.assert_array_equal(left, right)
        json.dumps(trace, allow_nan=False)

    def test_tesseract_raw_tokens_confidences_and_derived_text(self):
        pages = [
            [(" \t雪 e\u0301\n", "120"), ("next", "50"), ("bad", "oops"), ("", "-1")],
            [("tie", "75")],
            [(" \t", "90")], [("", "-1")],
            [("fallback", "nan")], [("last", "-1")],
        ]
        raw = tesseract_data(pages)
        diagnostics = collector()
        plain, plain_module = self.tess(raw, (TALL,))
        traced, module = self.tess(raw, (TALL,), diagnostics=diagnostics)
        self.assertEqual(plain, traced)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(trace["batches"][0]["raw_output"], raw)
        self.assertEqual(trace["attempts"][0]["raw_output"]["fields"]["text"], raw["text"][:4])
        self.assertEqual(trace["attempts"][0]["raw_output"]["row_indexes"], [0, 1, 2, 3])
        self.assertEqual(trace["attempts"][0]["adapter_result"], {
            "text": "雪 e\u0301 next bad", "confidence": .75, "text_kind": "derived"})
        self.assertEqual([a["selection"] for a in trace["attempts"]],
                         ["selected", "not_selected", "unreadable", "unreadable", "not_selected", "not_selected"])
        self.assertEqual(module.image_to_data.call_count, 1)
        self.assertEqual(len(module.submitted_pages), 6)
        for left, right in zip(module.submitted_pages, plain_module.submitted_pages):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(module.image_to_data.call_args.kwargs, plain_module.image_to_data.call_args.kwargs)
        self.assertEqual([a["preprocessing"]["variant"] for a in trace["attempts"]],
                         ["grayscale", "otsu_binary"] * 3)
        self.assertTrue(all(a["preprocessing"]["upscale_factor"] >= 1 for a in trace["attempts"]))
        json.dumps(trace, allow_nan=False)

    def test_malformed_rapid_output_is_retained_before_any_conversion(self):
        raw = types.SimpleNamespace(txts=("ok", 123, "last"), scores=(.9, .8, .7))
        diagnostics = collector()
        for current in (None, diagnostics):
            with self.assertRaisesRegex(RuntimeError, "non-text"):
                self.rapid(raw, (TALL,), diagnostics=current)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual([a["raw_output"]["text"] for a in trace["attempts"]], ["ok", 123, "last"])
        self.assertEqual([a["processing"] for a in trace["attempts"]],
                         ["completed", "invalid_output", "not_processed"])
        self.assertTrue(all(a["selection"] == "unresolved" for a in trace["attempts"]))
        self.assertIsNone(trace["results"][0]["application_result"])

    def test_rapid_absent_null_and_misaligned_output(self):
        cases = [types.SimpleNamespace(), types.SimpleNamespace(txts=None, scores=None),
                 types.SimpleNamespace(txts=(), scores=()),
                 types.SimpleNamespace(txts="bad", scores=(.9,))]
        for raw in cases:
            with self.subTest(raw=raw):
                diagnostics = collector()
                for current in (None, diagnostics):
                    with self.assertRaises(RuntimeError):
                        self.rapid(raw, diagnostics=current)
                trace = diagnostics.record["ocr_trace"]
                fields = trace["batches"][0]["raw_output"]
                self.assertEqual(fields["txts"]["presence"], "present" if hasattr(raw, "txts") else "absent")
                if hasattr(raw, "txts"):
                    self.assertEqual(fields["txts"]["value"], snapshot(raw.txts))
                self.assertEqual(trace["attempts"][0]["output_availability"], "unavailable")

    def test_invalid_rapid_confidences_preserve_error_and_raw_value(self):
        for score in (True, -.1, 1.1, "90", float("nan"), float("inf"),
                      np.longdouble('-.1'), np.longdouble('1.1'),
                      np.longdouble('nan'), np.longdouble('inf')):
            with self.subTest(score=score):
                diagnostics = collector()
                for current in (None, diagnostics):
                    with self.assertRaisesRegex(RuntimeError, "confidence"):
                        self.rapid(output("text", score), diagnostics=current)
                self.assertEqual(diagnostics.record["ocr_trace"]["attempts"][0]["raw_output"]["confidence"], snapshot(score))
                json.dumps(diagnostics.record, allow_nan=False)

    def test_extended_numpy_confidence_preserves_rapid_results_and_exports(self):
        raw = output(' label ', np.longdouble('.75'))
        diagnostics = collector()
        ordinary, plain_engine = self.rapid(raw)
        traced, engine = self.rapid(raw, diagnostics=diagnostics)
        self.assertEqual(ordinary, traced)
        self.assertEqual(traced.regions[0].confidence, .75)
        self.assertEqual(engine.recognize_txt.call_count, plain_engine.recognize_txt.call_count)
        self.assertEqual(len(engine.cls_calls), len(plain_engine.cls_calls))
        for left, right in zip(plain_engine.recognize_txt.call_args.args[0],
                               engine.recognize_txt.call_args.args[0], strict=True):
            np.testing.assert_array_equal(left, right)
        trace = diagnostics.record['ocr_trace']
        self.assertEqual(trace['attempts'][0]['raw_output']['confidence'], snapshot(raw.scores[0]))
        json.dumps(diagnostics.record, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics.finish()
            path = diagnostics.persist(root)
            self.assertIsNotNone(path)
            persisted = json.loads(path.read_text())
            self.assertEqual(persisted['ocr_trace'], trace)
            path = diagnostics.export_zip(root)
            self.assertIsNotNone(path)
            with ZipFile(path) as archive:
                self.assertEqual(json.loads(archive.read('ocr_trace.json')), trace)
                self.assertEqual(json.loads(archive.read('diagnostics.json')), persisted)

    def test_tesseract_preserves_odd_confidences_and_unmatched_tokens(self):
        raw = tesseract_data([[("A", "bad"), ("B", "-1")], [(" \t", "50")]])
        raw["text"].append("unmatched")
        diagnostics = collector()
        plain, _ = self.tess(raw)
        traced, _ = self.tess(raw, diagnostics=diagnostics)
        self.assertEqual(plain, traced)
        self.assertEqual(traced.regions[0].confidence, 0.0)
        self.assertEqual(diagnostics.record["ocr_trace"]["batches"][0]["raw_output"]["text"][-1], "unmatched")

    def test_tesseract_malformed_mapping_preserves_full_response(self):
        cases = [{}, {"page_num": ["bad"]}, {"page_num": [1], "text": ["A"], "conf": ["90"]},
                 {"page_num": [1, 2], "text": ["A"], "conf": ["90", "80"]},
                 {"page_num": [1, 2], "text": None, "conf": []}]
        for raw in cases:
            with self.subTest(raw=raw):
                diagnostics = collector()
                errors = []
                for current in (None, diagnostics):
                    with self.assertRaises(Exception) as caught:
                        self.tess(raw, diagnostics=current)
                    errors.append((type(caught.exception), str(caught.exception)))
                self.assertEqual(errors[0], errors[1])
                self.assertEqual(diagnostics.record["ocr_trace"]["batches"][0]["raw_output"], raw)
                self.assertTrue(all(a["selection"] == "unresolved" for a in diagnostics.record["ocr_trace"]["attempts"]))

    def test_engine_failures_are_batch_failures(self):
        for engine in ("rapid", "tess", "classifier"):
            with self.subTest(engine=engine):
                diagnostics = collector()
                error = RuntimeError("engine failed")
                with self.assertRaises(RuntimeError):
                    if engine == "tess":
                        self.tess(error, diagnostics=diagnostics)
                    else:
                        self.rapid(error, diagnostics=diagnostics,
                                   cls_error=error if engine == "classifier" else None)
                trace = diagnostics.record["ocr_trace"]
                self.assertEqual(len(trace["batches"]), 1)
                self.assertEqual(trace["batches"][0]["outcome"], "failed")
                self.assertEqual(trace["batches"][0]["operation"],
                                 {"rapid": "recognize_txt", "tess": "image_to_data", "classifier": "cls_and_rotate"}[engine])
                for attempt in trace["attempts"]:
                    self.assertEqual(attempt["submission"], "submitted")
                    self.assertEqual(attempt["output_availability"], "unavailable")
                    self.assertNotIn("error", attempt)
                    self.assertEqual(attempt["recognition_submission"],
                                     "not_submitted" if engine == "classifier" else "submitted")

    def test_preprocessing_failure_leaves_attempts_unsubmitted(self):
        diagnostics = collector()
        with patch("src.rapidocr_ocr._recognition_candidates", side_effect=ValueError("bad image")):
            with self.assertRaisesRegex(ValueError, "bad image"):
                self.rapid(output(), (SHORT, TALL), diagnostics=diagnostics)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(len(trace["attempts"]), 4)
        self.assertTrue(all(a["submission"] == "not_submitted" for a in trace["attempts"]))
        self.assertTrue(all(r["submission"] == "not_submitted" for r in trace["results"]))

    def test_legacy_sequential_retains_results_and_unsubmitted_detections(self):
        backend = types.SimpleNamespace(recognize=Mock(side_effect=[OCRBackendResult(" A ", .8), RuntimeError("failed")]))
        third = Detection("third", "legend_label", Box(32, 2, 45, 10), .9)
        diagnostics = collector()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            recognize(backend, (SHORT, TALL, third), diagnostics)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(backend.recognize.call_count, 2)
        self.assertEqual(trace["attempts"], [])
        self.assertEqual([r["submission"] for r in trace["results"]], ["submitted", "submitted", "not_submitted"])
        self.assertEqual(trace["results"][0]["adapter_result"]["text"], " A ")
        self.assertTrue(all(b["internal_attempts"] == "unavailable" for b in trace["batches"]))

    def test_legacy_none_empty_whitespace_remain_distinct_from_application(self):
        for result in (None, OCRBackendResult("", .8), OCRBackendResult(" \t\n", .7)):
            with self.subTest(result=result):
                backend = types.SimpleNamespace(recognize=Mock(return_value=result))
                diagnostics = collector()
                value = recognize(backend, (SHORT,), diagnostics)
                record = diagnostics.record["ocr_trace"]["results"][0]
                self.assertEqual(value.regions[0].text, "")
                self.assertEqual(record["application_result"], {"text": "", "confidence": 0.0})
                self.assertEqual(record["adapter_result"], None if result is None else {
                    "text": result.text, "confidence": result.confidence,
                    "text_state": "empty" if not result.text else "whitespace"})

    def test_legacy_batch_compatibility_and_failures(self):
        backend = types.SimpleNamespace(recognize=Mock(), recognize_many=Mock(return_value=[OCRBackendResult(" A ", .8)]))
        diagnostics = collector()
        value = recognize(backend, (SHORT,), diagnostics)
        self.assertEqual(value.regions[0].text, " A ")
        backend.recognize.assert_not_called()
        backend.recognize_many.assert_called_once()
        self.assertEqual(set(backend.recognize_many.call_args.kwargs), {"language"})
        self.assertEqual(diagnostics.record["ocr_trace"]["attempts"], [])
        backend.recognize_many.side_effect = RuntimeError("batch failed")
        with self.assertRaisesRegex(RuntimeError, "batch failed"):
            recognize(backend, (SHORT,), diagnostics)
        self.assertEqual(diagnostics.record["ocr_trace"]["batches"][-1]["outcome"], "failed")

    def test_legacy_batch_invalid_responses_fail_without_changing_exceptions(self):
        good = OCRBackendResult("first", .8)
        cases = [None, {}, "bad", [], [good], [good, None, None],
                 [good, ("bad", .9)], [good, OCRBackendResult(123, .9)]]
        cases.extend([good, OCRBackendResult("bad confidence", value)]
                     for value in (True, -.1, 1.1, float("nan"), float("inf")))
        for response in cases:
            with self.subTest(response=response):
                diagnostics = collector()
                errors = []
                for current in (None, diagnostics):
                    backend = types.SimpleNamespace(
                        recognize=Mock(), recognize_many=Mock(return_value=response))
                    with self.assertRaises((TypeError, ValueError)) as caught:
                        recognize(backend, (SHORT, TALL), current)
                    errors.append((type(caught.exception), str(caught.exception)))
                    backend.recognize.assert_not_called()
                    backend.recognize_many.assert_called_once()
                    self.assertEqual(set(backend.recognize_many.call_args.kwargs), {"language"})
                self.assertEqual(errors[0], errors[1])
                trace = diagnostics.record["ocr_trace"]
                batch, = trace["batches"]
                self.assertEqual(batch["outcome"], "failed")
                self.assertEqual(batch["operation"], "interpret_output")
                self.assertEqual(batch["error"], {
                    "type": errors[0][0].__name__, "message": errors[0][1]})
                expected = ([adapter_evidence(r) for r in response]
                            if isinstance(response, (list, tuple)) else adapter_evidence(response))
                self.assertEqual(batch["raw_output"], expected)
                self.assertEqual(batch["raw_output_availability"], "available")
                self.assertEqual(trace["attempts"], [])
                for result in trace["results"]:
                    self.assertEqual(result["submission"], "submitted")
                    self.assertEqual(result["selection"], "unresolved")
                    self.assertIsNone(result["application_result"])
                    self.assertIsNone(result["selected_attempt_id"])
                diagnostics.finish(failed=True)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    record = json.loads(diagnostics.persist(root).read_text())
                    with ZipFile(diagnostics.export_zip(root)) as archive:
                        self.assertEqual(json.loads(archive.read("diagnostics.json")), record)
                        self.assertEqual(json.loads(archive.read("ocr_trace.json")), trace)

    def test_legacy_batch_success_preserves_blank_confidence_behavior(self):
        for response in ([None, OCRBackendResult(" \t\n", float("nan"))],
                         (OCRBackendResult("", True), OCRBackendResult(" valid ", .8))):
            with self.subTest(response=response):
                diagnostics = collector()
                results = []
                for current in (None, diagnostics):
                    backend = types.SimpleNamespace(
                        recognize=Mock(), recognize_many=Mock(return_value=response))
                    results.append(recognize(backend, (SHORT, TALL), current))
                    backend.recognize_many.assert_called_once()
                    backend.recognize.assert_not_called()
                self.assertEqual(results[0], results[1])
                trace = diagnostics.record["ocr_trace"]
                self.assertEqual(trace["batches"][0]["outcome"], "succeeded")
                for record, region in zip(trace["results"], results[0].regions, strict=True):
                    self.assertEqual(record["application_result"], {
                        "text": region.text, "confidence": region.confidence})

    def test_legacy_batch_engine_exception_remains_an_inference_failure(self):
        diagnostics = collector()
        for current in (None, diagnostics):
            backend = types.SimpleNamespace(
                recognize=Mock(), recognize_many=Mock(side_effect=RuntimeError("engine failed")))
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                recognize(backend, (SHORT,), current)
            backend.recognize_many.assert_called_once()
            backend.recognize.assert_not_called()
        batch, = diagnostics.record["ocr_trace"]["batches"]
        self.assertEqual(batch["operation"], "recognize_many")
        self.assertEqual(batch["outcome"], "failed")
        self.assertEqual(batch["raw_output_availability"], "unavailable")

    def test_ids_are_unique_across_calls_and_runs_and_exports_roundtrip(self):
        first, second = collector(), collector()
        self.rapid(output("雪", .9), diagnostics=first)
        self.rapid(output("other", .8), diagnostics=first)
        self.rapid(output("雪", .9), diagnostics=second)
        ids = [a["attempt_id"] for c in (first, second) for a in c.record["ocr_trace"]["attempts"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(second.record["ocr_trace"]["attempts"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first.finish()
            persisted = json.loads(first.persist(root).read_text())
            with ZipFile(first.export_zip(root)) as archive:
                exported = json.loads(archive.read("ocr_trace.json"))
                self.assertEqual(exported, first.record["ocr_trace"])
                self.assertEqual(json.loads(archive.read("diagnostics.json")), persisted)

    def test_empty_detection_batch_does_not_infer(self):
        for kind in ("rapid", "tess"):
            diagnostics = collector()
            if kind == "rapid":
                result, engine = self.rapid(output(), (), diagnostics=diagnostics)
                engine.recognize_txt.assert_not_called()
                self.assertEqual(engine.cls_calls, [])
            else:
                result, module = self.tess({}, (), diagnostics=diagnostics)
                module.image_to_data.assert_not_called()
            self.assertEqual(result.regions, ())
            self.assertEqual(diagnostics.record["ocr_trace"]["attempts"], [])

    def test_tesseract_missing_text_or_confidence_is_distinct_from_empty_tokens(self):
        for missing in ("text", "conf"):
            with self.subTest(missing=missing):
                raw = tesseract_data([[("A", "90")], [("", "-1")]])
                del raw[missing]
                diagnostics = collector()
                plain, _ = self.tess(raw)
                traced, _ = self.tess(raw, diagnostics=diagnostics)
                self.assertEqual(plain, traced)
                trace = diagnostics.record["ocr_trace"]
                self.assertNotIn(missing, trace["batches"][0]["raw_output"])
                self.assertIsNone(trace["results"][0]["adapter_result"])
                self.assertTrue(all(a["output_availability"] == "available" for a in trace["attempts"]))

    def test_tesseract_preprocessing_failure_never_submits(self):
        diagnostics = collector()
        with patch("src.tesseract_ocr._recognition_candidates", side_effect=ValueError("bad image")):
            with self.assertRaisesRegex(ValueError, "bad image"):
                self.tess({}, (SHORT, TALL), diagnostics=diagnostics)
        attempts = diagnostics.record["ocr_trace"]["attempts"]
        self.assertEqual(len(attempts), 8)
        self.assertTrue(all(a["submission"] == "not_submitted" for a in attempts))

    def test_app_passes_collector_to_real_ocr_orchestration(self):
        from src.gradio_app import analyze_image
        module, _, _ = fake_rapidocr([output(" label ", .8)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        frame = DetectionFrame("input/chart.png", 60, 40, (SHORT,))
        diagnostics = collector()
        with patch("src.gradio_app.decode_upload", return_value=(Path("chart.png"), IMAGE)), \
             patch("src.gradio_app.run_detection", return_value=frame):
            result = analyze_image(object(), "chart.png", .25, .7,
                                   ocr_backend=backend, diagnostics=diagnostics)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(result.ocr_frame.regions[0].text, "label")
        self.assertEqual(trace["attempts"][0]["raw_output"]["text"], " label ")
        self.assertEqual(trace["results"][0]["application_result"]["text"], "label")

    def test_snapshot_handles_non_json_values_without_coercing_strings(self):
        value = {"string": " nan \t", "float": float("inf"), "odd": object(),
                 "array": np.array([float("nan")])}
        saved = snapshot(value)
        json.dumps(saved, allow_nan=False)
        self.assertEqual(saved["string"], value["string"])
        self.assertEqual(saved["odd"]["diagnostic_type"], "unsupported")

    def test_snapshot_extended_numpy_scalars_and_nested_arrays(self):
        for scalar_type in (np.longdouble, np.clongdouble):
            for number in ('.75', 'nan', 'inf', '-inf'):
                value = scalar_type(number)
                with self.subTest(dtype=str(value.dtype), number=number):
                    converted = value.item()
                    if type(converted) is type(value):
                        inner = {'diagnostic_type': 'unsupported', 'python_type':
                                 f'{type(value).__module__}.{type(value).__qualname__}'}
                    else:
                        inner = snapshot(converted)
                    expected = {'diagnostic_type': str(value.dtype), 'value': inner}
                    self.assertEqual(snapshot(value), expected)
                    array = np.array([value])
                    saved = snapshot({'nested': [(value,)], 'array': array})
                    self.assertEqual(saved['nested'], [[expected]])
                    # Some platforms convert extended array elements to native
                    # Python scalars even though scalar .item() preserves NumPy.
                    array_item = array.tolist()[0]
                    expected_item = expected if type(array_item) is type(value) else snapshot(array_item)
                    self.assertEqual(saved['array'], {
                        'diagnostic_type': 'ndarray', 'dtype': str(value.dtype), 'value': [expected_item]})
                    json.dumps(saved, allow_nan=False)

    def test_snapshot_ordinary_numpy_scalar_formats_are_unchanged(self):
        self.assertEqual(snapshot(np.float32(.75)), {'diagnostic_type': 'float32', 'value': .75})
        self.assertEqual(snapshot(np.float64(.75)), .75)
        self.assertEqual(snapshot(np.int64(3)), {'diagnostic_type': 'int64', 'value': 3})


if __name__ == "__main__":
    unittest.main()
