"""RapidOCR PP-OCRv6 adapter configuration, crops, and output handling."""

import builtins
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.contracts import Box, Detection
from src.ocr import OCRBackendResult
from src.rapidocr_ocr import RapidOCRBackend, _load_rapidocr


def detection(box, role="chart_title"):
    return Detection("text", role, Box(*box), 0.9)


def output(text="Label", score=0.9):
    return types.SimpleNamespace(txts=(text,), scores=(score,))


class FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.cls_calls = []
        self.rec_calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return next(self.responses)

    def cls_and_rotate(self, images):
        self.cls_calls.append(images)
        return images, object()

    def recognize_txt(self, images):
        self.rec_calls.append(images)
        responses = [next(self.responses) for _ in images]
        return types.SimpleNamespace(
            txts=tuple(item.txts[0] for item in responses),
            scores=tuple(item.scores[0] for item in responses),
        )


def fake_rapidocr(responses):
    engine = FakeEngine(responses)
    constructor_calls = []

    def constructor(**kwargs):
        constructor_calls.append(kwargs)
        return engine

    module = types.SimpleNamespace(
        RapidOCR=constructor,
        EngineType=types.SimpleNamespace(ONNXRUNTIME="onnxruntime"),
        ModelType=types.SimpleNamespace(SMALL="small"),
        OCRVersion=types.SimpleNamespace(PPOCRV6="PP-OCRv6"),
    )
    return module, engine, constructor_calls


class RapidOCRBackendTest(unittest.TestCase):
    def test_configures_ppocrv6_small_onnx_and_recognition_only(self):
        module, engine, constructor_calls = fake_rapidocr([output(" Sales ", 0.8)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)

        recognized = backend.recognize(
            image, detection((0.2, 0.4, 8.1, 6.1)), language="eng"
        )

        self.assertEqual(recognized, OCRBackendResult("Sales", 0.8))
        self.assertEqual(constructor_calls, [{"params": {
            "Global.text_score": 0.0,
            "Rec.engine_type": "onnxruntime",
            "Rec.model_type": "small",
            "Rec.ocr_version": "PP-OCRv6",
        }}])
        crop, options = engine.calls[0]
        self.assertEqual(crop.shape, (9, 11, 3))
        self.assertEqual(options, {
            "use_det": False, "use_cls": True, "use_rec": True,
        })

    def test_tall_crop_tries_both_rotations_and_keeps_best_confidence(self):
        module, engine, _ = fake_rapidocr([
            output("vertical", 0.4), output("clockwise", 0.95),
            output("counter", 0.7),
        ])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((25, 30, 4), dtype=np.uint8)
        original = image.copy()

        recognized = backend.recognize(
            image, detection((10, 2, 14, 18), "y_axis_title"), language="eng"
        )

        self.assertEqual(recognized, OCRBackendResult("clockwise", 0.95))
        self.assertEqual([call[0].shape for call in engine.calls], [
            (20, 8, 4), (8, 20, 4), (8, 20, 4),
        ])
        self.assertTrue(np.array_equal(image, original))

    def test_blank_is_unreadable_and_bad_output_fails(self):
        module, _, _ = fake_rapidocr([output("  ", 0.9)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        item = detection((2, 2, 20, 10))
        self.assertIsNone(backend.recognize(image, item, language="eng"))

        bad_cases = (
            (types.SimpleNamespace(txts="bad", scores=(0.8,)), "malformed"),
            (types.SimpleNamespace(txts=("a", "b"), scores=(0.8, 0.7)), "one result"),
            (output(123, 0.8), "non-text"),
            (output("text", float("nan")), "confidence"),
        )
        for bad_output, message in bad_cases:
            with self.subTest(message=message):
                module, _, _ = fake_rapidocr([bad_output])
                with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
                    backend = RapidOCRBackend()
                with self.assertRaisesRegex(RuntimeError, message):
                    backend.recognize(image, item, language="eng")

    def test_recognize_many_preserves_detection_order(self):
        module, _, _ = fake_rapidocr([output("first", 0.9), output("second", 0.8)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        items = (
            detection((1, 1, 10, 7)),
            Detection("second", "legend_label", Box(12, 1, 21, 7), 0.9),
        )
        results = backend.recognize_many(
            np.zeros((20, 30, 3), dtype=np.uint8), items, language="eng"
        )
        self.assertEqual([item.text for item in results], ["first", "second"])
        self.assertEqual(len(backend._engine.cls_calls), 1)
        self.assertEqual(len(backend._engine.rec_calls), 1)

    def test_missing_dependency_error_is_actionable(self):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "rapidocr":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, "requirements-rapidocr.txt"):
                _load_rapidocr()


if __name__ == "__main__":
    unittest.main()
