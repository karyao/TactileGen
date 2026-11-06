"""Pytesseract adapter preprocessing, selection, and failure behavior."""

import builtins
import importlib.util
import shutil
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.contracts import Box, Detection
from src.ocr import OCRBackendResult
from src.tesseract_ocr import TesseractBackend, _load_pytesseract


class FakeTesseractNotFoundError(Exception):
    pass


def fake_pytesseract(*, languages=("eng",), results=()):
    module = types.SimpleNamespace()
    module.Output = types.SimpleNamespace(DICT="dict-output")
    module.TesseractNotFoundError = FakeTesseractNotFoundError
    module.get_languages = Mock(return_value=list(languages))
    module.image_to_data = Mock(side_effect=list(results))
    return module


def result(text="Label", confidence="90"):
    return {"text": [text], "conf": [confidence]}


def empty_result():
    return {"text": [], "conf": []}


def detection(box, role="x_tick_label", id="text"):
    return Detection(id, role, Box(*box), 0.9)


class TesseractBackendTest(unittest.TestCase):
    def test_horizontal_crop_rounds_pads_clamps_upscales_and_forwards_options(self):
        module = fake_pytesseract(
            results=(result("  Annual\t Sales ", "80"), empty_result())
        )
        backend = TesseractBackend(timeout_seconds=2.5)
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        original = image.copy()

        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                image, detection((0.2, 0.4, 8.1, 6.1)), language="eng"
            )

        self.assertEqual(recognized, OCRBackendResult("Annual Sales", 0.8))
        self.assertEqual(module.image_to_data.call_count, 2)
        for call in module.image_to_data.call_args_list:
            candidate = call.args[0]
            self.assertEqual(candidate.shape, (36, 44))
            self.assertEqual(candidate.ndim, 2)
            self.assertEqual(call.kwargs, {
                "lang": "eng", "config": "--psm 7",
                "output_type": "dict-output", "timeout": 2.5,
            })
        self.assertTrue(np.array_equal(image, original))

    def test_title_is_read_from_full_width_strip_and_other_roles_are_unchanged(self):
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        title = detection((80, 10, 100, 26), "chart_title")  # narrower than the text
        shapes = {}
        for label, item in (("title", title), ("tick", detection((80, 10, 100, 26)))):
            module = fake_pytesseract(results=(result(), empty_result()))
            with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
                TesseractBackend().recognize(image, item, language="eng")
            shapes[label] = module.image_to_data.call_args_list[0].args[0].shape
        # 16 px box + 2 px padding each side = 20 rows, doubled to reach 32+.
        self.assertEqual(shapes["title"], (40, 400))
        self.assertEqual(shapes["tick"], (40, 48))

        with patch("src.ocr_crop._TITLE_FULL_WIDTH_STRIP", False):
            module = fake_pytesseract(results=(result(), empty_result()))
            with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
                TesseractBackend().recognize(image, title, language="eng")
        self.assertEqual(module.image_to_data.call_args_list[0].args[0].shape, (40, 48))

    def test_title_strip_stops_at_other_text_in_its_band(self):
        from src.ocr_crop import _title_strip_bounds

        title = detection((80, 10, 100, 26), "chart_title", "title")
        neighbors = (
            title,
            detection((5, 5, 20, 30), "y_axis_title", "ytitle"),
            detection((25, 12, 40, 24), "y_tick_label", "ytick"),
            detection((150, 8, 190, 28), "legend_label", "legend"),
            detection((30, 100, 60, 120), "x_tick_label", "below"),  # other band
            detection((90, 12, 95, 24), "legend_label", "inside"),  # inside title box
        )
        self.assertEqual(_title_strip_bounds(title, neighbors, 200), (42, 110))
        self.assertEqual(_title_strip_bounds(title, (title,), 200), (0, 200))

    def test_recognize_many_gives_title_its_neighbors(self):
        from src.tesseract_ocr import _recognition_candidates

        title = detection((80, 10, 100, 26), "chart_title", "title")
        tick = detection((10, 12, 40, 24), "y_tick_label", "tick")
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        (strip, _), = [_recognition_candidates(image, title, (title, tick))]
        self.assertEqual(strip.shape, (40, 316))  # x from 42 to 200, doubled

    def test_tall_crop_tries_three_orientations_and_keeps_first_confidence_tie(self):
        responses = [
            result("first", "75"), empty_result(),
            result("second", "75"), empty_result(),
            result("third", "70"), empty_result(),
        ]
        module = fake_pytesseract(results=responses)
        backend = TesseractBackend()
        image = np.zeros((25, 30, 4), dtype=np.uint8)

        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                image, detection((10, 2, 14, 18), "y_axis_title"), language="eng"
            )

        self.assertEqual(recognized, OCRBackendResult("first", 0.75))
        self.assertEqual(module.image_to_data.call_count, 6)
        shapes = [call.args[0].shape for call in module.image_to_data.call_args_list]
        self.assertEqual(shapes, [(40, 16), (40, 16), (32, 80), (32, 80),
                                  (32, 80), (32, 80)])

    def test_confidence_uses_valid_words_and_text_without_confidence_is_retained(self):
        module = fake_pytesseract(results=(
            {"text": ["One", "Two", "Three"], "conf": ["-1", "50", "bad"]},
            {"text": ["Fallback"], "conf": ["bad"]},
        ))
        backend = TesseractBackend()
        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                np.zeros((20, 30), dtype=np.uint8),
                detection((2, 2, 20, 10)),
                language="eng",
            )
        self.assertEqual(recognized, OCRBackendResult("One Two Three", 0.5))

        empty_confidence = fake_pytesseract(
            results=(result("Text", "bad"), empty_result())
        )
        with patch("src.tesseract_ocr._load_pytesseract", return_value=empty_confidence):
            recognized = TesseractBackend().recognize(
                np.zeros((20, 30), dtype=np.uint8),
                detection((2, 2, 20, 10)),
                language="eng",
            )
        self.assertEqual(recognized, OCRBackendResult("Text", 0.0))

    def test_empty_candidates_return_none_and_grayscale_single_channel_is_supported(self):
        module = fake_pytesseract(results=(result(" ", "90"), result("", "-1")))
        single_channel = np.zeros((20, 30, 1), dtype=np.uint8)
        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            recognized = TesseractBackend().recognize(
                single_channel, detection((2, 2, 20, 10)), language="eng"
            )
        self.assertIsNone(recognized)

    def test_language_validation_and_installed_language_cache(self):
        module = fake_pytesseract(
            languages=("eng", "fra"),
            results=(result(), result(), result(), result()),
        )
        backend = TesseractBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        item = detection((2, 2, 20, 10))
        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            backend.recognize(image, item, language="eng+fra")
            backend.recognize(image, item, language="eng")
        module.get_languages.assert_called_once_with(config="")

        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            with self.assertRaisesRegex(ValueError, "codes joined"):
                backend.recognize(image, item, language="eng + fra")
            with self.assertRaisesRegex(RuntimeError, "not installed: deu"):
                backend.recognize(image, item, language="deu")

    def test_missing_dependency_executable_and_timeout_are_actionable(self):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, "requirements-ocr.txt"):
                _load_pytesseract()

        missing = fake_pytesseract()
        missing.get_languages.side_effect = FakeTesseractNotFoundError("missing")
        with patch("src.tesseract_ocr._load_pytesseract", return_value=missing):
            with self.assertRaisesRegex(RuntimeError, "not available on PATH"):
                TesseractBackend().recognize(
                    np.zeros((20, 30), dtype=np.uint8),
                    detection((2, 2, 20, 10)),
                    language="eng",
                )

        timed_out = fake_pytesseract(results=(RuntimeError("process timeout"),))
        with patch("src.tesseract_ocr._load_pytesseract", return_value=timed_out):
            with self.assertRaisesRegex(RuntimeError, "recognition timed out"):
                TesseractBackend().recognize(
                    np.zeros((20, 30), dtype=np.uint8),
                    detection((2, 2, 20, 10)),
                    language="eng",
                )

    def test_rejects_bad_timeout_and_unsupported_channel_count(self):
        for timeout in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    TesseractBackend(timeout)
        module = fake_pytesseract(results=(result(), result()))
        with patch("src.tesseract_ocr._load_pytesseract", return_value=module):
            with self.assertRaisesRegex(ValueError, "grayscale, BGR, or BGRA"):
                TesseractBackend().recognize(
                    np.zeros((20, 30, 2), dtype=np.uint8),
                    detection((2, 2, 20, 10)),
                    language="eng",
                )

    @unittest.skipUnless(
        shutil.which("tesseract") and importlib.util.find_spec("pytesseract"),
        "Tesseract and pytesseract are not installed",
    )
    def test_installed_tesseract_smoke_reads_fixture_title(self):
        image = cv2.imread("tests/fixtures/contracts/val/example.png", cv2.IMREAD_COLOR)
        recognized = TesseractBackend().recognize(
            image,
            Detection("title", "chart_title", Box(155, 10, 167, 17), 0.99),
            language="eng",
        )
        self.assertIsInstance(recognized, OCRBackendResult)
        self.assertTrue(recognized.text)
        self.assertGreaterEqual(recognized.confidence, 0.0)
        self.assertLessEqual(recognized.confidence, 1.0)

    @unittest.skipUnless(
        shutil.which("tesseract") and importlib.util.find_spec("pytesseract"),
        "Tesseract and pytesseract are not installed",
    )
    def test_installed_tesseract_reads_title_wider_than_its_box(self):
        text = "Percentage of female workers employed in Agricultural sector"
        image = np.full((50, 800, 3), 255, dtype=np.uint8)
        cv2.putText(image, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                    cv2.LINE_AA)
        narrow = Detection("title", "chart_title", Box(120, 10, 500, 42), 0.9)
        recognized = TesseractBackend().recognize(image, narrow, language="eng")
        self.assertEqual(recognized.text, text)
        with patch("src.ocr_crop._TITLE_FULL_WIDTH_STRIP", False):
            clipped = TesseractBackend().recognize(image, narrow, language="eng")
        self.assertNotEqual(clipped.text, text)


if __name__ == "__main__":
    unittest.main()
