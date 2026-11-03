"""Engine-neutral OCR pipeline behavior and contract integration."""

import json
import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.contract_io import (
    load_detection_frame,
    load_ocr_frame,
    write_detection_frame,
    write_ocr_frame,
)
from src.contracts import (
    TEXT_ROLES,
    Box,
    Detection,
    DetectionFrame,
    join_text_regions,
)
from src.ocr import OCRBackendResult, _run_command, parse_args, run_ocr


class RecordingBackend:
    def __init__(self, results=None):
        self.results = {} if results is None else results
        self.calls = []

    def recognize(self, image, detection, *, language):
        self.calls.append((image, detection, language))
        result = self.results.get(detection.id)
        if isinstance(result, BaseException):
            raise result
        return result


def detection(identifier: str, role: str, index: int = 0) -> Detection:
    left = float(index)
    return Detection(identifier, role, Box(left, 0.0, left + 1.0, 1.0), 0.9)


def frame(*detections: Detection, width: int = 20, height: int = 10) -> DetectionFrame:
    return DetectionFrame("input/chart.png", width, height, detections)


class OCRPipelineTest(unittest.TestCase):
    def test_processes_every_text_role_once_and_preserves_filtered_order(self):
        roles = tuple(sorted(TEXT_ROLES))
        items = [detection("bar-before", "vertical_bar", 0)]
        for index, role in enumerate(roles, start=1):
            items.append(detection(f"text-{index}", role, index))
            if index == 3:
                items.append(detection("plot-middle", "plot_area", 10))
        backend = RecordingBackend({
            item.id: OCRBackendResult(item.id, 0.5)
            for item in items
            if item.role in TEXT_ROLES
        })
        image = np.zeros((10, 20, 3), dtype=np.uint8)

        result = run_ocr(image, frame(*items), backend, "eng")

        expected = [item for item in items if item.role in TEXT_ROLES]
        self.assertEqual([call[1] for call in backend.calls], expected)
        self.assertEqual([region.detection_id for region in result.regions],
                         [item.id for item in expected])
        self.assertEqual(len(result.regions), len(TEXT_ROLES))
        self.assertEqual(result.source_key, "input/chart.png")
        self.assertEqual(len(join_text_regions(frame(*items), result)), len(TEXT_ROLES))

    def test_backend_receives_readonly_complete_image_exact_detection_and_language(self):
        item = detection("title", "chart_title")
        source = np.zeros((10, 20), dtype=np.uint8)
        backend = RecordingBackend({"title": OCRBackendResult("  Annual total  ", 1.0)})

        result = run_ocr(source, frame(item), backend, " eng+fra ")

        passed_image, passed_detection, passed_language = backend.calls[0]
        self.assertTrue(np.shares_memory(source, passed_image))
        self.assertFalse(passed_image.flags.writeable)
        self.assertTrue(source.flags.writeable)
        self.assertIs(passed_detection, item)
        self.assertEqual(passed_language, " eng+fra ")
        self.assertEqual(result.regions[0].text, "  Annual total  ")
        self.assertEqual(result.regions[0].confidence, 1.0)

    def test_none_empty_and_whitespace_are_canonical_unreadable_records(self):
        items = (
            detection("none", "chart_title", 0),
            detection("empty", "x_axis_title", 1),
            detection("spaces", "y_axis_title", 2),
        )
        backend = RecordingBackend({
            "none": None,
            "empty": OCRBackendResult("", 0.8),
            "spaces": OCRBackendResult(" \t\n", 0.7),
        })

        result = run_ocr(np.zeros((10, 20, 3), dtype=np.uint8), frame(*items),
                         backend, "eng")

        self.assertEqual(
            [(region.text, region.confidence) for region in result.regions],
            [("", 0.0), ("", 0.0), ("", 0.0)],
        )

    def test_no_text_detections_returns_empty_frame_without_backend_call(self):
        backend = RecordingBackend()
        result = run_ocr(
            np.zeros((10, 20, 3), dtype=np.uint8),
            frame(detection("bar", "vertical_bar")),
            backend,
            "eng",
        )
        self.assertEqual(result.regions, ())
        self.assertEqual(backend.calls, [])

    def test_confidence_boundaries_pass_and_invalid_values_fail(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        items = (
            detection("low", "chart_title", 0),
            detection("high", "x_axis_title", 1),
        )
        result = run_ocr(image, frame(*items), RecordingBackend({
            "low": OCRBackendResult("A", 0.0),
            "high": OCRBackendResult("B", 1.0),
        }), "eng")
        self.assertEqual([region.confidence for region in result.regions], [0.0, 1.0])

        for bad in (True, math.nan, math.inf, -0.1, 1.1):
            with self.subTest(confidence=bad):
                backend = RecordingBackend({"low": OCRBackendResult("A", bad)})
                with self.assertRaisesRegex(ValueError, "confidence"):
                    run_ocr(image, frame(items[0]), backend, "eng")

    def test_rejects_malformed_backend_values_and_propagates_exceptions(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        item = detection("title", "chart_title")
        cases = (
            (("text", 0.9), TypeError, "OCRBackendResult"),
            (OCRBackendResult(123, 0.9), TypeError, "text must be a string"),
        )
        for value, error, message in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(error, message):
                    run_ocr(image, frame(item), RecordingBackend({"title": value}), "eng")

        backend = RecordingBackend({"title": RuntimeError("engine unavailable")})
        with self.assertRaisesRegex(RuntimeError, "engine unavailable"):
            run_ocr(image, frame(item), backend, "eng")
        self.assertEqual(len(backend.calls), 1)

    def test_backend_exception_stops_processing_without_a_partial_result(self):
        items = (
            detection("first", "chart_title", 0),
            detection("second", "x_axis_title", 1),
            detection("third", "y_axis_title", 2),
        )
        backend = RecordingBackend({
            "first": OCRBackendResult("A", 0.9),
            "second": RuntimeError("recognition failed"),
            "third": OCRBackendResult("C", 0.9),
        })
        with self.assertRaisesRegex(RuntimeError, "recognition failed"):
            run_ocr(np.zeros((10, 20, 3), dtype=np.uint8), frame(*items), backend, "eng")
        self.assertEqual([call[1].id for call in backend.calls], ["first", "second"])

    def test_validates_all_pipeline_inputs_before_recognition(self):
        item = detection("title", "chart_title")
        valid_frame = frame(item)
        valid_image = np.zeros((10, 20, 3), dtype=np.uint8)
        cases = (
            (object(), valid_frame, RecordingBackend(), "eng", TypeError, "NumPy"),
            (np.zeros((10,), dtype=np.uint8), valid_frame, RecordingBackend(), "eng",
             ValueError, "2D grayscale"),
            (np.zeros((10, 20, 0), dtype=np.uint8), valid_frame, RecordingBackend(), "eng",
             ValueError, "positive"),
            (np.zeros((9, 20, 3), dtype=np.uint8), valid_frame, RecordingBackend(), "eng",
             ValueError, "do not match"),
            (valid_image, valid_frame, RecordingBackend(), " ", ValueError, "nonempty"),
            (valid_image, valid_frame, object(), "eng", TypeError, "recognize"),
        )
        for image, current_frame, backend, language, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    run_ocr(image, current_frame, backend, language)
                self.assertEqual(getattr(backend, "calls", []), [])

        backend = RecordingBackend()
        with self.assertRaisesRegex(TypeError, "DetectionFrame"):
            run_ocr(valid_image, object(), backend, "eng")
        self.assertEqual(backend.calls, [])

    def test_json_round_trip_joins_and_contains_no_geometry(self):
        items = (
            detection("title", "chart_title", 0),
            detection("bar", "vertical_bar", 1),
            detection("tick", "x_tick_label", 2),
        )
        detection_frame = frame(*items)
        ocr = run_ocr(
            np.zeros((10, 20, 3), dtype=np.uint8),
            detection_frame,
            RecordingBackend({
                "title": OCRBackendResult("Sales", 0.95),
                "tick": OCRBackendResult("2025", 0.85),
            }),
            "eng",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.json"
            write_ocr_frame(path, ocr)
            serialized = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_ocr_frame(path)

        self.assertEqual(loaded, ocr)
        self.assertEqual([pair[0].id for pair in join_text_regions(detection_frame, loaded)],
                         ["title", "tick"])
        self.assertEqual(set(serialized), {"schema_version", "source_key", "regions"})
        for region in serialized["regions"]:
            self.assertEqual(set(region), {"detection_id", "text", "confidence"})


class OCRCommandTest(unittest.TestCase):
    def test_argument_defaults(self):
        args = parse_args([
            "--image", "chart.png", "--detections", "chart_detections.json",
            "--output", "chart_ocr.json",
        ])
        self.assertEqual(args.engine, "tesseract")
        self.assertEqual(args.language, "eng")

    def test_command_loads_yolo_json_and_writes_joinable_ocr_without_geometry(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"

        class Backend:
            def recognize(self, image, item, *, language):
                return OCRBackendResult(f"read-{item.id}", 0.75)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/chart_ocr.json"
            args = Namespace(
                engine="tesseract",
                image=fixture_root / "val/example.png",
                detections=fixture_root / "detections.json",
                output=output,
                language="eng",
            )
            with patch("src.tesseract_ocr.TesseractBackend", return_value=Backend()):
                result = _run_command(args)
            loaded = load_ocr_frame(output)
            serialized = json.loads(output.read_text(encoding="utf-8"))

        detections = load_detection_frame(fixture_root / "detections.json")
        self.assertEqual(loaded, result)
        expected_count = sum(item.role in TEXT_ROLES for item in detections.detections)
        self.assertEqual(len(join_text_regions(detections, loaded)), expected_count)
        for region in serialized["regions"]:
            self.assertEqual(set(region), {"detection_id", "text", "confidence"})

    def test_dimension_mismatch_fails_without_creating_output(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detections = frame(
                detection("title", "chart_title"), width=319, height=240
            )
            detection_path = root / "detections.json"
            output = root / "ocr.json"
            write_detection_frame(detection_path, detections)
            args = Namespace(
                engine="tesseract", image=fixture_root / "val/example.png",
                detections=detection_path, output=output, language="eng",
            )
            with patch("src.tesseract_ocr.TesseractBackend", return_value=RecordingBackend()):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    _run_command(args)
            self.assertFalse(output.exists())

    def test_backend_failure_preserves_existing_output(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"

        class FailingBackend:
            def recognize(self, image, item, *, language):
                raise RuntimeError("OCR failed")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ocr.json"
            output.write_text("keep me", encoding="utf-8")
            args = Namespace(
                engine="tesseract", image=fixture_root / "val/example.png",
                detections=fixture_root / "detections.json", output=output,
                language="eng",
            )
            with patch("src.tesseract_ocr.TesseractBackend", return_value=FailingBackend()):
                with self.assertRaisesRegex(RuntimeError, "OCR failed"):
                    _run_command(args)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_missing_inputs_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ocr.json"
            args = Namespace(
                engine="tesseract", image=root / "missing.png",
                detections=root / "missing.json", output=output, language="eng",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing detections"):
                _run_command(args)
            self.assertFalse(output.exists())

            fixture_root = Path(__file__).parent / "fixtures/contracts"
            args.detections = fixture_root / "detections.json"
            with self.assertRaisesRegex(FileNotFoundError, "Missing image"):
                _run_command(args)
            self.assertFalse(output.exists())

    def test_invalid_detection_json_fails_without_output(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "detections.json"
            invalid.write_text('{"schema_version": 1, "detections": "bad"}',
                               encoding="utf-8")
            output = root / "ocr.json"
            args = Namespace(
                engine="tesseract", image=fixture_root / "val/example.png",
                detections=invalid, output=output, language="eng",
            )
            with self.assertRaises((KeyError, ValueError)):
                _run_command(args)
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        shutil.which("tesseract") and importlib.util.find_spec("pytesseract"),
        "Tesseract and pytesseract are not installed",
    )
    def test_module_command_uses_one_result_class_and_writes_valid_output(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ocr.json"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "src.ocr",
                    "--image", str(fixture_root / "val/example.png"),
                    "--detections", str(fixture_root / "detections.json"),
                    "--output", str(output),
                    "--language", "eng",
                ],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            loaded = load_ocr_frame(output)
        self.assertEqual(summary["regions"], 10)
        self.assertEqual(len(loaded.regions), 10)


if __name__ == "__main__":
    unittest.main()
