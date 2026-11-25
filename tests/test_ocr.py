"""OCR orchestration, backends, crops, traces, evidence, and benchmarks."""

from argparse import Namespace
import builtins
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import cv2
import numpy as np

from src.baseline.ocr import (
    compare_text,
    edit_distance,
    evaluate,
    load_truth_images,
    match_regions,
    normalize_text,
    register_engine,
)
from src.contract_io import (
    load_detection_frame,
    load_ocr_frame,
    write_detection_frame,
    write_ocr_frame,
)
from src.contracts import (
    Box,
    CLASS_NAMES,
    Detection,
    DetectionFrame,
    TEXT_ROLES,
    join_text_regions,
)
from src.diagnostics.tracking import AnalysisDiagnostics
from src.ocr import OCRBackendResult, run_ocr
from src.ocr.pipeline import _run_command, parse_args
from src.ocr.crop import _title_strip_bounds, crop_text_bounds, crop_text_region
from src.ocr.trace import adapter_evidence, snapshot
from src.ocr.rapidocr import RapidOCRBackend, _candidate_rotations, _load_rapidocr
from src.ocr.tesseract import TesseractBackend, _load_pytesseract


# Ocr

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
            with patch("src.ocr.tesseract.TesseractBackend", return_value=Backend()):
                result = _run_command(args)
            loaded = load_ocr_frame(output)
            serialized = json.loads(output.read_text(encoding="utf-8"))

        detections = load_detection_frame(fixture_root / "detections.json")
        self.assertEqual(loaded, result)
        expected_count = sum(item.role in TEXT_ROLES for item in detections.detections)
        self.assertEqual(len(join_text_regions(detections, loaded)), expected_count)
        for region in serialized["regions"]:
            self.assertEqual(set(region), {"detection_id", "text", "confidence"})

    def test_command_dispatches_rapidocr_backend(self):
        fixture_root = Path(__file__).parent / "fixtures/contracts"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ocr.json"
            args = Namespace(
                engine="rapidocr",
                image=fixture_root / "val/example.png",
                detections=fixture_root / "detections.json",
                output=output,
                language="eng",
            )
            backend = RecordingBackend()
            with patch(
                "src.ocr.rapidocr.RapidOCRBackend", return_value=backend
            ) as constructor:
                result = _run_command(args)
        constructor.assert_called_once_with()
        self.assertEqual(len(result.regions), len(backend.calls))

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
            with patch("src.ocr.tesseract.TesseractBackend", return_value=RecordingBackend()):
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
            with patch("src.ocr.tesseract.TesseractBackend", return_value=FailingBackend()):
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


# Ocr baseline

class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    names = dict(enumerate(CLASS_NAMES))

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.results)


class RoleBackend:
    TEXT = {
        "chart_title": " SALES ",
        "x_axis_title": "Year",
        "x_tick_label": "A",
        "y_axis_title": "Value",
        "y_tick_label": "0",
        "legend_label": "North",
    }

    def recognize(self, image, detection, *, language):
        return OCRBackendResult(self.TEXT[detection.role], 0.8)


def annotation(image_id=1):
    return {
        "image_index": image_id,
        "type": "vbar_categorical",
        "models": [],
        "general_figure_info": {
            "title": {"text": "Sales", "bbox": {"x": 10, "y": 5, "w": 30, "h": 10}},
            "plot_info": {"bbox": {"x": 10, "y": 20, "w": 80, "h": 60}},
            "x_axis": {
                "label": {"text": "Year", "bbox": {"x": 40, "y": 85, "w": 20, "h": 10}},
                "major_labels": {
                    # PlotQA commonly duplicates identical label/value/box entries.
                    "values": ["A", "A"],
                    "bboxes": [
                        {"x": 20, "y": 75, "w": 10, "h": 8},
                        {"x": 20, "y": 75, "w": 10, "h": 8},
                    ],
                },
            },
            "y_axis": {
                "label": {"text": "Value", "bbox": {"x": 1, "y": 30, "w": 8, "h": 30}},
                "major_labels": {
                    "values": ["0"],
                    "bboxes": [{"x": 10, "y": 65, "w": 8, "h": 8}],
                },
            },
            "legend": {
                "items": [{
                    "label": {
                        "text": "North", "bbox": {"x": 65, "y": 20, "w": 20, "h": 8}
                    }
                }]
            },
        },
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    image = root / "images/test/1.png"
    image.parent.mkdir(parents=True)
    cv2.imwrite(str(image), np.full((100, 100, 3), 255, dtype=np.uint8))
    annotations = root / "annotations.json"
    annotations.write_text(json.dumps([annotation()]), encoding="utf-8")
    return image, annotations


def fake_result(path: Path):
    # Four matches, one unmatched legend prediction, and two missed truth boxes.
    boxes = [
        [10, 5, 40, 15],
        [40, 85, 60, 95],
        [20, 75, 30, 83],
        [10, 65, 18, 73],
        [80, 80, 95, 90],
    ]
    classes = [
        CLASS_NAMES.index("chart_title"),
        CLASS_NAMES.index("x_axis_title"),
        CLASS_NAMES.index("x_tick_label"),
        CLASS_NAMES.index("y_tick_label"),
        CLASS_NAMES.index("legend_label"),
    ]
    return type("Result", (), {
        "path": str(path),
        "orig_shape": (100, 100),
        "boxes": type("Boxes", (), {
            "xyxy": FakeTensor(boxes),
            "cls": FakeTensor(classes),
            "conf": FakeTensor([0.9] * len(boxes)),
        })(),
    })()


class TextMetricTest(unittest.TestCase):
    def test_normalization_preserves_numeric_punctuation(self):
        self.assertEqual(normalize_text("  TOTAL\tSales  "), "total sales")
        self.assertNotEqual(normalize_text("-1.5%"), normalize_text("1.5"))
        comparison = compare_text("Annual Sales", " annual   SALES ")
        self.assertFalse(comparison.raw_exact)
        self.assertTrue(comparison.normalized_exact)
        self.assertEqual((comparison.character_edits, comparison.word_edits), (0, 0))

    def test_edit_distance_handles_characters_and_words(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance(["north", "sales"], ["south", "sales"]), 1)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_matching_is_one_to_one_and_role_constrained(self):
        truth = [
            Detection("a", "chart_title", Box(0, 0, 10, 10), 1),
            Detection("b", "legend_label", Box(20, 0, 30, 10), 1),
        ]
        predicted = [
            Detection("p1", "chart_title", Box(0, 0, 10, 10), 1),
            Detection("p2", "chart_title", Box(0, 0, 10, 10), 1),
            Detection("p3", "chart_title", Box(20, 0, 30, 10), 1),
        ]
        matches, false_positives, false_negatives = match_regions(predicted, truth, 0.5)
        self.assertEqual([(p.id, t.id) for p, t, _ in matches], [("p1", "a")])
        self.assertEqual([item.id for item in false_positives], ["p2", "p3"])
        self.assertEqual([item.id for item in false_negatives], ["b"])


class TruthExtractionTest(unittest.TestCase):
    def test_extracts_text_roles_and_removes_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, annotations = write_fixture(root)
            images, report = load_truth_images(annotations, root, "test")
            self.assertEqual(report, {
                "images": 1,
                "raw_regions": 7,
                "unique_regions": 6,
                "duplicate_regions_removed": 1,
                "empty_boxes_excluded": 0,
            })
            self.assertEqual(
                [item.role for item in images[0].frame.detections],
                ["chart_title", "x_axis_title", "x_tick_label", "y_axis_title",
                 "y_tick_label", "legend_label"],
            )
            self.assertEqual(images[0].text_by_id["gt-00003"], "A")

    def test_rejects_mismatched_tick_values_and_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, annotations = write_fixture(root)
            broken = annotation()
            broken["general_figure_info"]["x_axis"]["major_labels"]["bboxes"].pop()
            annotations.write_text(json.dumps([broken]), encoding="utf-8")
            self.assertTrue(image.is_file())
            with self.assertRaisesRegex(ValueError, "differ in length"):
                load_truth_images(annotations, root, "test")


class OCRBaselineIntegrationTest(unittest.TestCase):
    def test_both_tracks_write_metrics_and_region_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, annotations = write_fixture(root)
            output = root / "output"
            model = FakeModel([fake_result(image)])
            report = evaluate(
                model, RoleBackend, annotations=annotations, data_root=root,
                output_dir=output, weights=Path("fake.pt"), workers=2,
            )
            self.assertEqual(report["dataset"]["unique_regions"], 6)
            self.assertEqual(report["oracle"]["overall"]["normalized_exact_accuracy"], 1)
            self.assertEqual(report["oracle"]["overall"]["raw_exact_accuracy"], 5 / 6)
            detection = report["text_detection"]["overall"]
            self.assertEqual(
                (detection["true_positive"], detection["false_positive"],
                 detection["false_negative"]),
                (4, 1, 2),
            )
            self.assertEqual(
                report["end_to_end"]["overall"]["normalized_exact_accuracy"], 4 / 6
            )
            self.assertEqual(report["end_to_end"]["overall"]["spurious_regions"], 1)
            self.assertEqual(report["matched_box_ocr"]["overall"]["regions"], 4)
            self.assertEqual(len(model.calls), 1)
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset"], report["dataset"])
            with (output / "regions.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 13)  # six oracle + six truth + one false positive
            self.assertEqual(
                {row["status"] for row in rows if row["track"] == "end_to_end"},
                {"true_positive", "false_positive", "false_negative"},
            )

    def test_engine_registry_accepts_another_backend(self):
        from src.baseline.ocr import ENGINE_FACTORIES

        self.assertIn("rapidocr", ENGINE_FACTORIES)
        try:
            register_engine("fake-model", RoleBackend)
            self.assertIs(ENGINE_FACTORIES["fake-model"], RoleBackend)
            with self.assertRaisesRegex(ValueError, "engine name"):
                register_engine("Bad Engine", RoleBackend)
        finally:
            ENGINE_FACTORIES.pop("fake-model", None)


# Ocr crop

def text_detection(box, role="x_tick_label", id="text"):
    return Detection(id, role, Box(*box), 0.9)


class OCRCropTest(unittest.TestCase):
    def test_non_title_crop_rounds_pads_clamps_and_returns_copy(self):
        image = np.arange(20 * 30 * 3, dtype=np.int32).reshape((20, 30, 3))
        crop = crop_text_region(image, text_detection((0.2, 0.4, 8.1, 6.1)))

        self.assertEqual(crop.shape, (9, 11, 3))
        crop[:] = -1
        self.assertFalse(np.any(image == -1))

    def test_title_uses_full_width_strip_and_can_be_disabled(self):
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        title = text_detection((80, 10, 100, 26), "chart_title")

        self.assertEqual(crop_text_region(image, title).shape, (20, 200, 3))
        with patch("src.ocr.crop._TITLE_FULL_WIDTH_STRIP", False):
            self.assertEqual(crop_text_region(image, title).shape, (20, 24, 3))

    def test_title_strip_stops_at_other_text_in_its_band(self):
        title = text_detection((80, 10, 100, 26), "chart_title", "title")
        neighbors = (
            title,
            text_detection((5, 5, 20, 30), "y_axis_title", "ytitle"),
            text_detection((25, 12, 40, 24), "y_tick_label", "ytick"),
            text_detection((150, 8, 190, 28), "legend_label", "legend"),
            text_detection((30, 100, 60, 120), "x_tick_label", "below"),
            text_detection((90, 12, 95, 24), "legend_label", "inside"),
        )

        self.assertEqual(_title_strip_bounds(title, neighbors, 200), (42, 110))
        self.assertEqual(_title_strip_bounds(title, (title,), 200), (0, 200))
        self.assertEqual(
            crop_text_region(np.zeros((60, 200), dtype=np.uint8), title, neighbors).shape,
            (20, 68),
        )


# Tesseract ocr

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


class TesseractBackendTest(unittest.TestCase):
    def test_horizontal_crop_rounds_pads_clamps_upscales_and_forwards_options(self):
        module = fake_pytesseract(
            results=(result("  Annual\t Sales ", "80"), empty_result())
        )
        backend = TesseractBackend(timeout_seconds=2.5)
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        original = image.copy()

        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                image, text_detection((0.2, 0.4, 8.1, 6.1)), language="eng"
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
        title = text_detection((80, 10, 100, 26), "chart_title")  # narrower than the text
        shapes = {}
        for label, item in (("title", title), ("tick", text_detection((80, 10, 100, 26)))):
            module = fake_pytesseract(results=(result(), empty_result()))
            with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
                TesseractBackend().recognize(image, item, language="eng")
            shapes[label] = module.image_to_data.call_args_list[0].args[0].shape
        # 16 px box + 2 px padding each side = 20 rows, doubled to reach 32+.
        self.assertEqual(shapes["title"], (40, 400))
        self.assertEqual(shapes["tick"], (40, 48))

        with patch("src.ocr.crop._TITLE_FULL_WIDTH_STRIP", False):
            module = fake_pytesseract(results=(result(), empty_result()))
            with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
                TesseractBackend().recognize(image, title, language="eng")
        self.assertEqual(module.image_to_data.call_args_list[0].args[0].shape, (40, 48))

    def test_title_strip_stops_at_other_text_in_its_band(self):
        from src.ocr.crop import _title_strip_bounds

        title = text_detection((80, 10, 100, 26), "chart_title", "title")
        neighbors = (
            title,
            text_detection((5, 5, 20, 30), "y_axis_title", "ytitle"),
            text_detection((25, 12, 40, 24), "y_tick_label", "ytick"),
            text_detection((150, 8, 190, 28), "legend_label", "legend"),
            text_detection((30, 100, 60, 120), "x_tick_label", "below"),  # other band
            text_detection((90, 12, 95, 24), "legend_label", "inside"),  # inside title box
        )
        self.assertEqual(_title_strip_bounds(title, neighbors, 200), (42, 110))
        self.assertEqual(_title_strip_bounds(title, (title,), 200), (0, 200))

    def test_recognize_many_gives_title_its_neighbors(self):
        from src.ocr.tesseract import _recognition_candidates

        title = text_detection((80, 10, 100, 26), "chart_title", "title")
        tick = text_detection((10, 12, 40, 24), "y_tick_label", "tick")
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

        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                image, text_detection((10, 2, 14, 18), "y_axis_title"), language="eng"
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
        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            recognized = backend.recognize(
                np.zeros((20, 30), dtype=np.uint8),
                text_detection((2, 2, 20, 10)),
                language="eng",
            )
        self.assertEqual(recognized, OCRBackendResult("One Two Three", 0.5))

        empty_confidence = fake_pytesseract(
            results=(result("Text", "bad"), empty_result())
        )
        with patch("src.ocr.tesseract._load_pytesseract", return_value=empty_confidence):
            recognized = TesseractBackend().recognize(
                np.zeros((20, 30), dtype=np.uint8),
                text_detection((2, 2, 20, 10)),
                language="eng",
            )
        self.assertEqual(recognized, OCRBackendResult("Text", 0.0))

    def test_empty_candidates_return_none_and_grayscale_single_channel_is_supported(self):
        module = fake_pytesseract(results=(result(" ", "90"), result("", "-1")))
        single_channel = np.zeros((20, 30, 1), dtype=np.uint8)
        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            recognized = TesseractBackend().recognize(
                single_channel, text_detection((2, 2, 20, 10)), language="eng"
            )
        self.assertIsNone(recognized)

    def test_language_validation_and_installed_language_cache(self):
        module = fake_pytesseract(
            languages=("eng", "fra"),
            results=(result(), result(), result(), result()),
        )
        backend = TesseractBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        item = text_detection((2, 2, 20, 10))
        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            backend.recognize(image, item, language="eng+fra")
            backend.recognize(image, item, language="eng")
        module.get_languages.assert_called_once_with(config="")

        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
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
        with patch("src.ocr.tesseract._load_pytesseract", return_value=missing):
            with self.assertRaisesRegex(RuntimeError, "not available on PATH"):
                TesseractBackend().recognize(
                    np.zeros((20, 30), dtype=np.uint8),
                    text_detection((2, 2, 20, 10)),
                    language="eng",
                )

        timed_out = fake_pytesseract(results=(RuntimeError("process timeout"),))
        with patch("src.ocr.tesseract._load_pytesseract", return_value=timed_out):
            with self.assertRaisesRegex(RuntimeError, "recognition timed out"):
                TesseractBackend().recognize(
                    np.zeros((20, 30), dtype=np.uint8),
                    text_detection((2, 2, 20, 10)),
                    language="eng",
                )

    def test_rejects_bad_timeout_and_unsupported_channel_count(self):
        for timeout in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    TesseractBackend(timeout)
        module = fake_pytesseract(results=(result(), result()))
        with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
            with self.assertRaisesRegex(ValueError, "grayscale, BGR, or BGRA"):
                TesseractBackend().recognize(
                    np.zeros((20, 30, 2), dtype=np.uint8),
                    text_detection((2, 2, 20, 10)),
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
        with patch("src.ocr.crop._TITLE_FULL_WIDTH_STRIP", False):
            clipped = TesseractBackend().recognize(image, narrow, language="eng")
        self.assertNotEqual(clipped.text, text)


# Rapidocr ocr

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
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)

        recognized = backend.recognize(
            image, text_detection((0.2, 0.4, 8.1, 6.1)), language="eng"
        )

        self.assertEqual(recognized, OCRBackendResult("Sales", 0.8))
        self.assertEqual(constructor_calls, [{"params": {
            "Global.text_score": 0.0,
            "Rec.engine_type": "onnxruntime",
            "Rec.model_type": "small",
            "Rec.ocr_version": "PP-OCRv6",
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }}])
        crop, options = engine.calls[0]
        self.assertEqual(crop.shape, (9, 11, 3))
        self.assertEqual(options, {
            "use_det": False, "use_cls": True, "use_rec": True,
        })

    def test_title_uses_full_width_strip_and_other_roles_keep_padded_crop(self):
        module, engine, _ = fake_rapidocr([output(), output()])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        title = text_detection((80, 10, 100, 26), "chart_title")
        tick = text_detection((80, 10, 100, 26))

        backend.recognize(image, title, language="eng")
        backend.recognize(image, tick, language="eng")

        self.assertEqual([call[0].shape for call in engine.calls], [
            (20, 200, 3), (20, 24, 3),
        ])

    def test_recognize_accepts_neighbors_for_title_strip_bounds(self):
        module, engine, _ = fake_rapidocr([output()])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        title = text_detection((80, 10, 100, 26), "chart_title", "title")
        tick = text_detection((10, 12, 40, 24), id="tick")

        backend.recognize(
            np.zeros((60, 200, 3), dtype=np.uint8),
            title,
            language="eng",
            neighbors=(title, tick),
        )

        self.assertEqual(engine.calls[0][0].shape, (20, 158, 3))

    def test_tall_crop_tries_both_rotations_and_keeps_best_confidence(self):
        module, engine, _ = fake_rapidocr([
            output("vertical", 0.4), output("clockwise", 0.95),
            output("counter", 0.7),
        ])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((25, 30, 4), dtype=np.uint8)
        original = image.copy()

        recognized = backend.recognize(
            image, text_detection((10, 2, 14, 18), "y_axis_title"), language="eng"
        )

        self.assertEqual(recognized, OCRBackendResult("clockwise", 0.95))
        self.assertEqual([call[0].shape for call in engine.calls], [
            (20, 8, 3), (8, 20, 3), (8, 20, 3),
        ])
        self.assertTrue(np.array_equal(image, original))

    def test_recognize_normalizes_supported_images_to_contiguous_bgr(self):
        gray = np.arange(8 * 8, dtype=np.uint16).reshape((8, 8))
        cases = (
            ("grayscale", gray, np.repeat(gray[:, :, None], 3, axis=2)),
            (
                "single-channel",
                gray[:, :, None],
                np.repeat(gray[:, :, None], 3, axis=2),
            ),
            (
                "bgr",
                np.dstack((gray, gray + 100, gray + 200))[:, ::-1],
                np.dstack((gray, gray + 100, gray + 200))[:, ::-1],
            ),
            (
                "bgra",
                np.dstack((gray, gray + 100, gray + 200, gray + 300)),
                np.dstack((gray, gray + 100, gray + 200)),
            ),
        )
        for label, image, expected in cases:
            with self.subTest(label=label):
                module, engine, _ = fake_rapidocr([output()])
                with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
                    backend = RapidOCRBackend()

                backend.recognize(
                    image, text_detection((2, 2, 6, 6)), language="eng"
                )

                candidate = engine.calls[0][0]
                self.assertEqual(candidate.shape, (8, 8, 3))
                self.assertEqual(candidate.dtype, image.dtype)
                self.assertTrue(candidate.flags.c_contiguous)
                self.assertTrue(np.array_equal(candidate, expected))

    def test_recognize_many_normalizes_candidates_and_rejects_bad_channels(self):
        module, engine, _ = fake_rapidocr([output()])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10
        bgra[:, :, 1] = 20
        bgra[:, :, 2] = 30
        bgra[:, :, 3] = 255

        backend.recognize_many(
            bgra, (text_detection((2, 2, 6, 6)),), language="eng"
        )

        candidate = engine.cls_calls[0][0]
        self.assertEqual(candidate.shape, (8, 8, 3))
        self.assertTrue(candidate.flags.c_contiguous)
        self.assertTrue(np.all(candidate == np.array([10, 20, 30])))

        with self.assertRaisesRegex(ValueError, "grayscale, BGR, or BGRA"):
            backend.recognize_many(
                np.zeros((8, 8, 2), dtype=np.uint8),
                (text_detection((2, 2, 6, 6)),),
                language="eng",
            )

    def test_recognize_many_empty_batch_does_not_call_engine(self):
        module, engine, _ = fake_rapidocr([])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()

        self.assertEqual(
            backend.recognize_many(
                np.zeros((8, 8, 2), dtype=np.uint8), (), language="eng"
            ),
            (),
        )
        self.assertEqual(engine.cls_calls, [])
        self.assertEqual(engine.rec_calls, [])

    def test_blank_is_unreadable_and_bad_output_fails(self):
        module, _, _ = fake_rapidocr([output("  ", 0.9)])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        item = text_detection((2, 2, 20, 10))
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
                with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
                    backend = RapidOCRBackend()
                with self.assertRaisesRegex(RuntimeError, message):
                    backend.recognize(image, item, language="eng")

    def test_recognize_many_preserves_detection_order(self):
        module, _, _ = fake_rapidocr([output("first", 0.9), output("second", 0.8)])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        items = (
            text_detection((1, 1, 10, 7)),
            Detection("second", "legend_label", Box(12, 1, 21, 7), 0.9),
        )
        results = backend.recognize_many(
            np.zeros((20, 30, 3), dtype=np.uint8), items, language="eng"
        )
        self.assertEqual([item.text for item in results], ["first", "second"])
        # one crop at a time: batched recognition pads narrow crops to the widest and misreads them
        self.assertEqual([len(call) for call in backend._engine.cls_calls], [1, 1])
        self.assertEqual([len(call) for call in backend._engine.rec_calls], [1, 1])

    def test_recognize_many_reads_each_crop_alone(self):
        """A batch pads narrow crops to the widest and misreads them; every crop must be its own call."""
        class PaddingEngine(FakeEngine):
            def recognize_txt(self, images):
                self.rec_calls.append(images)
                widest = max(image.shape[1] for image in images)
                texts = tuple("wide" if image.shape[1] == widest else "garbled" for image in images)
                return types.SimpleNamespace(txts=texts, scores=(0.9,) * len(images))

        engine = PaddingEngine([])
        module = types.SimpleNamespace(
            RapidOCR=lambda **kwargs: engine,
            EngineType=types.SimpleNamespace(ONNXRUNTIME="onnxruntime"),
            ModelType=types.SimpleNamespace(SMALL="small"),
            OCRVersion=types.SimpleNamespace(PPOCRV6="PP-OCRv6"),
        )
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        image = np.zeros((40, 120, 3), dtype=np.uint8)
        items = (text_detection((5, 5, 60, 15), "legend_label", "wide"),
                 Detection("narrow", "legend_label", Box(70, 5, 80, 15), 0.9))
        many = backend.recognize_many(image, items, language="eng")
        self.assertEqual([item.text for item in many], ["wide", "wide"])  # each crop was the widest in its own call
        self.assertTrue(all(len(call) == 1 for call in engine.rec_calls))

    def test_recognize_many_gives_title_all_detection_neighbors(self):
        module, engine, _ = fake_rapidocr([output("title"), output("tick")])
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        title = text_detection((80, 10, 100, 26), "chart_title", "title")
        tick = text_detection((10, 12, 40, 24), id="tick")

        backend.recognize_many(
            np.zeros((60, 200, 3), dtype=np.uint8),
            (title, tick),
            language="eng",
        )

        self.assertEqual([candidate.shape for call in engine.cls_calls for candidate in call], [
            (20, 158, 3), (16, 34, 3),
        ])

    def test_missing_dependency_error_is_actionable(self):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "rapidocr":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, "requirements-ocr.txt"):
                _load_rapidocr()


# Ocr trace

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


def recognize_with_fake_rapidocr(raw, items=(SHORT,), *, diagnostics=None, cls_error=None):
    """Run OCR with a fake RapidOCR engine and return its result and engine."""
    module, engine, _ = fake_rapidocr([])
    total = sum(len(_candidate_rotations(item)) for item in items)
    texts, scores = getattr(raw, "txts", None), getattr(raw, "scores", None)
    aligned = (isinstance(texts, (tuple, list)) and isinstance(scores, (tuple, list))
               and len(texts) == len(scores) == total > 1)
    sliced = iter(range(total))

    def respond(images):
        # The engine is called once per crop; a batch-shaped fake response is handed out one item at a time.
        if isinstance(raw, Exception):
            raise raw
        if not aligned:
            return raw
        index = next(sliced)
        return types.SimpleNamespace(txts=(texts[index],), scores=(scores[index],))

    engine.recognize_txt = Mock(side_effect=respond)
    if cls_error:
        engine.cls_and_rotate = Mock(side_effect=cls_error)
    with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
        backend = RapidOCRBackend()
    result = recognize(backend, items, diagnostics)
    return result, engine


def recognize_with_fake_tesseract(raw, items=(SHORT,), *, diagnostics=None):
    """Run OCR with fake Tesseract and retain the submitted image pages."""
    module = fake_pytesseract(results=[raw])
    module.submitted_pages = []
    if not isinstance(raw, Exception):
        def response(path, **kwargs):
            from PIL import Image, ImageSequence
            with Image.open(path) as image:
                module.submitted_pages.extend(np.array(page) for page in ImageSequence.Iterator(image))
            return raw
        module.image_to_data.side_effect = response
    with patch("src.ocr.tesseract._load_pytesseract", return_value=module):
        result = recognize(TesseractBackend(), items, diagnostics)
    return result, module


class OCRTraceTest(unittest.TestCase):
    def test_rapid_exact_strings_all_attempts_and_first_tie(self):
        raw = types.SimpleNamespace(txts=(" \t雪 e\u0301\n", "tie", " \n", ""),
                                    scores=(np.float32(.75), .75, float("nan"), .9))
        diagnostics = collector()
        ordinary, plain_engine = recognize_with_fake_rapidocr(raw, (TALL, SHORT))
        traced, engine = recognize_with_fake_rapidocr(raw, (TALL, SHORT), diagnostics=diagnostics)
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
        self.assertEqual(engine.recognize_txt.call_count, 4)
        self.assertEqual([len(call) for call in engine.cls_calls], [1, 1, 1, 1])
        for left, right in zip(plain_engine.recognize_txt.call_args_list, engine.recognize_txt.call_args_list):
            for left_image, right_image in zip(left.args[0], right.args[0]):
                np.testing.assert_array_equal(left_image, right_image)
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
        plain, plain_module = recognize_with_fake_tesseract(raw, (TALL,))
        traced, module = recognize_with_fake_tesseract(raw, (TALL,), diagnostics=diagnostics)
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
                recognize_with_fake_rapidocr(raw, (TALL,), diagnostics=current)
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
                        recognize_with_fake_rapidocr(raw, diagnostics=current)
                trace = diagnostics.record["ocr_trace"]
                fields = trace["batches"][0]["raw_output"]
                self.assertEqual(fields["txts"]["presence"], "present" if hasattr(raw, "txts") else "absent")
                if hasattr(raw, "txts"):
                    self.assertEqual(fields["txts"]["value"], snapshot(raw.txts))
                self.assertEqual(trace["attempts"][0]["output_availability"], "available")
                self.assertEqual(set(trace["attempts"][0]["raw_output"]), {"txts", "scores"})

    def test_invalid_rapid_confidences_preserve_error_and_raw_value(self):
        for score in (True, -.1, 1.1, "90", float("nan"), float("inf"),
                      np.longdouble('-.1'), np.longdouble('1.1'),
                      np.longdouble('nan'), np.longdouble('inf')):
            with self.subTest(score=score):
                diagnostics = collector()
                for current in (None, diagnostics):
                    with self.assertRaisesRegex(RuntimeError, "confidence"):
                        recognize_with_fake_rapidocr(output("text", score), diagnostics=current)
                self.assertEqual(diagnostics.record["ocr_trace"]["attempts"][0]["raw_output"]["confidence"], snapshot(score))
                json.dumps(diagnostics.record, allow_nan=False)

    def test_extended_numpy_confidence_preserves_rapid_results_and_exports(self):
        raw = output(' label ', np.longdouble('.75'))
        diagnostics = collector()
        ordinary, plain_engine = recognize_with_fake_rapidocr(raw)
        traced, engine = recognize_with_fake_rapidocr(raw, diagnostics=diagnostics)
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
        plain, _ = recognize_with_fake_tesseract(raw)
        traced, _ = recognize_with_fake_tesseract(raw, diagnostics=diagnostics)
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
                        recognize_with_fake_tesseract(raw, diagnostics=current)
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
                        recognize_with_fake_tesseract(error, diagnostics=diagnostics)
                    else:
                        recognize_with_fake_rapidocr(
                            error, diagnostics=diagnostics,
                            cls_error=error if engine == "classifier" else None,
                        )
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

    def test_rapid_midway_failure_keeps_each_crops_submission_and_response(self):
        third = Detection("third", "legend_label", Box(32, 2, 45, 10), .9)
        items = (SHORT, Detection("second", "legend_label", Box(20, 2, 31, 10), .9), third)
        module, engine, _ = fake_rapidocr([])
        failure = RuntimeError("crop two failed")
        engine.recognize_txt = Mock(side_effect=[output(" first ", .75), failure])
        diagnostics = collector()
        with patch("src.ocr.rapidocr._load_rapidocr", return_value=module):
            with self.assertRaises(RuntimeError) as caught:
                recognize(RapidOCRBackend(), items, diagnostics)

        self.assertIs(caught.exception, failure)
        self.assertEqual(engine.recognize_txt.call_count, 2)
        trace = diagnostics.record["ocr_trace"]
        attempts = trace["attempts"]
        self.assertEqual([a["detection_id"] for a in attempts], [d.id for d in items])
        self.assertEqual([a["recognition_submission"] for a in attempts],
                         ["submitted", "submitted", "not_submitted"])
        self.assertEqual([a["output_availability"] for a in attempts],
                         ["available", "unavailable", "unavailable"])
        self.assertEqual(attempts[0]["raw_output"],
                         {"text": " first ", "text_state": "text", "confidence": .75})
        self.assertEqual([r["submission"] for r in trace["results"]],
                         ["submitted", "submitted", "not_submitted"])
        self.assertEqual(trace["batches"][0]["outcome"], "failed")
        self.assertEqual(trace["batches"][0]["operation"], "recognize_txt")
        json.dumps(trace, allow_nan=False)

    def test_preprocessing_failure_leaves_attempts_unsubmitted(self):
        diagnostics = collector()
        with patch("src.ocr.rapidocr._recognition_candidates", side_effect=ValueError("bad image")):
            with self.assertRaisesRegex(ValueError, "bad image"):
                recognize_with_fake_rapidocr(output(), (SHORT, TALL), diagnostics=diagnostics)
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
        recognize_with_fake_rapidocr(output("雪", .9), diagnostics=first)
        recognize_with_fake_rapidocr(output("other", .8), diagnostics=first)
        recognize_with_fake_rapidocr(output("雪", .9), diagnostics=second)
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
                result, engine = recognize_with_fake_rapidocr(output(), (), diagnostics=diagnostics)
                engine.recognize_txt.assert_not_called()
                self.assertEqual(engine.cls_calls, [])
            else:
                result, module = recognize_with_fake_tesseract({}, (), diagnostics=diagnostics)
                module.image_to_data.assert_not_called()
            self.assertEqual(result.regions, ())
            self.assertEqual(diagnostics.record["ocr_trace"]["attempts"], [])

    def test_tesseract_missing_text_or_confidence_is_distinct_from_empty_tokens(self):
        for missing in ("text", "conf"):
            with self.subTest(missing=missing):
                raw = tesseract_data([[("A", "90")], [("", "-1")]])
                del raw[missing]
                diagnostics = collector()
                plain, _ = recognize_with_fake_tesseract(raw)
                traced, _ = recognize_with_fake_tesseract(raw, diagnostics=diagnostics)
                self.assertEqual(plain, traced)
                trace = diagnostics.record["ocr_trace"]
                self.assertNotIn(missing, trace["batches"][0]["raw_output"])
                self.assertIsNone(trace["results"][0]["adapter_result"])
                self.assertTrue(all(a["output_availability"] == "available" for a in trace["attempts"]))

    def test_tesseract_preprocessing_failure_never_submits(self):
        diagnostics = collector()
        with patch("src.ocr.tesseract._recognition_candidates", side_effect=ValueError("bad image")):
            with self.assertRaisesRegex(ValueError, "bad image"):
                recognize_with_fake_tesseract({}, (SHORT, TALL), diagnostics=diagnostics)
        attempts = diagnostics.record["ocr_trace"]["attempts"]
        self.assertEqual(len(attempts), 8)
        self.assertTrue(all(a["submission"] == "not_submitted" for a in attempts))


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


# Ocr image evidence

def decode(diagnostics, evidence):
    return cv2.imdecode(np.frombuffer(diagnostics.assets[evidence['filename']], np.uint8),
                        cv2.IMREAD_UNCHANGED)


class OCRImageEvidenceTest(unittest.TestCase):
    def test_source_coordinates_pixels_and_app_collection_is_idempotent(self):
        title = Detection('../../title', 'chart_title', Box(25, 3, 35, 12), .9)
        neighbor = Detection('neighbor', 'legend_label', Box(55, 2, 59, 14), .9)
        edge = Detection('edge', 'x_tick_label', Box(.2, 34, 5.1, 39.9), .9)
        frame = DetectionFrame('input/chart.png', 60, 40, (title, neighbor, edge, TALL))
        diagnostics = AnalysisDiagnostics(settings={})
        diagnostics.source_images(IMAGE, frame)
        backend = types.SimpleNamespace(recognize=lambda *args, **kwargs: None)
        run_ocr(IMAGE, frame, backend, 'eng', diagnostics=diagnostics)
        sources = diagnostics.record['ocr_source_crops']
        self.assertEqual(len(sources), 4)
        self.assertEqual(sources[0]['crop_policy'], 'neighbor_aware_title_strip')
        self.assertEqual(sources[2]['crop_bounds'], [0, 32, 8, 40])
        for source, detection in zip(sources, frame.detections):
            bounds = crop_text_bounds(IMAGE, detection, frame.detections)
            self.assertEqual(source['crop_bounds'], list(bounds))
            left, top, right, bottom = bounds
            np.testing.assert_array_equal(decode(diagnostics, source['image']),
                                          IMAGE[top:bottom, left:right])
            np.testing.assert_array_equal(crop_text_region(IMAGE, detection, frame.detections),
                                          IMAGE[top:bottom, left:right])
            self.assertNotIn('..', source['image']['filename'])
            self.assertEqual(source['detection_box']['x1'], detection.box.x1)

    def test_rapid_images_are_after_classifier_rotation(self):
        module, engine, _ = fake_rapidocr([output()] * 4)
        def classify(images):
            oriented = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
            return oriented, types.SimpleNamespace(cls_res=[('180', .99)] * len(images),
                                                   elapse=.01)
        engine.cls_and_rotate = Mock(side_effect=classify)
        diagnostics = AnalysisDiagnostics(settings={})
        with patch('src.ocr.rapidocr._load_rapidocr', return_value=module):
            backend = RapidOCRBackend()
        frame = DetectionFrame('input/chart.png', 60, 40, (SHORT, TALL))
        run_ocr(IMAGE, frame, backend, 'eng', diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        self.assertEqual(len(attempts), 4)
        submitted_images = [image for call in engine.rec_calls for image in call]
        for attempt, submitted in zip(attempts, submitted_images):
            np.testing.assert_array_equal(decode(diagnostics, attempt['recognition_image']), submitted)
            self.assertEqual(attempt['preprocessing']['classifier_orientation'], ['180', .99])
            self.assertIsNotNone(attempt['source_crop_id'])
        self.assertEqual(engine.cls_and_rotate.call_count, 4)
        self.assertEqual(len(engine.rec_calls), 4)
        self.assertTrue(all(len(call) == 1 for call in engine.rec_calls))
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics.finish()
            path = diagnostics.export_zip(Path(temporary))
            with ZipFile(path) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                self.assertNotIn('input.png', archive.namelist())
                for attempt in record['ocr_trace']['attempts']:
                    self.assertIn(attempt['recognition_image']['filename'], archive.namelist())

    def test_tesseract_pngs_equal_actual_multipage_tiff_pixels(self):
        diagnostics = AnalysisDiagnostics(settings={})
        raw = tesseract_data([[('label', '90')]] * 8)
        _, module = recognize_with_fake_tesseract(raw, (SHORT, TALL), diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        self.assertEqual(len(attempts), 8)
        for attempt, submitted in zip(attempts, module.submitted_pages):
            np.testing.assert_array_equal(decode(diagnostics, attempt['recognition_image']), submitted)
            self.assertEqual(attempt['recognition_image']['channel_layout'], 'GRAY')
        self.assertEqual({a['preprocessing']['variant'] for a in attempts},
                         {'grayscale', 'otsu_binary'})
        self.assertEqual(module.image_to_data.call_count, 1)

    def test_encoding_failure_retains_result_and_explicit_unavailability(self):
        diagnostics = AnalysisDiagnostics(settings={})
        with patch('cv2.imencode', side_effect=RuntimeError('encoder unavailable')):
            result, _ = recognize_with_fake_rapidocr(output('label'), diagnostics=diagnostics)
        self.assertEqual(result.regions[0].text, 'label')
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'unavailable')
        self.assertIsNone(attempt['recognition_image']['filename'])
        self.assertEqual(diagnostics.assets, {})
        self.assertTrue(diagnostics.record['exceptions'])

    def test_failures_retain_prepared_images_without_inventing_submission(self):
        diagnostics = AnalysisDiagnostics(settings={})
        with self.assertRaisesRegex(RuntimeError, 'classification failed'):
            recognize_with_fake_rapidocr(
                output(), diagnostics=diagnostics,
                cls_error=RuntimeError('classification failed'),
            )
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'unavailable')
        self.assertEqual(attempt['recognition_submission'], 'not_submitted')
        self.assertEqual(len(diagnostics.record['ocr_source_crops']), 1)
        diagnostics = AnalysisDiagnostics(settings={})
        with self.assertRaisesRegex(RuntimeError, 'recognition failed'):
            recognize_with_fake_rapidocr(RuntimeError('recognition failed'), diagnostics=diagnostics)
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'available')
        self.assertEqual(attempt['recognition_submission'], 'submitted')

    def test_repeated_calls_and_runs_have_distinct_input_paths(self):
        diagnostics = AnalysisDiagnostics(settings={})
        recognize_with_fake_rapidocr(output(), diagnostics=diagnostics)
        recognize_with_fake_rapidocr(output(), diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        paths = [a['recognition_image']['filename'] for a in attempts]
        self.assertEqual(len(set(paths)), 2)
        other = AnalysisDiagnostics(settings={})
        self.assertEqual(other.assets, {})
        self.assertNotEqual(diagnostics.run_id, other.run_id)

    def test_grayscale_sources_and_unavailable_classifier_metadata(self):
        gray = IMAGE[:, :, 0].copy()
        module, engine, _ = fake_rapidocr([output()])
        with patch('src.ocr.rapidocr._load_rapidocr', return_value=module):
            backend = RapidOCRBackend()
        diagnostics = AnalysisDiagnostics(settings={})
        frame = DetectionFrame('input/gray.png', 60, 40, (SHORT,))
        run_ocr(gray, frame, backend, 'eng', diagnostics=diagnostics)
        source = diagnostics.record['ocr_source_crops'][0]
        np.testing.assert_array_equal(decode(diagnostics, source['image']),
                                      crop_text_region(gray, SHORT, (SHORT,)))
        self.assertEqual(source['image']['channel_layout'], 'GRAY')
        trace = diagnostics.record['ocr_trace']
        self.assertEqual(trace['batches'][0]['classifier_output']['availability'], 'unavailable')
        self.assertEqual(trace['attempts'][0]['preprocessing']['classifier_orientation'], 'unavailable')
        np.testing.assert_array_equal(decode(diagnostics, trace['attempts'][0]['recognition_image']),
                                      engine.rec_calls[0][0])

    def test_unsupported_png_dtype_is_explicit_without_lossy_conversion(self):
        diagnostics = AnalysisDiagnostics(settings={})
        evidence = diagnostics.image_evidence('test.png', IMAGE.astype(np.float32),
                                              channel_layout='BGR')
        self.assertEqual(evidence['availability'], 'unavailable')
        self.assertEqual(evidence['dtype'], 'float32')
        self.assertIsNone(evidence['filename'])
        self.assertEqual(diagnostics.assets, {})


if __name__ == "__main__":
    unittest.main()
