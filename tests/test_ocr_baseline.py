"""Authoritative PlotQA extraction and engine-neutral OCR benchmark scoring."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

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
from src.contracts import CLASS_NAMES, Box, Detection
from src.ocr import OCRBackendResult


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

        try:
            register_engine("fake-model", RoleBackend)
            self.assertIs(ENGINE_FACTORIES["fake-model"], RoleBackend)
            with self.assertRaisesRegex(ValueError, "engine name"):
                register_engine("Bad Engine", RoleBackend)
        finally:
            ENGINE_FACTORIES.pop("fake-model", None)


if __name__ == "__main__":
    unittest.main()
