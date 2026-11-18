"""ChartQA conversion to the shared YOLO detection schema."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from src.contracts import CLASS_NAMES
from src.dataset.prepare_chartqa import convert


def png(path: Path, width: int = 100, height: int = 80) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", width, height))


def chart(kind: str = "v_bar") -> dict:
    return {
        "type": kind,
        "models": [{"bboxes": [{"x": 10, "y": 20, "w": 30, "h": 40}]}],
        "general_figure_info": {
            "title": {"bbox": {"x": 2, "y": 2, "w": 20, "h": 8}},
            "figure_info": {"bbox": {"x": 5, "y": 10, "w": 80, "h": 60}},
            "x_axis": {"label": {}, "major_labels": {"bboxes": []}},
            "y_axis": {"label": {}, "major_labels": {"bboxes": []}},
        },
    }


class PrepareChartQATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations"
        self.images = self.root / "png"
        self.annotations.mkdir()
        self.images.mkdir()

    def add(self, image_id: str, annotation: dict | None) -> None:
        if annotation is not None:
            (self.annotations / f"{image_id}.json").write_text(json.dumps(annotation))
        png(self.images / f"{image_id}.png")

    def test_maps_bars_and_figure_boxes_to_shared_classes(self):
        self.add("chart_a", chart())
        self.add("chart_b", chart("h_bar"))
        output = self.root / "out"
        counts = convert(self.annotations, self.images, output)
        self.assertEqual(counts["images"], 2)
        self.assertEqual((output / "classes.txt").read_text().splitlines(), list(CLASS_NAMES))
        rows = (output / "labels/train/chart_a.txt").read_text().splitlines()
        self.assertIn("1 0.25000000 0.50000000 0.30000000 0.50000000", rows)
        self.assertIn("11 0.45000000 0.50000000 0.80000000 0.75000000", rows)
        self.assertTrue((output / "images/train/chart_a.png").exists())
        self.assertIn("chart_b,h_bar", (output / "chart_types_train.csv").read_text())

    def test_skips_unsupported_chart_type_and_honors_filter(self):
        self.add("bar", chart())
        self.add("pie", chart("pie"))
        output = self.root / "out"
        counts = convert(self.annotations, self.images, output, ["h_bar"])
        self.assertEqual(counts["images"], 0)
        self.assertFalse(any((output / "images/train").iterdir()))
        self.assertEqual(counts["unsupported_chart_types"], 1)

    def test_rejects_nonempty_output(self):
        self.add("chart", chart())
        output = self.root / "out"
        (output / "images/train").mkdir(parents=True)
        (output / "images/train/old.png").touch()
        with self.assertRaises(FileExistsError):
            convert(self.annotations, self.images, output)

    def test_missing_annotation_directory_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            convert(self.root / "missing", self.images, self.root / "out")


if __name__ == "__main__":
    unittest.main()
