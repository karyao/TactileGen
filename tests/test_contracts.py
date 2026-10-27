"""Stage-handoff examples and compatibility checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.baseline.opencv_bars import Box as BaselineBox
from src.contract_io import (
    load_detection_frame,
    load_ocr_frame,
    load_vertical_bar_chart,
    write_detection_frame,
    write_ocr_frame,
    write_vertical_bar_chart,
)
from src.contracts import (
    CLASS_NAMES,
    BarSeries,
    Box,
    Detection,
    DetectionFrame,
    OCRFrame,
    RecognizedText,
    VerticalBarChart,
    join_text_regions,
)
from src.dataset.prepare_plotqa import CLASS_NAMES as ConverterClassNames


FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class ContractTest(unittest.TestCase):
    def test_existing_class_order_and_box_are_shared(self):
        self.assertEqual(CLASS_NAMES, (
            "horizontal_bar", "vertical_bar", "dot", "line_segment",
            "chart_title", "x_axis_title", "y_axis_title", "x_tick_label",
            "y_tick_label", "x_axis_rule", "y_axis_rule", "plot_area",
            "legend", "legend_swatch", "legend_label",
        ))
        self.assertIs(ConverterClassNames, CLASS_NAMES)
        self.assertIs(BaselineBox, Box)

    def test_fixture_handoffs_and_round_trip(self):
        detections = load_detection_frame(FIXTURES / "detections.json")
        ocr = load_ocr_frame(FIXTURES / "ocr.json")
        chart = load_vertical_bar_chart(FIXTURES / "vertical_bar_chart.json")
        self.assertEqual(detections.source_key, ocr.source_key)
        self.assertEqual(ocr.source_key, chart.source_key)
        self.assertEqual(detections.source_key, "val/example.png")
        self.assertEqual(len(join_text_regions(detections, ocr)), len(ocr.regions))
        self.assertEqual(len(chart.series), 2)
        self.assertEqual(chart.series[0].values, (4, 6))

        from src.dataset.prepare_plotqa import png_size
        self.assertEqual(
            png_size(FIXTURES / detections.source_key),
            (detections.width, detections.height),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_detection_frame(root / "detections.json", detections)
            write_ocr_frame(root / "ocr.json", ocr)
            write_vertical_bar_chart(root / "chart.json", chart)
            self.assertEqual(load_detection_frame(root / "detections.json"), detections)
            self.assertEqual(load_ocr_frame(root / "ocr.json"), ocr)
            self.assertEqual(load_vertical_bar_chart(root / "chart.json"), chart)

    def test_rejects_unknown_role_box_and_confidence(self):
        with self.assertRaisesRegex(ValueError, "source_key"):
            DetectionFrame("22410", 10, 10, ())
        with self.assertRaisesRegex(ValueError, "Unknown detection role"):
            Detection("a", "not_a_role", Box(0, 0, 1, 1), 0.5)
        with self.assertRaisesRegex(ValueError, "Box outside image"):
            DetectionFrame("val/a.png", 10, 10, (
                Detection("a", "vertical_bar", Box(9, 0, 11, 2), 0.5),
            ))
        with self.assertRaisesRegex(ValueError, "confidence"):
            Detection("a", "vertical_bar", Box(0, 0, 1, 1), math.nan)

    def test_rejects_duplicate_and_broken_ocr_links(self):
        detection = Detection("a", "chart_title", Box(0, 0, 5, 5), 0.9)
        with self.assertRaisesRegex(ValueError, "Duplicate detection ID"):
            DetectionFrame("val/a.png", 10, 10, (detection, detection))
        frame = DetectionFrame("val/a.png", 10, 10, (detection,))
        with self.assertRaisesRegex(ValueError, "different source images"):
            join_text_regions(frame, OCRFrame("test/a.png", ()))
        with self.assertRaisesRegex(ValueError, "unknown detection"):
            join_text_regions(frame, OCRFrame("val/a.png", (
                RecognizedText("missing", "A", 0.9),
            )))
        non_text = DetectionFrame("val/a.png", 10, 10, (
            Detection("bar", "vertical_bar", Box(0, 0, 5, 5), 0.9),
        ))
        with self.assertRaisesRegex(ValueError, "non-text role"):
            join_text_regions(non_text, OCRFrame("val/a.png", (
                RecognizedText("bar", "A", 0.9),
            )))

    def test_rejects_incomplete_chart(self):
        with self.assertRaisesRegex(ValueError, "category count"):
            VerticalBarChart("val/a.png", "Title", "X", "Y", ("A", "B"), (
                BarSeries("North", (1,)),
            ))

    def test_rejects_unsupported_json_version(self):
        data = json.loads((FIXTURES / "detections.json").read_text(encoding="utf-8"))
        data["schema_version"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version 1"):
                load_detection_frame(path)


if __name__ == "__main__":
    unittest.main()
