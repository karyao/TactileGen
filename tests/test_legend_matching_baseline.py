"""Legend-matching validation report and calibration smoke tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.baseline.legend_matching import (
    TruthChart,
    _report,
    _score_legend_to_bars,
    _score_end_to_end,
    calibrate,
    evaluate,
    load_truth_charts,
)
from src.contracts import (
    Box,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
)


class LegendMatchingBaselineTest(unittest.TestCase):
    def _dataset(self, root: Path):
        images = root / "dataset/images/val"
        images.mkdir(parents=True)
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[11:19, 11:19] = (40, 80, 120)
        image[21:49, 11:19] = (40, 80, 120)
        cv2.imwrite(str(images / "1.png"), image)
        annotation = [{
            "image_index": 1,
            "type": "vbar_categorical",
            "models": [{
                "name": "North",
                "bboxes": [{"x": 10, "y": 20, "w": 10, "h": 30}],
            }],
            "general_figure_info": {"legend": {
                "bbox": {"x": 8, "y": 8, "w": 32, "h": 14},
                "items": [{
                    "model": "North",
                    "preview": {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}},
                    "label": {
                        "text": "North",
                        "bbox": {"x": 22, "y": 10, "w": 15, "h": 10},
                    },
                }],
            }},
        }]
        annotations = root / "annotations.json"
        annotations.write_text(json.dumps(annotation), encoding="utf-8")
        return annotations, root / "dataset"

    def test_oracle_evaluation_writes_metrics_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, data_root = self._dataset(root)
            charts = load_truth_charts(annotations, data_root, "val")
            self.assertEqual(len(charts), 1)
            output = root / "output"
            report = evaluate(
                annotations=annotations,
                data_root=data_root,
                split="val",
                output_dir=output,
                track="oracle",
            )
            self.assertEqual(report["oracle"]["precision"], 1.0)
            self.assertEqual(report["oracle"]["recall"], 1.0)
            self.assertTrue((output / "metrics.json").is_file())
            rows = (output / "assignments.csv").read_text(encoding="utf-8")
            self.assertIn("North", rows)
            self.assertIn("matched", rows)

    def test_calibration_uses_documented_tie_break(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, data_root = self._dataset(root)
            original_imread = cv2.imread
            with patch(
                "src.baseline.legend_matching.cv2.imread",
                side_effect=original_imread,
            ) as imread, patch(
                "src.baseline.legend_matching._score_legend_to_bars",
                wraps=_score_legend_to_bars,
            ) as score:
                report = calibrate(
                    annotations=annotations,
                    data_root=data_root,
                    split="val",
                    output_dir=root / "calibration",
                )
            self.assertEqual(report["chosen"]["max_error"], 0.03)
            self.assertEqual(report["chosen"]["min_margin"], 0.03)
            self.assertEqual(report["chosen"]["f1"], 1.0)
            self.assertEqual(imread.call_count, 1)
            self.assertEqual(score.call_count, 1)

    def test_end_to_end_counts_correct_wrong_missed_and_false_positive_bars(self):
        def bar(identifier, x1, x2):
            return Detection(
                identifier, "vertical_bar", Box(x1, 20, x2, 50), 1.0
            )

        truth_frame = DetectionFrame("val/1.png", 100, 60, (
            bar("t1", 5, 15), bar("t2", 25, 35), bar("t3", 45, 55),
        ))
        chart = TruthChart(
            Path("unused.png"),
            truth_frame,
            OCRFrame("val/1.png", ()),
            {"t1": "A", "t2": "B", "t3": "C"},
        )
        predicted = DetectionFrame("val/1.png", 100, 60, (
            bar("p1", 5, 15), bar("p2", 25, 35), bar("fp", 75, 85),
        ))
        ocr = OCRFrame("val/1.png", (
            RecognizedText("l1", "A", 0.9),
            RecognizedText("l2", "Wrong", 0.8),
            RecognizedText("l3", "Ghost", 0.7),
        ))
        mapping = LegendMappingFrame("val/1.png", (
            LegendBarAssignment("p1", "s1", "l1", "A", 0.99, "matched"),
            LegendBarAssignment("p2", "s2", "l2", "Wrong", 0.98, "matched"),
            LegendBarAssignment("fp", "s3", "l3", "Ghost", 0.97, "matched"),
        ))

        with patch(
            "src.baseline.legend_matching.match_legend_to_bars",
            return_value=mapping,
        ):
            counts, rows = _score_end_to_end(
                chart,
                np.zeros((60, 100, 3), dtype=np.uint8),
                predicted,
                ocr,
                0.5,
                0.03,
                0.01,
            )
        report = _report(counts)

        self.assertEqual(report["truth_bars"], 3)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["incorrect"], 2)
        self.assertEqual(report["emitted_assignments"], 3)
        self.assertAlmostEqual(report["precision"], 1 / 3)
        self.assertAlmostEqual(report["recall"], 1 / 3)
        self.assertAlmostEqual(report["coverage"], 2 / 3)
        self.assertEqual(report["all_bars_correct_chart_accuracy"], 0.0)
        self.assertEqual(report["bar_detection"]["true_positive"], 2)
        self.assertEqual(report["bar_detection"]["false_positive"], 1)
        self.assertEqual(report["bar_detection"]["false_negative"], 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["status"] for row in rows}, {"matched", "missed_detection"}
        )

    def test_programmatic_evaluate_rejects_invalid_settings_before_io(self):
        common = dict(
            annotations=Path("missing.json"), data_root=Path("missing"),
            split="val", output_dir=Path("unused"),
        )
        for name, value in (
            ("track", "typo"), ("match_iou", -0.1),
            ("max_error", 1.1), ("min_margin", float("nan")),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    evaluate(**common, **{name: value})


if __name__ == "__main__":
    unittest.main()
