"""Framework-independent formatting and overlay helpers."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.contracts import Box, Detection, DetectionFrame
from src.presentation import detection_rows, render_detection_overlay, summarize_detections


def sample_frame(*detections: Detection, width: int = 120, height: int = 100):
    return DetectionFrame("input/chart.png", width, height, detections)


class PresentationTest(unittest.TestCase):
    def test_module_import_does_not_require_frontend_or_model_packages(self):
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'gradio', 'ultralytics'}:
        raise ImportError(f'blocked dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import src.presentation
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_summary_groups_roles_in_contract_order(self):
        frame = sample_frame(
            Detection("title", "chart_title", Box(5, 5, 60, 15), 0.8),
            Detection("bar-1", "vertical_bar", Box(20, 30, 35, 90), 0.95),
            Detection("bar-2", "vertical_bar", Box(50, 40, 65, 90), 0.9),
        )

        summary = summarize_detections(frame)

        self.assertIn("Detected 3 chart elements.", summary)
        self.assertIn("vertical bar: 2; chart title: 1", summary)
        self.assertIn("Confidence range: 80.0% to 95.0%.", summary)
        self.assertIn("Vertical bars: 2.", summary)
        self.assertTrue(summary.endswith("Chart text and numeric values have not been read."))

    def test_summary_treats_zero_detections_as_valid(self):
        summary = summarize_detections(sample_frame())
        self.assertIn("Detected 0 chart elements.", summary)
        self.assertIn("No roles were detected.", summary)
        self.assertIn("Vertical bars: 0.", summary)
        self.assertNotIn("Confidence range", summary)

    def test_table_preserves_order_and_numeric_values(self):
        frame = sample_frame(
            Detection("det-2", "x_tick_label", Box(1.5, 2.5, 10.5, 12.5), 0.875),
            Detection("det-1", "dot", Box(20, 21, 24, 25), 0.5),
        )

        rows = detection_rows(frame)

        self.assertEqual([row["Detection ID"] for row in rows], ["det-2", "det-1"])
        self.assertEqual(rows[0], {
            "Detection ID": "det-2", "Role": "x tick label", "Confidence": 0.875,
            "X1": 1.5, "Y1": 2.5, "X2": 10.5, "Y2": 12.5,
        })
        self.assertEqual(list(rows[0]), [
            "Detection ID", "Role", "Confidence", "X1", "Y1", "X2", "Y2",
        ])

    def test_overlay_draws_every_group_without_mutating_input(self):
        frame = sample_frame(
            Detection("bar", "vertical_bar", Box(10, 25, 25, 90), 0.91),
            Detection("title", "chart_title", Box(35, 10, 75, 20), 0.82),
            Detection("axis", "x_axis_rule", Box(5, 90, 100, 94), 0.77),
            Detection("legend", "legend_swatch", Box(100, 25, 115, 40), 0.88),
        )
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        original = image.copy()

        overlay = render_detection_overlay(image, frame)

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(overlay.shape, image.shape)
        self.assertFalse(np.array_equal(overlay, image))
        for x, y in ((10, 30), (35, 15), (5, 90), (100, 30)):
            self.assertTrue(np.any(overlay[y, x]))

    def test_structure_outline_is_dashed_and_solid_outline_is_continuous(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        solid = render_detection_overlay(image, sample_frame(
            Detection("bar", "vertical_bar", Box(10, 20, 100, 80), 0.9),
        ))
        dashed = render_detection_overlay(image, sample_frame(
            Detection("plot", "plot_area", Box(10, 20, 100, 80), 0.9),
        ))

        # Away from the label, a long solid edge has fewer black gaps than a dashed edge.
        solid_gaps = np.count_nonzero(np.all(solid[20, 20:95] == 0, axis=1))
        dashed_gaps = np.count_nonzero(np.all(dashed[20, 20:95] == 0, axis=1))
        self.assertLess(solid_gaps, dashed_gaps)

    def test_labels_and_boxes_remain_within_edge_boundaries(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        frame = sample_frame(
            Detection("edge", "legend_label", Box(45, 1, 60, 12), 0.99),
            width=60, height=40,
        )
        overlay = render_detection_overlay(image, frame)
        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.any(overlay[0:20, 40:60]))

    def test_overlay_rejects_invalid_or_mismatched_images(self):
        frame = sample_frame()
        cases = (
            (object(), TypeError),
            (np.zeros((100, 120), dtype=np.uint8), ValueError),
            (np.zeros((100, 120, 4), dtype=np.uint8), ValueError),
            (np.zeros((100, 120, 3), dtype=np.float32), ValueError),
            (np.zeros((99, 120, 3), dtype=np.uint8), ValueError),
        )
        for image, error in cases:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(error):
                    render_detection_overlay(image, frame)


if __name__ == "__main__":
    unittest.main()
