"""Engine-neutral OCR crop selection behavior."""

import unittest
from unittest.mock import patch

import numpy as np

from src.contracts import Box, Detection
from src.ocr_crop import _title_strip_bounds, crop_text_region


def detection(box, role="x_tick_label", id="text"):
    return Detection(id, role, Box(*box), 0.9)


class OCRCropTest(unittest.TestCase):
    def test_non_title_crop_rounds_pads_clamps_and_returns_copy(self):
        image = np.arange(20 * 30 * 3, dtype=np.int32).reshape((20, 30, 3))
        crop = crop_text_region(image, detection((0.2, 0.4, 8.1, 6.1)))

        self.assertEqual(crop.shape, (9, 11, 3))
        crop[:] = -1
        self.assertFalse(np.any(image == -1))

    def test_title_uses_full_width_strip_and_can_be_disabled(self):
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        title = detection((80, 10, 100, 26), "chart_title")

        self.assertEqual(crop_text_region(image, title).shape, (20, 200, 3))
        with patch("src.ocr_crop._TITLE_FULL_WIDTH_STRIP", False):
            self.assertEqual(crop_text_region(image, title).shape, (20, 24, 3))

    def test_title_strip_stops_at_other_text_in_its_band(self):
        title = detection((80, 10, 100, 26), "chart_title", "title")
        neighbors = (
            title,
            detection((5, 5, 20, 30), "y_axis_title", "ytitle"),
            detection((25, 12, 40, 24), "y_tick_label", "ytick"),
            detection((150, 8, 190, 28), "legend_label", "legend"),
            detection((30, 100, 60, 120), "x_tick_label", "below"),
            detection((90, 12, 95, 24), "legend_label", "inside"),
        )

        self.assertEqual(_title_strip_bounds(title, neighbors, 200), (42, 110))
        self.assertEqual(_title_strip_bounds(title, (title,), 200), (0, 200))
        self.assertEqual(
            crop_text_region(np.zeros((60, 200), dtype=np.uint8), title, neighbors).shape,
            (20, 68),
        )


if __name__ == "__main__":
    unittest.main()
