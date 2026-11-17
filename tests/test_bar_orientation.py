"""Orientation heuristic for PMC bar charts."""

import unittest

from src.contracts import Box
from src.dataset.bar_orientation import BAR, PLOT_AREA, YLABEL, orientation


PLOT = Box(100, 10, 300, 200)


class OrientationTest(unittest.TestCase):
    def test_value_axis_labels_left_of_plot_mean_vertical(self):
        boxes = {PLOT_AREA: [PLOT], YLABEL: [Box(60, 20, 95, 30), Box(60, 100, 95, 110)]}
        self.assertEqual(orientation(boxes), "vertical")

    def test_value_axis_labels_below_plot_mean_horizontal(self):
        boxes = {PLOT_AREA: [PLOT], YLABEL: [Box(100, 205, 120, 215), Box(200, 205, 220, 215)]}
        self.assertEqual(orientation(boxes), "horizontal")

    def test_bar_alignment_is_the_fallback(self):
        vertical = [Box(110, 50, 130, 200), Box(150, 80, 170, 200), Box(190, 120, 210, 200)]
        horizontal = [Box(100, 20, 250, 40), Box(100, 60, 180, 80), Box(100, 100, 220, 120)]
        self.assertEqual(orientation({BAR: vertical}), "vertical")
        self.assertEqual(orientation({BAR: horizontal}), "horizontal")

    def test_single_bar_without_labels_is_undecided(self):
        self.assertEqual(orientation({BAR: [Box(0, 0, 5, 5)]}), "undecided")


if __name__ == "__main__":
    unittest.main()
