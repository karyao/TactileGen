"""Robust tick fit and bar-label pairing."""

import unittest

from src.axis_fit import ols, pair_by_overlap, robust_fit


class AxisFitTest(unittest.TestCase):
    def test_ols_recovers_a_line_and_rejects_constant_y(self):
        a, b = ols([(0, 10), (10, 20), (20, 30)])
        self.assertAlmostEqual(a, 10)
        self.assertAlmostEqual(b, 1)
        self.assertIsNone(ols([(5, 1), (5, 2)]))

    def test_robust_fit_ignores_a_wrong_read(self):
        points = [(250, 0), (200, 20), (150, 40), (100, 600), (50, 80)]  # 60 misread as 600
        a, b, inliers, unique = robust_fit(points)
        self.assertEqual(len(inliers), 4)
        self.assertAlmostEqual(b, -0.4)
        self.assertAlmostEqual(a, 100)
        self.assertTrue(unique)

    def test_rising_lines_are_rejected(self):
        self.assertIsNone(robust_fit([(0, 0), (10, 5), (20, 10)]))  # value grows with y

    def test_two_equally_supported_axes_are_not_unique(self):
        # Two disjoint pairs, each a different falling line.
        points = [(0, 100), (10, 90), (20, 500), (30, 400)]
        self.assertFalse(robust_fit(points)[3])

    def test_pair_by_overlap_matches_each_bar_once(self):
        bars = [(0, 50, 20, 100), (40, 50, 20, 100)]
        labels = [(38, 10, 24, 12), (2, 10, 16, 12), (200, 10, 10, 12)]
        self.assertEqual(pair_by_overlap(labels, bars), {0: 1, 1: 0})


if __name__ == "__main__":
    unittest.main()
