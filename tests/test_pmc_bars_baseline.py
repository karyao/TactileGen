"""Scoring helpers of the PMC bar baselines."""

import unittest

from src.baseline.pmc_bars_baseline import prf


class ScoringTest(unittest.TestCase):
    def test_precision_recall_f1(self):
        self.assertEqual(prf({"tp": 0, "fp": 0, "fn": 0}), (0.0, 0.0, 0.0))
        precision, recall, f1 = prf({"tp": 3, "fp": 1, "fn": 1})
        self.assertEqual((precision, recall, f1), (0.75, 0.75, 0.75))


if __name__ == "__main__":
    unittest.main()
