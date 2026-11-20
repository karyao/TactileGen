import unittest

from scripts.eval_gemma import score, unique_matches
from src.contracts import BarSeries, VerticalBarChart


class GemmaEvaluationTests(unittest.TestCase):
    def test_unique_matching_rejects_duplicates(self):
        self.assertEqual(unique_matches([' A ', 'b', 'b'], ['a', 'b']), {0: 0})

    def test_values_and_failures(self):
        record = {'models': [{'x': [0, 1], 'y': [100, 0], 'name': 'sales'}],
                  'general_figure_info': {'x_axis': {'major_ticks': {'values': [0, 1]},
                                                        'major_labels': {'values': ['A', 'B']}}}}
        chart = VerticalBarChart('test/a', 'title', 'x', 'y', ('B', 'A'),
                                 (BarSeries('sales', (4, 105)),))
        result = score(record, chart)
        self.assertEqual((result['value_aligned'], result['value_correct']), (2, 2))
        self.assertEqual(result['bar_count_correct'], 1)
        self.assertEqual(score(record, None)['value_total'], 2)
        self.assertEqual(score(record, None)['value_correct'], 0)

    def test_zero_scale_requires_exact(self):
        record = {'models': [{'x': [0], 'y': [0], 'name': 's'}],
                  'general_figure_info': {'x_axis': {'major_ticks': {'values': [0]},
                                                        'major_labels': {'values': ['A']}}}}
        chart = VerticalBarChart('test/a', 't', 'x', 'y', ('A',), (BarSeries('s', (0.01,)),))
        self.assertEqual(score(record, chart)['value_correct'], 0)


if __name__ == '__main__':
    unittest.main()
