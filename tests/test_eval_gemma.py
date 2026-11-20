import unittest

from scripts.eval_gemma import aggregate_metrics, score, unique_matches, validate_record
from src.contracts import BarSeries, HorizontalBarChart, VerticalBarChart


class GemmaEvaluationTests(unittest.TestCase):
    def test_unique_matching_rejects_duplicates(self):
        self.assertEqual(unique_matches([' A ', 'b', 'b'], ['a', 'b']), {0: 0})

    def test_values_and_failures(self):
        record = {'type': 'vbar_categorical', 'models': [{'x': [0, 1], 'y': [100, 0], 'name': 'sales'}],
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
        record = {'type': 'vbar_categorical', 'models': [{'x': [0], 'y': [0], 'name': 's'}],
                  'general_figure_info': {'x_axis': {'major_ticks': {'values': [0]},
                                                        'major_labels': {'values': ['A']}}}}
        chart = VerticalBarChart('test/a', 't', 'x', 'y', ('A',), (BarSeries('s', (0.01,)),))
        self.assertEqual(score(record, chart)['value_correct'], 0)

    def test_horizontal_grouped_and_orientation_mismatch(self):
        record = {'type': 'hbar_categorical', 'models': [
            {'y': ['A', 'B'], 'x': [10, None], 'name': 'one'},
            {'y': ['A', 'B'], 'x': [20, 30], 'name': 'two'}],
            'general_figure_info': {'y_axis': {'major_ticks': {'values': []},
                                                 'major_labels': {'values': []}}}}
        validate_record(record)
        series = (BarSeries('one', (10, 0)), BarSeries('two', (20, 30)))
        chart = HorizontalBarChart('test/a', 't', 'x', 'y', ('A', 'B'), series)
        result = score(record, chart)
        self.assertEqual((result['value_total'], result['value_aligned'], result['value_correct']), (3, 3, 3))
        wrong = VerticalBarChart('test/a', 't', 'x', 'y', ('A', 'B'), series)
        self.assertEqual(score(record, wrong)['value_correct'], 0)
        self.assertEqual(score(record, wrong)['value_total'], 3)

    def test_invalid_annotations(self):
        with self.assertRaises(ValueError):
            validate_record({'type': 'line'})
        record = {'type': 'vbar_categorical', 'models': [{'x': [0], 'y': ['bad']}],
                  'general_figure_info': {'x_axis': {'major_ticks': {'values': []},
                                                       'major_labels': {'values': []}}}}
        with self.assertRaises(ValueError):
            validate_record(record)
        record['models'][0]['y'][0] = float('inf')
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_missing_values_and_aggregate(self):
        rows = []
        for chart_type, category_axis, value_axis, chart_class in (
                ('vbar_categorical', 'x', 'y', VerticalBarChart),
                ('hbar_categorical', 'y', 'x', HorizontalBarChart)):
            for missing in (float('nan'), None):
                record = {'type': chart_type, 'models': [{category_axis: ['A', 'B'],
                          value_axis: [10, missing], 'name': 's'}],
                          'general_figure_info': {category_axis + '_axis': {
                              'major_ticks': {'values': []}, 'major_labels': {'values': []}}}}
                validate_record(record)
                self.assertIsNone(record['models'][0][value_axis][1])
                chart = chart_class('test/a', 't', 'x', 'y', ('A', 'B'), (BarSeries('s', (10, 99)),))
                result = score(record, chart)
                self.assertEqual((result['value_total'], result['value_correct'], result['bar_count_eligible']), (1, 1, 0))
                rows.append({'status': 'complete', 'scores': result})
        complete = {'type': 'vbar_categorical', 'models': [{'x': ['A'], 'y': [10]}],
                    'general_figure_info': {'x_axis': {'major_ticks': {'values': []}, 'major_labels': {'values': []}}}}
        chart = VerticalBarChart('test/a', 't', 'x', 'y', ('A',), (BarSeries(None, (10,)),))
        rows.append({'status': 'complete', 'scores': score(complete, chart)})
        metrics = aggregate_metrics(rows)
        self.assertEqual((metrics['bar_count_accuracy'], metrics['bar_count_eligible']), (1, 1))
        self.assertEqual((metrics['charts_with_missing_values'], metrics['missing_value_count']), (4, 4))


if __name__ == '__main__':
    unittest.main()
