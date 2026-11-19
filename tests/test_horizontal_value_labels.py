"""Horizontal bar values share formatting, with rightward placement and leads."""
import unittest
import tempfile
from zipfile import ZipFile
from dataclasses import replace
from unittest.mock import patch

from src.contracts import BarSeries, HorizontalBarChart, SeriesSpec, VerticalBarSpec, YAxisSpec
from src.tactile.generation import generate_tactile_artifacts
from src.tactile.braille import has_numeric_indicator
from src.tactile.constants import GRID_WIDTH, LABEL_MIN_GAP, LEAD_LINE_LENGTH, LEAD_STRENGTH
from src.tactile.layout import Bar, Line, Rect, _label, _cut_grid
from src.tactile import layout_hbar
from src.tactile.layout_hbar import layout_horizontal_bar_chart as layout
from src.tactile.layout_hbar import layout_horizontal_bar_chart_pages as pages
from src.tactile.rules import check_layout
from src.tactile.value_labels import plan_value_labels


def chart(values=(10., 20., 30.), rescued=()):
    return HorizontalBarChart('demo/hvalues.png', 'Sales', 'Value', 'Region',
                              tuple('ABC'[:len(values)]),
                              (BarSeries(None, values, rescued=rescued),))


def values(page):
    return [label for label in page.labels if label.role == 'value']


def notes(page):
    return ' '.join(label.text for label in page.labels if label.role == 'note')


class HorizontalValuesTest(unittest.TestCase):
    def test_default_labels_are_right_of_bars_and_rule_clean(self):
        page = layout(chart())
        self.assertEqual(len(values(page)), 3)
        self.assertEqual(check_layout(page), [])
        for label in values(page):
            bar = page.by_id(label.owner_id)
            self.assertGreaterEqual(label.x - bar.bounds.x2, LABEL_MIN_GAP)
            self.assertAlmostEqual((label.bounds.y1 + label.bounds.y2) / 2,
                                   bar.y + bar.height / 2)
            self.assertTrue(has_numeric_indicator(label.cells))
        self.assertIn('estimated from the print', notes(page))

    def test_opt_out_and_pages(self):
        plain = layout(chart(), value_labels=False)
        self.assertFalse(values(plain))
        self.assertNotIn('estimated from the print', notes(plain))
        self.assertEqual(pages(chart(), value_labels=False), [plain])
        self.assertEqual(len(values(pages(chart())[0])), 3)
        self.assertGreater(plain.plot.x2, layout(chart()).plot.x2)

    def test_numeric_ranges_and_grouped_series(self):
        for numbers in ((.001, .2, .3), (100000., 200000., 300000.), (0., 5., 10.)):
            with self.subTest(numbers=numbers):
                page = layout(chart(numbers))
                for label in values(page):
                    bar = page.by_id('bar-' + label.id[len('value-'):])
                    if bar.value:
                        self.assertNotEqual(float(label.text), 0.)
                self.assertEqual(check_layout(page), [])
        grouped = replace(chart(), series=(BarSeries('One', (10., 20., 30.)),
                                           BarSeries('Two', (20., 30., 40.))))
        page = layout(grouped)
        self.assertEqual(len(values(page)), 6)
        self.assertEqual(check_layout(page), [])

    def test_rescued_and_absent_series(self):
        source = chart(rescued=(False, True, False))
        source = replace(source, series=source.series + (BarSeries('Absent', (), has_bars=False),))
        page = layout(source)
        self.assertEqual({label.id for label in values(page)}, {'value-0-0', 'value-0-2'})
        self.assertIn('length was measured', notes(page))
        self.assertNotIn('height was measured', notes(page))
        self.assertEqual(check_layout(page), [])

    def test_all_rescued_values_have_no_estimate_note(self):
        page = layout(chart(rescued=(True, True, True)))
        self.assertFalse(values(page))
        self.assertIn('length was measured', notes(page))
        self.assertNotIn('estimated from the print', notes(page))

    def test_no_room_falls_back_to_plain_geometry(self):
        original = layout_hbar._layout_pass
        def no_room(*args, **kwargs):
            if kwargs.get('value_labels'):
                raise ValueError('no room')
            return original(*args, **kwargs)
        with patch.object(layout_hbar, '_layout_pass', side_effect=no_room):
            page = layout(chart())
        plain = layout(chart(), value_labels=False)
        self.assertEqual(page.bars, plain.bars)
        self.assertEqual(page.plot, plain.plot)
        self.assertIn('bar values are not labelled: no room on the page for them', page.notes)

    def test_omission_notes_match_drawn_labels(self):
        original = layout_hbar.plan_value_labels
        def omit_one(*args, **kwargs):
            args = list(args)
            args[2] = args[2] | {'bar-0-1'}
            return original(*args, **kwargs)
        with patch.object(layout_hbar, 'plan_value_labels', side_effect=omit_one):
            page = layout(chart())
        self.assertEqual({label.id for label in values(page)}, {'value-0-0', 'value-0-2'})
        self.assertIn('No value is shown for bars lacking space: B.', notes(page))
        self.assertEqual(check_layout(page), [])

    def test_wrong_side_and_miscentred_labels_are_reported(self):
        page = layout(chart())
        label = values(page)[0]
        for broken in (replace(label, y=label.y + 1), replace(label, x=page.plot.x1)):
            altered = replace(page, labels=tuple(broken if l.id == label.id else l for l in page.labels))
            self.assertIn('value_label_clearance', [v.rule.id for v in check_layout(altered)])

    def test_vertical_grid_is_cut_with_clearance(self):
        label = _label('value-0-0', 'value', '20', 30, 40, numeric=True)
        grid = Line('grid-1', 'grid', 35, 10, 35, 90, GRID_WIDTH, 1)
        pieces = _cut_grid([grid], [label])
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(l.bounds.gap(label.bounds) >= LABEL_MIN_GAP for l in pieces))
        self.assertEqual([(l.x1, l.x2) for l in pieces], [(35, 35), (35, 35)])

    def test_crowded_grouped_chart_moves_key_and_notes_before_graph(self):
        source = HorizontalBarChart(
            'demo/air.png', 'Air quality index in northern Taiwan',
            'Air quality index', 'Time (in hrs)', ('09:00:00', '08:00:00', '07:00:00'),
            (BarSeries('NMHC', (.2, .15, .1)), BarSeries('CO', (.4, .45, .4)),
             BarSeries('CH4', (1.9, 2., 1.93))))
        single = layout(source)
        self.assertFalse(values(single))
        result = pages(source)
        self.assertEqual([p.page_kind for p in result], ['key', 'graph'])
        key, graph = result
        self.assertEqual(len(values(graph)), 9)
        self.assertTrue(graph.series_key_elsewhere)
        self.assertFalse(graph.key_samples)
        self.assertEqual(len(key.key_samples), 3)
        self.assertIn('estimated from the print', notes(key))
        for page in result:
            self.assertEqual(check_layout(page), [])
        self.assertEqual(len(pages(source, value_labels=False)), 1)

        spec = VerticalBarSpec(
            source.source_key, source.title, source.x_axis_title, source.y_axis_title,
            YAxisSpec(0, 2., [0, .5, 1., 1.5, 2.]), list(source.categories),
            [SeriesSpec(s.name, '#112233', list(s.values)) for s in source.series],
            False, [], {}, orientation='horizontal')
        with tempfile.TemporaryDirectory() as root:
            generated = generate_tactile_artifacts(spec, root)
            self.assertEqual([p.page_kind for p in generated.pages], ['key', 'graph'])
            self.assertEqual(generated.pages[0].caption, 'Page 1 of 2 — key')
            svg = generated.pages[1].review_svg.read_text()
            self.assertEqual(svg.count('<g id="value-'), 9)
            self.assertIn('id="print"', svg)
            self.assertFalse(generated.pages[1].violations)
            with ZipFile(generated.zip_path) as archive:
                self.assertEqual(len(archive.namelist()), 2)

    def test_precision_reduction_and_omission_at_page_edge(self):
        bar = Bar('bar-0-0', 20, 30, 30, 10, 'dots', 0, 0, .123)
        def plan(right):
            return plan_value_labels(
                [bar], set(), set(), [], [], [], Rect(0, 0, right, 100), .01, ['0', '.5'],
                lambda lid, text, x, y, owner: _label(lid, 'value', text, x, y,
                                                     owner_id=owner, numeric=True),
                lambda lid, x1, y1, x2, y2: Line(lid, 'lead', x1, y1, x2, y2,
                                                GRID_WIDTH, LEAD_STRENGTH),
                orientation='horizontal')
        reduced = plan(80)
        self.assertEqual(reduced.decimals, 1)
        self.assertEqual(reduced.labels[0].text, '0.1')
        self.assertFalse(reduced.leads)
        self.assertEqual(plan(60).omitted, (bar.id,))

    def test_horizontal_lead_clears_a_neighbouring_bar(self):
        bar = Bar('bar-0-0', 20, 30, 30, 10, 'dots', 0, 0, 10.)
        neighbour = replace(bar, id='bar-1-0', y=40, width=40, series_index=1)
        plan = plan_value_labels(
            [bar, neighbour], {neighbour.id}, set(), [], [], [], Rect(0, 0, 150, 100),
            1., ['0', '10'],
            lambda lid, text, x, y, owner: _label(lid, 'value', text, x, y,
                                                 owner_id=owner, numeric=True),
            lambda lid, x1, y1, x2, y2: Line(lid, 'lead', x1, y1, x2, y2,
                                            GRID_WIDTH, LEAD_STRENGTH),
            orientation='horizontal')
        self.assertFalse(plan.omitted)
        self.assertEqual(len(plan.leads), 1)
        lead = plan.leads[0]
        self.assertAlmostEqual(lead.x1, bar.bounds.x2)
        self.assertAlmostEqual(lead.y1, bar.y + bar.height / 2)
        self.assertEqual(lead.y1, lead.y2)
        self.assertAlmostEqual(lead.length, LEAD_LINE_LENGTH)
        page = layout(chart((10.,)))
        page = replace(page, bars=(bar, neighbour), labels=plan.labels, lines=plan.leads,
                       series_textures=('dots', 'dots'))
        self.assertNotIn('value_label_clearance', [v.rule.id for v in check_layout(page)])
        wrong = replace(lead, x1=lead.x1 + 1)
        self.assertIn('value_label_clearance', [v.rule.id for v in check_layout(
            replace(page, lines=(wrong,)))])


if __name__ == '__main__':
    unittest.main()
