"""Diagnostic evidence must explain decisions without changing chart results."""
from dataclasses import asdict, replace
import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from src.contracts import OCRFrame, RecognizedText
from src.diagnostics import AnalysisDiagnostics
from src.legend_matcher import match_legend_to_bars
from src.reconstruct import reconstruct_vbar
from tests.test_reconstruct import base_chart, y_of, Chart, PLOT, _grouped, SIMILAR
from tests.test_reconstruct import _grouped_with_third_entry, _custom_entries


def recorded(chart, **kwargs):
    diagnostics = AnalysisDiagnostics(settings={})
    before = reconstruct_vbar(chart.frame(), chart.texts, chart.image, **kwargs)
    after = reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                             diagnostics=diagnostics, **kwargs)
    assert asdict(after) == asdict(before)
    return after, diagnostics


def events(diagnostics, code):
    return [e for e in diagnostics.record['events'] if e['code'] == code]


def test_dropped_ticks_distinguish_parse_failure_from_outlier():
    chart = base_chart(ticks=(0, 50, 150, 200))
    outlier = chart.add('y_tick_label', (40, y_of(75)-6, 75, y_of(75)+6), text='900')
    unreadable = chart.add('y_tick_label', (40, y_of(125)-6, 75, y_of(125)+6), text='?')
    spec, diagnostic = recorded(chart)
    assert 'y_ticks_dropped:2' in spec.flags
    assert any(e['detection_id'] == unreadable and e['reason'] == 'unparseable'
               for e in events(diagnostic, 'y_tick_parse'))
    assert any(t['detection_id'] == outlier and t['reason'] == 'numeric_outlier'
               for t in events(diagnostic, 'y_tick_decisions')[0]['ticks'])
    assert events(diagnostic, 'y_ticks_not_arithmetic')[0]['differences'] == [50, 100, 50]
    calibration = events(diagnostic, 'y_calibration')[0]
    assert calibration['a'] == pytest.approx(-1 / 1.5)
    assert len(calibration['residuals_pixels']) == 5


def test_extreme_tick_residuals_preserve_both_report_exports(tmp_path):
    chart = base_chart()
    outlier = chart.add('y_tick_label', (40, y_of(75)-6, 75, y_of(75)+6), text='1.7e308')
    with pytest.warns(RuntimeWarning, match='overflow encountered in scalar divide'):
        spec, diagnostic = recorded(chart)
    assert spec.y_axis.tick_values == [0, 50, 100, 150, 200]
    assert 'y_ticks_dropped:1' in spec.flags
    assert any(t['detection_id'] == outlier and t['reason'] == 'numeric_outlier'
               for t in events(diagnostic, 'y_tick_decisions')[0]['ticks'])
    calibration = events(diagnostic, 'y_calibration')[0]
    for key in ('selection_residuals_pixels', 'residuals_pixels'):
        assert len(calibration[key]) == 6
        assert calibration[key][-1] == 'inf'
        assert all(isinstance(value, float) for value in calibration[key][:-1])

    diagnostic.finish()
    persisted = diagnostic.persist(tmp_path)
    assert persisted is not None
    persisted_record = json.loads(persisted.read_text())
    assert persisted_record == json.loads(json.dumps(diagnostic.record, allow_nan=False))
    exported = diagnostic.export_zip(tmp_path)
    assert exported is not None
    with ZipFile(exported) as archive:
        assert json.loads(archive.read('diagnostics.json')) == persisted_record
        report = archive.read('report.txt').decode()
        assert 'y_calibration' in report
        assert 'selection_residuals_pixels' in report
        assert 'residuals_pixels' in report
        assert '"inf"' in report
    assert not any(e['stage'] == 'diagnostic_export' for e in diagnostic.record['exceptions'])


def test_baseline_and_missing_slot_evidence():
    chart = Chart()
    chart.add('plot_area', PLOT)
    y_at = lambda value: PLOT[3] - 1.5 * (value - 100)
    for tick in (100, 150, 200):
        chart.add('y_tick_label', (40, y_at(tick)-6, 75, y_at(tick)+6), text=str(tick))
    for i, value in enumerate((150, 120)):
        box = (100+200*i, y_at(value), 160+200*i, PLOT[3])
        chart.add('vertical_bar', box)
        chart.paint(box, (200, 100, 30))
    for i in range(3):
        chart.add('x_tick_label', (100+200*i, 345, 160+200*i, 360), text=f'C{i}')
    spec, diagnostic = recorded(chart)
    zero = events(diagnostic, 'baseline_outside_plot')[0]
    assert zero['zero_y'] > zero['accepted_bounds'][1]
    count = events(diagnostic, 'category_count_mismatch')[0]
    assert count['detected_bars'] == 2 and count['category_count'] == 3
    assert count['missing_slots'] == [2]
    assert not events(diagnostic, 'series_grouping')[0]['rescue_enabled']


def test_category_collision_and_rescue_details():
    _, diagnostic = recorded(_grouped(SIMILAR, legend=False))
    collision = events(diagnostic, 'category_ambiguous')[0]
    assert collision['selected_bar_id'] in collision['competing_bar_ids']
    chart = base_chart()
    bar = next(d for d in chart.detections if d.role == 'vertical_bar')
    chart.detections.remove(bar)
    _, diagnostic = recorded(chart)
    assert events(diagnostic, 'pixel_rescue')[0]['value'] == pytest.approx(120, rel=.01)
    assert all(e['series_index'] == 0 for e in diagnostic.record['events'] if 'series_index' in e)


def test_legend_diagnostics_reuse_scores():
    chart = base_chart()
    diagnostic = AnalysisDiagnostics(settings={})
    from src.legend_matcher import _score_legend_to_bars
    with patch('src.legend_matcher._score_legend_to_bars', wraps=_score_legend_to_bars) as scoring:
        actual = match_legend_to_bars(chart.image, chart.frame(), OCRFrame(chart.frame().source_key, ()),
                                      diagnostics=diagnostic)
        assert scoring.call_count == 1
    assert all(e['status'] == 'no_legend' for e in events(diagnostic, 'legend_assignment'))
    assert len(actual.assignments) == 3


@pytest.mark.parametrize('order, names', [
    ((2, 0, 1), ['Gamma', 'Alpha', 'Beta']),
    ((0, 2, 1), ['Alpha', 'Gamma', 'Beta']),
    ((1, 0, 2), ['Alpha', 'Beta', 'Gamma']),
    ((0, 1, 2), ['Alpha', 'Beta', 'Gamma']),
])
@pytest.mark.parametrize('scenario', ['complete', 'rescue', 'missing', 'collision'])
def test_series_events_follow_final_legend_order(tmp_path, order, names, scenario):
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, order)
    if scenario in ('rescue', 'missing'):
        # Leave Beta's pixels intact so rescue can recover its first value.
        bars = [d for d in chart.detections if d.role == 'vertical_bar']
        chart.detections.remove(bars[1])
        mapping = replace(mapping, assignments=tuple(
            assignment for assignment in mapping.assignments if assignment.bar_id != bars[1].id))
    elif scenario == 'collision':
        box = (215, y_of(70), 235, 340)
        bar_id = chart.add('vertical_bar', box)
        chart.paint(box, (60, 180, 60))
        beta = next(a for a in mapping.assignments if a.series_name == 'Beta')
        mapping = replace(mapping, assignments=(*mapping.assignments, replace(beta, bar_id=bar_id)))
    spec, diagnostic = recorded(chart, legend_mapping=mapping,
                                rescue_bars=scenario != 'missing')
    assert [s.name for s in spec.series] == names
    by_bar = {assignment.bar_id: assignment.series_name for assignment in mapping.assignments}
    indexed = [event for event in diagnostic.record['events'] if 'series_index' in event]
    assert indexed
    for event in indexed:
        series = spec.series[event['series_index']]
        assert not series.no_visible_bars
        if event['code'] == 'category_assignment':
            assert series.name == by_bar[event['bar_id']]
        elif event['code'] == 'category_ambiguous':
            assert series.name == 'Beta'
            assert all(by_bar[bar_id] == series.name for bar_id in event['competing_bar_ids'])
        elif event['code'] == 'pixel_rescue':
            assert series.name == 'Beta'
            assert series.values[event['category_index']] == event['value']
    counts = [event for event in indexed
              if event['code'] in ('category_counts', 'category_count_mismatch')]
    assert [spec.series[event['series_index']].name for event in counts] == ['Alpha', 'Beta']
    assert len(events(diagnostic, 'pixel_rescue')) == (scenario == 'rescue')
    assert len(events(diagnostic, 'category_ambiguous')) == (scenario == 'collision')
    mismatches = events(diagnostic, 'category_count_mismatch')
    assert len(mismatches) == (scenario in ('missing', 'collision'))
    assert all(spec.series[event['series_index']].name == 'Beta' for event in mismatches)

    diagnostic.finish()
    path = diagnostic.persist(tmp_path)
    assert path is not None
    persisted = json.loads(path.read_text())
    assert [e for e in persisted['events'] if 'series_index' in e] == indexed
    path = diagnostic.export_zip(tmp_path)
    assert path is not None
    with ZipFile(path) as archive:
        assert json.loads(archive.read('diagnostics.json')) == persisted


@pytest.mark.parametrize('helper', ['_bar_less_series', '_in_legend_order'])
def test_series_events_survive_legend_failure(tmp_path, helper):
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    diagnostic = AnalysisDiagnostics(settings={})
    failure = RuntimeError('injected legend failure')
    with patch('src.reconstruct.' + helper, side_effect=failure), pytest.raises(RuntimeError) as caught:
        with diagnostic.stage('reconstruction'):
            reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                             legend_mapping=mapping, diagnostics=diagnostic)
    assert caught.value is failure
    assignments = events(diagnostic, 'category_assignment')
    counts = events(diagnostic, 'category_counts')
    assert len(assignments) == 6
    assert len({event['bar_id'] for event in assignments}) == 6
    by_bar = {assignment.bar_id: assignment.series_name for assignment in mapping.assignments}
    for event in assignments:
        assert ('Alpha', 'Beta')[event['series_index']] == by_bar[event['bar_id']]
    assert [event['series_index'] for event in counts] == [0, 1]
    indexed = [event for event in diagnostic.record['events'] if 'series_index' in event]
    assert len(indexed) == 8

    diagnostic.finish(failed=True)
    path = diagnostic.persist(tmp_path)
    assert path is not None
    persisted = json.loads(path.read_text())
    assert persisted['outcome'] == 'failed'
    assert [event for event in persisted['events'] if 'series_index' in event] == indexed
    path = diagnostic.export_zip(tmp_path)
    assert path is not None
    with ZipFile(path) as archive:
        assert json.loads(archive.read('diagnostics.json')) == persisted
        report = archive.read('report.txt').decode()
        assert 'category_assignment' in report
        assert 'category_counts' in report
        assert 'injected legend failure' in report


def test_series_events_survive_failure_while_building_later_series():
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    bars = [d for d in chart.detections if d.role == 'vertical_bar']
    chart.detections.remove(bars[1])
    mapping = replace(mapping, assignments=tuple(
        assignment for assignment in mapping.assignments if assignment.bar_id != bars[1].id))
    diagnostic = AnalysisDiagnostics(settings={})
    failure = RuntimeError('injected rescue failure')
    with patch('src.reconstruct._rescue_value', side_effect=failure), pytest.raises(RuntimeError) as caught:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                         legend_mapping=mapping, diagnostics=diagnostic)
    assert caught.value is failure
    assignments = events(diagnostic, 'category_assignment')
    assert len(assignments) == 5
    assert [event['series_index'] for event in assignments] == [0, 0, 0, 1, 1]
    assert [event['series_index'] for event in events(diagnostic, 'category_counts')] == [0]
    assert not events(diagnostic, 'pixel_rescue')


def test_export_failure_does_not_leave_partial_files(tmp_path):
    diagnostic = AnalysisDiagnostics(settings={})
    diagnostic.finish()
    with patch('src.diagnostics.os.replace', side_effect=OSError('disk full')):
        assert diagnostic.export_zip(tmp_path) is None
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.rglob('*.zip'))
    assert diagnostic.record['exceptions'][-1]['stage'] == 'diagnostic_export'


def test_ambiguous_series_exposes_legend_scores_and_unresolved_bars():
    chart = _grouped(((0, 165, 255), (255, 100, 0)))
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, confidence)
                     for k, (text, confidence) in chart.texts.items()))
    diagnostic = AnalysisDiagnostics(settings={})
    mapping = match_legend_to_bars(chart.image, frame, ocr,
                                   min_margin=1.0, diagnostics=diagnostic)
    scores = events(diagnostic, 'legend_assignment')
    assert all(e['best_error'] == pytest.approx(0) for e in scores)
    assert all(0 < e['runner_up_error'] < e['min_margin'] for e in scores)
    assert all(e['swatch_id'] and e['label_id'] and e['status'] == 'ambiguous' for e in scores)
    spec, reconstruction = recorded(chart, legend_mapping=mapping)
    assert 'series_ambiguous' in spec.flags
    evidence = events(reconstruction, 'series_ambiguous')[0]
    assert set(evidence['unresolved_bar_ids']) == {d.id for d in frame.detections if d.role == 'vertical_bar'}
    assert evidence['has_detected_swatches']
