"""Diagnostic evidence must explain decisions without changing chart results."""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from src.contracts import ChartInterpretationError, LegendMappingFrame, OCRFrame, RecognizedText
from src.diagnostics import AnalysisDiagnostics
from src.gradio_app import analyze_image, build_app
from src.legend_matcher import match_legend_to_bars
from src.reconstruct import reconstruct_vbar
from tests.test_reconstruct import base_chart, y_of, Chart, PLOT, _grouped, SIMILAR


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


@pytest.mark.parametrize('failure', [None, 'detection', 'ocr', 'legend_matching', 'reconstruction', 'presentation'])
def test_zip_preserves_partial_stage_outputs(tmp_path, failure):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    diagnostic = AnalysisDiagnostics(settings={'confidence': .25})
    functions = {'detection': 'run_detection', 'ocr': 'run_ocr',
                 'legend_matching': 'match_legend_to_bars', 'reconstruction': 'reconstruct_vbar',
                 'presentation': 'render_detection_overlay'}
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch('src.gradio_app.run_detection', return_value=frame))
        stack.enter_context(patch('src.gradio_app.run_ocr', return_value=ocr))
        if failure:
            stack.enter_context(patch('src.gradio_app.' + functions[failure], side_effect=RuntimeError('injected failure')))
        try:
            result = analyze_image(object(), image_path, .25, .7, diagnostics=diagnostic)
            assert result.diagnostics is diagnostic
        except RuntimeError:
            assert failure in ('detection', 'presentation')
        diagnostic.finish(failed=failure in ('detection', 'presentation'))
    path = diagnostic.export_zip(tmp_path)
    with ZipFile(path) as archive:
        report = json.loads(archive.read('diagnostics.json'))
        assert 'injected failure' in archive.read('report.txt').decode() if failure else not report['exceptions']
        assert 'input.png' in archive.namelist()
        if failure != 'detection':
            assert 'detections.json' in archive.namelist()
            crops = [e for e in report['events'] if e['code'] == 'source_crop']
            assert len(crops) == len(ocr.regions)
            assert all(e['filename'] in archive.namelist() for e in crops)
            assert cv2.imdecode(np.frombuffer(archive.read(crops[0]['filename']), dtype=np.uint8), 1) is not None
        if failure not in ('detection', 'ocr'):
            assert 'ocr.json' in archive.namelist()
        if failure in ('reconstruction', 'presentation', None):
            assert 'legend.json' in archive.namelist()
        if failure == 'presentation':
            assert 'spec.json' in archive.namelist()


def test_export_failure_does_not_leave_partial_files(tmp_path):
    diagnostic = AnalysisDiagnostics(settings={})
    diagnostic.finish()
    with patch('src.diagnostics.os.replace', side_effect=OSError('disk full')):
        assert diagnostic.export_zip(tmp_path) is None
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.rglob('*.zip'))
    assert diagnostic.record['exceptions'][-1]['stage'] == 'diagnostic_export'


def test_gradio_download_reset_and_export_failure(tmp_path):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    event = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(event.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_app.run_detection', return_value=frame), \
             patch('src.gradio_app.run_ocr', return_value=ocr):
            success = event.fn(str(image_path), .25, .7, 'eng')
            assert success[index]['visible']
            with ZipFile(success[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert 'annotated.png' in archive.namelist()
                assert record['outputs']['spec']['categories'] == ['C0', 'C1', 'C2']
            rejected = event.fn(None, .25, .7, 'eng')
            assert rejected[index]['visible']
            assert rejected[index]['value'] != success[index]['value']
            with ZipFile(rejected[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert record['outcome'] == 'failed'
                assert 'input.png' not in archive.namelist()
            with patch('src.diagnostics.os.replace', side_effect=OSError('disk full')):
                failed_export = event.fn(str(image_path), .25, .7, 'eng')
            assert not failed_export[index]['visible']
            assert 'Diagnostic export failed' in failed_export[index+1]['value']
            assert failed_export[8] is not None  # Review remains available.
        for name in ('ready_results', 'begin_analysis', 'clear_all'):
            callback = next(fn.fn for fn in demo.fns.values() if fn.api_name == name)
            updates = callback()
            offset = 6 if name in ('begin_analysis', 'clear_all') else 0
            assert not updates[index+offset]['visible']
            assert updates[index+offset]['value'] is None
    finally:
        demo.close(verbose=False)


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
