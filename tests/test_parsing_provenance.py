"""Source attribution and behavioral parity for OCR consumers and findings."""

from dataclasses import asdict, replace
import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from src.contracts import ChartInterpretationError, OCRFrame, RecognizedText
from src.diagnostics import AnalysisDiagnostics
from src.reconstruct import _calibrate, _text_table, parse_tick_value, reconstruct_vbar
from src.tactile.completeness import check_completeness, cut_off_flags
from tests.test_reconstruct import (base_chart, _grouped_with_third_entry, _custom_entries,
                                    _legend_chart, _row, BAR_BGR, GREEN)
from tests.test_completeness import make_spec


def events(diagnostics, code):
    return [e for e in diagnostics.record['events'] if e['code'] == code]


def recorded(chart, **kwargs):
    diagnostics = AnalysisDiagnostics(settings={})
    diagnostics.source_images(chart.image, chart.frame())
    plain = reconstruct_vbar(chart.frame(), chart.texts, chart.image, **kwargs)
    traced = reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                             diagnostics=diagnostics, **kwargs)
    assert asdict(traced) == asdict(plain)
    assert check_completeness(traced, diagnostics=diagnostics) == check_completeness(plain)
    return traced, diagnostics


@pytest.mark.parametrize('kind', ['frame', 'regions', 'strings', 'tuples'])
def test_original_consumer_text_survives_all_legacy_formats(kind):
    raw = ' \t雪 e\u0301\n'
    region = RecognizedText('label', raw, .9)
    data = {'frame': OCRFrame('input/chart.png', (region,)), 'regions': [region],
            'strings': {'label': raw}, 'tuples': {'label': (raw, '.9')}}[kind]
    diagnostics = AnalysisDiagnostics(settings={})
    assert _text_table(data, diagnostics=diagnostics) == _text_table(data)
    event, = events(diagnostics, 'consumer_text')
    assert event['input_text'] == raw
    assert event['consumer_text'] == '雪 e\u0301'
    assert event['transformations'] == [{'operation': 'strip', 'before': raw, 'after': '雪 e\u0301'}]


@pytest.mark.parametrize('text, expected, reason', [
    (' \t$−1, 250K%\n', (-1250000., '%'), 'parsed'),
    ('–.5M', (-500000., None), 'parsed'),
    ('1e309', None, 'nonfinite'), ('1\u00a0000', None, 'invalid_number'),
    ('', None, 'invalid_number'), (' \t', None, 'invalid_number'),
    ('3.5B', (3500000000., None), 'parsed'),
])
def test_numeric_steps_are_actual_and_do_not_expand_normalization(text, expected, reason):
    diagnostics = AnalysisDiagnostics(settings={})
    assert parse_tick_value(text) == expected
    assert parse_tick_value(text, diagnostics=diagnostics, detection_id='tick') == expected
    event, = events(diagnostics, 'numeric_parse')
    assert event['reason'] == reason
    steps = event['transformations']
    assert steps[0]['before'] == text
    for before, after in zip(steps, steps[1:]):
        assert before['after'] == after['before']
    assert steps[-1]['after'] == event['numeric_text']
    json.dumps(diagnostics.record, allow_nan=False)


@pytest.mark.parametrize('state, expected_reason', [
    ('absent_detection', 'no_detection'), ('below_confidence', 'filtered_out'),
    ('absent_ocr', 'absent_ocr'), ('whitespace', 'empty_text'),
])
def test_missing_title_distinguishes_filtering_and_text_availability(state, expected_reason):
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'chart_title')
    if state == 'absent_detection':
        chart.detections.remove(title)
        del chart.texts[title.id]
    elif state == 'below_confidence':
        chart.detections[chart.detections.index(title)] = replace(title, confidence=.1)
    elif state == 'absent_ocr':
        del chart.texts[title.id]
    else:
        chart.texts[title.id] = (' \t\n', .8)
    spec, diagnostics = recorded(chart)
    assert spec.title is None
    decision = next(e for e in events(diagnostics, 'title_selection') if e['field'] == 'title')
    assert decision['reason'] == expected_reason
    diagnostics.warning('title_missing', 'reconstruction', flag='title_missing')
    warning, = diagnostics.record['warnings']
    assert decision['event_id'] in warning['evidence_ids']
    source = diagnostics.record['spec_sources']['title']
    assert decision['event_id'] in source['evidence_ids']
    finding = next(e for e in events(diagnostics, 'completeness_check')
                   if e['rule'] == 'missing_item' and e['field'] == 'title')
    assert finding['source'] == source


def test_duplicates_and_spatial_exclusions_have_measured_causes():
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'chart_title')
    duplicate = chart.add('chart_title', (150, 5, 450, 30), conf=.8, text='wrong')
    outside = chart.add('y_tick_label', (5, 380, 30, 395), text='999')
    _, diagnostics = recorded(chart)
    excluded = next(e for e in events(diagnostics, 'detection_deduplication')
                    if e['detection_id'] == duplicate)
    assert excluded['retained_detection_id'] == title.id
    assert excluded['measured_iou'] == 1.
    assert excluded['measured_iou'] > excluded['duplicate_iou']
    selection = next(e for e in events(diagnostics, 'tick_spatial_selection')
                     if e['role'] == 'y_tick_label')
    decision = next(e for e in selection['decisions'] if e['detection_id'] == outside)
    assert decision['outcome'] == 'outside_plot_band'
    assert decision['coordinate'] > selection['accepted_bounds'][1]


def test_unreadable_category_links_to_original_text_and_source_crop():
    chart = base_chart()
    category = next(d for d in chart.detections if d.role == 'x_tick_label')
    chart.texts[category.id] = (' \t|\n', .96)
    _, diagnostics = recorded(chart)
    decision = next(e for e in events(diagnostics, 'category_text_check') if e['unreadable'])
    assert decision['text_evidence']['input_text'] == ' \t|\n'
    assert decision['text'] == '|'
    assert diagnostics.record['spec_sources']['category[0]']['detection_ids'] == [category.id]
    assert any(s['detection_id'] == category.id for s in diagnostics.record['ocr_source_crops'])
    assert any(e['rule'] == 'unreadable_category' and e['field'] == 'category[0]'
               for e in events(diagnostics, 'completeness_check'))


@pytest.mark.parametrize('categories, rule, index', [
    (['Germany', 'Greece', 'i'], 'single_character_category', 2),
    (['2007', '2008', '6002'], 'year_category_outlier', 2),
])
def test_suspicious_categories_and_supplied_exemptions(categories, rule, index):
    spec = make_spec(categories=categories)
    diagnostics = AnalysisDiagnostics(settings={})
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    check = next(e for e in events(diagnostics, 'completeness_check')
                 if e['rule'] == rule and e['field'] == f'category[{index}]')
    assert check['triggered'] and not check['supplied_exemption']
    assert check['source']['availability'] == 'unavailable'
    supplied = replace(spec, flags=[f'category_supplied:{index}'])
    other = AnalysisDiagnostics(settings={})
    assert check_completeness(supplied, diagnostics=other) == check_completeness(supplied)
    check = next(e for e in events(other, 'completeness_check')
                 if e['rule'] == rule and e['field'] == f'category[{index}]')
    assert check['supplied_exemption']


def test_braille_uses_actual_whitespace_normalization_and_original_call_counts():
    from src.tactile.braille import to_braille
    spec = make_spec(title=' \tBad ]\n')
    with patch('src.tactile.completeness.to_braille', wraps=to_braille) as encoder:
        plain = check_completeness(spec)
        original_calls = encoder.call_count
    diagnostics = AnalysisDiagnostics(settings={})
    with patch('src.tactile.completeness.to_braille', wraps=to_braille) as encoder:
        traced = check_completeness(spec, diagnostics=diagnostics)
        assert encoder.call_count == original_calls
    assert traced == plain
    encoding = next(e for e in events(diagnostics, 'braille_input') if e['field'] == 'title')
    assert encoding['input_text'] == spec.title
    assert encoding['encoder_text'] == 'Bad ]'
    rejection = next(e for e in events(diagnostics, 'completeness_check')
                     if e['rule'] == 'braille_encoding' and e['field'] == 'title')
    assert rejection['rejected_characters'] == [']']
    assert rejection['evidence_ids'] == [encoding['event_id']]


def test_blockers_preserve_early_return_without_inventing_braille_checks():
    spec = make_spec(title='Bad ]', flags=['y_tick_leading_zero'])
    diagnostics = AnalysisDiagnostics(settings={})
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    assert not events(diagnostics, 'braille_input')
    assert any(e['rule'] == 'blocked_early_return' for e in events(diagnostics, 'completeness_check'))


def test_competing_calibration_sets_and_failure_before_refit_are_retained():
    ticks = [(0., 0.), (10., 10.), (20., 40.)]
    diagnostics = AnalysisDiagnostics(settings={})
    plain = _calibrate(ticks)
    assert _calibrate(ticks, diagnostics=diagnostics) == plain
    selected, = events(diagnostics, 'calibration_selection')
    assert selected['ambiguous']
    assert len(selected['competing_inlier_sets']) > 1
    candidates = events(diagnostics, 'calibration_candidate')
    for candidate in candidates:
        expected = [abs(v - (candidate['a'] * y + candidate['b'])) / abs(candidate['a'])
                    for y, v in ticks]
        assert candidate['residuals_pixels'] == expected
    failed = AnalysisDiagnostics(settings={})
    failure = RuntimeError('refit failed')
    with patch('src.reconstruct.np.polyfit', side_effect=failure), pytest.raises(RuntimeError) as caught:
        _calibrate(ticks, diagnostics=failed)
    assert caught.value is failure
    assert events(failed, 'calibration_selection')
    assert not events(failed, 'y_calibration')


def test_reconstruction_failure_preserves_parse_evidence_and_original_exception():
    chart = base_chart()
    ticks = [d for d in chart.detections if d.role == 'y_tick_label']
    for tick in ticks:
        chart.texts[tick.id] = ('?', .9)
    diagnostics = AnalysisDiagnostics(settings={})
    with pytest.raises(ChartInterpretationError) as plain:
        chart.run()
    with pytest.raises(ChartInterpretationError) as traced:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image, diagnostics=diagnostics)
    assert str(plain.value) == str(traced.value)
    assert len(events(diagnostics, 'numeric_parse')) == len(ticks)
    assert events(diagnostics, 'reconstruction_blocked')[-1]['required_count'] == 2
    assert not events(diagnostics, 'calibration_inputs')


def test_title_cutoff_measurements_and_series_sources_follow_final_order():
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    spec, diagnostics = recorded(chart, legend_mapping=mapping)
    labels = {entry.name: entry.label_id for entry in mapping.entries}
    for index, series in enumerate(spec.series):
        assert diagnostics.record['spec_sources'][f'series_name[{index}]']['detection_ids'] == [labels[series.name]]
    chart.add("chart_title", (150, 5, 450, 30), text="Sales")
    title = next(d for d in chart.detections if d.role == 'chart_title')
    chart.image[int(title.box.y1):int(title.box.y2), -3:] = 0
    assert cut_off_flags(chart.frame(), chart.image, diagnostics=diagnostics) == cut_off_flags(chart.frame(), chart.image)
    cutoff = next(e for e in events(diagnostics, 'title_cutoff_check') if e['field'] == 'title')
    assert cutoff['dark_pixel_count'] >= cutoff['required_ink_pixels']
    assert cutoff['flag'] == 'title_cut_off'


def test_event_links_warning_deduplication_and_json_zip_roundtrip(tmp_path):
    chart = base_chart(titles=False)
    _, diagnostics = recorded(chart)
    for flag in ['title_missing', 'title_missing', 'x_axis_title_missing']:
        diagnostics.warning(flag, 'reconstruction', flag=flag)
    assert len(diagnostics.record['warnings']) == 2
    all_ids = {e['event_id'] for e in diagnostics.record['events']}
    assert len(all_ids) == len(diagnostics.record['events'])
    for warning in diagnostics.record['warnings']:
        assert warning['evidence_ids']
        assert set(warning['evidence_ids']) <= all_ids
    diagnostics.finish()
    persisted = diagnostics.persist(tmp_path)
    archive_path = diagnostics.export_zip(tmp_path)
    with ZipFile(archive_path) as archive:
        assert json.loads(archive.read('diagnostics.json')) == json.loads(persisted.read_text())
    other = AnalysisDiagnostics(settings={})
    new_id = other.event('test', 'event')
    assert new_id not in all_ids
    assert other.record.get('spec_sources', {}) == {}


@pytest.mark.parametrize('label, confidence, expected', [
    (' \t', .9, 'unreadable_text'), ('Ghost', .1, 'low_confidence'),
    (' KENYA ', .9, 'duplicate_name'),
])
def test_legend_skip_keeps_exact_flag_guard_and_text(label, confidence, expected):
    chart = _legend_chart([_row('Kenya', BAR_BGR), _row(label, GREEN, ocr_conf=confidence)])
    spec, diagnostics = recorded(chart)
    decision = next(e for e in events(diagnostics, 'legend_entry_decision') if e['outcome'] == 'skipped')
    assert decision['first_failing_guard'] == expected
    assert decision['text_evidence']['input_text'] == label
    assert decision['text'] == label.strip()
    assert decision['flag'] in spec.flags
    diagnostics.warning('legend_entry_skipped', 'reconstruction', flag=decision['flag'])
    assert decision['event_id'] in diagnostics.record['warnings'][0]['evidence_ids']
    if expected == 'duplicate_name':
        assert decision['duplicate_name_key'] == 'kenya'
        assert 'kenya' in decision['existing_name_keys']
    if expected == 'low_confidence':
        assert decision['ocr_confidence'] < decision['confidence_threshold']


def test_calibration_rejects_flat_ticks_with_original_error_and_evidence():
    ticks = [(0., 10.), (20., 10.)]
    diagnostics = AnalysisDiagnostics(settings={})
    with pytest.raises(ChartInterpretationError) as plain:
        _calibrate(ticks)
    with pytest.raises(ChartInterpretationError) as traced:
        _calibrate(ticks, diagnostics=diagnostics)
    assert str(plain.value) == str(traced.value)
    selected, = events(diagnostics, 'calibration_selection')
    assert selected['outcome'] == 'rejected'
    assert selected['flag'] == 'chart_interpretation_blocked'
    assert all(e['reason'] == 'equal_value' for e in events(diagnostics, 'calibration_candidate'))


def test_series_field_relationships_survive_later_reconstruction_failure():
    chart = _grouped_with_third_entry()
    _, mapping = _custom_entries(chart, (2, 0, 1))
    diagnostics = AnalysisDiagnostics(settings={})
    failure = RuntimeError('legend ordering failed')
    with patch('src.reconstruct._in_legend_order', side_effect=failure), pytest.raises(RuntimeError) as caught:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image,
                         legend_mapping=mapping, diagnostics=diagnostics)
    assert caught.value is failure
    assert diagnostics.record['spec_sources']['series_name[0]']['detection_ids'] == [mapping.entries[1].label_id]
    assert diagnostics.record['spec_sources']['series_name[1]']['detection_ids'] == [mapping.entries[2].label_id]
    assert 'category[0]' in diagnostics.record['spec_sources']


def test_cutoff_finding_links_to_the_actual_cutoff_flag_not_an_earlier_edge_note():
    spec = make_spec(flags=['title_touches_right_edge', 'title_cut_off'])
    diagnostics = AnalysisDiagnostics(settings={})
    edge_id = diagnostics.event('reconstruction', 'title_edge_check', flag='title_touches_right_edge')
    cutoff_id = diagnostics.event('reconstruction', 'title_cutoff_check', flag='title_cut_off')
    assert check_completeness(spec, diagnostics=diagnostics) == check_completeness(spec)
    finding = next(e for e in events(diagnostics, 'completeness_check')
                   if e['rule'] == 'missing_or_cutoff_title')
    assert finding['flag'] == 'title_cut_off'
    assert cutoff_id in finding['evidence_ids']
    assert edge_id not in finding['evidence_ids']


def test_multiple_legend_skips_keep_distinct_evidence_under_one_warning_code():
    chart = _legend_chart([_row('Kenya', BAR_BGR), _row('Ghost', GREEN, ocr_conf=.1),
                           _row(' KENYA ', (20, 120, 70))])
    spec, diagnostics = recorded(chart)
    decisions = [e for e in events(diagnostics, 'legend_entry_decision') if e['outcome'] == 'skipped']
    assert {e['first_failing_guard'] for e in decisions} == {'low_confidence', 'duplicate_name'}
    for flag in spec.flags:
        if flag.startswith('legend_entry_skipped:'):
            diagnostics.warning('legend_entry_skipped', 'reconstruction', flag=flag)
    warning, = diagnostics.record['warnings']
    assert {e['event_id'] for e in decisions} <= set(warning['evidence_ids'])


def test_numeric_normalization_is_not_repeated_for_diagnostics():
    from src.reconstruct import _normalized_tick_text
    chart = base_chart()
    with patch('src.reconstruct._normalized_tick_text', wraps=_normalized_tick_text) as normalize:
        chart.run()
        plain_calls = normalize.call_count
    diagnostics = AnalysisDiagnostics(settings={})
    with patch('src.reconstruct._normalized_tick_text', wraps=_normalized_tick_text) as normalize:
        reconstruct_vbar(chart.frame(), chart.texts, chart.image, diagnostics=diagnostics)
        assert normalize.call_count == plain_calls
