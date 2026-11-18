"""Build deterministic Plan 4 review evidence without loading ML models.

Run: .venv/bin/python -m scripts.review_parsing_provenance
"""

from dataclasses import asdict, replace
import html
import json
from pathlib import Path

import cv2

from src.contracts import Box, ChartInterpretationError, OCRFrame, RecognizedText, TEXT_ROLES
from src.diagnostics import AnalysisDiagnostics
from src.reconstruct import reconstruct_vbar
from src.tactile.completeness import check_completeness, completeness_from_error, cut_off_flags
from tests.test_reconstruct import base_chart, _legend_chart, _row, BAR_BGR, GREEN


def cases():
    yield 'clean', base_chart()
    for state in ('absent', 'low_confidence', 'whitespace'):
        chart = base_chart()
        title = next(d for d in chart.detections if d.role == 'chart_title')
        if state == 'absent':
            chart.detections.remove(title)
            del chart.texts[title.id]
        elif state == 'low_confidence':
            chart.detections[chart.detections.index(title)] = replace(title, confidence=.1)
        else:
            chart.texts[title.id] = (' \t\n', .9)
        yield 'title_' + state, chart
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'chart_title')
    chart.detections[chart.detections.index(title)] = replace(title, box=Box(150, 5, 600, 30))
    chart.image[5:30, -3:] = 0
    yield 'title_cutoff', chart
    chart = base_chart()
    title = next(d for d in chart.detections if d.role == 'y_axis_title')
    chart.detections[chart.detections.index(title)] = replace(title, box=Box(2, 1, 20, 100))
    yield 'y_title_cutoff', chart
    for name, labels in (
        ('category_unreadable', ['C0', '|', 'C2']),
        ('category_short', ['Germany', 'i', 'Greece']),
        ('category_year', ['2007', '6002', '2008']),
        ('braille_rejection', ['C0', 'Bad]', 'C2']),
    ):
        chart = base_chart()
        for detection, text in zip((d for d in chart.detections if d.role == 'x_tick_label'), labels):
            chart.texts[detection.id] = (' \t' + text + '\n', .95)
        yield name, chart
    for name, label, confidence in (
        ('legend_unreadable', ' \t', .95),
        ('legend_low_confidence', 'Ghost', .1),
        ('legend_duplicate_name', ' KENYA ', .95),
    ):
        yield name, _legend_chart([_row('Kenya', BAR_BGR), _row(label, GREEN, ocr_conf=confidence)])
    chart = base_chart(ticks=(0, 50, 100))
    for detection, text in zip((d for d in chart.detections if d.role == 'y_tick_label'),
                               (' $0K% ', ' $0.05K% ', ' $0.1K% ')):
        chart.texts[detection.id] = (text, .95)
    yield 'numeric_transformations', chart
    chart = base_chart(ticks=(0, 50, 100))
    ticks = [d for d in chart.detections if d.role == 'y_tick_label']
    chart.texts[ticks[-1].id] = ('200', .95)
    yield 'calibration_ambiguous', chart
    chart = base_chart()
    for detection in chart.detections:
        if detection.role == 'y_tick_label':
            chart.texts[detection.id] = ('?', .95)
    yield 'insufficient_ticks', chart


def render_fixture_labels(chart):
    """Draw fixture strings for crop review; these are not engine observations."""
    for detection in chart.detections:
        if detection.role not in TEXT_ROLES or detection.id not in chart.texts:
            continue
        text = chart.texts[detection.id][0].strip()
        if not text:
            continue
        box = detection.box
        width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        scale = min(.5, (box.x2 - box.x1 - 2) / max(width, 1),
                    (box.y2 - box.y1 - 2) / max(height, 1))
        cv2.putText(chart.image, text, (int(box.x1) + 1, int(box.y2) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.05, scale), (0, 0, 0), 1)


def review_html(name, diagnostics):
    def escaped(value):
        return html.escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))

    parts = ['<!doctype html><meta charset="utf-8"><title>Plan 4: ' + html.escape(name) + '</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:32px}'
             'pre{white-space:pre-wrap;background:#f5f6f7;padding:14px}'
             'img{max-width:100%;border:1px solid #aaa}article{border-top:1px solid #aaa;padding:16px 0}'
             ':target{outline:3px solid #789}</style>',
             '<h1>' + html.escape(name) + '</h1>',
             '<p>Synthetic detections and OCR text. Source labels are drawn from fixture strings; '
             'no detector or OCR engine ran. This demonstrates provenance, not recognition accuracy.</p>',
             '<h2>Findings</h2>']
    for warning in diagnostics.record['warnings']:
        parts.append('<h3>' + html.escape(warning['code']) + '</h3><ul>')
        for link in warning.get('evidence_ids', []):
            parts.append('<li><a href="#' + html.escape(link) + '">' + html.escape(link) + '</a></li>')
        parts.append('</ul>')
    for event in diagnostics.record['events']:
        if event['code'] == 'completeness_check' and event['rule'] == 'missing_item':
            parts.append('<p><a href="#' + html.escape(event['event_id']) + '">' +
                         html.escape(event['field'] + ': ' + event['reason']) + '</a></p>')
    parts.extend(['<h2>Field sources and outputs</h2>',
             '<pre>' + escaped({key: diagnostics.record.get(key) for key in
                 ('run_id', 'outcome', 'warnings', 'spec_sources', 'outputs')}) + '</pre>',
             '<h2>Decision evidence</h2><ul>'])
    for event in diagnostics.record['events']:
        parts.append('<li><a href="#' + html.escape(event['event_id']) + '">' +
                     html.escape(event['code']) + '</a></li>')
    parts.append('</ul>')
    sources = diagnostics.record.get('ocr_source_crops', [])
    for event in diagnostics.record['events']:
        parts.append('<article id="' + html.escape(event['event_id']) + '"><h3>' +
                     html.escape(event['code']) + '</h3><pre>' + escaped(event) + '</pre>')
        detection_ids = {event.get('detection_id'), event.get('label_id')}
        detection_ids.update(event.get('source', {}).get('detection_ids', []) if event.get('source') else [])
        for source in sources:
            if source['detection_id'] in detection_ids and source['image']['filename']:
                parts.append('<p>Source detection: ' + html.escape(source['detection_id']) + '</p>' +
                             '<img alt="Source crop" src="' + html.escape(source['image']['filename']) + '">')
        for link in event.get('evidence_ids', []):
            parts.append('<p><a href="#' + html.escape(link) + '">Earlier evidence: ' + html.escape(link) + '</a></p>')
        parts.append('</article>')
    return '\n'.join(parts)


def main():
    root = Path('output/ocr-plan4-review')
    root.mkdir(parents=True, exist_ok=True)
    links = []
    for name, chart in cases():
        render_fixture_labels(chart)
        frame = chart.frame()
        ocr = OCRFrame(frame.source_key, tuple(RecognizedText(key, text, confidence)
                       for key, (text, confidence) in chart.texts.items()))
        diagnostics = AnalysisDiagnostics(settings={'fixture': True})
        diagnostics.snapshot('detections', frame)
        diagnostics.snapshot('ocr', ocr)
        diagnostics.source_images(chart.image, frame)
        failed = False
        # Compare against an ordinary call on exactly the same fixture pixels/text.
        try:
            plain = reconstruct_vbar(frame, ocr, chart.image)
        except ChartInterpretationError as exc:
            plain = exc
        try:
            with diagnostics.stage('reconstruction', blocked_errors=(ChartInterpretationError,)):
                spec = reconstruct_vbar(frame, ocr, chart.image, diagnostics=diagnostics)
                assert not isinstance(plain, Exception)
                assert asdict(spec) == asdict(plain)
                spec = replace(spec, flags=[*spec.flags, *cut_off_flags(frame, chart.image,
                                                                       diagnostics=diagnostics)])
                diagnostics.snapshot('spec', spec)
                diagnostics.record['spec_flags'] = list(spec.flags)
                for flag in spec.flags:
                    diagnostics.warning(flag.split(':', 1)[0], 'reconstruction', flag=flag)
                completeness = check_completeness(spec, diagnostics=diagnostics)
                assert completeness == check_completeness(spec)
        except ChartInterpretationError as exc:
            assert isinstance(plain, ChartInterpretationError) and str(exc) == str(plain)
            failed = True
            diagnostics.warning('chart_interpretation_blocked', 'reconstruction')
            completeness = completeness_from_error(exc, frame)
            assert completeness == completeness_from_error(plain, frame)
        diagnostics.snapshot('completeness', completeness)
        diagnostics.record['completeness_status'] = completeness.status
        diagnostics.finish(failed=failed)
        destination = root / name
        destination.mkdir(exist_ok=True)
        record_path = diagnostics.persist(destination)
        archive_path = diagnostics.export_zip(destination)
        diagnostics.log_summary(record_path=record_path, archive_path=archive_path)
        assert archive_path is not None
        # Assets use generated relative paths, identical to their archive layout.
        for filename, data in diagnostics.assets.items():
            path = destination / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (destination / 'review.html').write_text(review_html(name, diagnostics), encoding='utf-8')
        (destination / 'verification.json').write_text(json.dumps({
            'spec_or_exception_parity': True, 'completeness_parity': True,
            'expected_reconstruction_exception': failed,
            'fixture': True, 'run_id': diagnostics.run_id,
        }, indent=2) + '\n', encoding='utf-8')
        links.append(f'- [{name}]({name}/review.html): {completeness.status}; '
                     f'{len(diagnostics.record["events"])} evidence events.')
    (root / 'REVIEW.md').write_text('# Plan 4 review checkpoint\n\n'
        'Synthetic consumer-input fixtures; no ML inference or accuracy evaluation. '
        'Each example includes source crops, exact text transformations, field sources, '
        'decision evidence, JSON/ZIP exports, and parity verification.\n\n' +
        '\n'.join(links) + '\n', encoding='utf-8')
    print(root / 'REVIEW.md')


if __name__ == '__main__':
    main()
