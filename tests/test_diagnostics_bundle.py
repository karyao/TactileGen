"""Archive integrity, navigable explanations, and export failure isolation."""

from html.parser import HTMLParser
from contextlib import nullcontext
import json
import logging
from unittest.mock import patch
from urllib.parse import unquote
from zipfile import ZipFile

import pytest

from src.contracts import DetectionFrame
from src.diagnostics import AnalysisDiagnostics
from src.diagnostics_report import DiagnosticReport, literal
from src.reconstruct import reconstruct_vbar
from src.tactile.completeness import check_completeness
from tests import test_ocr_trace as trace_helpers
from tests.test_reconstruct import base_chart


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images, self.tags = set(), [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs.get('href', ''))
        if tag == 'img':
            self.images.append(attrs['src'])


def assert_links(report, members):
    MarkdownIt = pytest.importorskip('markdown_it').MarkdownIt
    parser = Links()
    parser.feed(MarkdownIt('commonmark', {'html': True}).render(report))
    for link in [*parser.links, *parser.images]:
        if link.startswith('#'):
            assert link[1:] in parser.ids, link
        else:
            assert unquote(link) in members, link
    assert 'script' not in parser.tags
    return parser


def recognized_run():
    diagnostics = AnalysisDiagnostics(settings={'language': 'eng'})
    frame = DetectionFrame('input/chart.png', 60, 40, (trace_helpers.SHORT, trace_helpers.TALL))
    diagnostics.snapshot('detections', frame)
    result, _ = trace_helpers.OCRTraceTest().rapid(
        type('Output', (), {'txts': (' Sales ', 'tie', '\t', ''), 'scores': (.8, .8, .5, .2)})(),
        (trace_helpers.SHORT, trace_helpers.TALL), diagnostics=diagnostics)
    diagnostics.snapshot('ocr', result)
    diagnostics.finish()
    return diagnostics


def test_complete_archive_manifest_links_outputs_and_all_attempts(tmp_path):
    diagnostics = recognized_run()
    diagnostics.assets['input.png'] = b'full source must be omitted'
    path = diagnostics.export_zip(tmp_path)
    assert path.name == f'analysis-diagnostics-{diagnostics.run_id}.zip'
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        assert {'manifest.json', 'report.md', 'report.txt', 'diagnostics.json', 'analysis.log',
                'ocr.json', 'ocr_trace.json', 'detections.json'} <= names
        assert 'input.png' not in names
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['run_id'] == diagnostics.run_id
        assert {a['path'] for a in manifest['artifacts'] if a['availability'] == 'available'} == names
        assert manifest['unavailable_evidence'] == []
        assert json.loads(archive.read('ocr.json')) == {
            'schema_version': 1, **json.loads(json.dumps(diagnostics.record['outputs']['ocr']))}
        from src.contract_io import load_ocr_frame, load_detection_frame
        archive.extract('ocr.json', tmp_path)
        archive.extract('detections.json', tmp_path)
        assert len(load_ocr_frame(tmp_path / 'ocr.json').regions) == 2
        assert len(load_detection_frame(tmp_path / 'detections.json').detections) == 2
        report = archive.read('report.md').decode()
        assert_links(report, names)
        for attempt in diagnostics.record['ocr_trace']['attempts']:
            assert attempt['attempt_id'] in report
            assert attempt['recognition_image']['filename'] in names
        for source in diagnostics.record['ocr_source_crops']:
            assert source['image']['filename'] in names


def test_early_failure_archive_marks_missing_evidence_and_records_traceback(tmp_path):
    diagnostics = AnalysisDiagnostics(settings={})
    with pytest.raises(ValueError):
        with diagnostics.stage('validation'):
            raise ValueError('invalid chart bytes')
    diagnostics.warning('upload_rejected', 'validation')
    diagnostics.finish(failed=True)
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        missing = {item['path']: item for item in manifest['artifacts'] if item['availability'] == 'unavailable'}
        assert {'ocr.json', 'detections.json', 'ocr_trace.json'} <= missing.keys()
        assert all(item['reason'] for item in missing.values())
        report = archive.read('report.md').decode()
        assert 'invalid chart bytes' in report
        assert 'Detailed provenance is unavailable' in report
        assert_links(report, archive.namelist())
        log = archive.read('analysis.log').decode()
        assert 'Traceback' in log and 'invalid chart bytes' in log


def test_detection_measurements_survive_json_and_both_reports(tmp_path):
    from tests.test_detection import ObservedModel, fake_result
    from src.detection import run_detection

    image = tmp_path / 'chart.png'
    image.touch()
    diagnostics = AnalysisDiagnostics(settings={})
    model = ObservedModel([fake_result(image)])
    with diagnostics.stage('detection'):
        frame = run_detection(model, image, imgsz=641, diagnostics=diagnostics)
        diagnostics.snapshot('detections', frame)
    diagnostics.finish()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        record = json.loads(archive.read('diagnostics.json'))
        assert record['settings']['imgsz'] == 641
        assert record['metadata']['detection'] == {
            'requested_imgsz': 641, 'effective_imgsz': [672, 672],
            'tensor_shape': [1, 3, 352, 672], 'unavailable_reasons': {},
        }
        for name in ('report.md', 'report.txt', 'analysis.log'):
            content = archive.read(name).decode()
            assert 'effective_imgsz' in content and 'tensor_shape' in content
        # The interchange contract still contains only original-image detections.
        payload = json.loads(archive.read('detections.json'))
        assert payload['width'] == 200 and payload['height'] == 100
        assert 'tensor_shape' not in payload


def test_missing_asset_never_creates_a_broken_report_link(tmp_path):
    diagnostics = recognized_run()
    attempt = diagnostics.record['ocr_trace']['attempts'][0]
    filename = attempt['recognition_image']['filename']
    del diagnostics.assets[filename]
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert any(item['owner_id'] == attempt['attempt_id'] for item in manifest['unavailable_evidence'])
        report = archive.read('report.md').decode()
        assert 'Recognition input — unavailable' in report
        assert_links(report, archive.namelist())


def test_findings_link_to_sources_and_preserve_literal_text(tmp_path):
    chart = base_chart()
    category = next(d for d in chart.detections if d.role == 'x_tick_label')
    chart.texts[category.id] = (' \t|\n', .9)
    diagnostics = AnalysisDiagnostics(settings={})
    diagnostics.source_images(chart.image, chart.frame())
    spec = reconstruct_vbar(chart.frame(), chart.texts, chart.image, diagnostics=diagnostics)
    diagnostics.snapshot('spec', spec)
    for flag in spec.flags:
        diagnostics.warning(flag.split(':', 1)[0], 'reconstruction', flag=flag)
    completeness = check_completeness(spec, diagnostics=diagnostics)
    diagnostics.snapshot('completeness', completeness)
    diagnostics.record['completeness_status'] = completeness.status
    diagnostics.finish()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        report = archive.read('report.md').decode()
        assert 'category_unreadable' in report
        assert 'Input needed: category' in report
        assert 'Source detection' in report
        assert '\\t|\\n' in report
        assert_links(report, archive.namelist())


@pytest.mark.parametrize('text', ['雪 e\u0301\t\n', '```<script>alert(1)</script>```',
                                  '[bad](https://example.com) ![x](missing.png)', 'x|y\\z'])
def test_exact_strings_cannot_inject_markdown_or_html(text):
    MarkdownIt = pytest.importorskip('markdown_it').MarkdownIt
    rendered = MarkdownIt('commonmark', {'html': True}).render(literal(text))
    parser = Links()
    parser.feed(rendered)
    assert parser.links == [] and parser.images == []
    assert 'script' not in parser.tags
    # JSON escapes make whitespace visible while Unicode remains intact.
    if '雪' in text:
        assert '雪' in rendered and '\\t\\n' in rendered


@pytest.mark.parametrize('operation, target, error', [
    ('report generation', 'src.diagnostics_report.DiagnosticReport.render', RuntimeError('bad report')),
    ('archive writing', 'src.diagnostics.ZipFile.writestr', OSError('disk full')),
    ('archive finalization', 'src.diagnostics.os.replace', OSError('rename failed')),
])
def test_export_failure_retains_analysis_and_leaves_no_partial_archive(tmp_path, operation, target, error):
    diagnostics = recognized_run()
    original = diagnostics.record['outputs']['ocr'].copy()
    with patch(target, side_effect=error):
        assert diagnostics.export_zip(tmp_path) is None
    assert operation in diagnostics.export_error
    assert diagnostics.record['outputs']['ocr'] == original
    assert diagnostics.record['outcome'] == 'succeeded'
    assert list(tmp_path.rglob('*.tmp')) == []
    assert list(tmp_path.rglob('*.zip')) == []


def test_serialization_failure_is_specific_and_does_not_escape(tmp_path):
    diagnostics = AnalysisDiagnostics(settings={})
    diagnostics.record['unexpected'] = object()
    assert diagnostics.export_zip(tmp_path) is None
    assert 'snapshot serialization' in diagnostics.export_error
    assert diagnostics.record['exceptions'][-1]['stage'] == 'diagnostic_export'


@pytest.mark.parametrize('case', ['success', 'warnings', 'early_failure', 'storage_failure', 'export_failure'])
def test_final_summary_reports_context_and_actual_artifacts(tmp_path, caplog, case):
    diagnostics = AnalysisDiagnostics(settings={})
    if case != 'early_failure':
        diagnostics.record['completeness_status'] = 'needs_input' if case == 'warnings' else 'ready'
    if case == 'warnings':
        diagnostics.warning('sample_warning', 'reconstruction')
    diagnostics.finish(failed=case == 'early_failure')
    original_outcome = diagnostics.record['outcome']
    # Capture directly, independent of handlers installed by application tests.
    caplog.clear()
    logger = logging.getLogger('src.gradio_app')
    with patch.object(logger, 'handlers', [caplog.handler]), \
         patch.object(logger, 'propagate', False), \
         caplog.at_level(logging.INFO, logger='src.gradio_app'):
        with (patch('src.diagnostics.os.replace', side_effect=OSError('disk full'))
              if case == 'storage_failure' else nullcontext()):
            record_path = diagnostics.persist(tmp_path)
        with (patch('src.diagnostics.os.replace', side_effect=OSError('rename failed'))
              if case == 'export_failure' else nullcontext()):
            archive_path = diagnostics.export_zip(tmp_path)
        diagnostics.log_summary(record_path=record_path, archive_path=archive_path)
    messages = [record.getMessage() for record in caplog.records]
    summaries = [message for message in messages if 'analysis outcome=' in message]
    assert len(summaries) == 1
    summary = summaries[0]
    assert messages[-1] == summary
    assert f'[run {diagnostics.run_id}]' in summary
    assert f'outcome={original_outcome} duration=' in summary
    assert f"completeness={diagnostics.record['completeness_status'] or 'unavailable'}" in summary
    assert f"warnings={len(diagnostics.record['warnings'])}" in summary
    assert f"record={record_path or 'unavailable'} zip={archive_path or 'unavailable'}" in summary
    assert (record_path is None) == (case == 'storage_failure')
    assert (archive_path is None) == (case == 'export_failure')
    assert diagnostics.record['outcome'] == original_outcome
    created = [message for message in messages if 'diagnostic ZIP created' in message]
    if archive_path is None:
        assert not created
        assert any('diagnostic ZIP export failed' in message for message in messages)
    else:
        assert archive_path.is_file()
        assert created == [f'[run {diagnostics.run_id}] diagnostic ZIP created path={archive_path}']
        assert messages[-2] == created[0]
        with ZipFile(archive_path) as archive:
            log = archive.read('analysis.log').decode()
        assert 'diagnostic ZIP created' not in log
        assert 'analysis outcome=' not in log
        assert any(summary in line for line in diagnostics._log_entries)


def test_logs_are_run_local_and_archives_snapshot_the_original_analysis(tmp_path):
    first, second = recognized_run(), recognized_run()
    first.log('first-marker %s', 'value', level=logging.WARNING)
    second.log('second-marker')
    render = DiagnosticReport.render
    def mutate_after_snapshot(report):
        first.record['outputs']['ocr']['regions'][0]['text'] = 'later review edit'
        first.assets.clear()
        first.log('later-marker')
        return render(report)
    with patch.object(DiagnosticReport, 'render', mutate_after_snapshot):
        path = first.export_zip(tmp_path)
    with ZipFile(path) as archive:
        assert json.loads(archive.read('ocr.json'))['regions'][0]['text'] != 'later review edit'
        assert any(name.startswith('ocr_inputs/') for name in archive.namelist())
        log = archive.read('analysis.log').decode()
        assert 'WARNING' in log and 'first-marker value' in log and first.run_id in log
        assert 'second-marker' not in log and second.run_id not in log and 'later-marker' not in log


def test_unavailable_legacy_backend_internals_are_explained(tmp_path):
    from src.ocr import run_ocr, OCRBackendResult
    backend = type('Legacy', (), {'recognize': lambda *a, **k: OCRBackendResult('label', .8)})()
    diagnostics = AnalysisDiagnostics(settings={})
    frame = DetectionFrame('input/chart.png', 60, 40, (trace_helpers.SHORT,))
    result = run_ocr(trace_helpers.IMAGE, frame, backend, 'eng', diagnostics=diagnostics)
    diagnostics.snapshot('ocr', result)
    diagnostics.finish()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        report = archive.read('report.md').decode()
        assert 'Internal OCR attempts are unavailable' in report
        assert_links(report, archive.namelist())


def test_report_failure_keeps_app_analysis_and_shows_specific_error(tmp_path):
    import cv2
    from src.contracts import OCRFrame, RecognizedText
    from src.gradio_app import build_app

    chart = base_chart()
    image = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(key, text, confidence)
                   for key, (text, confidence) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    binding = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(binding.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_ui.analysis.run_detection', return_value=frame), \
             patch('src.gradio_ui.analysis.run_ocr', return_value=ocr), \
             patch('src.diagnostics_report.DiagnosticReport.render', side_effect=RuntimeError('formatting failed')):
            updates = binding.fn(str(image), .25, .7, 'eng')
        assert updates[8] is not None  # Analysis review state survived packaging failure.
        assert not updates[index]['visible'] and updates[index]['value'] is None
        assert 'Diagnostic export failed' in updates[index + 1]['value']
        assert 'report generation' in updates[index + 1]['value']
        assert 'formatting failed' in updates[index + 1]['value']
    finally:
        demo.close(verbose=False)
