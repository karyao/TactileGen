"""Diagnostic records, report archives, and analysis evidence."""

import contextlib
from contextlib import nullcontext
from dataclasses import asdict, replace
from html.parser import HTMLParser
import io
import json
import logging
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.parse import unquote
import uuid
from zipfile import ZipFile

import pytest

from src.contracts import DetectionFrame, OCRFrame, RecognizedText
from src.diagnostics.tracking import AnalysisDiagnostics, configure_analysis_logging
from src.diagnostics.report import DiagnosticReport, literal
from src.reconstruction.legend_matcher import match_legend_to_bars
from src.ocr.rapidocr import RapidOCRBackend
from src.reconstruction.vertical import reconstruct_vbar
from src.tactile.completeness import check_completeness
from src.ocr.tesseract import TesseractBackend

from tests import test_ocr as trace_helpers
from tests.test_reconstruct import (
    Chart,
    PLOT,
    SIMILAR,
    _custom_entries,
    _grouped,
    _grouped_with_third_entry,
    base_chart,
    y_of,
)


# Diagnostics

class DiagnosticsTest(unittest.TestCase):
    def test_stage_start_is_logged_before_execution_and_completion(self):
        for outcome in ("succeeded", "failed", "blocked"):
            with self.subTest(outcome=outcome):
                collector = AnalysisDiagnostics(settings={})
                blocked_errors = (ValueError,) if outcome == "blocked" else ()
                with self.assertLogs("src.gradio_app", level="INFO") as logs:
                    with (contextlib.nullcontext() if outcome == "succeeded"
                          else self.assertRaisesRegex(ValueError, "OCR unavailable")):
                        with collector.stage("ocr", blocked_errors=blocked_errors):
                            self.assertEqual(len(logs.records), 1)
                            self.assertEqual(logs.records[0].levelno, logging.INFO)
                            self.assertEqual(logs.records[0].getMessage(),
                                             f"[run {collector.run_id}] stage=ocr outcome=started")
                            stage = collector.record["stages"]["ocr"]
                            self.assertIsNotNone(stage["started_at"])
                            self.assertIsNone(stage["outcome"])
                            self.assertIsNone(stage["ended_at"])
                            if outcome != "succeeded":
                                raise ValueError("OCR unavailable")
                self.assertEqual(len(logs.records), 2)
                self.assertIn(f"stage=ocr outcome={outcome} duration=", logs.output[1])
                self.assertEqual(logs.records[1].levelno,
                                 logging.ERROR if outcome == "failed" else logging.INFO)
                self.assertEqual(collector.record["stages"]["ocr"]["outcome"], outcome)
                self.assertTrue(all(f"[run {collector.run_id}]" in line for line in logs.output))
                self.assertIn("stage=ocr outcome=started", collector._log_entries[1])

    def test_skipped_stage_has_no_start_message(self):
        collector = AnalysisDiagnostics(settings={})
        with self.assertLogs("src.gradio_app", level="INFO") as logs:
            collector.skip("ocr", "no text regions")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("stage=ocr outcome=skipped reason=no text regions", logs.output[0])
        self.assertIsNone(collector.record["stages"]["ocr"]["started_at"])

    def test_event_normalizes_nested_non_finite_evidence(self):
        collector = AnalysisDiagnostics(settings={})
        nested = {"values": [float("inf"), {"negative": float("-inf")}],
                  "tuple": (float("nan"), 1.7e308, 0.0, -2, True, None, "text")}
        collector.event("reconstruction", "numeric_evidence", nested=nested)
        event = collector.record["events"][0]
        self.assertEqual(event, {
            "stage": "reconstruction", "code": "numeric_evidence",
            "event_id": f"{collector.run_id}:event-00001",
            "nested": {"values": ["inf", {"negative": "-inf"}],
                       "tuple": ("nan", 1.7e308, 0.0, -2, True, None, "text")},
        })
        json.dumps(collector.record, allow_nan=False)
        self.assertEqual(nested["values"][0], float("inf"))

    def test_interleaved_collectors_and_chained_exceptions_round_trip(self):
        first = AnalysisDiagnostics(settings={"language": "eng"})
        second = AnalysisDiagnostics(settings={"language": "fra"})
        with first.stage("validation"):
            with second.stage("validation"):
                second.warning("sample_warning", "validation")
        try:
            with second.stage("detection"):
                try:
                    raise ValueError("underlying cause")
                except ValueError as cause:
                    raise RuntimeError("detection failed") from cause
        except RuntimeError:
            pass
        first.finish()
        second.finish(failed=True)
        with tempfile.TemporaryDirectory() as temporary:
            one = first.persist(Path(temporary))
            two = second.persist(Path(temporary))
            record = json.loads(two.read_text())
            self.assertEqual(json.loads(one.read_text()), first.record)
            self.assertEqual(record, second.record)
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])
        self.assertNotEqual(uuid.UUID(first.run_id), uuid.UUID(second.run_id))
        self.assertEqual(first.record["warnings"], [])
        self.assertEqual(first.record["exceptions"], [])
        self.assertEqual(record["outcome"], "failed")
        self.assertIn("underlying cause", record["exceptions"][0]["traceback"])
        self.assertIn("direct cause", record["exceptions"][0]["traceback"])
        self.assertNotIn("locals", record["exceptions"][0])
        self.assertIn("detection", record["stages"]["ocr"]["reason"])
        self.assertGreaterEqual(record["duration_seconds"], 0)
        self.assertTrue(record["ended_at"].endswith("+00:00"))

    def test_metadata_is_best_effort_and_does_not_initialize_engines(self):
        collector = AnalysisDiagnostics(settings={})
        model = types.SimpleNamespace(
            names={0: "chart_title"}, ckpt_path=Path("weights.pt"), task="detect",
            overrides={"imgsz": 640, "unrelated": object()},
        )
        with patch("src.diagnostics.tracking.metadata.version", side_effect=RuntimeError("missing metadata")):
            collector.capture_metadata(model, TesseractBackend(2), "eng+fra")
        self.assertEqual(collector.record["metadata"]["model"]["checkpoint"], "weights.pt")
        self.assertEqual(collector.record["metadata"]["model"]["configured_prediction_options"],
                         {"imgsz": 640})
        self.assertTrue(all(value is None for value in collector.record["metadata"]["packages"].values()))
        self.assertEqual(collector.record["metadata"]["ocr"]["effective_language"], "eng+fra")
        # Bypass construction: metadata must not require an initialized engine.
        collector.capture_metadata(model, object.__new__(RapidOCRBackend), "eng")
        self.assertEqual(collector.record["metadata"]["ocr"]["effective_language"], "multilingual")
        self.assertIn("not applied", collector.record["metadata"]["ocr"]["language_behavior"])
        collector.capture_metadata(object(), object(), "eng")
        self.assertEqual(collector.record["metadata"]["ocr"]["language_behavior"], "unknown")
        class BrokenMetadata:
            def diagnostic_metadata(self, **kwargs):
                raise RuntimeError("metadata unavailable")
        collector.capture_metadata(object(), BrokenMetadata(), "eng")
        self.assertIn("metadata unavailable", collector.record["metadata"]["ocr"]["metadata_error"])


    def test_logging_setup_is_idempotent_and_terminal_summaries_are_visible(self):
        logger = logging.getLogger("src.gradio_app")
        original = logger.handlers[:], logger.level, logger.propagate
        third_party_level = logging.getLogger("ultralytics").level
        try:
            logger.handlers = []
            terminal = io.StringIO()
            with contextlib.redirect_stderr(terminal):
                configure_analysis_logging()
                configure_analysis_logging()
                collector = AnalysisDiagnostics(settings={})
                with collector.stage("validation"):
                    pass
                collector.finish()
                with tempfile.TemporaryDirectory() as temporary:
                    path = collector.persist(Path(temporary))
                    archive_path = collector.export_zip(Path(temporary))
                    collector.log_summary(record_path=path, archive_path=archive_path)
            self.assertEqual(len(logger.handlers), 1)
            output = terminal.getvalue()
            self.assertEqual(output.count("analysis started"), 1)
            self.assertEqual(output.count("analysis outcome=succeeded"), 1)
            self.assertIn(str(path), output)
            self.assertTrue(all(f"[run {collector.run_id}]" in line for line in output.splitlines()))
            self.assertEqual(logging.getLogger("ultralytics").level, third_party_level)
        finally:
            logger.handlers, logger.level, logger.propagate = original

    def test_storage_error_retains_record_and_cleans_up(self):
        collector = AnalysisDiagnostics(settings={})
        collector.finish()
        with tempfile.TemporaryDirectory() as temporary,\
             patch("src.diagnostics.tracking.os.replace", side_effect=OSError("disk full")),\
             self.assertLogs("src.gradio_app", level="INFO") as logs:
            self.assertIsNone(collector.persist(Path(temporary)))
            collector.log_summary(record_path=None, archive_path=None)
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])
        self.assertEqual(collector.record["outcome"], "succeeded")
        self.assertIn("disk full", "\n".join(logs.output))
        self.assertIn("record=unavailable", logs.output[-1])


# Diagnostics bundle

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
    result, _ = trace_helpers.recognize_with_fake_rapidocr(
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


def test_orientation_section_reports_missing_evidence_without_a_decision(tmp_path):
    diagnostics = recognized_run()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        report = archive.read('report.md').decode()
        assert '## Bar orientation' in report
        assert 'Bar orientation evidence is unavailable; see the stage outcomes.' in report
        assert_links(report, archive.namelist())


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
    ('report generation', 'src.diagnostics.report.DiagnosticReport.render', RuntimeError('bad report')),
    ('archive writing', 'src.diagnostics.tracking.ZipFile.writestr', OSError('disk full')),
    ('archive finalization', 'src.diagnostics.tracking.os.replace', OSError('rename failed')),
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
    with patch.object(logger, 'handlers', [caplog.handler]),\
         patch.object(logger, 'propagate', False),\
         caplog.at_level(logging.INFO, logger='src.gradio_app'):
        with (patch('src.diagnostics.tracking.os.replace', side_effect=OSError('disk full'))
              if case == 'storage_failure' else nullcontext()):
            record_path = diagnostics.persist(tmp_path)
        with (patch('src.diagnostics.tracking.os.replace', side_effect=OSError('rename failed'))
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


# Analysis evidence

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
    from src.reconstruction.legend_matcher import _score_legend_to_bars
    with patch('src.reconstruction.legend_matcher._score_legend_to_bars', wraps=_score_legend_to_bars) as scoring:
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
    with patch('src.reconstruction.vertical.' + helper, side_effect=failure), pytest.raises(RuntimeError) as caught:
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
    with patch('src.reconstruction.vertical._rescue_value', side_effect=failure), pytest.raises(RuntimeError) as caught:
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
    with patch('src.diagnostics.tracking.os.replace', side_effect=OSError('disk full')):
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
                                   min_margin=1.0, diagnostics=diagnostic, resolve_by_colour=False)
    scores = events(diagnostic, 'legend_assignment')
    assert all(e['best_error'] == pytest.approx(0) for e in scores)
    assert all(0 < e['runner_up_error'] < e['min_margin'] for e in scores)
    assert all(e['swatch_id'] and e['label_id'] and e['status'] == 'ambiguous' for e in scores)
    spec, reconstruction = recorded(chart, legend_mapping=mapping)
    assert 'series_ambiguous' in spec.flags
    evidence = events(reconstruction, 'series_ambiguous')[0]
    assert set(evidence['unresolved_bar_ids']) == {d.id for d in frame.detections if d.role == 'vertical_bar'}
    assert evidence['has_detected_swatches']


if __name__ == "__main__":
    unittest.main()
