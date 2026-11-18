"""Application gate integration, using the real orientation resolver."""
from contextlib import ExitStack
from dataclasses import asdict
import json
from unittest.mock import Mock, patch
from zipfile import ZipFile

from PIL import Image
import pytest

from src.contract_io import load_detection_frame
from src.contracts import Box, Detection, DetectionFrame, LegendBarAssignment, LegendMappingFrame, OCRFrame
from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.presentation import render_detection_overlay, summarize_detections
from src.tactile.completeness import Completeness, READY
from tests.gradio.session_helpers import make_spec
from tests.test_diagnostics import assert_links


def bar(identifier, orientation, confidence):
    return Detection(identifier, orientation + "_bar", Box(5, 20, 20, 60), confidence)


TITLE = Detection("title", "chart_title", Box(5, 2, 50, 15), .9)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "chart.png"
    Image.new("RGB", (100, 80), "white").save(path)
    return path


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("preference,bars,reason", [
    ("auto", (), "no_qualifying_bars"),
    ("auto", (bar("v", "vertical", .4),), "insufficient_evidence"),
    ("auto", (bar("v", "vertical", .9), bar("h", "horizontal", .9)), "ambiguous_orientation"),
    ("horizontal", (bar("v", "vertical", .9),), "requested_orientation_unavailable"),
])
def test_uncertainty_skips_every_downstream_call(image_path, diagnostic, preference, bars, reason):
    raw = DetectionFrame("input/chart.png", 100, 80, (TITLE, *bars))
    diagnostics = AnalysisDiagnostics(settings={}) if diagnostic else None
    factory = Mock()
    names = ("TesseractBackend", "run_ocr", "match_legend_to_bars", "match_legend_to_hbars",
             "reconstruct_vbar", "reconstruct_hbar", "cut_off_flags", "check_completeness")
    with ExitStack() as stack:
        stack.enter_context(patch("src.gradio_ui.analysis.run_detection", return_value=raw))
        calls = [stack.enter_context(patch("src.gradio_ui.analysis." + name)) for name in names]
        overlay = stack.enter_context(patch("src.gradio_ui.analysis.render_detection_overlay",
                                            wraps=render_detection_overlay))
        result = analyze_image(object(), image_path, .25, .7, bar_orientation=preference,
                               ocr_backend_factory=factory, diagnostics=diagnostics)
    factory.assert_not_called()
    for call in calls:
        call.assert_not_called()
    assert result.orientation_result.reason == reason
    assert result.orientation_result.effective_frame is None
    assert result.detection_frame.detections == (TITLE,)
    assert result.detection_frame.detections[0] is TITLE
    assert overlay.call_args.args[1] is result.detection_frame
    assert result.detection_count == 1
    assert result.overlay_rgb.shape == (80, 100, 3)
    assert "Bar orientation is unclear" in result.summary
    for field in ("ocr_frame", "legend_mapping", "spec", "completeness",
                  "ocr_warning", "legend_warning", "reconstruction_warning"):
        assert getattr(result, field) is None
    assert raw.detections == (TITLE, *bars)
    if diagnostics:
        record = diagnostics.record
        assert record["outputs"]["detections"] == asdict(raw)
        assert record["stages"]["orientation_resolution"]["outcome"] == "succeeded"
        for stage in ("ocr", "legend_matching", "reconstruction"):
            assert record["stages"][stage]["outcome"] == "skipped"
            assert record["stages"][stage]["reason"] == "bar_orientation_uncertain: " + reason
        uncertain = [e for e in record["events"] if e["code"] == "bar_orientation_uncertain"]
        assert len(uncertain) == 1
        event = uncertain[0]
        assert event["stage"] == "orientation_resolution"
        assert event["preference"] == preference
        assert event["status"] == "uncertain"
        assert event["reason"] == reason
        assert event["orientation"] is None
        assert event["suppressed_ids"] == []
        assert event["raw_detections"] == len(raw.detections)
        assert event["effective_detections"] is None
        warnings = [w for w in record["warnings"]
                    if w["code"] == "bar_orientation_uncertain"]
        assert len(warnings) == 1
        assert warnings[0]["stage"] == "orientation_resolution"
        assert warnings[0]["evidence_ids"] == [event["event_id"]]
        assert record["counts"]["detections"] == len(raw.detections)
        assert record["counts"]["effective_detections"] is None
        diagnostics.finish()
        assert record["outcome"] == "completed_with_warnings"


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("preference,vertical,horizontal,selected", [
    ("auto", .9, 0, "vertical"),
    ("auto", 0, .9, "horizontal"),
    ("auto", .9, .2, "vertical"),
    ("auto", .2, .9, "horizontal"),
    ("horizontal", .9, .3, "horizontal"),
    ("vertical", .3, .9, "vertical"),
])
def test_effective_frame_reaches_all_consumers(image_path, diagnostic, preference, vertical, horizontal, selected):
    bars = tuple(bar(name, orientation, score) for name, orientation, score in
                 (("v", "vertical", vertical), ("h", "horizontal", horizontal)) if score)
    raw = DetectionFrame("input/chart.png", 100, 80, (TITLE, *bars))
    before = asdict(raw)
    diagnostics = AnalysisDiagnostics(settings={}) if diagnostic else None
    with ExitStack() as stack:
        stack.enter_context(patch("src.gradio_ui.analysis.run_detection", return_value=raw))
        ocr = stack.enter_context(patch("src.gradio_ui.analysis.run_ocr", return_value=OCRFrame("input/chart.png", ())))
        calls = {}
        for orientation, match_name, reconstruct_name in (
            ("vertical", "match_legend_to_bars", "reconstruct_vbar"),
            ("horizontal", "match_legend_to_hbars", "reconstruct_hbar"),
        ):
            calls[orientation] = (
                stack.enter_context(patch("src.gradio_ui.analysis." + match_name,
                                          return_value=LegendMappingFrame("input/chart.png", (LegendBarAssignment("v" if orientation == "vertical" else "h"),)))),
                stack.enter_context(patch("src.gradio_ui.analysis." + reconstruct_name,
                                          return_value=make_spec(orientation=orientation))),
            )
        cutoff = stack.enter_context(patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]))
        complete = stack.enter_context(patch("src.gradio_ui.analysis.check_completeness", return_value=Completeness(READY)))
        overlay = stack.enter_context(patch("src.gradio_ui.analysis.render_detection_overlay", wraps=render_detection_overlay))
        summary = stack.enter_context(patch("src.gradio_ui.analysis.summarize_detections", wraps=summarize_detections))
        result = analyze_image(object(), image_path, .25, .7, bar_orientation=preference,
                               ocr_backend=object(), diagnostics=diagnostics)
    effective = result.orientation_result.effective_frame
    assert result.detection_frame is effective
    assert result.detection_count == len(effective.detections)
    assert all(d.role != ("horizontal_bar" if selected == "vertical" else "vertical_bar") for d in effective.detections)
    matcher, reconstruct = calls[selected]
    for call, index in ((ocr, 1), (matcher, 1), (reconstruct, 0), (cutoff, 0), (overlay, 1), (summary, 0)):
        assert call.call_args.args[index] is effective
    for call in calls["horizontal" if selected == "vertical" else "vertical"]:
        call.assert_not_called()
    assert complete.call_args.args[0] is result.spec
    assert result.spec.orientation == selected
    assert "Bar orientation: " + selected in result.summary
    if selected == "horizontal":
        assert "diagnostics" not in matcher.call_args.kwargs
        assert "diagnostics" not in reconstruct.call_args.kwargs
    assert asdict(raw) == before
    if diagnostics:
        record = diagnostics.record
        assert record["outputs"]["detections"] == before
        assert record["stages"]["orientation_resolution"]["outcome"] == "succeeded"
        resolved = [e for e in record["events"] if e["code"] == "bar_orientation_resolved"]
        assert len(resolved) == 1
        event = resolved[0]
        assert event["stage"] == "orientation_resolution"
        assert event["preference"] == preference
        assert event["status"] == "resolved"
        assert event["source"] == ("automatic" if preference == "auto" else "manual")
        assert event["orientation"] == selected
        opposite = "horizontal_bar" if selected == "vertical" else "vertical_bar"
        assert event["suppressed_ids"] == [d.id for d in raw.detections
                                           if d.role == opposite]
        assert event["raw_detections"] == len(raw.detections)
        assert event["effective_detections"] == len(effective.detections)
        assert record["outputs"]["effective_detections"] == asdict(effective)
        assert record["counts"]["effective_detections"] == len(effective.detections)
        assert not [w for w in record["warnings"]
                    if w["code"] == "bar_orientation_uncertain"]


def test_invalid_preference_is_not_an_ocr_failure(image_path):
    raw = DetectionFrame("input/chart.png", 100, 80, ())
    factory = Mock()
    with patch("src.gradio_ui.analysis.run_detection", return_value=raw):
        with pytest.raises(ValueError, match="preference"):
            analyze_image(object(), image_path, .25, .7, bar_orientation=[], ocr_backend_factory=factory)
    factory.assert_not_called()


def _run_with_diagnostics(image_path, raw, diagnostics, *, preference="auto", resolved=True):
    """Drive analyze_image through the real resolver with downstream calls stubbed."""
    factory = Mock()
    with ExitStack() as stack:
        stack.enter_context(patch("src.gradio_ui.analysis.run_detection", return_value=raw))
        stack.enter_context(patch("src.gradio_ui.analysis.run_ocr",
                                  return_value=OCRFrame("input/chart.png", ())))
        if resolved:
            stack.enter_context(patch("src.gradio_ui.analysis.match_legend_to_bars",
                                      return_value=LegendMappingFrame(
                                          "input/chart.png", (LegendBarAssignment("v"),))))
            stack.enter_context(patch("src.gradio_ui.analysis.reconstruct_vbar",
                                      return_value=make_spec(orientation="vertical")))
            stack.enter_context(patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]))
            stack.enter_context(patch("src.gradio_ui.analysis.check_completeness",
                                      return_value=Completeness(READY)))
            backend = {"ocr_backend": object()}
        else:
            backend = {"ocr_backend_factory": factory}
        result = analyze_image(object(), image_path, .25, .7, bar_orientation=preference,
                               diagnostics=diagnostics, **backend)
    factory.assert_not_called()
    return result


def test_resolved_bundle_explains_the_orientation_decision(image_path, tmp_path):
    raw = DetectionFrame("input/chart.png", 100, 80,
                         (TITLE, bar("v", "vertical", .9), bar("h", "horizontal", .2)))
    diagnostics = AnalysisDiagnostics(settings={})
    _run_with_diagnostics(image_path, raw, diagnostics)
    diagnostics.finish()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        names = set(archive.namelist())
        report = archive.read("report.md").decode()
        assert ("## Bar orientation\n\n- Decision: resolved **vertical**; "
                "preference ` \"auto\" `, source ` \"automatic\" `, "
                "reason ` \"automatic_selection\" `.") in report
        assert ("Evidence scores: vertical ` 0.9 `, horizontal ` 0.0 `; thresholds: "
                "bar confidence at or above ` 0.3 `, winning score at or above ` 0.6 `, "
                "share at or above ` 0.75 `.") in report
        assert ("Detections: raw ` 3 `, effective ` 2 `; "
                "suppressed IDs: ` \"h\" `.") in report
        assert ("Frames: raw [detections\\.json](detections.json); "
                "effective [effective\\_detections\\.json](effective_detections.json).") in report
        assert_links(report, names)
        manifest = json.loads(archive.read("manifest.json"))
        entry = next(a for a in manifest["artifacts"]
                     if a["path"] == "effective_detections.json")
        assert entry["availability"] == "available"
        payload = json.loads(archive.read("effective_detections.json"))
        assert payload["schema_version"] == 1
        archive.extract("effective_detections.json", tmp_path)
        effective = load_detection_frame(tmp_path / "effective_detections.json")
        assert [d.id for d in effective.detections] == ["title", "v"]
        record = json.loads(archive.read("diagnostics.json"))
        assert record["counts"]["effective_detections"] == 2
        assert [e["code"] for e in record["events"] if "bar_orientation" in e["code"]] \
            == ["bar_orientation_resolved"]


def test_uncertain_bundle_explains_the_missing_decision(image_path, tmp_path):
    raw = DetectionFrame("input/chart.png", 100, 80,
                         (TITLE, bar("v", "vertical", .9), bar("h", "horizontal", .9)))
    diagnostics = AnalysisDiagnostics(settings={})
    _run_with_diagnostics(image_path, raw, diagnostics, resolved=False)
    diagnostics.finish()
    with ZipFile(diagnostics.export_zip(tmp_path)) as archive:
        names = set(archive.namelist())
        report = archive.read("report.md").decode()
        assert ("## Bar orientation\n\n- Decision: **uncertain**; "
                "preference ` \"auto\" `, source ` \"automatic\" `, "
                "reason ` \"ambiguous_orientation\" `.") in report
        assert ("Detections: raw ` 3 `, effective none (orientation unresolved); "
                "suppressed IDs: none.") in report
        assert ("Frames: raw [detections\\.json](detections.json); "
                "effective: unavailable (orientation unresolved).") in report
        assert "The shared orientation resolver did not select an orientation" in report
        assert "Recorded checks and measurements:" in report
        assert_links(report, names)
        assert "effective_detections.json" not in names
        manifest = json.loads(archive.read("manifest.json"))
        entry = next(a for a in manifest["artifacts"]
                     if a["path"] == "effective_detections.json")
        assert entry["availability"] == "unavailable"
        assert entry["reason"] == ("Bar orientation was not resolved; "
                                   "no effective frame was recorded")
        record = json.loads(archive.read("diagnostics.json"))
        assert record["counts"]["effective_detections"] is None
        warnings = [w for w in record["warnings"]
                    if w["code"] == "bar_orientation_uncertain"]
        assert warnings[0]["stage"] == "orientation_resolution"
        assert warnings[0]["evidence_ids"]
