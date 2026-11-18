"""Application gate integration, using the real orientation resolver."""
from contextlib import ExitStack
from dataclasses import asdict
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from src.contracts import Box, Detection, DetectionFrame, LegendBarAssignment, LegendMappingFrame, OCRFrame
from src.diagnostics import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.presentation import render_detection_overlay, summarize_detections
from src.tactile.completeness import Completeness, READY
from tests.gradio.test_gradio_app import make_spec


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
        assert diagnostics.record["outputs"]["detections"] == asdict(raw)
        for stage in ("ocr", "legend_matching", "reconstruction"):
            assert diagnostics.record["stages"][stage]["outcome"] == "skipped"
            assert diagnostics.record["stages"][stage]["reason"] == "bar_orientation_uncertain: " + reason
        diagnostics.finish()
        assert diagnostics.record["outcome"] == "completed_with_warnings"


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
        assert diagnostics.record["outputs"]["detections"] == before


def test_invalid_preference_is_not_an_ocr_failure(image_path):
    raw = DetectionFrame("input/chart.png", 100, 80, ())
    factory = Mock()
    with patch("src.gradio_ui.analysis.run_detection", return_value=raw):
        with pytest.raises(ValueError, match="preference"):
            analyze_image(object(), image_path, .25, .7, bar_orientation=[], ocr_backend_factory=factory)
    factory.assert_not_called()
