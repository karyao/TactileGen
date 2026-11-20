"""The app and CLI forward every bar and its mismatch evidence."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import pytest

from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.presentation import render_detection_overlay, summarize_detections
from src.reconstruction.legend_matcher import filter_bars_by_legend
from src.tactile.completeness import READY, Completeness
from src.tactile.pipeline import analyse
from tests.gradio.session_helpers import make_spec
from tests.test_legend_filter import chart


@pytest.fixture(params=["vertical", "horizontal"])
def orientation(request):
    return request.param


@pytest.fixture(params=["none", "some", "all"])
def mismatch(request):
    return request.param


@pytest.fixture
def legend_case(chart, tmp_path, orientation, mismatch):
    image, raw, ocr = chart
    if mismatch == "none":
        image[25:55, 40:50] = (0, 0, 255)
    elif mismatch == "all":
        image[25:55, 5:15] = (0, 255, 0)
    bar_role = orientation + "_bar"
    raw = replace(raw, detections=tuple(
        replace(d, role=bar_role) if d.role == "vertical_bar" else d
        for d in raw.detections))
    filtered = filter_bars_by_legend(image, raw, ocr, bar_role=bar_role)
    assert len(filtered.mismatched_bar_ids) == {"none": 0, "some": 1, "all": 2}[mismatch]
    path = tmp_path / "chart.png"
    assert cv2.imwrite(str(path), image)
    return SimpleNamespace(
        raw=raw, ocr=ocr, filtered=filtered, path=path,
        bar_role=bar_role, orientation=orientation,
        spec=make_spec(orientation=orientation),
    )


def test_app_preserves_orientation_frame(legend_case):
    case = legend_case
    filtered = case.filtered
    diagnostics = AnalysisDiagnostics(settings={})
    with (
        patch("src.gradio_ui.analysis.run_detection", return_value=case.raw),
        patch("src.gradio_ui.analysis.run_ocr", return_value=case.ocr),
        patch("src.gradio_ui.analysis.filter_bars_by_legend", return_value=filtered) as filter_call,
        patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=case.spec) as vertical,
        patch("src.gradio_ui.analysis.reconstruct_hbar", return_value=case.spec) as horizontal,
        patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]) as cutoff,
        patch("src.gradio_ui.analysis.check_completeness",
              return_value=Completeness(READY)) as completeness,
        patch("src.gradio_ui.analysis.render_detection_overlay",
              wraps=render_detection_overlay) as overlay,
        patch("src.gradio_ui.analysis.summarize_detections",
              wraps=summarize_detections) as summary,
    ):
        result = analyze_image(
            object(), case.path, .25, .7, ocr_backend=object(), diagnostics=diagnostics)

    reconstruct = horizontal if case.orientation == "horizontal" else vertical
    other_reconstruct = vertical if case.orientation == "horizontal" else horizontal
    other_reconstruct.assert_not_called()
    assert filter_call.call_args.kwargs["bar_role"] == case.bar_role
    assert filter_call.call_args.args[1] == case.raw
    assert result.orientation_result.effective_frame is filter_call.call_args.args[1]
    assert overlay.call_args.args[1] is filter_call.call_args.args[1]
    assert summary.call_args.args[0] is filter_call.call_args.args[1]
    assert result.detection_frame is filter_call.call_args.args[1]
    assert result.detection_count == len(filtered.frame.detections)
    assert result.legend_mapping is filtered.mapping
    assert diagnostics.record["outputs"]["detections"] == asdict(case.raw)
    assert diagnostics.record["counts"]["colour_mismatched_bars"] == len(filtered.mismatched_bar_ids)
    assert bool(result.legend_warning) == bool(filtered.mismatched_bar_ids)
    if filtered.mismatched_bar_ids:
        assert result.legend_warning == (
            f"{len(filtered.mismatched_bar_ids)} candidate bar detection(s) could not be "
            "matched to the legend. Reconstruction may exclude some candidates; "
            "review the series assignments."
        )
    assert result.spec == case.spec
    assert reconstruct.call_args.args[0] is filter_call.call_args.args[1]
    assert reconstruct.call_args.kwargs["legend_mapping"] is filtered.mapping
    assert reconstruct.call_args.kwargs["mismatched_bar_ids"] == filtered.mismatched_bar_ids
    assert cutoff.call_args.args[0] is filter_call.call_args.args[1]
    assert completeness.call_args.args[0] is result.spec


def test_cli_preserves_orientation_frame(legend_case):
    case = legend_case
    filtered = case.filtered
    with (
        patch("src.detection.load_model", return_value=object()),
        patch("src.detection.run_detection", return_value=case.raw),
        patch("src.ocr.run_ocr", return_value=case.ocr),
        patch("src.ocr.tesseract.TesseractBackend", return_value=object()),
        patch("src.reconstruction.legend_matcher.filter_bars_by_legend",
              return_value=filtered) as filter_call,
        patch("src.reconstruction.vertical.reconstruct_vbar",
              return_value=case.spec) as vertical,
        patch("src.reconstruction.horizontal.reconstruct_hbar",
              return_value=case.spec) as horizontal,
        patch("src.tactile.pipeline.cut_off_flags", return_value=[]) as cutoff,
    ):
        result_spec, error, result_frame = analyse(case.path, Path("weights.pt"))

    reconstruct = horizontal if case.orientation == "horizontal" else vertical
    other_reconstruct = vertical if case.orientation == "horizontal" else horizontal
    other_reconstruct.assert_not_called()
    assert filter_call.call_args.kwargs["bar_role"] == case.bar_role
    assert filter_call.call_args.args[1] == case.raw
    assert result_frame is filter_call.call_args.args[1]
    assert error is None
    assert result_spec == case.spec
    reconstruct.assert_called_once()
    assert reconstruct.call_args.args[0] is filter_call.call_args.args[1]
    assert reconstruct.call_args.args[1] is case.ocr
    assert reconstruct.call_args.kwargs["legend_mapping"] is filtered.mapping
    assert reconstruct.call_args.kwargs["mismatched_bar_ids"] == filtered.mismatched_bar_ids
    assert cutoff.call_args.args[0] is filter_call.call_args.args[1]
