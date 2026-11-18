"""The app and CLI share the filtered frame and stop when all bars are rejected."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from src.contracts import ChartInterpretationError, NO_BARS_MATCHED_LEGEND
from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.presentation import render_detection_overlay, summarize_detections
from src.reconstruction.legend_matcher import filter_bars_by_legend
from src.reconstruction.horizontal import filter_hbars_by_legend
from src.tactile.completeness import CANNOT_BUILD, READY, Completeness
from src.tactile.pipeline import analyse
from tests.gradio.session_helpers import make_spec
from tests.test_legend_filter import chart


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("consumer", ["app", "cli"])
def test_consumers_use_filtered_frame(chart, tmp_path, horizontal, blocked, consumer):
    image, raw, ocr = chart
    if blocked:
        image[25:55, 5:15] = (0, 255, 0)
    if horizontal:
        raw = replace(raw, detections=tuple(
            replace(d, role="horizontal_bar") if d.role == "vertical_bar" else d
            for d in raw.detections))
    filtered = (filter_hbars_by_legend if horizontal else filter_bars_by_legend)(image, raw, ocr)
    assert filtered.blocked == blocked
    path = tmp_path / "chart.png"
    assert cv2.imwrite(str(path), image)
    spec = make_spec(orientation="horizontal" if horizontal else "vertical")
    diagnostics = AnalysisDiagnostics(settings={})
    module = "src.gradio_ui.analysis" if consumer == "app" else "src.reconstruction"
    filter_module = module if consumer == "app" else module + (".horizontal" if horizontal else ".legend_matcher")
    reconstruct_module = module if consumer == "app" else module + (".horizontal" if horizontal else ".vertical")
    with ExitStack() as stack:
        stack.enter_context(patch((module if consumer == "app" else "src.detection") + ".run_detection", return_value=raw))
        stack.enter_context(patch((module if consumer == "app" else "src.ocr") + ".run_ocr", return_value=ocr))
        filter_call = stack.enter_context(patch(filter_module + (".filter_hbars_by_legend" if horizontal else ".filter_bars_by_legend"), return_value=filtered))
        reconstruct = stack.enter_context(patch(reconstruct_module + (".reconstruct_hbar" if horizontal else ".reconstruct_vbar"), return_value=spec))
        cutoff = stack.enter_context(patch((module if consumer == "app" else "src.tactile.pipeline") + ".cut_off_flags", return_value=[]))
        if consumer == "app":
            stack.enter_context(patch(module + ".check_completeness", return_value=Completeness(READY)))
            overlay = stack.enter_context(patch(module + ".render_detection_overlay", wraps=render_detection_overlay))
            summary = stack.enter_context(patch(module + ".summarize_detections", wraps=summarize_detections))
            result = analyze_image(object(), path, .25, .7, ocr_backend=object(), diagnostics=diagnostics)
            assert overlay.call_args.args[1] is filtered.frame
            assert summary.call_args.args[0] is filtered.frame
            assert result.orientation_result.effective_frame is filter_call.call_args.args[1]
            assert result.detection_frame is filtered.frame
            assert result.detection_count == len(filtered.frame.detections)
            assert result.legend_mapping is filtered.mapping
            assert diagnostics.record["outputs"]["detections"] == asdict(raw)
            assert diagnostics.record["outputs"]["legend_filtered_detections"] == asdict(filtered.frame)
            assert diagnostics.record["counts"]["colour_rejected_bars"] == len(filtered.rejected)
            if blocked:
                assert result.spec is None
                assert result.completeness.status == CANNOT_BUILD
                assert "No detected bar matched the legend colours" in result.completeness.blockers[0]
        else:
            stack.enter_context(patch("src.detection.load_model", return_value=object()))
            stack.enter_context(patch("src.ocr.tesseract.TesseractBackend", return_value=object()))
            result_spec, error, result_frame = analyse(path, Path("weights.pt"))
            assert result_frame is filtered.frame
            if blocked:
                assert result_spec is None
                assert isinstance(error, ChartInterpretationError)
                assert str(error) == NO_BARS_MATCHED_LEGEND
            else:
                assert error is None
                assert result_spec == spec
        assert filter_call.call_args.args[1] == raw
        if blocked:
            reconstruct.assert_not_called()
            cutoff.assert_not_called()
        else:
            assert reconstruct.call_args.args[0] is filtered.frame
            assert reconstruct.call_args.kwargs["legend_mapping"] is filtered.mapping
            assert reconstruct.call_args.kwargs["rejected_bars"] == tuple(r.detection for r in filtered.rejected)
            assert cutoff.call_args.args[0] is filtered.frame
