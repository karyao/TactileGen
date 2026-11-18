"""Explicitly retaining legend colour matches removes unwanted series."""
import asyncio
from dataclasses import replace
from unittest.mock import patch

import cv2
import pytest

from src.contracts import Box, OCRFrame, RecognizedText
from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.reconstruction.horizontal import transpose_frame, transpose_image
from src.tactile.completeness import Overrides, apply_overrides
from src.tactile.generation import generate_tactile_artifacts
from src.tactile.layout import layout_vertical_bar_chart_pages
from src.tactile.layout_hbar import layout_horizontal_bar_chart_pages
from src.tactile.completeness import spec_to_chart
from tests.gradio.session_helpers import AppSession, make_analysis_result
from tests.test_reconstruct import BLUE, GREEN, _grouped


def selection_case(tmp_path, horizontal, legend=True):
    chart = _grouped((BLUE, GREEN), legend=legend)
    if legend:
        # Only the blue series is represented in the legend.
        del chart.detections[-2:]
    frame, image = chart.frame(), chart.image
    if horizontal:
        frame, image = transpose_frame(frame), transpose_image(image)
        # Turn the plot, but keep the legend in its usual swatch-then-label layout.
        frame = replace(frame, detections=tuple(
            replace(d, box=Box(360, 20, 374, 34)) if d.role == "legend_swatch" else
            replace(d, box=Box(380, 20, 398, 34)) if d.role == "legend_label" else d
            for d in frame.detections))
        image[20:34, 360:374] = BLUE
    ocr = OCRFrame(frame.source_key, tuple(
        RecognizedText(identifier, text, confidence)
        for identifier, (text, confidence) in chart.texts.items()
        if any(d.id == identifier for d in frame.detections)))
    path = tmp_path / "chart.png"
    assert cv2.imwrite(str(path), image)
    return path, frame, ocr


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_only_matched_blue_bars_reach_review_and_generation(tmp_path, horizontal, selected):
    path, frame, ocr = selection_case(tmp_path, horizontal)
    diagnostics = AnalysisDiagnostics(settings={})
    with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
         patch("src.gradio_ui.analysis.run_ocr", return_value=ocr):
        result = analyze_image(object(), path, .25, .7, ocr_backend=object(),
                               keep_legend_matches=selected, diagnostics=diagnostics)
    spec = result.spec
    assert spec is not None
    assert len(spec.series) == (1 if selected else 2)
    assert spec.series[0].name == "Alpha"
    assert spec.series[0].values == pytest.approx([100, 150, 40])
    bars = [d for d in result.detection_frame.detections if d.role.endswith("_bar")]
    assert len(bars) == (3 if selected else 6)
    assert len(result.legend_mapping.assignments) == len(bars)
    if selected:
        assert "legend_bars_removed:3" in spec.flags
        assert not any(f.startswith("bar_colour_mismatch:") for f in spec.flags)
        assert "removed 3" in result.summary
        assert result.legend_warning is None
        assert result.completeness.status == "needs_input"
        event, = [e for e in diagnostics.record["events"] if e["code"] == "legend_bar_selection"]
        assert len(event["removed_bar_ids"]) == 3
        assert set(event["kept_bar_ids"]) == {bar.id for bar in bars}
        spec, _ = apply_overrides(spec, Overrides(
            title="Sales", x_axis_title="Group", y_axis_title="Revenue"))
        generation = generate_tactile_artifacts(spec, tmp_path)
        assert generation.zip_path.exists()
        chart = spec_to_chart(spec)
        layouts = (layout_horizontal_bar_chart_pages(chart) if horizontal
                   else layout_vertical_bar_chart_pages(chart))
        assert sum(len(page.bars) for page in layouts) == 3
    else:
        assert spec.series[1].values == pytest.approx([50, 80, 120])
        assert "bar_colour_mismatch:3" in spec.flags


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("legend", [False, True])
def test_no_legend_matches_blocks_instead_of_generating_empty_chart(tmp_path, horizontal, legend):
    path, frame, ocr = selection_case(tmp_path, horizontal, legend=legend)
    if legend:
        # A visible swatch of a third colour matches neither series.
        image = cv2.imread(str(path))
        swatch = next(d for d in frame.detections if d.role == "legend_swatch")
        box = swatch.box
        image[int(box.y1):int(box.y2), int(box.x1):int(box.x2)] = (0, 0, 255)
        assert cv2.imwrite(str(path), image)
    with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
         patch("src.gradio_ui.analysis.run_ocr", return_value=ocr):
        result = analyze_image(object(), path, .25, .7, ocr_backend=object(),
                               keep_legend_matches=True)
    assert result.spec is None
    assert result.completeness.status == "cannot_build"
    assert "Turn off Keep only legend-matched bars" in result.completeness.blockers[0]


@pytest.mark.parametrize("horizontal", [False, True])
def test_selection_does_not_rescue_missing_bars_from_the_original_image(tmp_path, horizontal):
    path, frame, ocr = selection_case(tmp_path, horizontal)
    missing = next(d for d in frame.detections if d.role.endswith("_bar"))
    frame = replace(frame, detections=tuple(d for d in frame.detections if d.id != missing.id))
    with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
         patch("src.gradio_ui.analysis.run_ocr", return_value=ocr):
        result = analyze_image(object(), path, .25, .7, ocr_backend=object(),
                               keep_legend_matches=True)
    assert result.completeness.status == "cannot_build"
    assert len(result.spec.series) == 1
    assert result.spec.series[0].values[0] is None
    assert not any(result.spec.series[0].value_rescued)


def test_analysis_admission_captures_bar_selection_and_clear_resets_it(app):
    async def scenario():
        session = AppSession(app, "legend-selection")
        inputs = [None, .25, .7, "eng", 640, "rapidocr", "auto", True]
        admitted = await session.call("admit_analysis", inputs)
        token = admitted["data"][-2]["value"]
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=make_analysis_result()) as analyze:
            args = [*inputs[:7], token, False]
            first = await session.call("analyze", args)
            await session.call("analyze", args, first["iterator"])
        # The admission owns the selected setting even if the input later changes.
        assert analyze.call_args.kwargs["keep_legend_matches"] is True
        clear = session.find_callback("clear_all")
        selection = next(c for c in session.find_callback("admit_analysis").inputs
                         if c.label == "Keep only legend-matched bars")
        reset = await session.call("clear_all")
        assert reset["data"][clear.outputs.index(selection)]["value"] is False
    asyncio.run(scenario())
