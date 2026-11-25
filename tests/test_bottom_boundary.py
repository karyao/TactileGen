"""Shared-bottom selection must avoid changing ambiguous chart geometry."""

from src.contracts import Box, Detection, DetectionFrame
from src.reconstruction.bottom_boundary import filter_vertical_bottom


def test_ui_and_cli_use_filtered_frame_after_ocr(tmp_path):
    from contextlib import ExitStack
    from pathlib import Path
    from unittest.mock import patch

    from PIL import Image
    from src.contracts import OCRFrame
    from src.diagnostics.tracking import AnalysisDiagnostics
    from src.gradio_ui.analysis import analyze_image
    from src.tactile.pipeline import analyse
    from src.tactile.completeness import Completeness, READY
    from tests.gradio.session_helpers import make_spec
    from tests.legend_filter_helpers import passthrough_filter

    path = tmp_path / "chart.png"
    Image.new("RGB", (1296, 1098), "white").save(path)
    raw = archive_frame()
    ocr = OCRFrame(raw.source_key, ())
    spec = make_spec()
    diagnostics = AnalysisDiagnostics(settings={})
    with ExitStack() as stack:
        stack.enter_context(patch("src.gradio_ui.analysis.run_detection", return_value=raw))
        ocr_call = stack.enter_context(patch("src.gradio_ui.analysis.run_ocr", return_value=ocr))
        stack.enter_context(patch("src.gradio_ui.analysis.drop_unmatched_bars", return_value=None))
        filter_call = stack.enter_context(patch("src.gradio_ui.analysis.filter_bars_by_legend",
                                                side_effect=passthrough_filter))
        reconstruct = stack.enter_context(patch("src.gradio_ui.analysis.reconstruct_vbar",
                                                return_value=spec))
        stack.enter_context(patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]))
        stack.enter_context(patch("src.gradio_ui.analysis.check_completeness",
                                  return_value=Completeness(READY)))
        result = analyze_image(object(), path, .25, .7, ocr_backend=object(),
                               diagnostics=diagnostics)
    assert ocr_call.call_args.args[1] is result.ocr_detection_frame
    assert result.ocr_detection_frame is result.orientation_result.effective_frame
    assert "false1" in {d.id for d in result.ocr_detection_frame.detections}
    assert "false1" not in {d.id for d in result.detection_frame.detections}
    assert "of" not in {d.id for d in filter_call.call_args.args[1].detections}
    assert reconstruct.call_args.args[0] is result.detection_frame
    assert abs(reconstruct.call_args.kwargs["provisional_baseline"] - 892.2) < 1
    boundary_event = next(e for e in diagnostics.record["events"]
                          if e["code"] == "vertical_bottom_boundary")
    assert boundary_event["rejected_bar_ids"] == [
        "false1", "false2", "false3", "false4", "false5"]
    assert boundary_event["rejected_tick_ids"] == ["of"]

    with ExitStack() as stack:
        stack.enter_context(patch("src.detection.load_model", return_value=object()))
        stack.enter_context(patch("src.detection.run_detection", return_value=raw))
        cli_ocr = stack.enter_context(patch("src.ocr.run_ocr", return_value=ocr))
        stack.enter_context(patch("src.ocr.tesseract.TesseractBackend", return_value=object()))
        stack.enter_context(patch("src.bar_filter.drop_unmatched_bars", return_value=None))
        cli_filter = stack.enter_context(patch(
            "src.reconstruction.legend_matcher.filter_bars_by_legend",
            side_effect=passthrough_filter))
        cli_reconstruct = stack.enter_context(patch(
            "src.reconstruction.vertical.reconstruct_vbar", return_value=spec))
        reread = stack.enter_context(patch("src.tactile.pipeline.reread_categories",
                                           return_value=spec))
        stack.enter_context(patch("src.tactile.pipeline.cut_off_flags", return_value=[]))
        _, error, frame = analyse(path, Path("weights.pt"))
    assert error is None
    assert cli_ocr.call_args.args[1] is not frame
    assert "false1" not in {d.id for d in cli_filter.call_args.args[1].detections}
    assert cli_reconstruct.call_args.kwargs["provisional_baseline"] == filter_vertical_bottom(raw).baseline_y
    assert reread.call_args.args[1] is frame


def detection(identifier, role, box, confidence=.9):
    return Detection(identifier, role, Box(*box), confidence)


def archive_frame():
    return DetectionFrame("input/chart.png", 1296, 1098, (
        detection("a1", "vertical_bar", (394, 637, 448, 891.76), .97),
        detection("a2", "vertical_bar", (447, 720, 498, 892.30), .96),
        detection("m1", "vertical_bar", (576, 465, 626, 892.11), .96),
        detection("m2", "vertical_bar", (626, 377, 683, 892.57), .97),
        detection("j1", "vertical_bar", (758, 295, 812, 892.08), .97),
        detection("j2", "vertical_bar", (812, 379, 863, 892.41), .97),
        detection("false1", "vertical_bar", (736, 63, 748, 92), .49),
        detection("false2", "vertical_bar", (752, 69, 773, 87), .43),
        detection("false3", "vertical_bar", (751, 73, 773, 92), .41),
        detection("false4", "vertical_bar", (753, 68, 773, 82), .34),
        detection("false5", "vertical_bar", (709, 68, 722, 91), .26),
        detection("april", "x_tick_label", (416, 919, 489, 958), .83),
        detection("may", "x_tick_label", (599, 920, 667, 958), .80),
        detection("june", "x_tick_label", (776, 919, 853, 958), .52),
        detection("of", "x_tick_label", (110, 451, 148, 486), .37),
    ))


def test_archive_retains_real_bars_and_months():
    frame = archive_frame()
    result = filter_vertical_bottom(frame)
    assert result is not None
    assert .77 < result.confidence_share < .79
    assert abs(result.baseline_y - 892.2) < 1
    assert set(result.retained_bar_ids) == {"a1", "a2", "m1", "m2", "j1", "j2"}
    assert set(result.rejected_bar_ids) == {"false1", "false2", "false3", "false4", "false5"}
    assert result.rejected_tick_ids == ("of",)
    assert {d.id for d in result.frame.detections} == {
        "a1", "a2", "m1", "m2", "j1", "j2", "april", "may", "june"}
    assert len(frame.detections) == 15


def test_ambiguous_or_plausible_bars_do_not_filter():
    base = archive_frame()
    bars = [d for d in base.detections if d.id in {"a1", "a2", "false1", "false2"}]
    assert filter_vertical_bottom(DetectionFrame(base.source_key, base.width, base.height,
                                                 tuple(bars))) is None
    wide = detection("wide", "vertical_bar", (700, 63, 755, 92), .49)
    frame = DetectionFrame(base.source_key, base.width, base.height,
                           tuple(d for d in base.detections if d.id != "false1") + (wide,))
    assert filter_vertical_bottom(frame) is None


def test_plot_area_and_horizontal_bars_keep_existing_paths():
    base = archive_frame()
    plot = detection("plot", "plot_area", (350, 190, 900, 900))
    assert filter_vertical_bottom(DetectionFrame(base.source_key, base.width, base.height,
                                                 (*base.detections, plot))) is None
    horizontal = DetectionFrame(base.source_key, base.width, base.height, (
        detection("h1", "horizontal_bar", (200, 300, 700, 350)),
        detection("h2", "horizontal_bar", (200, 400, 800, 450)),
    ))
    assert filter_vertical_bottom(horizontal) is None
