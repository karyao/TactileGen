"""The CLI and Gradio apply the same optional category-label fallback."""

from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np
import pytest

from src.contracts import Box, Detection, DetectionFrame, OCRFrame, RecognizedText, SeriesSpec
from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.analysis import analyze_image
from src.gradio_ui.artifacts import write_spec_download
from src.gradio_ui.review import build_review_model, review_text_rows
from src.presentation import ocr_rows
from src.tactile import pipeline
from src.tactile.category_reread import reread_categories
from src.tactile.completeness import check_completeness
from tests.gradio.session_helpers import make_spec
from tests.legend_filter_helpers import passthrough_filter


def chart_fixture(tmp_path, orientation="vertical", categories=None):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (400, 200), "white").save(image_path)
    detections = [Detection("plot", "plot_area", Box(20, 10, 390, 150), .9),
                  Detection("bar", orientation + "_bar", Box(40, 50, 60, 150), .9)]
    for j in range(4):
        x = 40 + 90 * j
        detections.append(Detection(f"x{j}", "x_tick_label", Box(x, 160, x + 40, 195), .9))
    frame = DetectionFrame("input/chart.png", 400, 200, tuple(detections))
    categories = categories or ["Germany", "|", "|", "India"]
    spec = make_spec(categories=categories, series=[SeriesSpec(
        None, "#112233", [1.0, 2.0, 3.0, 4.0])],
        flags=["category_unreadable"] if "|" in categories else [],
        confidence={f"category[{j}]": .1 for j, name in enumerate(categories) if name == "|"},
        orientation=orientation)
    return image_path, frame, spec


def run_both(image_path, frame, spec, engine, retry):
    ocr = OCRFrame(frame.source_key, (RecognizedText("x1", "|", .1),))
    diagnostics = AnalysisDiagnostics(settings={})
    with (patch("cv2.imread", return_value=np.zeros((200, 400, 3), dtype="uint8")),
          patch("src.detection.load_model"),
          patch("src.detection.run_detection", return_value=frame),
          patch.object(pipeline, "_ocr_backend", return_value=(object(), "eng")),
          patch("src.ocr.run_ocr", return_value=ocr),
          patch("src.reconstruction.legend_matcher.match_legend_to_bars"),
          patch("src.bar_filter.drop_unmatched_bars", return_value=None),
          patch("src.reconstruction.legend_matcher.filter_bars_by_legend", side_effect=passthrough_filter),
          patch("src.reconstruction.vertical.reconstruct_vbar", return_value=spec),
          patch("src.reconstruction.horizontal.reconstruct_hbar", return_value=spec),
          patch.object(pipeline, "cut_off_flags", return_value=[]),
          patch.object(pipeline, "reread_categories", side_effect=retry) as cli_retry):
        cli_spec, error, _ = pipeline.analyse(image_path, Path("weights.pt"),
                                               ocr_engine=engine, bar_orientation=spec.orientation)
    assert error is None
    with (patch("src.gradio_ui.analysis.run_detection", return_value=frame),
          patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),
          patch("src.gradio_ui.analysis.match_legend_to_bars"),
          patch("src.gradio_ui.analysis.drop_unmatched_bars", return_value=None),
          patch("src.gradio_ui.analysis.filter_bars_by_legend", side_effect=passthrough_filter),
          patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=spec),
          patch("src.gradio_ui.analysis.reconstruct_hbar", return_value=spec),
          patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]),
          patch("src.gradio_ui.analysis.reread_categories", side_effect=retry) as ui_retry):
        result = analyze_image(object(), image_path, .25, .7, ocr_engine=engine,
                               bar_orientation=spec.orientation, ocr_backend=object(),
                               diagnostics=diagnostics)
    return cli_spec, result, diagnostics, cli_retry, ui_retry


@pytest.mark.parametrize("engine", ["tesseract", "hybrid"])
def test_partial_recovery_matches_cli_and_reaches_review_export_and_diagnostics(tmp_path, engine):
    image_path, frame, spec = chart_fixture(tmp_path)
    reader = Mock(side_effect=lambda image, label:
                  ("Hungary", .9) if label.id == "x1" else None)
    retry = lambda spec, frame, image: reread_categories(spec, frame, image, reader=reader)
    cli, ui, diagnostics, cli_retry, ui_retry = run_both(image_path, frame, spec, engine, retry)
    assert cli_retry.call_args.args[1] == frame
    assert ui_retry.call_args.args[1] is ui.orientation_result.effective_frame
    assert cli.categories == ui.spec.categories == ["Germany", "Hungary", "|", "India"]
    assert [m.field for m in check_completeness(cli).missing] == [
        m.field for m in ui.completeness.missing] == ["category[2]"]
    assert ui.spec.series == spec.series
    assert ui.ocr_frame == OCRFrame(frame.source_key, (RecognizedText("x1", "|", .1),))
    assert any(row[0] == "x1" and "|" in row for row in ocr_rows(ui.detection_frame, ui.ocr_frame))
    rows = review_text_rows(build_review_model(ui.spec, ui.completeness))
    assert ["Category 1", "Hungary"] in rows
    assert '"Hungary"' in write_spec_download(tmp_path, image_path, ui.spec).read_text()
    assert diagnostics.record["outputs"]["spec"]["categories"] == ui.spec.categories
    assert any(e["code"] == "category_reread" and e["category_index"] == 1
               and e["original_text"] == "|" and e["recovered_text"] == "Hungary"
               for e in diagnostics.record["events"])
    assert any("re-read" in note for note in ui.completeness.notes)


@pytest.mark.parametrize("engine,orientation,categories", [
    ("rapidocr", "vertical", None),
    ("tesseract", "horizontal", None),
    ("hybrid", "vertical", ["Germany", "Greece", "Spain", "India"]),
])
def test_ineligible_or_readable_charts_make_no_reader_calls(tmp_path, engine, orientation, categories):
    image_path, frame, spec = chart_fixture(tmp_path, orientation, categories)
    reader = Mock()
    retry = lambda spec, frame, image: reread_categories(spec, frame, image, reader=reader)
    cli, ui, _, cli_retry, ui_retry = run_both(image_path, frame, spec, engine, retry)
    reader.assert_not_called()
    assert cli.categories == ui.spec.categories == spec.categories
    if engine == "rapidocr" or orientation == "horizontal":
        cli_retry.assert_not_called()
        ui_retry.assert_not_called()
    if categories is not None:
        cli_retry.assert_not_called()
        ui_retry.assert_not_called()


def test_failed_retry_keeps_spec_and_reports_warning(tmp_path, capsys):
    image_path, frame, spec = chart_fixture(tmp_path)
    retry = Mock(side_effect=RuntimeError("Tesseract missing"))
    cli, ui, diagnostics, _, _ = run_both(image_path, frame, spec, "tesseract", retry)
    assert cli.categories == ui.spec.categories == spec.categories
    assert [m.field for m in ui.completeness.missing] == ["category[1]", "category[2]"]
    assert "retry was unavailable" in capsys.readouterr().err
    assert any("retry was unavailable" in note for note in ui.completeness.notes)
    assert any(w["code"] == "category_reread_unavailable" for w in diagnostics.record["warnings"])


def test_mismatched_label_count_preserves_original(tmp_path):
    image_path, frame, spec = chart_fixture(tmp_path)
    frame = DetectionFrame(frame.source_key, frame.width, frame.height, frame.detections[:-1])
    reader = Mock()
    retry = lambda spec, frame, image: reread_categories(spec, frame, image, reader=reader)
    cli, ui, _, _, _ = run_both(image_path, frame, spec, "hybrid", retry)
    reader.assert_not_called()
    assert cli.categories == ui.spec.categories == spec.categories
