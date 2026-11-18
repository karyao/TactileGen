from tests.gradio.callback_helpers import invoke
"""Orientation controls, reruns, diagnostics settings, and stale-result resets."""
from dataclasses import fields
import json
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
import pytest

from src.contracts import Box, Detection, DetectionFrame, LegendBarAssignment, LegendMappingFrame, OCRFrame
from src.gradio_app import build_app
from src.gradio_ui.components import AnalysisComponents, GenerationComponents
from src.gradio_ui.review import review_text_rows
from tests.gradio.test_gradio_app import make_spec


@pytest.fixture
def app():
    demo = build_app(object(), ocr_backend=object())
    yield demo
    demo.close(verbose=False)


def handler(app, name):
    return next(fn for fn in app.fns.values() if fn.api_name == name)


def named(updates):
    return dict(zip((field.name for field in fields(AnalysisComponents)), updates, strict=True))


def assert_cleared(updates):
    assert updates["review_state"] is None
    assert updates["review_fields"]["value"] == []
    assert updates["review_values"]["value"] == []
    assert updates["generate_button"]["interactive"] is False
    assert updates["spec_output"]["value"] == {}
    assert updates["spec_download"]["value"] is None
    assert updates["generation_gallery"]["value"] == []
    assert updates["generation_report_output"]["value"] == ""
    assert updates["generation_download"]["value"] is None
    for name in ("review_panel", "spec_output", "spec_download", "generation_gallery",
                 "generation_report_output", "generation_download", "generation_error_output"):
        assert updates[name]["visible"] is False


def test_orientation_control_events_default_busy_and_clear(app):
    analysis = handler(app, "analyze")
    selector = next(c for c in analysis.inputs if c.label == "Bar orientation")
    assert selector.value == "auto"
    assert selector.choices == [("Auto", "auto"), ("Vertical", "vertical"), ("Horizontal", "horizontal")]
    clear = handler(app, "clear_all")
    assert invoke(clear.fn)[clear.outputs.index(selector)]["value"] == "auto"
    changed = handler(app, "orientation_changed")
    event = next(d for d in app.get_config_file()["dependencies"]
                 if [selector._id, "input"] in [list(t) for t in d["targets"]])
    assert event["outputs"] == [c._id for c in analysis.outputs]
    assert event["queue"] is False
    assert event["api_visibility"] == "private"
    reset = named(invoke(changed.fn))
    assert_cleared(reset)
    assert "Analyze chart" in reset["empty_output"]["value"]
    assert reset["orientation_output"]["visible"] is False
    assert reset["diagnostics_download"]["value"] is None
    assert named(invoke(handler(app, "ready_results").fn))["orientation_output"]["visible"] is False


def test_preference_reruns_uncertainty_and_generation_reset(app, tmp_path):
    path = tmp_path / "chart.png"
    Image.new("RGB", (100, 80), "white").save(path)
    analysis = handler(app, "analyze")

    def frame(vertical, horizontal):
        return DetectionFrame("input/chart.png", 100, 80, tuple(
            Detection(identifier, role, Box(10, 20, 25, 60), score)
            for identifier, role, score in (("v", "vertical_bar", vertical), ("h", "horizontal_bar", horizontal))
            if score
        ))

    def mapping(image, effective, ocr, **kwargs):
        return LegendMappingFrame(effective.source_key, tuple(LegendBarAssignment(d.id) for d in effective.detections))

    def run(preference):
        return named(invoke(analysis.fn, str(path), .25, .7, "eng", 640, "rapidocr", preference))

    def settings(updates):
        with ZipFile(updates["diagnostics_download"]["value"]) as archive:
            return json.loads(archive.read("diagnostics.json"))["settings"]

    with patch("src.gradio_ui.analysis.run_detection", return_value=frame(.9, .2)) as detect, \
         patch("src.gradio_ui.analysis.run_ocr", return_value=OCRFrame("input/chart.png", ())) as ocr, \
         patch("src.gradio_ui.analysis.match_legend_to_bars", side_effect=mapping), \
         patch("src.gradio_ui.analysis.match_legend_to_hbars", side_effect=mapping), \
         patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=make_spec()), \
         patch("src.gradio_ui.analysis.reconstruct_hbar", return_value=make_spec(orientation="horizontal")), \
         patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]):
        success = run("auto")
        assert success["generate_button"]["interactive"] is True
        assert "vertical" in success["orientation_output"]["value"]
        assert "automatic" in success["orientation_output"]["value"]
        assert "Suppressed 1" in success["orientation_output"]["value"]
        assert settings(success)["bar_orientation"] == "auto"
        model = success["review_state"]
        generation = invoke(handler(app, "generate_tactile").fn, model, review_text_rows(model))
        generated = dict(zip((f.name for f in fields(GenerationComponents)), generation, strict=True))
        assert generated["generation_gallery"]["visible"] is True
        assert generated["generation_download"]["value"] is not None

        # The selector's actual event handler clears both analysis and generation.
        assert_cleared(named(invoke(handler(app, "orientation_changed").fn)))
        detect.return_value = frame(.9, .9)
        uncertain = run("auto")
        assert_cleared(uncertain)
        assert uncertain["annotated_output"]["visible"] is True
        assert "Bar orientation is unclear. Choose Vertical or Horizontal and analyze again." in uncertain["orientation_output"]["value"]
        assert ocr.call_count == 1

        horizontal = run("horizontal")
        assert horizontal["generate_button"]["interactive"] is True
        assert "horizontal" in horizontal["orientation_output"]["value"]
        assert "manual" in horizontal["orientation_output"]["value"]
        assert settings(horizontal)["bar_orientation"] == "horizontal"
        assert run("horizontal")["spec_output"]["value"]["orientation"] == "horizontal"

        detect.return_value = frame(0, .9)
        unavailable = run("vertical")
        assert_cleared(unavailable)
        assert "No usable vertical bars were detected" in unavailable["orientation_output"]["value"]
        assert "Check the requested orientation or image" in unavailable["orientation_output"]["value"]
        assert ocr.call_count == 3
