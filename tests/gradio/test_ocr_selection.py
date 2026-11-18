from tests.gradio.callback_helpers import invoke
"""Engine selection, lazy initialization, and partial-result recovery."""

import json
from unittest.mock import Mock, patch
from zipfile import ZipFile

from PIL import Image
import pytest

from src.contracts import Box, Detection, DetectionFrame, OCRFrame
from src.gradio_app import analyze_image, build_app, parse_args
from src.gradio_ui.ocr import OCRBackendResolver


def test_resolver_is_lazy_reuses_success_and_retries_failure():
    rapid = object()
    tesseract = object()
    with patch("src.ocr.rapidocr.RapidOCRBackend", side_effect=[RuntimeError("offline"), rapid]) as load_rapid, \
         patch("src.ocr.tesseract.TesseractBackend", return_value=tesseract) as load_tesseract:
        resolver = OCRBackendResolver()
        load_rapid.assert_not_called()
        load_tesseract.assert_not_called()
        with pytest.raises(RuntimeError, match="offline"):
            resolver.resolve("rapidocr")
        assert resolver.resolve("tesseract") is tesseract
        assert resolver.resolve("rapidocr") is rapid
        assert resolver.resolve("tesseract") is tesseract
        assert resolver.resolve("rapidocr") is rapid
        assert load_rapid.call_count == 2
        load_tesseract.assert_called_once_with()
        with pytest.raises(ValueError, match="Unsupported OCR engine"):
            resolver.resolve("unknown")


@pytest.mark.parametrize("engine", ["rapidocr", "tesseract"])
def test_ui_initial_selection_clear_and_controls_are_lazy(engine):
    assert parse_args([]).ocr == "rapidocr"
    assert parse_args(["--ocr", engine]).ocr == engine
    with patch("src.ocr.rapidocr.RapidOCRBackend") as rapid, \
         patch("src.ocr.tesseract.TesseractBackend") as tesseract:
        demo = build_app(object(), ocr_engine=engine)
        try:
            rapid.assert_not_called()
            tesseract.assert_not_called()
            event = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            selector = next(c for c in event.inputs if c.label == "OCR engine")
            assert selector.label == "OCR engine"
            assert selector.value == engine
            clear = next(fn for fn in demo.fns.values() if fn.api_name == "clear_all")
            assert invoke(clear.fn)[clear.outputs.index(selector)]["value"] == engine
        finally:
            demo.close(verbose=False)


def test_injected_backend_serves_initial_choice_only():
    custom = object()
    resolver = OCRBackendResolver("tesseract", custom)
    with patch("src.ocr.rapidocr.RapidOCRBackend", return_value=object()) as rapid:
        assert resolver.resolve("tesseract") is custom
        assert resolver.resolve("rapidocr") is rapid.return_value


def test_analysis_rejects_conflicting_backend_arguments():
    factory = Mock()
    with pytest.raises(ValueError, match="either ocr_backend or ocr_backend_factory"):
        analyze_image(object(), None, .25, .7, ocr_backend=object(), ocr_backend_factory=factory)
    factory.assert_not_called()


def test_engine_failure_switch_and_retry_preserve_results_and_diagnostics(tmp_path):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    frame = DetectionFrame("input/chart.png", 100, 80, (
        Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
    ))
    ocr = OCRFrame(frame.source_key, ())
    # Real adapter instances provide diagnostic metadata; their constructors and
    # recognition are mocked so this test requires neither models nor binaries.
    from src.ocr.rapidocr import RapidOCRBackend
    from src.ocr.tesseract import TesseractBackend

    rapid_backend = object.__new__(RapidOCRBackend)
    tesseract_backend = TesseractBackend()
    demo = build_app(object())
    event = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
    archive_index = next(i for i, c in enumerate(event.outputs)
                         if getattr(c, "label", None) == "Download analysis diagnostics (ZIP)")
    overlay_index = next(i for i, c in enumerate(event.outputs)
                         if getattr(c, "label", None) == "Annotated detections")
    try:
        with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
             patch("src.gradio_ui.analysis.run_ocr", return_value=ocr) as recognize, \
             patch("src.ocr.rapidocr.RapidOCRBackend", side_effect=[RuntimeError("model unavailable"), rapid_backend]) as rapid, \
             patch("src.ocr.tesseract.TesseractBackend", return_value=tesseract_backend) as tesseract:
            for engine, expected in (("rapidocr", None), ("tesseract", tesseract_backend),
                                     ("rapidocr", rapid_backend), ("tesseract", tesseract_backend)):
                updates = invoke(event.fn, str(image_path), .25, .7, "eng+fra", 640, engine)
                assert updates[overlay_index]["value"] is not None
                with ZipFile(updates[archive_index]["value"]) as archive:
                    record = json.loads(archive.read("diagnostics.json"))
                    assert record["settings"]["ocr_engine"] == engine
                    assert record["metadata"]["detection"]["requested_imgsz"] == 640
                    assert "detections.json" in archive.namelist()
                    if expected is None:
                        recognize.assert_not_called()
                        assert record["stages"]["ocr"]["outcome"] == "failed"
                        assert any(w["code"] == "ocr_unavailable" for w in record["warnings"])
                        assert "model unavailable" in record["exceptions"][0]["message"]
                    else:
                        assert recognize.call_args.args[2] is expected
                        assert recognize.call_args.args[3] == "eng+fra"
                        assert record["metadata"]["ocr"]["class"].endswith(type(expected).__name__)
                        assert "ocr.json" in archive.namelist()
            assert rapid.call_count == 2
            tesseract.assert_called_once_with()
    finally:
        demo.close(verbose=False)
