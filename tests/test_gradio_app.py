"""Runnable Gradio vertical slice without loading a real checkpoint."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from src.borealis import BOREALIS_CSS
from src.contract_io import load_ocr_frame
from src.contracts import (
    Box,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
)
from src.gradio_app import (
    AnalysisResult,
    DEFAULT_CONFIDENCE,
    DEFAULT_HOST,
    DEFAULT_NMS_IOU,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_PORT,
    DEFAULT_WEIGHTS,
    MAX_UPLOAD_BYTES,
    LEGEND_WARNING,
    OCR_WARNING,
    UploadValidationError,
    alert_message,
    analyze_image,
    analyzing_status,
    build_app,
    complete_status,
    decode_upload,
    main,
    parse_args,
    ready_status,
)


class GradioAppTest(unittest.TestCase):
    def _image(
        self,
        root: Path,
        name: str = "chart.png",
        *,
        size=(100, 80),
        format: str | None = None,
        rgb=(255, 255, 255),
    ) -> Path:
        path = root / name
        Image.new("RGB", size, rgb).save(path, format=format)
        return path

    def test_argument_defaults_and_port_validation(self):
        args = parse_args([])
        self.assertEqual(args.weights, DEFAULT_WEIGHTS)
        self.assertEqual(args.host, DEFAULT_HOST)
        self.assertEqual(args.port, DEFAULT_PORT)
        self.assertIsNone(args.device)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--port", "0"])
            with self.assertRaises(SystemExit):
                parse_args(["--port", "not-a-port"])

    def test_decode_accepts_supported_content_and_ignores_filename_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("chart.png", "PNG"),
                ("chart.jpg", "JPEG"),
                ("chart.webp", "WEBP"),
                ("renamed.data", "PNG"),
            )
            for name, image_format in cases:
                with self.subTest(name=name):
                    path = self._image(root, name, format=image_format)
                    returned_path, image = decode_upload(path)
                    self.assertEqual(returned_path, path)
                    self.assertEqual(image.shape, (80, 100, 3))

    def test_valid_analysis_forwards_settings_and_swaps_bgr_to_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(
                Path(temporary), rgb=(30, 20, 10)  # OpenCV reads this as BGR 10,20,30.
            )
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            progress = Mock()
            model = object()
            backend = object()
            ocr = OCRFrame("input/chart.png", ())
            with patch("src.gradio_app.run_detection", return_value=frame) as run, \
                 patch("src.gradio_app.run_ocr", return_value=ocr) as recognize:
                result = analyze_image(
                    model,
                    image_path,
                    0.4,
                    0.6,
                    language="eng",
                    ocr_backend=backend,
                    device="cpu",
                    progress=progress,
                )
            run.assert_called_once_with(
                model, image_path, confidence=0.4, nms_iou=0.6, device="cpu"
            )
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.detection_count, 0)
            self.assertEqual(result.ocr_frame, ocr)
            recognize.assert_called_once()
            self.assertIs(recognize.call_args.args[2], backend)
            self.assertEqual(recognize.call_args.args[3], "eng")
            self.assertIn("Detected 0 chart elements.", result.summary)
            self.assertIn("OCR attempted 0 text regions", result.summary)
            self.assertEqual(tuple(result.overlay_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(progress.call_count, 6)

    def test_ocr_failure_keeps_successful_detection_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("title", "chart_title", Box(5, 5, 50, 15), 0.9),
            ))
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", side_effect=RuntimeError("missing")), \
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.detection_frame, frame)
            self.assertIsNone(result.ocr_frame)
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertIn("Detected 1 chart element.", result.summary)
            self.assertIsNotNone(result.overlay_rgb)

    def test_matcher_failure_is_independent_and_can_coexist_with_ocr_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(5, 5, 20, 70), 0.9),
            ))
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", side_effect=RuntimeError("ocr")), \
                 patch("src.gradio_app.match_legend_to_bars",
                       side_effect=RuntimeError("matcher")), \
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.legend_warning, LEGEND_WARNING)
            self.assertIsNone(result.legend_mapping)
            self.assertNotIn("remained unmatched", result.summary)

    def test_invalid_upload_never_calls_model_and_mismatched_frame_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "broken.png"
            broken.write_bytes(b"not an image")
            with patch("src.gradio_app.run_detection") as run:
                with self.assertRaises(UploadValidationError):
                    analyze_image(object(), broken, 0.25, 0.70)
                run.assert_not_called()

            valid = self._image(root)
            mismatched = DetectionFrame("input/chart.png", 99, 80, ())
            with patch("src.gradio_app.run_detection", return_value=mismatched):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    analyze_image(object(), valid, 0.25, 0.70)

    def test_upload_validation_rejects_missing_size_dimensions_and_bad_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "directory"
            directory.mkdir()
            for missing in (None, root / "missing.png", directory):
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(UploadValidationError, "Choose"):
                        decode_upload(missing)

            empty = root / "empty.png"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(UploadValidationError, "could not be read"):
                decode_upload(empty)

            text = root / "chart.png"
            text.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(UploadValidationError, "could not be read"):
                decode_upload(text)

            gif = self._image(root, "chart.gif", format="GIF")
            with self.assertRaisesRegex(UploadValidationError, "PNG, JPEG, or WebP"):
                decode_upload(gif)

            too_large = root / "large.png"
            with too_large.open("wb") as stream:
                stream.truncate(MAX_UPLOAD_BYTES + 1)
            with self.assertRaisesRegex(UploadValidationError, "20 MB"):
                decode_upload(too_large)

            normal = self._image(root, "dimensions.png")
            fake_source = Mock()
            fake_source.__enter__ = Mock(return_value=fake_source)
            fake_source.__exit__ = Mock(return_value=False)
            fake_source.format = "PNG"
            fake_source.size = (10_000, 4_001)
            with patch("PIL.Image.open", return_value=fake_source):
                with self.assertRaisesRegex(UploadValidationError, "40 megapixels"):
                    decode_upload(normal)

    def test_accessible_status_helpers_cover_ready_busy_success_zero_and_error(self):
        self.assertIn('role="status"', ready_status())
        self.assertIn('aria-live="polite"', analyzing_status())
        self.assertIn("No detections found", complete_status(0))
        self.assertIn("1 detection found", complete_status(1))
        self.assertIn("3 detections found", complete_status(3))
        self.assertIn("OCR attempted 2 text regions", complete_status(3, 2))
        self.assertIn('role="alert"', alert_message("Try again"))

    def test_app_factory_wires_controls_states_and_private_events(self):
        with patch("src.gradio_app.load_model") as load, \
             patch("gradio.Blocks.launch") as launch:
            demo = build_app(object(), device="cpu")
        load.assert_not_called()
        launch.assert_not_called()
        config = demo.get_config_file()
        components = config["components"]
        labels = {component["props"].get("label") for component in components}
        self.assertIn("Vertical bar chart image", labels)
        self.assertIn("Minimum detection confidence", labels)
        self.assertIn("NMS IoU for duplicate suppression", labels)
        self.assertIn("OCR language", labels)
        self.assertIn("Annotated detections", labels)
        self.assertIn("Recognized text", labels)
        self.assertIn("Legend mappings", labels)

        by_class = {}
        for component in components:
            for class_name in component["props"].get("elem_classes", []):
                by_class[class_name] = component
        for class_name in (
            "borealis-scope", "borealis-control-rail", "borealis-workspace",
            "borealis-status", "borealis-empty", "borealis-summary",
            "borealis-error",
        ):
            self.assertIn(class_name, by_class)
        self.assertTrue(by_class["borealis-empty"]["props"]["visible"])
        self.assertFalse(by_class["borealis-summary"]["props"]["visible"])
        self.assertFalse(by_class["borealis-error"]["props"]["visible"])
        annotated = next(c for c in components
                         if c["props"].get("label") == "Annotated detections")
        self.assertFalse(annotated["props"]["visible"])
        recognized = next(c for c in components
                          if c["props"].get("label") == "Recognized text")
        self.assertFalse(recognized["props"]["visible"])

        inference = [dependency for dependency in config["dependencies"]
                     if dependency.get("api_name") == "analyze"]
        self.assertEqual(len(inference), 1)
        self.assertEqual(inference[0]["api_visibility"], "private")
        self.assertEqual(inference[0]["show_progress"], "full")
        inference_fn = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        self.assertEqual(inference_fn.concurrency_limit, 1)
        self.assertEqual(inference_fn.concurrency_id, "tactilegen-inference")

        begin = next(fn.fn for fn in demo.fns.values() if fn.api_name == "begin_analysis")
        begin_updates = begin()
        self.assertEqual(len(begin_updates), 16)
        self.assertTrue(all(update["interactive"] is False for update in begin_updates[:6]))
        finish = next(fn.fn for fn in demo.fns.values() if fn.api_name == "finish_analysis")
        self.assertTrue(all(update["interactive"] is True for update in finish()))
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = ready()
        self.assertEqual(len(ready_updates), 10)
        self.assertTrue(ready_updates[1]["visible"])
        self.assertFalse(ready_updates[2]["visible"])
        self.assertFalse(ready_updates[4]["visible"])

        analyze = inference_fn.fn
        display_frame = DetectionFrame("input/chart.png", 2, 2, (
            Detection("title", "chart_title", Box(0, 0, 2, 1), 0.9),
        ))
        display_ocr = OCRFrame("input/chart.png", (
            RecognizedText("title", "Sales", 0.8),
        ))
        success = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, display_ocr, None,
        )
        with patch("src.gradio_app.analyze_image", return_value=success):
            success_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertIn("1 detection found", success_updates[0]["value"])
        self.assertTrue(success_updates[2]["visible"])
        self.assertTrue(success_updates[3]["visible"])
        self.assertTrue(success_updates[4]["visible"])
        self.assertEqual(success_updates[4]["value"][0][:4],
                         ["title", "chart title", "Sales", 0.8])
        self.assertTrue(success_updates[5]["visible"])
        self.assertEqual(load_ocr_frame(Path(success_updates[5]["value"])), display_ocr)
        self.assertFalse(success_updates[6]["visible"])
        self.assertFalse(success_updates[7]["visible"])
        self.assertFalse(success_updates[8]["visible"])
        self.assertFalse(success_updates[9]["visible"])

        mapped_frame = DetectionFrame("input/chart.png", 20, 20, (
            Detection("bar", "vertical_bar", Box(0, 5, 5, 20), 0.9),
            Detection("swatch", "legend_swatch", Box(8, 0, 12, 4), 0.9),
            Detection("label", "legend_label", Box(13, 0, 20, 4), 0.9),
        ))
        mapped = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment(
                "bar", "swatch", "label", "North", 0.98, "matched"
            ),
        ))
        mapped_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None,
        )
        with patch("src.gradio_app.analyze_image", return_value=mapped_result):
            mapped_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(mapped_updates[6]["visible"])
        self.assertEqual(
            mapped_updates[6]["value"][0],
            ["bar", "North", 0.98, "swatch", "label", "matched"],
        )

        warning = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, None, OCR_WARNING,
        )
        with patch("src.gradio_app.analyze_image", return_value=warning):
            warning_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(warning_updates[2]["visible"])
        self.assertTrue(warning_updates[3]["visible"])
        self.assertFalse(warning_updates[4]["visible"])
        self.assertTrue(warning_updates[7]["visible"])
        self.assertFalse(warning_updates[8]["visible"])
        self.assertFalse(warning_updates[9]["visible"])
        with patch(
            "src.gradio_app.analyze_image",
            side_effect=UploadValidationError("Upload a supported image."),
        ):
            failure_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertFalse(failure_updates[2]["visible"])
        self.assertFalse(failure_updates[3]["visible"])
        self.assertFalse(failure_updates[4]["visible"])
        self.assertTrue(failure_updates[9]["visible"])
        self.assertIn('role="alert"', failure_updates[9]["value"])

        private_path = "/private/tmp/gradio/secret-chart.png"
        with self.assertLogs("src.gradio_app", level="ERROR"), patch(
            "src.gradio_app.analyze_image",
            side_effect=RuntimeError(private_path),
        ):
            unexpected_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(unexpected_updates[9]["visible"])
        self.assertNotIn(private_path, unexpected_updates[9]["value"])
        self.assertNotIn("RuntimeError", unexpected_updates[9]["value"])

        change_events = [dependency for dependency in config["dependencies"]
                         if dependency["targets"] and dependency["targets"][0][1] == "change"]
        self.assertEqual(len(change_events), 1)
        self.assertFalse(change_events[0]["queue"])
        demo.close(verbose=False)

    def test_main_loads_once_and_uses_private_styled_launch(self):
        model = object()
        backend = object()
        demo = Mock()
        theme = object()
        with patch("src.gradio_app._import_gradio"), \
             patch("src.gradio_app.load_model", return_value=model) as load, \
             patch("src.gradio_app.TesseractBackend", return_value=backend) as load_ocr, \
             patch("src.gradio_app.build_app", return_value=demo) as build, \
             patch("src.gradio_app.borealis_theme", return_value=theme):
            main(["--weights", "custom.pt", "--device", "cpu",
                  "--host", "0.0.0.0", "--port", "9000"])
        load.assert_called_once_with(Path("custom.pt"))
        load_ocr.assert_called_once_with()
        build.assert_called_once_with(model, ocr_backend=backend, device="cpu")
        kwargs = demo.launch.call_args.kwargs
        self.assertEqual(kwargs["server_name"], "0.0.0.0")
        self.assertEqual(kwargs["server_port"], 9000)
        self.assertIs(kwargs["share"], False)
        self.assertIs(kwargs["show_error"], False)
        self.assertEqual(kwargs["max_file_size"], MAX_UPLOAD_BYTES)
        self.assertIs(kwargs["theme"], theme)
        self.assertEqual(kwargs["css"], BOREALIS_CSS)
        self.assertIs(kwargs["run_history"], False)
        self.assertIs(kwargs["enable_monitoring"], False)


if __name__ == "__main__":
    unittest.main()
