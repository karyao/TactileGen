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
from src.contracts import Box, Detection, DetectionFrame
from src.gradio_app import (
    AnalysisResult,
    DEFAULT_CONFIDENCE,
    DEFAULT_HOST,
    DEFAULT_NMS_IOU,
    DEFAULT_PORT,
    DEFAULT_WEIGHTS,
    MAX_UPLOAD_BYTES,
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
            with patch("src.gradio_app.run_detection", return_value=frame) as run:
                result = analyze_image(
                    model,
                    image_path,
                    0.4,
                    0.6,
                    device="cpu",
                    progress=progress,
                )
            run.assert_called_once_with(
                model, image_path, confidence=0.4, nms_iou=0.6, device="cpu"
            )
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.detection_count, 0)
            self.assertIn("Detected 0 chart elements.", result.summary)
            self.assertEqual(tuple(result.overlay_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(progress.call_count, 4)

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
        self.assertIn("Annotated detections", labels)

        by_class = {}
        for component in components:
            for class_name in component["props"].get("elem_classes", []):
                by_class[class_name] = component
        for class_name in (
            "borealis-scope", "borealis-upload", "borealis-results",
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
        self.assertEqual(len(begin_updates), 10)
        self.assertTrue(all(update["interactive"] is False for update in begin_updates[:5]))
        finish = next(fn.fn for fn in demo.fns.values() if fn.api_name == "finish_analysis")
        self.assertTrue(all(update["interactive"] is True for update in finish()))
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = ready()
        self.assertTrue(ready_updates[1]["visible"])
        self.assertFalse(ready_updates[2]["visible"])

        analyze = inference_fn.fn
        success = AnalysisResult("summary", np.zeros((2, 2, 3), dtype=np.uint8), 2)
        with patch("src.gradio_app.analyze_image", return_value=success):
            success_updates = analyze("chart.png", 0.25, 0.70)
        self.assertIn("2 detections found", success_updates[0]["value"])
        self.assertTrue(success_updates[2]["visible"])
        self.assertTrue(success_updates[3]["visible"])
        self.assertFalse(success_updates[4]["visible"])
        with patch(
            "src.gradio_app.analyze_image",
            side_effect=UploadValidationError("Upload a supported image."),
        ):
            failure_updates = analyze("chart.png", 0.25, 0.70)
        self.assertFalse(failure_updates[2]["visible"])
        self.assertFalse(failure_updates[3]["visible"])
        self.assertTrue(failure_updates[4]["visible"])
        self.assertIn('role="alert"', failure_updates[4]["value"])

        private_path = "/private/tmp/gradio/secret-chart.png"
        with self.assertLogs("src.gradio_app", level="ERROR"), patch(
            "src.gradio_app.analyze_image",
            side_effect=RuntimeError(private_path),
        ):
            unexpected_updates = analyze("chart.png", 0.25, 0.70)
        self.assertTrue(unexpected_updates[4]["visible"])
        self.assertNotIn(private_path, unexpected_updates[4]["value"])
        self.assertNotIn("RuntimeError", unexpected_updates[4]["value"])

        change_events = [dependency for dependency in config["dependencies"]
                         if dependency["targets"] and dependency["targets"][0][1] == "change"]
        self.assertEqual(len(change_events), 1)
        self.assertFalse(change_events[0]["queue"])
        demo.close(verbose=False)

    def test_main_loads_once_and_uses_private_styled_launch(self):
        model = object()
        demo = Mock()
        theme = object()
        with patch("src.gradio_app._import_gradio"), \
             patch("src.gradio_app.load_model", return_value=model) as load, \
             patch("src.gradio_app.build_app", return_value=demo) as build, \
             patch("src.gradio_app.borealis_theme", return_value=theme):
            main(["--weights", "custom.pt", "--device", "cpu",
                  "--host", "0.0.0.0", "--port", "9000"])
        load.assert_called_once_with(Path("custom.pt"))
        build.assert_called_once_with(model, device="cpu")
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
