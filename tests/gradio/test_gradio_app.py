"""Command-line startup and launch settings."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.borealis import BOREALIS_CSS
from src.gradio_app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_WEIGHTS,
    MAX_UPLOAD_BYTES,
    main,
    parse_args,
)

from tests.gradio.helpers import GradioTestCase


class GradioCliTest(GradioTestCase):
    def test_argument_defaults_and_port_validation(self):
        args = parse_args([])
        self.assertEqual(args.weights, DEFAULT_WEIGHTS)
        self.assertEqual(args.host, DEFAULT_HOST)
        self.assertEqual(args.port, DEFAULT_PORT)
        self.assertIsNone(args.device)
        self.assertEqual(args.ocr, "tesseract")
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(parse_args(["--imgsz", "641"]).imgsz, 641)
        self.assertEqual(parse_args(["--imgsz", "1280"]).imgsz, 1280)
        for value in ("0", "-1", "640.5", "no", "True", "1281", "100000"):
            with self.subTest(imgsz=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--imgsz", value])
        self.assertEqual(parse_args(["--ocr", "rapidocr"]).ocr, "rapidocr")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--port", "0"])
            with self.assertRaises(SystemExit):
                parse_args(["--port", "not-a-port"])

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
                  "--host", "0.0.0.0", "--port", "9000", "--imgsz", "960"])
        load.assert_called_once_with(Path("custom.pt"))
        load_ocr.assert_called_once_with()
        build.assert_called_once_with(model, ocr_backend=backend, device="cpu", imgsz=960)
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
