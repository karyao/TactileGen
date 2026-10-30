"""Reusable one-image detection without frontend or model dependencies."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.contracts import CLASS_NAMES
from src.detection import run_detection


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def fake_result(path, boxes=None, classes=None, confidences=None):
    return SimpleNamespace(
        path=str(path),
        orig_shape=(100, 200),
        boxes=SimpleNamespace(
            xyxy=FakeTensor(boxes or []),
            cls=FakeTensor(classes or []),
            conf=FakeTensor(confidences or []),
        ),
    )


class FakeModel:
    names = dict(enumerate(CLASS_NAMES))

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class DetectionTest(unittest.TestCase):
    def test_module_import_does_not_require_model_or_frontend_packages(self):
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'cv2', 'gradio', 'ultralytics'}:
        raise ImportError(f'blocked dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import src.detection
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_forwards_explicit_inference_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.write_bytes(b"not decoded by the fake model")
            model = FakeModel([fake_result(image, [[10, 20, 40, 80]], [1], [0.91])])

            frame = run_detection(model, image, confidence=0.4, nms_iou=0.6, device="cpu")

            self.assertEqual(model.calls, [{
                "source": str(image), "conf": 0.4, "iou": 0.6,
                "device": "cpu", "verbose": False,
            }])
            self.assertEqual(frame.source_key, "input/chart.png")
            self.assertEqual(frame.detections[0].role, "vertical_bar")

    def test_thresholds_are_validated_independently_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.write_bytes(b"")
            invalid = (True, "0.5", -0.1, 1.1, math.nan, math.inf)
            for name in ("confidence", "nms_iou"):
                for value in invalid:
                    with self.subTest(name=name, value=value):
                        model = FakeModel([fake_result(image)])
                        settings = {"confidence": 0.25, "nms_iou": 0.70, name: value}
                        with self.assertRaisesRegex(ValueError, name):
                            run_detection(model, image, **settings)
                        self.assertEqual(model.calls, [])
            for boundary in (0.0, 1.0):
                model = FakeModel([fake_result(image)])
                run_detection(model, image, confidence=boundary, nms_iou=boundary)

    def test_sanitizes_only_the_source_filename(self):
        cases = {
            "chart final!.PNG": "input/chart_final.PNG",
            "图表.png": "input/upload.png",
            "...---___.jpg": "input/upload.jpg",
            ".hidden": "input/hidden",
        }
        with tempfile.TemporaryDirectory() as temporary:
            nested = Path(temporary) / "private" / "gradio" / "temp"
            nested.mkdir(parents=True)
            for filename, expected in cases.items():
                with self.subTest(filename=filename):
                    image = nested / filename
                    image.write_bytes(b"")
                    frame = run_detection(FakeModel([fake_result(image)]), image)
                    self.assertEqual(frame.source_key, expected)
                    self.assertNotIn(str(nested), frame.source_key)

    def test_rejects_missing_wrong_or_ambiguous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "chart.png"
            other = root / "other.png"
            image.write_bytes(b"")
            other.write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "Missing image"):
                run_detection(FakeModel([]), root / "missing.png")
            with self.assertRaisesRegex(ValueError, "Expected one"):
                run_detection(FakeModel([]), image)
            with self.assertRaisesRegex(ValueError, "Expected one"):
                run_detection(FakeModel([fake_result(image), fake_result(image)]), image)
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_detection(FakeModel([fake_result(other)]), image)

    def test_rejects_incompatible_model_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.write_bytes(b"")
            model = FakeModel([fake_result(image)])
            model.names = {0: "vertical_bar"}
            with self.assertRaisesRegex(ValueError, "class names/order"):
                run_detection(model, image)
            self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
