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
from src.diagnostics import AnalysisDiagnostics


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


class ObservedModel(FakeModel):
    """Exercise the actual callback/hook lifecycle without Torch or Ultralytics."""

    def __init__(self, results):
        super().__init__(results)
        self.callbacks = {"on_predict_start": [lambda predictor: None]}
        self.hooks = []
        self.fail = False
        self.warmed = False
        self.predictor = None

    def register_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def forward(self, shape):
        for hook in self.hooks:
            hook(self, (SimpleNamespace(shape=shape),))

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        size = math.ceil(kwargs["imgsz"] / 32) * 32
        if self.predictor is None:
            self.predictor = SimpleNamespace(
                callbacks=self.callbacks, model=SimpleNamespace(
                    register_forward_pre_hook=self.register_hook), done_warmup=False,
            )
        self.predictor.imgsz = [size, size]
        for callback in self.callbacks["on_predict_start"]:
            callback(self.predictor)
        if not self.warmed:
            self.forward((1, 3, size, size))
            self.warmed = True
            self.predictor.done_warmup = True
        if self.fail == "before_forward":
            raise RuntimeError("before forward")
        self.forward((1, 3, math.ceil(size / 64) * 32, size))
        if self.fail:
            raise RuntimeError("after forward")
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
                "device": "cpu", "verbose": False, "imgsz": 640,
            }])
            self.assertEqual(frame.source_key, "input/chart.png")
            self.assertEqual(frame.detections[0].role, "vertical_bar")

    def test_custom_sizes_and_invalid_sizes_before_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.touch()
            for size in (1, 641, 1280):
                model = FakeModel([fake_result(image)])
                run_detection(model, image, imgsz=size)
                self.assertEqual(model.calls[0]["imgsz"], size)
            for size in (True, False, "640", 640.0, 640.5, 0, -1, None, math.nan, math.inf):
                with self.subTest(size=size):
                    model = FakeModel([fake_result(image)])
                    with self.assertRaisesRegex(ValueError, "imgsz"):
                        run_detection(model, image, imgsz=size)
                    self.assertEqual(model.calls, [])

    def test_measures_rectangular_tensor_and_cleans_up_between_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.touch()
            model = ObservedModel([fake_result(image)])
            callbacks = list(model.callbacks["on_predict_start"])
            original_hooks = [lambda backend, inputs: None]
            model.hooks.extend(original_hooks)
            records = []
            for requested, effective, height in ((640, 640, 320), (641, 672, 352), (1280, 1280, 640)):
                diagnostics = AnalysisDiagnostics(settings={})
                run_detection(model, image, imgsz=requested, diagnostics=diagnostics)
                evidence = diagnostics.record["metadata"]["detection"]
                records.append(evidence)
                self.assertEqual(evidence["requested_imgsz"], requested)
                self.assertEqual(evidence["effective_imgsz"], [effective, effective])
                self.assertEqual(evidence["tensor_shape"], [1, 3, height, effective])
                self.assertEqual(evidence["unavailable_reasons"], {})
                self.assertEqual(diagnostics.record["settings"]["imgsz"], requested)
                self.assertIn("tensor_shape=", diagnostics._log_entries[-1])
                self.assertEqual(model.callbacks["on_predict_start"], callbacks)
                self.assertEqual(model.hooks, original_hooks)
            self.assertEqual(records[0]["tensor_shape"], [1, 3, 320, 640])

    def test_failure_keeps_observed_evidence_but_never_records_warmup_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.touch()
            for failure in ("before_forward", "after_forward"):
                with self.subTest(failure=failure):
                    model = ObservedModel([fake_result(image)])
                    model.fail = failure
                    diagnostics = AnalysisDiagnostics(settings={})
                    with self.assertRaises(RuntimeError):
                        run_detection(model, image, diagnostics=diagnostics)
                    evidence = diagnostics.record["metadata"]["detection"]
                    self.assertEqual(evidence["effective_imgsz"], [640, 640])
                    self.assertEqual(evidence["tensor_shape"],
                                     None if failure == "before_forward" else [1, 3, 320, 640])
                    self.assertEqual(len(model.callbacks["on_predict_start"]), 1)
                    self.assertEqual(model.hooks, [])
                    model.fail = False
                    recovered = AnalysisDiagnostics(settings={})
                    run_detection(model, image, imgsz=1280, diagnostics=recovered)
                    self.assertEqual(recovered.record["metadata"]["detection"]["tensor_shape"],
                                     [1, 3, 640, 1280])
                    self.assertEqual(model.hooks, [])

    def test_unavailable_backend_hook_keeps_effective_size_and_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.touch()
            model = ObservedModel([fake_result(image)])

            def unsupported(hook):
                raise RuntimeError("hooks unsupported")

            model.register_hook = unsupported
            diagnostics = AnalysisDiagnostics(settings={})
            run_detection(model, image, diagnostics=diagnostics)
            evidence = diagnostics.record["metadata"]["detection"]
            self.assertEqual(evidence["effective_imgsz"], [640, 640])
            self.assertIsNone(evidence["tensor_shape"])
            self.assertIn("hooks unsupported", evidence["unavailable_reasons"]["tensor_shape"])
            self.assertEqual(len(model.callbacks["on_predict_start"]), 1)

    def test_custom_model_without_observation_support_still_detects(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.touch()
            diagnostics = AnalysisDiagnostics(settings={})
            run_detection(FakeModel([fake_result(image)]), image, diagnostics=diagnostics)
            evidence = diagnostics.record["metadata"]["detection"]
            self.assertIsNone(evidence["effective_imgsz"])
            self.assertIsNone(evidence["tensor_shape"])
            self.assertIn("callbacks", evidence["unavailable_reasons"]["tensor_shape"])

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
