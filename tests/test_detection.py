"""Detection orchestration, OpenCV baseline, and YOLO evaluation."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.baseline.opencv_bars import (
    Box as OpenCVBox,
    detect_vertical_bars,
    load_yolo_boxes,
    match_boxes,
)
from src.baseline.opencv_charts import predict_chart
from src.baseline.yolo_bars import check_model_classes, detection_frame, evaluate, main, validate
from src.contract_io import load_detection_frame
from src.contracts import Box, CLASS_NAMES, Detection, DetectionFrame
from src.detection import run_detection
from src.diagnostics import AnalysisDiagnostics


# Detection

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
            for size in (True, False, "640", 640.0, 640.5, 0, -1, None, math.nan, math.inf, 1281, 100000):
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


# Opencv baseline

class BaselineScoringTest(unittest.TestCase):
    def test_one_to_one_matching(self):
        actual = [OpenCVBox(10, 10, 30, 80), OpenCVBox(50, 10, 70, 80)]
        predicted = [OpenCVBox(10, 10, 30, 80), OpenCVBox(11, 11, 31, 81)]
        self.assertEqual(match_boxes(predicted, actual), 1)

    def test_yolo_decoder_filters_other_chart_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.txt"
            path.write_text("1 0.2 0.5 0.1 0.4\n4 0.5 0.1 0.5 0.1\n", encoding="utf-8")
            boxes = load_yolo_boxes(path, 1, (100, 100), 15)
            self.assertEqual(len(boxes), 1)
            self.assertAlmostEqual(boxes[0].x1, 15)
            self.assertAlmostEqual(boxes[0].y2, 70)

    def test_missing_label_is_error(self):
        with self.assertRaises(FileNotFoundError):
            load_yolo_boxes(Path("/missing/0.txt"), 1, (100, 100), 15)

    def test_detector_uses_image_pixels(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are not installed")
        image = np.full((400, 500, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 30), (50, 350), (0, 0, 0), 2)
        cv2.line(image, (50, 350), (450, 350), (0, 0, 0), 2)
        cv2.rectangle(image, (100, 150), (124, 349), (255, 0, 0), -1)
        cv2.rectangle(image, (200, 100), (224, 349), (0, 0, 0), -1)
        boxes = detect_vertical_bars(image)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(any(abs(box.x1 - 100) <= 2 for box in boxes))
        self.assertTrue(any(abs(box.x1 - 200) <= 2 for box in boxes))

    def test_four_type_router_uses_image_geometry(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are not installed")
        for kind in ("hbar_categorical", "vbar_categorical", "dot_line", "line"):
            with self.subTest(kind=kind):
                image = np.full((400, 500, 3), 255, dtype=np.uint8)
                cv2.line(image, (50, 30), (50, 350), (0, 0, 0), 2)
                cv2.line(image, (50, 350), (450, 350), (0, 0, 0), 2)
                if kind == "hbar_categorical":
                    cv2.rectangle(image, (51, 80), (250, 103), (255, 0, 0), -1)
                    cv2.rectangle(image, (51, 170), (350, 193), (0, 0, 0), -1)
                elif kind == "vbar_categorical":
                    cv2.rectangle(image, (100, 150), (124, 349), (255, 0, 0), -1)
                    cv2.rectangle(image, (200, 100), (224, 349), (0, 0, 0), -1)
                elif kind == "dot_line":
                    for x, y in ((100, 100), (180, 130), (260, 90), (340, 150)):
                        cv2.circle(image, (x, y), 7, (0, 100, 230), -1)
                else:
                    cv2.line(image, (90, 100), (180, 120), (255, 0, 0), 2)
                    cv2.line(image, (180, 120), (270, 90), (255, 0, 0), 2)
                    cv2.line(image, (270, 90), (360, 150), (255, 0, 0), 2)
                predicted_type, boxes = predict_chart(image)
                self.assertEqual(predicted_type, kind)
                self.assertGreaterEqual(len(boxes), 2)


# Yolo baseline

def yolo_fake_result(path, boxes, classes, confidences):
    return SimpleNamespace(
        path=str(path), orig_shape=(100, 100),
        boxes=SimpleNamespace(xyxy=FakeTensor(boxes), cls=FakeTensor(classes),
                              conf=FakeTensor(confidences)),
    )


class YoloFakeModel:
    names = dict(enumerate(CLASS_NAMES))

    def __init__(self, results):
        self.results = results

    def predict(self, **kwargs):
        return iter(self.results)

    def val(self, **kwargs):
        self.validation_args = kwargs
        return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.6,
                                                    maps=[0.4, 0.7] + [0.5] * 13))


class YoloBaselineTest(unittest.TestCase):
    def test_model_class_order_must_match_contract(self):
        check_model_classes(YoloFakeModel([]))
        with self.assertRaisesRegex(ValueError, "class names/order"):
            check_model_classes(SimpleNamespace(names={0: "vertical_bar"}))

    def test_result_to_detection_contract(self):
        result = yolo_fake_result("test/1.png", [[10, 20, 30, 80], [-0.1, 3, 101, 9]],
                             [1, 4], [0.9, 0.7])
        frame = detection_frame(result, "test/1.png")
        self.assertEqual((frame.width, frame.height), (100, 100))
        self.assertEqual([item.id for item in frame.detections], ["det-00001", "det-00002"])
        self.assertEqual([item.role for item in frame.detections], ["vertical_bar", "chart_title"])
        self.assertEqual((frame.detections[1].box.x1, frame.detections[1].box.x2), (0, 100))
        with self.assertRaisesRegex(ValueError, "Invalid model class ID"):
            detection_frame(yolo_fake_result("test/1.png", [[0, 0, 1, 1]], [15], [0.5]), "test/1.png")

    def test_evaluation_scores_bars_and_saves_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/test/1.png"
            label = root / "labels/test/1.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"png contents are unnecessary for mocked overlay")
            label.write_text("1 0.2 0.5 0.2 0.6\n", encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            result = yolo_fake_result(image, [[10, 20, 30, 80], [60, 10, 80, 20]], [1, 4], [0.9, 0.8])
            with patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(YoloFakeModel([result]), Path("best.pt"), root, "test", root / "output")
            self.assertEqual((report["true_positive"], report["false_positive"],
                              report["false_negative"]), (1, 0, 0))
            self.assertEqual(report["exact_bar_count_accuracy"], 1)
            self.assertEqual(report["imgsz"], 640)
            self.assertEqual(load_detection_frame(root / "output/1_detections.json").source_key,
                             "test/1.png")
            self.assertEqual(json.loads((root / "output/metrics.json").read_text())["images"], 1)

    def test_evaluation_uses_configurable_matching_iou(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/test/1.png"
            label = root / "labels/test/1.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"")
            label.write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            # The prediction has IoU 0.8 with the label: it passes 0.75 but fails 0.90.
            result = yolo_fake_result(image, [[34, 30, 74, 70]], [1], [0.9])
            with patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"):
                report_75 = evaluate(YoloFakeModel([result]), Path("best.pt"), root, "test",
                                     root / "iou75", match_iou=0.75)
                report_90 = evaluate(YoloFakeModel([result]), Path("best.pt"), root, "test",
                                     root / "iou90", match_iou=0.90)
            self.assertEqual(report_75["iou"], 0.75)
            self.assertEqual(report_75["true_positive"], 1)
            self.assertEqual(report_90["iou"], 0.90)
            self.assertEqual(report_90["true_positive"], 0)
            self.assertEqual(report_75["exact_bar_count_accuracy"], 1)
            self.assertEqual(report_90["exact_bar_count_accuracy"], 1)

    def test_evaluation_scores_multiple_matching_ious_in_one_inference_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/test/1.png"
            label = root / "labels/test/1.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"")
            label.write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            model = YoloFakeModel([yolo_fake_result(image, [[34, 30, 74, 70]], [1], [0.9])])
            with patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(model, Path("best.pt"), root, "test", root / "strict",
                                  match_iou=(0.75, 0.90))
            results = report["matching_results"]
            self.assertEqual([item["iou"] for item in results], [0.75, 0.90])
            self.assertEqual([item["true_positive"] for item in results], [1, 0])
            self.assertEqual([item["imgsz"] for item in results], [640, 640])
            self.assertEqual(len(model.results), 1)

    def test_validation_records_standard_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/test/1.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            (root / "data.yaml").write_text("test: images/test\n", encoding="utf-8")
            model = YoloFakeModel([])
            report = validate(model, Path("best.pt"), root, "test", root / "map", imgsz=1280)
            self.assertEqual(report["map50"], 0.8)
            self.assertEqual(report["vertical_bar_map50_95"], 0.7)
            self.assertEqual(model.validation_args["split"], "test")
            self.assertEqual(model.validation_args["imgsz"], 1280)
            self.assertEqual(report["imgsz"], 1280)
            self.assertEqual(json.loads((root / "map/metrics.json").read_text())["imgsz"], 1280)
            self.assertEqual(json.loads((root / "map/metrics.json").read_text())["images"], 1)

    def test_evaluation_scores_horizontal_bars_on_hbar_charts_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for image_id in ("1", "2"):
                (root / "images/test").mkdir(parents=True, exist_ok=True)
                (root / "labels/test").mkdir(parents=True, exist_ok=True)
                (root / f"images/test/{image_id}.png").write_bytes(b"")
            # Image 1 is a horizontal-bar chart (class 0), image 2 a vertical one.
            (root / "labels/test/1.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            (root / "labels/test/2.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "chart_types_test.csv").write_text(
                "image_id,chart_type\n1,hbar_categorical\n2,vbar_categorical\n", encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            result = yolo_fake_result(root / "images/test/1.png", [[30, 40, 70, 60]], [0], [0.9])
            model = YoloFakeModel([result])
            with patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(model, Path("best.pt"), root, "test", root / "hbar",
                                  role="horizontal_bar")
            self.assertEqual(report["class"], "horizontal_bar")
            self.assertEqual(report["images"], 1)
            self.assertEqual((report["true_positive"], report["false_positive"],
                              report["false_negative"]), (1, 0, 0))

    def test_unknown_bar_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bar role"):
                evaluate(YoloFakeModel([]), Path("best.pt"), root, "test", root / "out", role="dot")

    def test_validation_reports_each_present_chart_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images/test").mkdir(parents=True)
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            (root / "data.yaml").write_text("test: images/test\n", encoding="utf-8")
            model = YoloFakeModel([])
            metrics = SimpleNamespace(box=SimpleNamespace(
                map50=0.8, map=0.6, maps=[0.65, 0.7] + [0.5] * 13, ap_class_index=[0, 1, 4]))
            model.val = lambda **kwargs: metrics
            report = validate(model, Path("best.pt"), root, "test", root / "map")
            self.assertEqual(report["per_class_map50_95"],
                             {"horizontal_bar": 0.65, "vertical_bar": 0.7, "chart_title": 0.5})
            self.assertEqual(report["horizontal_bar_map50_95"], 0.65)

    def test_evaluation_limits_inference_input_to_one_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for image_id in ("1", "2"):
                image = root / "images/test" / f"{image_id}.png"
                label = root / "labels/test" / f"{image_id}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"")
                label.write_text("1 0.2 0.5 0.2 0.6\n", encoding="utf-8")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

            class TrackingModel:
                calls = []

                def predict(self, **kwargs):
                    self.calls.append(kwargs)
                    return iter(yolo_fake_result(path, [[10, 20, 30, 80]], [1], [0.9])
                                for path in kwargs["source"])

            model = TrackingModel()
            with patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(model, Path("best.pt"), root, "test", root / "output",
                                  batch=1, imgsz=960)
            self.assertEqual(report["images"], 2)
            self.assertEqual([len(call["source"]) for call in model.calls], [1, 1])
            self.assertEqual([call["iou"] for call in model.calls], [0.70, 0.70])
            self.assertEqual([call["imgsz"] for call in model.calls], [960, 960])
            self.assertEqual(json.loads((root / "output/metrics.json").read_text())["imgsz"], 960)

    def test_prediction_cli_uses_shared_detection_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "chart final.png"
            output = root / "output"
            image.write_bytes(b"")
            frame = DetectionFrame("input/chart_final.png", 100, 100, (
                Detection("det-00001", "vertical_bar", Box(10, 20, 30, 80), 0.9),
            ))
            model = object()
            argv = [
                "yolo_bars", "predict", "--image", str(image), "--output-dir", str(output),
                "--source-key", "test/chart.png", "--confidence", "0.4", "--nms-iou", "0.6",
            ]
            with patch("sys.argv", argv),\
                 patch("src.baseline.yolo_bars.load_model", return_value=model),\
                 patch("src.baseline.yolo_bars.run_detection", return_value=frame) as run,\
                 patch("src.baseline.yolo_bars.read_image", return_value=None),\
                 patch("src.baseline.yolo_bars._write_overlay"), redirect_stdout(io.StringIO()) as stdout:
                main()
            run.assert_called_once_with(model, image, confidence=0.4,
                                        nms_iou=0.6, device=None, imgsz=640)
            detection_path = output / "chart final_detections.json"
            self.assertTrue(detection_path.is_file())
            data = json.loads(detection_path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"schema_version", "source_key", "width", "height",
                                         "detections"})
            self.assertEqual(data["source_key"], "test/chart.png")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(set(summary), {"source_key", "detections", "vertical_bars", "output_dir"})

    def test_cli_forwards_selected_size_for_all_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "chart.png"
            image.touch()
            frame = DetectionFrame("input/chart.png", 100, 100, ())
            for command in ("predict", "evaluate", "validate"):
                for size in (640, 960, 1280):
                    with self.subTest(command=command, size=size):
                        argv = ["yolo_bars", command, "--output-dir", str(root / "out"),
                                "--imgsz", str(size)]
                        if command == "predict":
                            argv += ["--image", str(image)]
                        with patch("sys.argv", argv),\
                             patch("src.baseline.yolo_bars.load_model"),\
                             patch("src.baseline.yolo_bars.run_detection", return_value=frame) as predict,\
                             patch("src.baseline.yolo_bars.evaluate", return_value={}) as evaluate_fn,\
                             patch("src.baseline.yolo_bars.validate", return_value={}) as validate_fn,\
                             patch("src.baseline.yolo_bars.read_image"),\
                             patch("src.baseline.yolo_bars._write_overlay"),\
                             redirect_stdout(io.StringIO()):
                            main()
                        called = {"predict": predict, "evaluate": evaluate_fn,
                                  "validate": validate_fn}[command]
                        self.assertEqual(called.call_count, 1)
                        self.assertEqual(called.call_args.kwargs["imgsz"], size)

    def test_all_cli_commands_reject_invalid_sizes_before_loading_model(self):
        for command in ("predict", "evaluate", "validate"):
            for size in ("0", "-1", "1281", "100000", "640.5", "no"):
                with self.subTest(command=command, size=size):
                    argv = ["yolo_bars", command, "--output-dir", "unused", "--imgsz", size]
                    if command == "predict":
                        argv += ["--image", "unused.png"]
                    with patch("sys.argv", argv),\
                         patch("src.baseline.yolo_bars.load_model") as load,\
                         redirect_stderr(io.StringIO()):
                        with self.assertRaisesRegex(SystemExit, "2"):
                            main()
                        load.assert_not_called()

    def test_scoring_apis_reject_invalid_sizes_before_model_or_dataset_access(self):
        for score in (evaluate, validate):
            for size in (0, -1, 1281, 100000, 640.5, True):
                with self.subTest(function=score.__name__, size=size):
                    model = Mock()
                    with self.assertRaisesRegex(ValueError, "imgsz"):
                        score(model, Path("best.pt"), Path("missing"), "test",
                              Path("unused"), imgsz=size)
                    model.predict.assert_not_called()
                    model.val.assert_not_called()

    def test_prediction_cli_rejects_invalid_input_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "chart.png"
            image.write_bytes(b"")
            cases = (
                ["--confidence", "nan"],
                ["--nms-iou", "1.1"],
                ["--source-key", "../chart.png"],
            )
            for extra in cases:
                with self.subTest(extra=extra),\
                     patch("sys.argv", ["yolo_bars", "predict", "--image", str(image),
                                        "--output-dir", str(Path(temporary) / "out"), *extra]),\
                     patch("src.baseline.yolo_bars.load_model") as load,\
                     redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        main()
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
