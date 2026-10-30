"""YOLO result adapter and comparable bar scoring without model dependencies."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.baseline.yolo_bars import check_model_classes, detection_frame, evaluate, main, validate
from src.contract_io import load_detection_frame
from src.contracts import CLASS_NAMES, Box, Detection, DetectionFrame


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def fake_result(path, boxes, classes, confidences):
    return SimpleNamespace(
        path=str(path), orig_shape=(100, 100),
        boxes=SimpleNamespace(xyxy=FakeTensor(boxes), cls=FakeTensor(classes),
                              conf=FakeTensor(confidences)),
    )


class FakeModel:
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
        check_model_classes(FakeModel([]))
        with self.assertRaisesRegex(ValueError, "class names/order"):
            check_model_classes(SimpleNamespace(names={0: "vertical_bar"}))

    def test_result_to_detection_contract(self):
        result = fake_result("test/1.png", [[10, 20, 30, 80], [-0.1, 3, 101, 9]],
                             [1, 4], [0.9, 0.7])
        frame = detection_frame(result, "test/1.png")
        self.assertEqual((frame.width, frame.height), (100, 100))
        self.assertEqual([item.id for item in frame.detections], ["det-00001", "det-00002"])
        self.assertEqual([item.role for item in frame.detections], ["vertical_bar", "chart_title"])
        self.assertEqual((frame.detections[1].box.x1, frame.detections[1].box.x2), (0, 100))
        with self.assertRaisesRegex(ValueError, "Invalid model class ID"):
            detection_frame(fake_result("test/1.png", [[0, 0, 1, 1]], [15], [0.5]), "test/1.png")

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
            result = fake_result(image, [[10, 20, 30, 80], [60, 10, 80, 20]], [1, 4], [0.9, 0.8])
            with patch("src.baseline.yolo_bars.read_image", return_value=None), \
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(FakeModel([result]), Path("best.pt"), root, "test", root / "output")
            self.assertEqual((report["true_positive"], report["false_positive"],
                              report["false_negative"]), (1, 0, 0))
            self.assertEqual(report["exact_bar_count_accuracy"], 1)
            self.assertEqual(load_detection_frame(root / "output/1_detections.json").source_key,
                             "test/1.png")
            self.assertEqual(json.loads((root / "output/metrics.json").read_text())["images"], 1)

    def test_validation_records_standard_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "images/test/1.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"")
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            (root / "data.yaml").write_text("test: images/test\n", encoding="utf-8")
            model = FakeModel([])
            report = validate(model, Path("best.pt"), root, "test", root / "map")
            self.assertEqual(report["map50"], 0.8)
            self.assertEqual(report["vertical_bar_map50_95"], 0.7)
            self.assertEqual(model.validation_args["split"], "test")
            self.assertEqual(json.loads((root / "map/metrics.json").read_text())["images"], 1)

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
                    return iter(fake_result(path, [[10, 20, 30, 80]], [1], [0.9])
                                for path in kwargs["source"])

            model = TrackingModel()
            with patch("src.baseline.yolo_bars.read_image", return_value=None), \
                 patch("src.baseline.yolo_bars._write_overlay"):
                report = evaluate(model, Path("best.pt"), root, "test", root / "output", batch=1)
            self.assertEqual(report["images"], 2)
            self.assertEqual([len(call["source"]) for call in model.calls], [1, 1])
            self.assertEqual([call["iou"] for call in model.calls], [0.70, 0.70])

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
            with patch("sys.argv", argv), \
                 patch("src.baseline.yolo_bars.load_model", return_value=model), \
                 patch("src.baseline.yolo_bars.run_detection", return_value=frame) as run, \
                 patch("src.baseline.yolo_bars.read_image", return_value=None), \
                 patch("src.baseline.yolo_bars._write_overlay"), redirect_stdout(io.StringIO()) as stdout:
                main()
            run.assert_called_once_with(model, image, confidence=0.4,
                                        nms_iou=0.6, device=None)
            detection_path = output / "chart final_detections.json"
            self.assertTrue(detection_path.is_file())
            data = json.loads(detection_path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"schema_version", "source_key", "width", "height",
                                         "detections"})
            self.assertEqual(data["source_key"], "test/chart.png")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(set(summary), {"source_key", "detections", "vertical_bars", "output_dir"})

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
                with self.subTest(extra=extra), \
                     patch("sys.argv", ["yolo_bars", "predict", "--image", str(image),
                                        "--output-dir", str(Path(temporary) / "out"), *extra]), \
                     patch("src.baseline.yolo_bars.load_model") as load, \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        main()
                    load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
