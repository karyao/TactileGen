"""Dataset selection and scoring for the four bar-chart OCR benchmark groups."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.baseline.ocr import evaluate, load_truth_images, parse_args
from tests.test_ocr import FakeModel, RoleBackend, annotation, fake_result
from scripts.eval_ocr_bars import main as run_four_groups


class BarDatasetOCRTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def dataset(self, source):
        root = self.root / source
        (root / "images/test").mkdir(parents=True)
        if source == "chartqa":
            annotations = root / "annotations"
            annotations.mkdir()
            ids, types = ("two_col_995", "00339007006077"), ("v_bar", "h_bar")
        else:
            annotations = root / "annotations.json"
            ids, types = ("1", "2"), ("vbar_categorical", "hbar_categorical")
        records = []
        for image_id, kind in zip(ids, types, strict=True):
            record = annotation(image_id)
            record["type"] = kind
            cv2.imwrite(str(root / f"images/test/{image_id}.png"),
                        np.full((100, 100, 3), 255, dtype=np.uint8))
            if source == "chartqa":
                record.pop("image_index")
                (annotations / f"{image_id}.json").write_text(json.dumps(record))
            records.append(record)
        if source == "plotqa":
            annotations.write_text(json.dumps(records))
        with (root / "chart_types_test.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "chart_type"))
            writer.writerows(zip(ids, types, strict=True))
        return root, annotations, ids, types

    def test_all_four_groups_score_and_preserve_image_identity(self):
        for source in ("plotqa", "chartqa"):
            root, annotations, ids, types = self.dataset(source)
            for image_id, kind in zip(ids, types, strict=True):
                with self.subTest(source=source, kind=kind):
                    image = root / f"images/test/{image_id}.png"
                    model = FakeModel([fake_result(image)])
                    output = root / kind
                    report = evaluate(
                        model, RoleBackend, annotations=annotations, data_root=root,
                        dataset_format=source, chart_types=[kind], output_dir=output,
                        engine="fake", imgsz=960, workers=2,
                    )
                    self.assertEqual(report["dataset"]["images"], 1)
                    self.assertEqual(report["dataset"]["unique_regions"], 6)
                    self.assertEqual(report["dataset_format"], source)
                    self.assertEqual(report["oracle"]["overall"]["normalized_exact_accuracy"], 1)
                    self.assertEqual(report["end_to_end"]["overall"]["normalized_exact_accuracy"], 4 / 6)
                    self.assertEqual(model.calls[0]["source"], [str(image)])
                    self.assertEqual(model.calls[0]["imgsz"], 960)
                    with (output / "regions.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual({row["source_key"] for row in rows}, {f"test/{image_id}.png"})

    def test_filters_before_limit_and_supports_both_orientations(self):
        root, annotations, ids, _ = self.dataset("chartqa")
        images, _ = load_truth_images(
            annotations, root, "test", 1, dataset_format="chartqa", chart_types=["v_bar"])
        self.assertEqual([image.image_id for image in images], [ids[0]])
        images, report = load_truth_images(
            annotations, root, "test", dataset_format="chartqa",
            chart_types=["vbar_categorical", "hbar_categorical"])
        self.assertEqual({image.image_id for image in images}, set(ids))
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["raw_regions"], 14)

    def test_chartqa_optional_sections_and_list_axis_titles(self):
        root, annotations, ids, _ = self.dataset("chartqa")
        path = annotations / f"{ids[0]}.json"
        record = json.loads(path.read_text())
        info = record["general_figure_info"]
        del info["title"]
        del info["legend"]
        del info["y_axis"]["label"]
        info["x_axis"]["label"] = {
            "text": ["Year", "of sales"],
            "bbox": [{"x": 40, "y": 85, "w": 20, "h": 5},
                     {"x": 40, "y": 90, "w": 20, "h": 5}],
        }
        path.write_text(json.dumps(record))
        images, report = load_truth_images(
            annotations, root, "test", dataset_format="chartqa", chart_types=["v_bar"])
        self.assertEqual(report["unique_regions"], 4)
        self.assertEqual(list(images[0].text_by_id.values()), ["Year", "of sales", "A", "0"])

    def test_missing_chartqa_annotation_is_not_silently_skipped(self):
        root, annotations, ids, _ = self.dataset("chartqa")
        (annotations / f"{ids[0]}.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing ChartQA annotation"):
            load_truth_images(annotations, root, "test", dataset_format="chartqa")

    def test_manifest_annotation_orientation_mismatch_is_rejected(self):
        root, annotations, ids, _ = self.dataset("chartqa")
        path = annotations / f"{ids[0]}.json"
        record = json.loads(path.read_text())
        record["type"] = "h_bar"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "chart types differ"):
            load_truth_images(annotations, root, "test", dataset_format="chartqa",
                              chart_types=["v_bar", "h_bar"])

    def test_chartqa_missing_gold_text_is_not_stringified(self):
        root, annotations, ids, _ = self.dataset("chartqa")
        path = annotations / f"{ids[0]}.json"
        record = json.loads(path.read_text())
        record["general_figure_info"]["title"]["text"] = None
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Missing authoritative ChartQA text"):
            load_truth_images(annotations, root, "test", dataset_format="chartqa")

    def test_cli_keeps_legacy_defaults_and_accepts_new_dataset_options(self):
        default = parse_args(["evaluate", "--output-dir", "out"])
        self.assertEqual(default.dataset_format, "plotqa")
        self.assertEqual(default.chart_types, ["vbar_categorical"])
        args = parse_args(["evaluate", "--dataset-format", "chartqa", "--chart-types",
                           "v_bar", "h_bar", "--imgsz", "960", "--output-dir", "out"])
        self.assertEqual(args.chart_types, ["v_bar", "h_bar"])
        self.assertEqual(args.imgsz, 960)

    def test_four_group_runner_writes_summary_and_each_report(self):
        plotqa, plotqa_annotations, _, _ = self.dataset("plotqa")
        chartqa, chartqa_annotations, _, _ = self.dataset("chartqa")
        output = self.root / "results"

        class Model(FakeModel):
            def predict(self, **kwargs):
                self.calls.append(kwargs)
                return iter(fake_result(Path(path)) for path in kwargs["source"])

        model = Model([])
        argv = ["eval_ocr_bars", "--weights", "fake.pt", "--plotqa-root", str(plotqa),
                "--plotqa-annotations", str(plotqa_annotations), "--chartqa-root", str(chartqa),
                "--chartqa-annotations", str(chartqa_annotations), "--output-dir", str(output),
                "--limit", "1", "--workers", "1"]
        with patch("sys.argv", argv), patch("scripts.eval_ocr_bars.load_model", return_value=model), \
                patch.dict("scripts.eval_ocr_bars.ENGINE_FACTORIES", {"rapidocr": RoleBackend}):
            run_four_groups()
        with (output / "summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["group"] for row in rows], [
            "plotqa_vertical", "plotqa_horizontal", "chartqa_vertical", "chartqa_horizontal"])
        self.assertEqual(len(model.calls), 4)
        for row in rows:
            self.assertEqual(row["images"], "1")
            self.assertAlmostEqual(float(row["normalized_exact_accuracy"]), 4 / 6)
            report = json.loads((output / row["group"] / "metrics.json").read_text())
            self.assertEqual(report["limit"], 1)


if __name__ == "__main__":
    unittest.main()
