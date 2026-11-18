"""Combined dataset must feed cleanup and training without filename collisions."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.contracts import CLASS_NAMES
from src.dataset.combine import combine
from src.dataset.remove_split_leakage import clean_dataset
from src.baseline.train_yolo import prepare_run


class CombineDatasetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plotqa, self.chartqa = self.root / "plotqa", self.root / "chartqa"
        self.output = self.root / "combined"
        for dataset, root in (("plotqa", self.plotqa), ("chartqa", self.chartqa)):
            root.mkdir()
            (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                ids = ("0", "1") if dataset == "plotqa" else ("0", "chart-a")
                types = ("vbar_categorical", "hbar_categorical") if dataset == "plotqa" else ("v_bar", "h_bar")
                with (root / f"chart_types_{split}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("image_id", "chart_type"))
                    for image_id, chart_type in zip(ids, types):
                        (root / "images" / split / f"{image_id}.png").write_bytes(f"{dataset}-{split}-{image_id}".encode())
                        (root / "labels" / split / f"{image_id}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
                        writer.writerow((image_id, chart_type))

    def test_joint_cleanup_and_training_preparation(self):
        # ChartQA training duplicates a PlotQA held-out test image.
        test_bytes = (self.plotqa / "images/test/0.png").read_bytes()
        (self.chartqa / "images/train/0.png").write_bytes(test_bytes)
        counts = combine(self.plotqa, self.chartqa, self.output)
        self.assertEqual(counts["train"], {"plotqa": 2, "chartqa": 2})
        index = json.loads((self.output / "source_index.json").read_text())
        self.assertEqual(len(index), 12)
        self.assertEqual(len(list((self.output / "images/train").glob("*.png"))), 4)
        clean = self.root / "clean"
        report = clean_dataset(self.output, clean)
        self.assertEqual(report["counts"]["train"], {"kept": 3, "removed": 1})
        removed = report["removed"]["train"][0]
        self.assertEqual(index[removed["image_id"]]["dataset"], "chartqa")
        self.assertEqual(removed["matches_split"], "test")
        run = prepare_run(clean, self.root / "run", ("vbar_categorical", "hbar_categorical"), None, None, 0)
        self.assertEqual(run["counts"]["train"], {"vbar_categorical": 1, "hbar_categorical": 2})
        self.assertEqual(run["counts"]["val"], {"vbar_categorical": 2, "hbar_categorical": 2})
        for split in ("train", "val"):
            lines = (self.root / "run" / f"{split}_images.txt").read_text().splitlines()
            self.assertTrue(all(f"/images/{split}/" in line for line in lines))

    def test_validation_preserves_inputs_and_publishes_nothing(self):
        (self.chartqa / "labels/train/chart-a.txt").unlink()
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            combine(self.plotqa, self.chartqa, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue((self.plotqa / "images/train/0.png").is_file())

    def test_rejects_mismatched_classes_and_existing_output(self):
        (self.chartqa / "classes.txt").write_text("wrong\n")
        with self.assertRaisesRegex(ValueError, "class IDs"):
            combine(self.plotqa, self.chartqa, self.output)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            combine(self.plotqa, self.chartqa, self.output)

    def test_failed_link_leaves_no_partial_output(self):
        with patch.object(Path, "hardlink_to", side_effect=OSError("different device")):
            with self.assertRaisesRegex(OSError, "same filesystem"):
                combine(self.plotqa, self.chartqa, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".combined-*")), [])


if __name__ == "__main__":
    unittest.main()
