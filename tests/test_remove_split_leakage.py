"""Checks for cross-split removal without changing the source dataset."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.dataset.remove_split_leakage import clean_dataset


class RemoveSplitLeakageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "clean"
        (self.source / "classes.txt").parent.mkdir()
        (self.source / "classes.txt").write_text("vertical_bar\n")
        for split in ("train", "val", "test"):
            (self.source / "images" / split).mkdir(parents=True)
            (self.source / "labels" / split).mkdir(parents=True)

    def add_image(self, split, image_id, content):
        (self.source / "images" / split / f"{image_id}.png").write_bytes(content)
        (self.source / "labels" / split / f"{image_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    def add_manifest(self, split, ids):
        with (self.source / f"chart_types_{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "chart_type"))
            writer.writerows((image_id, "vbar_categorical") for image_id in ids)

    def test_preserves_test_then_val_then_train(self):
        for split, image_id, content in (
            ("test", 1, b"shared all"),
            ("test", 2, b"shared val"),
            ("val", 3, b"shared all"),
            ("val", 4, b"shared val"),
            ("val", 5, b"shared train"),
            ("train", 6, b"shared all"),
            ("train", 7, b"shared train"),
            ("train", 8, b"train only"),
            ("train", 9, b"train only"),
        ):
            self.add_image(split, image_id, content)
        self.add_manifest("test", (1, 2))
        self.add_manifest("val", (3, 4, 5))

        report = clean_dataset(self.source, self.output)

        self.assertEqual(
            report["counts"],
            {
                "test": {"kept": 2, "removed": 0},
                "val": {"kept": 1, "removed": 2},
                "train": {"kept": 2, "removed": 2},
            },
        )
        self.assertEqual(
            {
                split: [(row["image_id"], row["matches_split"]) for row in rows]
                for split, rows in report["removed"].items()
            },
            {
                "test": [],
                "val": [("3", "test"), ("4", "test")],
                "train": [("6", "test"), ("7", "val")],
            },
        )
        for split, expected in (
            ("test", {"1", "2"}),
            ("val", {"5"}),
            ("train", {"8", "9"}),
        ):
            for kind in ("images", "labels"):
                actual = {path.stem for path in (self.output / kind / split).iterdir()}
                self.assertEqual(actual, expected)
        self.assertEqual(
            (self.output / "chart_types_val.csv").read_text().splitlines(),
            ["image_id,chart_type", "5,vbar_categorical"],
        )
        self.assertEqual(len(list((self.source / "images" / "train").iterdir())), 4)
        self.assertTrue((self.output / "leakage_report.json").exists())
        self.assertEqual(
            (self.output / "data.yaml").read_text().splitlines()[1:],
            [
                f"path: {self.output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: vertical_bar",
            ],
        )

    def test_rejects_missing_label_before_creating_output(self):
        (self.source / "images" / "test" / "1.png").write_bytes(b"image")
        with self.assertRaisesRegex(ValueError, "Image/label IDs differ"):
            clean_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
