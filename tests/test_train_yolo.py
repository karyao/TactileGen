"""Subset selection and run files for the training script; no training runs."""

import json
import tempfile
import unittest
from pathlib import Path

from src.baseline.train_yolo import main, prepare_run, select_subset
from src.contracts import CLASS_NAMES


def make_dataset(root: Path, per_split: dict[str, dict[str, int]]) -> None:
    root.mkdir(parents=True)
    (root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    image_id = 0
    for split, counts in per_split.items():
        (root / "images" / split).mkdir(parents=True)
        rows = ["image_id,chart_type"]
        for chart_type, count in counts.items():
            for _ in range(count):
                (root / "images" / split / f"{image_id}.png").write_bytes(b"")
                rows.append(f"{image_id},{chart_type}")
                image_id += 1
        (root / f"chart_types_{split}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


class SubsetTest(unittest.TestCase):
    types = {str(i): ("vbar_categorical" if i % 3 else "hbar_categorical") for i in range(30)}

    def test_balanced_and_reproducible(self):
        a = select_subset(self.types, ("vbar_categorical", "hbar_categorical"), 5, seed=0)
        b = select_subset(self.types, ("vbar_categorical", "hbar_categorical"), 5, seed=0)
        self.assertEqual(a, b)
        kinds = [self.types[i] for i in a]
        self.assertEqual(kinds.count("vbar_categorical"), 5)
        self.assertEqual(kinds.count("hbar_categorical"), 5)

    def test_seed_changes_the_subset(self):
        a = select_subset(self.types, ("vbar_categorical",), 5, seed=0)
        b = select_subset(self.types, ("vbar_categorical",), 5, seed=1)
        self.assertNotEqual(a, b)

    def test_none_keeps_everything_and_small_types_are_kept_whole(self):
        everything = select_subset(self.types, ("hbar_categorical",), None, seed=0)
        self.assertEqual(len(everything), 10)
        capped = select_subset(self.types, ("hbar_categorical",), 50, seed=0)
        self.assertEqual(len(capped), 10)

    def test_other_chart_types_are_never_chosen(self):
        types = dict(self.types, **{"99": "line"})
        chosen = select_subset(types, ("vbar_categorical", "hbar_categorical"), None, seed=0)
        self.assertNotIn("99", chosen)

    def test_missing_type_and_bad_count_are_errors(self):
        with self.assertRaises(ValueError):
            select_subset(self.types, ("dot_line",), 5, seed=0)
        with self.assertRaises(ValueError):
            select_subset(self.types, ("vbar_categorical",), 0, seed=0)


class RunFilesTest(unittest.TestCase):
    def test_prepare_writes_lists_yaml_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            make_dataset(root, {
                "train": {"vbar_categorical": 6, "hbar_categorical": 6, "line": 3},
                "val": {"vbar_categorical": 4, "hbar_categorical": 4},
            })
            out = Path(temporary) / "run"
            run = prepare_run(root, out, ("vbar_categorical", "hbar_categorical"), 3, 2, seed=0)
            self.assertEqual(run["counts"]["train"], {"vbar_categorical": 3, "hbar_categorical": 3})
            self.assertEqual(run["counts"]["val"], {"vbar_categorical": 2, "hbar_categorical": 2})
            train_lines = (out / "train_images.txt").read_text().splitlines()
            self.assertEqual(len(train_lines), 6)
            self.assertTrue(all("/images/train/" in line for line in train_lines))
            yaml = (out / "data.yaml").read_text()
            self.assertIn("nc: 15", yaml)
            self.assertIn("1: vertical_bar", yaml)
            self.assertIn("train_images.txt", yaml)

    def test_prepare_only_cli_records_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            make_dataset(root, {"train": {"vbar_categorical": 3, "hbar_categorical": 3},
                                "val": {"vbar_categorical": 2, "hbar_categorical": 2}})
            weights = Path(temporary) / "best.pt"
            weights.write_bytes(b"weights")
            out = Path(temporary) / "run"
            main(["--data-root", str(root), "--output", str(out), "--weights", str(weights),
                  "--per-type", "2", "--prepare-only"])
            config = json.loads((out / "run_config.json").read_text())
            self.assertEqual(config["per_type"], 2)
            self.assertEqual(config["mosaic"], 0.0)
            self.assertEqual(len(config["weights_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
