"""Dataset preparation, split leakage, and training setup."""

import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.baseline.train_yolo import main, prepare_run, select_subset
from src.contracts import CLASS_NAMES
from src.dataset.prepare_plotqa import CLASS_IDS, convert
from src.dataset.remove_split_leakage import clean_dataset


# Prepare plotqa

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "dataset" / "prepare_plotqa.py"
CHART_TYPES = ("vbar_categorical", "hbar_categorical", "dot_line", "line")


def box():
    return {"x": 1, "y": 1, "w": 2, "h": 2}


def annotation(index, chart_type):
    axis = {
        "label": {"bbox": box()},
        "rule": {"bbox": box()},
        "major_labels": {"bboxes": [box()]},
    }
    return {
        "image_index": index,
        "type": chart_type,
        "models": [{"bboxes": [box()]}],
        "general_figure_info": {
            "title": {"bbox": box()},
            "plot_info": {"bbox": box()},
            "x_axis": axis,
            "y_axis": axis,
        },
    }


class PlotqaChartTypesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "png"
        self.images.mkdir()
        self.annotations = self.root / "annotations.json"
        self.annotations.write_text(
            json.dumps([annotation(i, chart_type) for i, chart_type in enumerate(CHART_TYPES)]),
            encoding="utf-8",
        )
        for index in range(len(CHART_TYPES)):
            (self.images / f"{index}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 10, 10)
            )

    def assert_output_indices(self, output, indices):
        expected = {f"{index}.png" for index in indices}
        self.assertEqual({path.name for path in (output / "images/train").iterdir()}, expected)
        self.assertEqual(
            {path.name for path in (output / "labels/train").iterdir()},
            {f"{index}.txt" for index in indices},
        )

    def test_vertical_bars_only(self):
        output = self.root / "vertical"
        for index in (1, 2, 3):
            (self.images / f"{index}.png").unlink()
        counts = convert(self.annotations, self.images, output, ["vbar_categorical"])
        self.assertEqual(counts["images"], 1)
        self.assert_output_indices(output, [0])
        self.assertTrue(
            (output / "labels/train/0.txt")
            .read_text(encoding="utf-8")
            .startswith(f"{CLASS_IDS['vertical_bar']} ")
        )

    def test_both_bar_types(self):
        output = self.root / "bars"
        convert(
            self.annotations,
            self.images,
            output,
            ["vbar_categorical", "hbar_categorical"],
        )
        self.assert_output_indices(output, [0, 1])

    def test_no_filter_includes_all_types(self):
        output = self.root / "all"
        convert(self.annotations, self.images, output)
        self.assert_output_indices(output, range(len(CHART_TYPES)))

    def test_validation_split_and_chart_type_manifest(self):
        output = self.root / "validation"
        counts = convert(
            self.annotations, self.images, output, ["vbar_categorical"], "val"
        )
        self.assertEqual(counts["images"], 1)
        self.assertEqual(
            {path.name for path in (output / "images/val").iterdir()}, {"0.png"}
        )
        self.assertEqual(
            {path.name for path in (output / "labels/val").iterdir()}, {"0.txt"}
        )
        self.assertEqual(
            (output / "chart_types_val.csv").read_text(encoding="utf-8").splitlines(),
            ["image_id,chart_type", "0,vbar_categorical"],
        )

    def test_filtered_output_must_not_contain_prior_images_or_labels(self):
        for directory, filename in (("images/train", "old.png"), ("labels/train", "old.txt")):
            with self.subTest(directory=directory):
                output = self.root / filename
                (output / directory).mkdir(parents=True)
                prior_file = output / directory / filename
                prior_file.write_text("old", encoding="utf-8")
                with self.assertRaisesRegex(FileExistsError, "new or empty --output"):
                    convert(self.annotations, self.images, output, ["vbar_categorical"])
                self.assertEqual(prior_file.read_text(), "old")
                self.assertFalse((output / "classes.txt").exists())

    def test_invalid_chart_type_is_rejected_by_cli(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--chart-types", "pie"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_cli_selects_both_bar_types(self):
        output = self.root / "cli_bars"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--annotations", str(self.annotations),
                "--images", str(self.images),
                "--output", str(output),
                "--chart-types", "vbar_categorical", "hbar_categorical",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("images: 2", result.stdout)
        self.assert_output_indices(output, [0, 1])


# Remove split leakage

LEAKAGE_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "dataset" / "remove_split_leakage.py"


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

    def test_link_failure_leaves_no_partial_dataset(self):
        self.add_image("test", 1, b"image")
        with patch.object(Path, "hardlink_to", side_effect=OSError("cross-device link")):
            with self.assertRaisesRegex(OSError, "same filesystem"):
                clean_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".clean-*")), [])

    def test_cli_explains_existing_output_without_changing_it(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        result = subprocess.run(
            [sys.executable, str(LEAKAGE_SCRIPT), "--source", str(self.source), "--output", str(self.output)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Choose a new --output path", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(marker.read_text(), "keep")


# Train yolo

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
