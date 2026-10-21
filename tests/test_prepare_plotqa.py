"""Small fixtures for PlotQA chart-type selection."""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.dataset.prepare_plotqa import CLASS_IDS, convert


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


if __name__ == "__main__":
    unittest.main()
