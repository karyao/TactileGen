"""Combine prepared PlotQA and ChartQA bar splits before leakage cleanup.

Inputs must contain all three YOLO splits and chart_types_<split>.csv files.
Images are hard-linked; source and output must be on the same filesystem.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from src.contracts import CLASS_NAMES
from src.dataset.remove_split_leakage import read_manifest, write_data_yaml

SPLITS = ("train", "val", "test")
BAR_TYPES = {
    "vbar_categorical": "vbar_categorical",
    "hbar_categorical": "hbar_categorical",
    "v_bar": "vbar_categorical",
    "h_bar": "hbar_categorical",
}


def combine(plotqa: Path, chartqa: Path, output: Path) -> dict:
    """Assign deterministic numeric IDs and preserve original source provenance."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output path.")
    records = {split: [] for split in SPLITS}
    for dataset, root in (("plotqa", plotqa.resolve()), ("chartqa", chartqa.resolve())):
        if (root / "classes.txt").read_text().splitlines() != list(CLASS_NAMES):
            raise ValueError(f"{dataset}: classes.txt does not match the shared class IDs")
        for split in SPLITS:
            images = root / "images" / split
            labels = root / "labels" / split
            if not images.is_dir() or not labels.is_dir():
                raise FileNotFoundError(f"{dataset}: missing images or labels for {split}")
            image_ids = {p.stem for p in images.glob("*.png")}
            if image_ids != {p.stem for p in labels.glob("*.txt")}:
                raise ValueError(f"{dataset}: image/label IDs differ in {split}")
            manifest = read_manifest(root / f"chart_types_{split}.csv", image_ids, split)
            if manifest is None or "chart_type" not in manifest[0]:
                raise ValueError(f"{dataset}: missing chart-type manifest for {split}")
            for row in sorted(manifest[1], key=lambda row: row["image_id"]):
                image_id = row["image_id"]
                # Input filenames can be nonnumeric; only bar charts enter this dataset.
                if row["chart_type"] not in BAR_TYPES:
                    continue
                records[split].append((dataset, image_id, BAR_TYPES[row["chart_type"]],
                                       images / f"{image_id}.png", labels / f"{image_id}.txt"))

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {}
    counts = {}
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
        write_data_yaml(staging, list(CLASS_NAMES))
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            counts[split] = {"plotqa": 0, "chartqa": 0}
            with (staging / f"chart_types_{split}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("image_id", "chart_type"))
                for dataset, original_id, chart_type, image, label in records[split]:
                    image_id = str(len(provenance))
                    try:
                        (staging / "images" / split / f"{image_id}.png").hardlink_to(image)
                    except OSError as exc:
                        raise OSError("Cannot hard-link images; keep inputs and output on the same filesystem") from exc
                    shutil.copy2(label, staging / "labels" / split / f"{image_id}.txt")
                    writer.writerow((image_id, chart_type))
                    provenance[image_id] = {
                        "dataset": dataset, "split": split, "original_id": original_id,
                        "image": str(image), "label": str(label),
                    }
                    counts[split][dataset] += 1
        (staging / "source_index.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        staging.rename(output)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotqa", required=True, type=Path, help="prepared PlotQA YOLO root")
    parser.add_argument("--chartqa", required=True, type=Path, help="prepared ChartQA YOLO root")
    parser.add_argument("--output", required=True, type=Path, help="new combined YOLO root")
    args = parser.parse_args()
    try:
        counts = combine(args.plotqa, args.chartqa, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(counts, indent=2))
    print(f"Combined dataset: {args.output}. Run remove_split_leakage before training.")


if __name__ == "__main__":
    main()
