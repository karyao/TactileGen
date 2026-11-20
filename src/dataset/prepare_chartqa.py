"""Convert one ChartQA split to the repository's YOLO detection layout.

ChartQA stores one annotation JSON per image. Images are hard-linked so keep
the input and output directories on the same filesystem.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.contracts import CLASS_IDS, CLASS_NAMES
from src.dataset.prepare_plotqa import png_size, yolo_row


ROOT = Path(__file__).resolve().parents[2]
CHART_TYPES = {"v_bar": "vertical_bar", "h_bar": "horizontal_bar"}
DEFAULT_CHART_TYPES = ("v_bar", "h_bar")


def iter_boxes(annotation: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield model and figure boxes supported by the shared class schema."""
    chart_type = annotation.get("type")
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported ChartQA chart type: {chart_type!r}")
    role = CHART_TYPES[chart_type]
    for model in annotation.get("models", []):
        for box in model.get("bboxes", []):
            yield role, box

    info = annotation.get("general_figure_info", {})
    for key in ("title", "x_axis", "y_axis", "legend"):
        section = info.get(key, {})
        if key == "title":
            box = section.get("bbox")
            if box is not None:
                yield "chart_title", box
        elif key in ("x_axis", "y_axis"):
            axis = key[0]
            label = section.get("label", {}).get("bbox")
            if label is not None:
                for box in label if isinstance(label, list) else [label]:
                    yield f"{axis}_axis_title", box
            for box in section.get("major_labels", {}).get("bboxes", []):
                yield f"{axis}_tick_label", box
    legend = info.get("legend", {})
    if legend.get("bbox") is not None:
        yield "legend", legend["bbox"]
    for item in legend.get("items", []):
        for key, role in (("preview", "legend_swatch"), ("label", "legend_label")):
            box = item.get(key, {}).get("bbox")
            if box is not None:
                yield role, box

    figure = info.get("figure_info", {}).get("bbox")
    if figure is not None:
        yield "plot_area", figure


def convert(
    annotations: Path,
    images: Path,
    output: Path,
    chart_types: Collection[str] | None = None,
    split: str = "train",
) -> Counter[str]:
    """Convert per-image ChartQA annotation files to YOLO labels."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    selected = set(chart_types) if chart_types is not None else set(DEFAULT_CHART_TYPES)
    unsupported = selected - CHART_TYPES.keys()
    if unsupported:
        raise ValueError(f"Unsupported chart types: {', '.join(sorted(unsupported))}")
    annotation_files = sorted(annotations.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No ChartQA annotations found in {annotations}")

    split_images, split_labels = output / "images" / split, output / "labels" / split
    if any(path.is_dir() and any(path.iterdir()) for path in (split_images, split_labels)):
        raise FileExistsError("Output split already contains files; choose a new or empty --output")
    split_images.mkdir(parents=True, exist_ok=True)
    split_labels.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    counts: Counter[str] = Counter()
    manifest: list[tuple[str, str]] = []
    for annotation_path in annotation_files:
        image_id = annotation_path.stem
        if Path(image_id).name != image_id or image_id in ("", ".", ".."):
            raise ValueError(f"Invalid ChartQA image ID: {image_id!r}")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotation, dict):
            raise ValueError(f"Expected an annotation object in {annotation_path}")
        chart_type = annotation.get("type")
        if chart_type not in CHART_TYPES:
            counts["unsupported_chart_types"] += 1
            continue
        if chart_type not in selected:
            continue
        image = images / f"{image_id}.png"
        size = png_size(image)
        destination = split_images / image.name
        try:
            destination.hardlink_to(image)
        except FileExistsError:
            if not os.path.samefile(destination, image):
                raise
        except OSError as exc:
            raise OSError(f"Cannot hard-link {image} to {destination}; use the same filesystem") from exc

        rows: list[str] = []
        seen: set[str] = set()
        for role, box in iter_boxes(annotation):
            row = yolo_row(role, box, size, counts)
            if row is None:
                continue
            if row in seen:
                counts["duplicate_boxes"] += 1
                continue
            seen.add(row)
            rows.append(row)
        (split_labels / f"{image_id}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        manifest.append((image_id, chart_type))
        counts["images"] += 1
        counts["boxes"] += len(rows)

    with (output / f"chart_types_{split}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "chart_type"))
        writer.writerows(manifest)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/chartqa/train/annotations")
    parser.add_argument("--images", type=Path, default=ROOT / "data/chartqa/train/png")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_yolo_chartqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--chart-types", nargs="+", choices=tuple(CHART_TYPES),
                        default=list(DEFAULT_CHART_TYPES))
    args = parser.parse_args()
    counts = convert(args.annotations, args.images, args.output, args.chart_types, args.split)
    for name in ("images", "boxes", "invalid_boxes", "clipped_boxes", "empty_boxes",
                 "duplicate_boxes", "unsupported_chart_types"):
        print(f"{name}: {counts[name]}")


if __name__ == "__main__":
    main()
