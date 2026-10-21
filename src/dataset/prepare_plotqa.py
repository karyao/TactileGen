"""Convert PlotQA chart annotations to YOLO detection labels.

The generated dataset contains a train split only. Images are hard-linked from
the source PNG directory, so conversion requires both directories to be on the
same filesystem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    "horizontal_bar",
    "vertical_bar",
    "dot",
    "line_segment",
    "chart_title",
    "x_axis_title",
    "y_axis_title",
    "x_tick_label",
    "y_tick_label",
    "x_axis_rule",
    "y_axis_rule",
    "plot_area",
    "legend",
    "legend_swatch",
    "legend_label",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
MARK_CLASS = {
    "hbar_categorical": "horizontal_bar",
    "vbar_categorical": "vertical_bar",
    "dot_line": "dot",
    "line": "line_segment",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ANNOTATION_CHARS = 16 * 1024 * 1024


def iter_annotations(path: Path) -> Iterator[dict[str, Any]]:
    """Decode a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        buffer = ""
        position = 0
        finished = False

        def refill() -> None:
            nonlocal buffer, position, finished
            chunk = source.read(1024 * 1024)
            buffer = buffer[position:] + chunk
            position = 0
            finished = not chunk

        refill()
        if not buffer.lstrip().startswith("["):
            raise ValueError(f"Expected a JSON array in {path}")
        position = buffer.index("[") + 1

        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or finished:
                    break
                refill()

            if position >= len(buffer):
                raise ValueError(f"Unterminated JSON array in {path}")
            if buffer[position] == "]":
                return
            if buffer[position] == ",":
                position += 1
                continue

            while True:
                try:
                    item, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if finished:
                        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
                    if len(buffer) - position >= MAX_ANNOTATION_CHARS:
                        raise ValueError(
                            f"Invalid or oversized annotation in {path} "
                            f"(over {MAX_ANNOTATION_CHARS} characters)"
                        ) from exc
                    refill()
            if not isinstance(item, dict):
                raise ValueError("Each annotation must be a JSON object")
            yield item


def png_size(path: Path) -> tuple[int, int]:
    """Read dimensions directly from the PNG header."""
    with path.open("rb") as image:
        header = image.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        raise ValueError(f"Invalid PNG dimensions: {path}")
    return width, height


def iter_boxes(annotation: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield the chart elements represented by bounding boxes in PlotQA."""
    chart_type = annotation["type"]
    if chart_type not in MARK_CLASS:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    for model in annotation["models"]:
        for box in model["bboxes"]:
            yield MARK_CLASS[chart_type], box

    info = annotation["general_figure_info"]
    yield "chart_title", info["title"]["bbox"]
    yield "plot_area", info["plot_info"]["bbox"]

    for axis in ("x", "y"):
        axis_info = info[f"{axis}_axis"]
        yield f"{axis}_axis_title", axis_info["label"]["bbox"]
        yield f"{axis}_axis_rule", axis_info["rule"]["bbox"]
        for box in axis_info["major_labels"]["bboxes"]:
            yield f"{axis}_tick_label", box

    if "legend" in info:
        legend = info["legend"]
        yield "legend", legend["bbox"]
        for item in legend["items"]:
            yield "legend_swatch", item["preview"]["bbox"]
            yield "legend_label", item["label"]["bbox"]


def yolo_row(
    class_name: str, box: Any, image_size: tuple[int, int], counts: Counter[str]
) -> str | None:
    """Convert and clip one pixel-space box to a normalized YOLO row."""
    try:
        x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
    except (TypeError, KeyError, ValueError):
        counts["invalid_boxes"] += 1
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        counts["invalid_boxes"] += 1
        return None

    if width < 0 or height < 0:
        counts["reversed_boxes"] += 1
    x1, x2 = sorted((x, x + width))
    y1, y2 = sorted((y, y + height))
    image_width, image_height = image_size
    clipped_x1 = min(max(x1, 0.0), image_width)
    clipped_x2 = min(max(x2, 0.0), image_width)
    clipped_y1 = min(max(y1, 0.0), image_height)
    clipped_y2 = min(max(y2, 0.0), image_height)
    if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1, y1, x2, y2):
        counts["clipped_boxes"] += 1
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        counts["empty_boxes"] += 1
        return None

    center_x = (clipped_x1 + clipped_x2) / (2 * image_width)
    center_y = (clipped_y1 + clipped_y2) / (2 * image_height)
    box_width = (clipped_x2 - clipped_x1) / image_width
    box_height = (clipped_y2 - clipped_y1) / image_height
    return (
        f"{CLASS_IDS[class_name]} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def convert(
    annotations: Path,
    images: Path,
    output: Path,
    chart_types: Collection[str] | None = None,
) -> Counter[str]:
    """Write a train-only YOLO dataset from PlotQA files."""
    selected_types = set(chart_types) if chart_types is not None else None
    if selected_types is not None:
        unsupported = selected_types - MARK_CLASS.keys()
        if unsupported:
            raise ValueError(f"Unsupported chart types: {', '.join(sorted(unsupported))}")

    train_images = output / "images" / "train"
    train_labels = output / "labels" / "train"
    if selected_types is not None and any(
        directory.is_dir() and any(directory.iterdir())
        for directory in (train_images, train_labels)
    ):
        raise FileExistsError(
            "Filtered output already contains images or labels; choose a new or empty --output directory"
        )
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    counts: Counter[str] = Counter()
    seen_indices: set[int] = set()
    for annotation in iter_annotations(annotations):
        chart_type = annotation["type"]
        if chart_type not in MARK_CLASS:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        if selected_types is not None and chart_type not in selected_types:
            continue

        index = annotation["image_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"Invalid image_index: {index!r}")
        if index in seen_indices:
            raise ValueError(f"Duplicate image_index: {index}")
        seen_indices.add(index)

        source_image = images / f"{index}.png"
        image_size = png_size(source_image)
        destination_image = train_images / source_image.name
        if destination_image.exists():
            if not os.path.samefile(source_image, destination_image):
                raise FileExistsError(f"Existing image is not linked to source: {destination_image}")
        else:
            try:
                destination_image.hardlink_to(source_image)
            except OSError as exc:
                raise OSError(
                    f"Cannot hard-link {source_image} to {destination_image}; "
                    "source and output must share a filesystem"
                ) from exc

        rows: list[str] = []
        seen_rows: set[str] = set()
        for class_name, box in iter_boxes(annotation):
            row = yolo_row(class_name, box, image_size, counts)
            if row is None:
                continue
            if row in seen_rows:
                counts["duplicate_boxes"] += 1
                continue
            seen_rows.add(row)
            rows.append(row)

        (train_labels / f"{index}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        counts["images"] += 1
        counts["boxes"] += len(rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "annotations.json")
    parser.add_argument("--images", type=Path, default=ROOT / "png")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_yolo")
    parser.add_argument(
        "--chart-types",
        nargs="+",
        choices=tuple(MARK_CLASS),
        metavar="TYPE",
        help="Chart types to include: " + ", ".join(MARK_CLASS),
    )
    args = parser.parse_args()
    counts = convert(args.annotations, args.images, args.output, args.chart_types)
    for name in (
        "images",
        "boxes",
        "reversed_boxes",
        "clipped_boxes",
        "empty_boxes",
        "invalid_boxes",
        "duplicate_boxes",
    ):
        print(f"{name}: {counts[name]}")


if __name__ == "__main__":
    main()
