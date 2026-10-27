"""Four-type PlotQA OpenCV baseline; run as ``python -m src.baseline.opencv_charts``.

Prediction uses image pixels only. Evaluation reads chart-type and YOLO labels
after prediction, so it can measure both routing and mark detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.contracts import CHART_MARK_ROLE
from src.baseline.opencv_bars import (
    Box,
    _plot_bounds,
    _write_overlay,
    detect_vertical_bars,
    load_yolo_boxes,
    match_boxes,
    read_image,
)


TYPE_CLASS = CHART_MARK_ROLE


def _plot_masks(image):
    import cv2
    import numpy as np

    left, top, right, bottom = _plot_bounds(image)
    plot = image[top:bottom, left + 3:right]
    if plot.size == 0:
        return (left + 3, top), plot, []
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    masks = [((gray < 210) & (hsv[:, :, 1] < 45)).astype("uint8") * 255]
    for low in range(0, 180, 15):
        masks.append(((hsv[:, :, 1] >= 45) & (hsv[:, :, 0] >= low) & (hsv[:, :, 0] < low + 15)).astype("uint8") * 255)
    return (left + 3, top), plot, masks


def _remove_long_lines(mask, orientation: str):
    import cv2

    height, width = mask.shape
    if orientation == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 7)))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 7), 1))
    return cv2.subtract(mask, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel))


def _deduplicate(boxes: list[Box]) -> list[Box]:
    from src.baseline.opencv_bars import intersection_over_union

    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True):
        if not any(intersection_over_union(box, other) > 0.5 for other in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: (b.x1, b.y1))


def detect_horizontal_bars(image) -> list[Box]:
    import cv2
    import numpy as np

    (offset_x, offset_y), plot, masks = _plot_masks(image)
    if not masks:
        return []
    plot_height, plot_width = plot.shape[:2]
    found = []
    for original in masks:
        if not np.any(original):
            continue
        mask = _remove_long_lines(original, "vertical")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component in range(1, count):
            x, y, width, height, pixels = map(int, stats[component])
            if width < 4 or height < 3 or height > plot_height * 0.16:
                continue
            if pixels / (width * height) < 0.72:
                component_mask = labels[y:y + height, x:x + width] == component
                dense_rows = np.flatnonzero(np.count_nonzero(component_mask, axis=1) >= width * 0.8)
                if dense_rows.size < 3:
                    continue
                runs = np.split(dense_rows, np.flatnonzero(np.diff(dense_rows) > 1) + 1)
                widest = max(runs, key=len)
                y += int(widest[0])
                height = len(widest)
            if width < height * 0.8:
                continue
            found.append(Box(offset_x + x, offset_y + y, offset_x + x + width, offset_y + y + height))
    return _deduplicate(found)


def detect_dots(image) -> list[Box]:
    import cv2
    import numpy as np

    (offset_x, offset_y), plot, masks = _plot_masks(image)
    if not masks:
        return []
    found = []
    for original in masks:
        if not np.any(original):
            continue
        mask = _remove_long_lines(_remove_long_lines(original, "vertical"), "horizontal")
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component in range(1, count):
            x, y, width, height, pixels = map(int, stats[component])
            if not (4 <= width <= 36 and 4 <= height <= 36):
                continue
            if not (0.5 <= width / height <= 2.0) or pixels / (width * height) < 0.4:
                continue
            found.append(Box(offset_x + x, offset_y + y, offset_x + x + width, offset_y + y + height))
    return _deduplicate(found)


def detect_line_segments(image) -> list[Box]:
    import cv2
    import numpy as np

    (offset_x, offset_y), plot, masks = _plot_masks(image)
    if not masks:
        return []
    found = []
    minimum_length = max(20, plot.shape[1] // 15)
    for original in masks:
        if not np.any(original):
            continue
        mask = _remove_long_lines(_remove_long_lines(original, "vertical"), "horizontal")
        lines = cv2.HoughLinesP(mask, 1, 3.141592653589793 / 180, threshold=15,
                               minLineLength=minimum_length, maxLineGap=12)
        if lines is None:
            continue
        for item in lines:
            x1, y1, x2, y2 = map(int, item.reshape(-1))
            if abs(x2 - x1) < minimum_length:
                continue
            found.append(Box(offset_x + min(x1, x2), offset_y + min(y1, y2),
                             offset_x + max(x1, x2) + 1, offset_y + max(y1, y2) + 1))
    return _deduplicate(found)


DETECTORS = {
    "hbar_categorical": detect_horizontal_bars,
    "vbar_categorical": detect_vertical_bars,
    "dot_line": detect_dots,
    "line": detect_line_segments,
}


def predict_chart(image) -> tuple[str, list[Box]]:
    """Choose a chart type from pixels and return its detected data marks."""
    horizontal = detect_horizontal_bars(image)
    vertical = detect_vertical_bars(image)
    image_area = image.shape[0] * image.shape[1]
    bar_scores = {
        "hbar_categorical": sum((b.x2 - b.x1) * (b.y2 - b.y1) for b in horizontal) / image_area,
        "vbar_categorical": sum((b.x2 - b.x1) * (b.y2 - b.y1) for b in vertical) / image_area,
    }
    best_bar = max(bar_scores, key=bar_scores.get)
    if bar_scores[best_bar] >= 0.007 and len(horizontal if best_bar == "hbar_categorical" else vertical) >= 2:
        return best_bar, horizontal if best_bar == "hbar_categorical" else vertical
    lines = detect_line_segments(image)
    if len(lines) >= 2 and sum(box.x2 - box.x1 for box in lines) >= image.shape[1] * 0.5:
        return "line", lines
    dots = detect_dots(image)
    if dots:
        return "dot_line", dots
    return "line", lines


def _read_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "chart_type"]:
            raise ValueError(f"Invalid manifest: {path}")
        rows = list(reader)
    mapping = {row["image_id"]: row["chart_type"] for row in rows}
    if len(mapping) != len(rows) or any(value not in TYPE_CLASS for value in mapping.values()):
        raise ValueError(f"Duplicate image IDs or unknown chart types in {path}")
    return mapping


def _scores(counts: Counter) -> dict:
    tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"images": counts["images"], "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate(data_root: Path, split: str, limit_per_type: int | None = None, output_dir: Path | None = None) -> dict:
    classes = (data_root / "classes.txt").read_text(encoding="utf-8").splitlines()
    for mark_class in TYPE_CLASS.values():
        if classes.count(mark_class) != 1:
            raise ValueError(f"Missing or duplicate class {mark_class} in classes.txt")
    manifest = _read_manifest(data_root / f"chart_types_{split}.csv")
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    by_type = {chart_type: [] for chart_type in TYPE_CLASS}
    for image_id, chart_type in manifest.items():
        by_type[chart_type].append(image_id)
    selected = []
    rng = random.Random(0)
    for chart_type, ids in by_type.items():
        ids.sort(key=int)
        if limit_per_type is not None and len(ids) > limit_per_type:
            ids = sorted(rng.sample(ids, limit_per_type), key=int)
        for image_id in ids:
            path = images_dir / f"{image_id}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")
            selected.append((path, chart_type))
    if not selected:
        raise ValueError("No charts selected")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    classification = Counter()
    end_to_end = {chart_type: Counter() for chart_type in TYPE_CLASS}
    oracle = {chart_type: Counter() for chart_type in TYPE_CLASS}
    examples = Counter()
    for path, actual_type in selected:
        image = read_image(path)
        height, width = image.shape[:2]
        actual = load_yolo_boxes(labels_dir / f"{path.stem}.txt", classes.index(TYPE_CLASS[actual_type]),
                                 (width, height), len(classes))
        predicted_type, predicted = predict_chart(image)
        classification[(actual_type, predicted_type)] += 1
        correct_type = predicted_type == actual_type
        matched = match_boxes(predicted, actual) if correct_type else 0
        counts = end_to_end[actual_type]
        counts["images"] += 1
        counts["tp"] += matched
        counts["fn"] += len(actual) - matched
        end_to_end[predicted_type]["fp"] += len(predicted) - matched

        oracle_predicted = predicted if correct_type else DETECTORS[actual_type](image)
        oracle_matches = match_boxes(oracle_predicted, actual)
        oracle_counts = oracle[actual_type]
        oracle_counts["images"] += 1
        oracle_counts["tp"] += oracle_matches
        oracle_counts["fp"] += len(oracle_predicted) - oracle_matches
        oracle_counts["fn"] += len(actual) - oracle_matches
        if output_dir and examples[actual_type] < 2:
            _write_overlay(output_dir / f"{actual_type}_{path.stem}.png", image, predicted, actual)
            examples[actual_type] += 1

    total = len(selected)
    report = {
        "split": split,
        "images": total,
        "chart_type_accuracy": sum(classification[(kind, kind)] for kind in TYPE_CLASS) / total,
        "confusion": {actual: {predicted: classification[(actual, predicted)] for predicted in TYPE_CLASS}
                      for actual in TYPE_CLASS},
        "mark_detection_end_to_end": {kind: _scores(end_to_end[kind]) for kind in TYPE_CLASS},
        "mark_detection_known_type": {kind: _scores(oracle[kind]) for kind in TYPE_CLASS},
    }
    if output_dir:
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output-dir", type=Path, required=True)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--data-root", type=Path, required=True)
    scoring.add_argument("--split", required=True)
    scoring.add_argument("--limit-per-type", type=int)
    scoring.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        image = read_image(args.image)
        chart_type, boxes = predict_chart(image)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {"image": str(args.image), "chart_type": chart_type,
                  "mark_class": TYPE_CLASS[chart_type], "boxes": [asdict(box) for box in boxes]}
        (args.output_dir / f"{args.image.stem}_predictions.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_overlay(args.output_dir / f"{args.image.stem}_overlay.png", image, boxes)
        print(json.dumps({"chart_type": chart_type, "marks": len(boxes)}))
    else:
        if args.limit_per_type is not None and args.limit_per_type < 1:
            parser.error("--limit-per-type must be positive")
        print(json.dumps(evaluate(args.data_root, args.split, args.limit_per_type, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
