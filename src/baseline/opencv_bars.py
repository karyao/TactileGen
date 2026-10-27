"""Image-only OpenCV vertical-bar baseline and YOLO-label evaluator.

Run with ``python3 -m src.baseline.opencv_bars --help``. The predictor never
reads annotations; only the evaluation command opens YOLO label files.

Compares vertical-bar predictions to YOLO annotation levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from src.contracts import Box


def intersection_over_union(left: Box, right: Box) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predicted: list[Box], actual: list[Box], threshold: float = 0.5) -> int:
    """Find maximum one-to-one matches without reusing a prediction or label."""
    choices = []
    for prediction in predicted:
        overlaps = [(intersection_over_union(prediction, label), ai) for ai, label in enumerate(actual)]
        choices.append([ai for overlap, ai in sorted(overlaps, reverse=True) if overlap >= threshold])
    assigned: dict[int, int] = {}

    def assign(prediction_index: int, seen: set[int]) -> bool:
        for actual_index in choices[prediction_index]:
            if actual_index in seen:
                continue
            seen.add(actual_index)
            if actual_index not in assigned or assign(assigned[actual_index], seen):
                assigned[actual_index] = prediction_index
                return True
        return False

    return sum(assign(pi, set()) for pi in range(len(predicted)))


def load_yolo_boxes(path: Path, class_id: int, size: tuple[int, int], class_count: int) -> list[Box]:
    """Read selected class boxes from a YOLO label file in pixel coordinates."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    image_width, image_height = size
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        try:
            label_class = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid YOLO value") from exc
        if not 0 <= label_class < class_count:
            raise ValueError(f"{path}:{line_number}: unknown class {label_class}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)):
            raise ValueError(f"{path}:{line_number}: coordinates must be finite and normalized")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: box has no area")
        if label_class == class_id:
            boxes.append(Box(
                (cx - width / 2) * image_width,
                (cy - height / 2) * image_height,
                (cx + width / 2) * image_width,
                (cy + height / 2) * image_height,
            ))
    return boxes


def _plot_bounds(image):
    """Estimate the left and bottom plot axes from long dark pixel runs."""
    import cv2
    import numpy as np

    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 115
    row_counts = np.count_nonzero(dark, axis=1)
    row_candidates = np.flatnonzero(
        (row_counts > image_width * 0.35)
        & (np.arange(image_height) > image_height * 0.45)
        & (np.arange(image_height) < image_height * 0.94)
    )
    bottom = int(row_candidates.max()) if row_candidates.size else int(image_height * 0.86)
    column_counts = np.count_nonzero(dark[:bottom], axis=0)
    column_candidates = np.flatnonzero(
        (column_counts > bottom * 0.35)
        & (np.arange(image_width) > image_width * 0.015)
        & (np.arange(image_width) < image_width * 0.3)
    )
    left = int(column_candidates.min()) if column_candidates.size else int(image_width * 0.06)
    axis_pixels = np.flatnonzero(dark[:bottom, left])
    if axis_pixels.size:
        runs = np.split(axis_pixels, np.flatnonzero(np.diff(axis_pixels) > 1) + 1)
        top = int(max(runs, key=len)[0])
    else:
        top = max(0, int(image_height * 0.035))
    bottom_line = np.flatnonzero(dark[bottom, left:])
    right = left + int(bottom_line.max()) if bottom_line.size else int(image_width * 0.97)
    right = min(image_width, max(left + 20, right))
    return left, top, right, bottom


def detect_vertical_bars(image) -> list[Box]:
    """Find tall, filled components inside an estimated plot area."""
    import cv2
    import numpy as np

    left, top, right, bottom = _plot_bounds(image)
    if right - left < 20 or bottom - top < 20:
        return []
    plot = image[top:bottom, left + 3:right]
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    colored = hsv[:, :, 1] >= 45
    dark = (gray < 210) & (hsv[:, :, 1] < 45)
    masks = [dark.astype("uint8") * 255]
    # Separate colors by hue so adjacent series do not become one component.
    for low in range(0, 180, 15):
        masks.append((colored & (hsv[:, :, 0] >= low) & (hsv[:, :, 0] < low + 15)).astype("uint8") * 255)

    plot_height, plot_width = plot.shape[:2]
    found: list[Box] = []
    for mask in masks:
        if not np.any(mask):
            continue
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, plot_width // 8), 1))
        horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        mask = cv2.subtract(mask, horizontal_lines)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component in range(1, count):
            x, y, width, height, pixels = map(int, stats[component])
            if width < 3 or height < 3 or width > plot_width * 0.16:
                continue
            if pixels / (width * height) < 0.72:
                # A neighboring outlined bar can join a filled bar into one
                # component. Keep the columns filled for most of its height.
                component_mask = labels[y:y + height, x:x + width] == component
                dense_columns = np.flatnonzero(
                    np.count_nonzero(component_mask, axis=0) >= height * 0.8
                )
                if dense_columns.size < 3:
                    continue
                gaps = np.flatnonzero(np.diff(dense_columns) > 1)
                runs = np.split(dense_columns, gaps + 1)
                widest = max(runs, key=len)
                x += int(widest[0])
                width = len(widest)
            if height < width * 0.8:
                continue
            if y + height < plot_height * 0.3:
                continue
            found.append(Box(left + 3 + x, top + y, left + 3 + x + width, top + y + height))

    # The dark and color masks can both see a colored bar. Keep the larger box.
    result: list[Box] = []
    for box in sorted(found, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True):
        if not any(intersection_over_union(box, kept) > 0.5 for kept in result):
            result.append(box)
    return sorted(result, key=lambda b: (b.x1, b.y1))


def read_image(path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python-headless numpy") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read PNG: {path}")
    return image


def draw_overlay(image, predicted: list[Box], actual: list[Box] | None = None):
    import cv2

    overlay = image.copy()
    if actual is not None:
        for box in actual:
            cv2.rectangle(overlay, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 180, 0), 2)
    for box in predicted:
        cv2.rectangle(overlay, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 0, 255), 2)
    return overlay


def _write_overlay(path: Path, image, predicted: list[Box], actual: list[Box] | None = None):
    import cv2

    if not cv2.imwrite(str(path), draw_overlay(image, predicted, actual)):
        raise OSError(f"Could not write overlay: {path}")


def _manifest_ids(path: Path, chart_type: str) -> set[str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "chart_type"]:
            raise ValueError(f"Invalid chart-type manifest: {path}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows if row["chart_type"] == chart_type]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image ID in {path}")
    return set(ids)


def evaluate(data_root: Path, split: str, limit: int | None = None, output_dir: Path | None = None) -> dict:
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    classes_path = data_root / "classes.txt"
    classes = classes_path.read_text(encoding="utf-8").splitlines()
    if classes.count("vertical_bar") != 1:
        raise ValueError(f"Expected one vertical_bar class in {classes_path}")
    class_id = classes.index("vertical_bar")
    selected_ids = _manifest_ids(data_root / f"chart_types_{split}.csv", "vbar_categorical")
    images = sorted(images_dir.glob("*.png"), key=lambda path: int(path.stem))
    if selected_ids is not None:
        images = [path for path in images if path.stem in selected_ids]
        missing = selected_ids - {path.stem for path in images}
        if missing:
            raise FileNotFoundError(f"Manifest images missing from {images_dir}: {sorted(missing)[:5]}")
    if not images:
        raise ValueError(f"No PNG images found in {images_dir}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        images = images[:limit]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    true_positive = false_positive = false_negative = exact_count = 0
    sample_rows = []
    for path in images:
        image = read_image(path)
        image_height, image_width = image.shape[:2]
        actual = load_yolo_boxes(labels_dir / f"{path.stem}.txt", class_id, (image_width, image_height), len(classes))
        predicted = detect_vertical_bars(image)
        matched = match_boxes(predicted, actual)
        true_positive += matched
        false_positive += len(predicted) - matched
        false_negative += len(actual) - matched
        exact_count += len(predicted) == len(actual)
        if len(sample_rows) < 8:
            sample_rows.append({"image_id": path.stem, "predicted": len(predicted), "actual": len(actual), "matched": matched})
            if output_dir is not None:
                _write_overlay(output_dir / f"{path.stem}_overlay.png", image, predicted, actual)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    report = {
        "split": split,
        "class": "vertical_bar",
        "images": len(images),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_bar_count_accuracy": exact_count / len(images),
        "sample_images": sample_rows,
    }
    if output_dir is not None:
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict", help="Detect bars from one PNG without annotations")
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output-dir", type=Path, required=True)
    scoring = commands.add_parser("evaluate", help="Score predictions against YOLO labels")
    scoring.add_argument("--data-root", type=Path, required=True)
    scoring.add_argument("--split", required=True)
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.command == "predict":
        image = read_image(args.image)
        boxes = detect_vertical_bars(image)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {"image": str(args.image), "chart_type": "vbar_categorical", "boxes": [asdict(box) for box in boxes]}
        (args.output_dir / f"{args.image.stem}_predictions.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_overlay(args.output_dir / f"{args.image.stem}_overlay.png", image, boxes)
        print(json.dumps({"bars": len(boxes), "output_dir": str(args.output_dir)}))
    else:
        print(json.dumps(evaluate(args.data_root, args.split, args.limit, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
