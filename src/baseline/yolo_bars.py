"""Run the trained YOLO detector and score its vertical-bar predictions.

Use ``python -m src.baseline.yolo_bars --help`` for the CLI. Inference reads
images only; evaluation reads labels after predictions have been made.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.baseline.opencv_bars import (
    _write_overlay,
    load_yolo_boxes,
    match_boxes,
    read_image,
)
from src.contract_io import write_detection_frame
from src.contracts import CLASS_IDS, CLASS_NAMES, Box, Detection, DetectionFrame


DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
DEFAULT_DATA_ROOT = Path("dataset_yolo_vbars_clean")
DEFAULT_CONFIDENCE = 0.25


def _class_names(names: dict[int, str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(names, dict):
        try:
            return tuple(names[index] for index in range(len(names)))
        except KeyError as exc:
            raise ValueError("Model class IDs must be contiguous from zero") from exc
    return tuple(names)


def check_model_classes(model) -> None:
    if _class_names(model.names) != CLASS_NAMES:
        raise ValueError("Model class names/order differ from the chart contract")


def load_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the model dependencies: python -m pip install -r requirements-model.txt") from exc
    if not weights.is_file():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {weights}")
    model = YOLO(str(weights))
    check_model_classes(model)
    return model


def detection_frame(result, source_key: str) -> DetectionFrame:
    """Adapt an Ultralytics result to the shared pixel-coordinate contract."""
    height, width = result.orig_shape
    boxes = result.boxes
    detections = []
    for index, (xyxy, class_id, confidence) in enumerate(zip(
        boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
        strict=True,
    )):
        if int(class_id) != class_id or not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"Invalid model class ID: {class_id}")
        x1, y1, x2, y2 = xyxy
        box = Box(max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                  max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        detections.append(Detection(f"det-{index + 1:05d}", CLASS_NAMES[int(class_id)],
                                    box, float(confidence)))
    return DetectionFrame(source_key, width, height, tuple(detections))


def _test_images(data_root: Path, split: str, limit: int | None) -> list[Path]:
    images_dir = data_root / "images" / split
    manifest = data_root / f"chart_types_{split}.csv"
    images = sorted(images_dir.glob("*.png"), key=lambda path: int(path.stem))
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "chart_type"]:
                raise ValueError(f"Invalid chart-type manifest: {manifest}")
            selected = {row["image_id"] for row in reader if row["chart_type"] == "vbar_categorical"}
        available = {path.stem for path in images}
        if selected - available:
            raise FileNotFoundError(f"Manifest references missing images in {images_dir}")
        images = [path for path in images if path.stem in selected]
    if not images:
        raise ValueError(f"No vertical-bar PNG images found in {images_dir}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        images = images[:limit]
    return images


def _class_id(data_root: Path) -> int:
    names = tuple((data_root / "classes.txt").read_text(encoding="utf-8").splitlines())
    if names != CLASS_NAMES:
        raise ValueError("Dataset class names/order differ from the chart contract")
    return CLASS_IDS["vertical_bar"]


def evaluate(model, weights: Path, data_root: Path, split: str, output_dir: Path,
             limit: int | None = None, confidence: float = DEFAULT_CONFIDENCE,
             batch: int = 16, device: str | None = None) -> dict:
    if not 0 <= confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    if batch < 1:
        raise ValueError("--batch must be positive")
    class_id = _class_id(data_root)
    images = _test_images(data_root, split, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    tp = fp = fn = exact_count = processed = 0
    sample_rows = []
    # Ultralytics opens every path in a list before yielding results. Limit each
    # call to one batch so a full test split does not fill memory with PNGs.
    for start in range(0, len(images), batch):
        batch_images = images[start:start + batch]
        predictions = model.predict(source=[str(path) for path in batch_images], stream=True,
                                    batch=batch, conf=confidence, device=device, verbose=False)
        batch_processed = 0
        for path, result in zip(batch_images, predictions, strict=True):
            if Path(result.path).resolve() != path.resolve():
                raise ValueError(f"Prediction order differs from input images: {result.path}")
            frame = detection_frame(result, f"{split}/{path.name}")
            predicted = [item.box for item in frame.detections if item.role == "vertical_bar"]
            actual = load_yolo_boxes(data_root / "labels" / split / f"{path.stem}.txt",
                                     class_id, (frame.width, frame.height), len(CLASS_NAMES))
            matched = match_boxes(predicted, actual)
            tp += matched
            fp += len(predicted) - matched
            fn += len(actual) - matched
            exact_count += len(predicted) == len(actual)
            processed += 1
            batch_processed += 1
            if processed % 500 == 0 or processed == len(images):
                print(f"Scored {processed}/{len(images)} images", file=sys.stderr, flush=True)
            if len(sample_rows) < 8:
                sample_rows.append({"image_id": path.stem, "predicted": len(predicted),
                                    "actual": len(actual), "matched": matched})
                write_detection_frame(output_dir / f"{path.stem}_detections.json", frame)
                _write_overlay(output_dir / f"{path.stem}_overlay.png", read_image(path),
                               predicted, actual)
        if batch_processed != len(batch_images):
            raise ValueError(f"Expected {len(batch_images)} model results, got {batch_processed}")
    if processed != len(images):
        raise ValueError(f"Expected {len(images)} model results, got {processed}")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    report = {
        "model": str(weights), "data_root": str(data_root), "split": split,
        "class": "vertical_bar", "confidence": confidence, "iou": 0.5,
        "images": processed, "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_bar_count_accuracy": exact_count / processed,
        "sample_images": sample_rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate(model, weights: Path, data_root: Path, split: str, output_dir: Path,
             batch: int = 16, device: str | None = None) -> dict:
    """Record standard Ultralytics detection mAP on the specified split."""
    class_id = _class_id(data_root)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml}")
    if batch < 1:
        raise ValueError("--batch must be positive")
    metrics = model.val(data=str(data_yaml.resolve()), split=split, batch=batch,
                        device=device, project=str(output_dir.resolve().parent),
                        name=output_dir.name, exist_ok=True, plots=False)
    image_count = len(list((data_root / "images" / split).glob("*.png")))
    report = {
        "model": str(weights), "data_root": str(data_root), "split": split,
        "images": image_count,
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "vertical_bar_map50_95": float(metrics.box.maps[class_id]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict", help="Run the checkpoint on one image")
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--source-key", help="Namespaced source key, e.g. test/22415.png")
    prediction.add_argument("--output-dir", type=Path, required=True)
    scoring = commands.add_parser("evaluate", help="Score vertical bars on a labeled split")
    scoring.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    scoring.add_argument("--split", choices=("train", "val", "test"), default="test")
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--batch", type=int, default=16)
    validation = commands.add_parser("validate", help="Compute standard detection mAP")
    validation.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    validation.add_argument("--split", choices=("train", "val", "test"), default="test")
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--batch", type=int, default=16)
    for command in (prediction, scoring, validation):
        command.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
        command.add_argument("--device", help="Ultralytics device, e.g. cpu or mps")
    for command in (prediction, scoring):
        command.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()
    model = load_model(args.weights)
    if args.command == "predict":
        if not 0 <= args.confidence <= 1:
            parser.error("--confidence must be between 0 and 1")
        if not args.image.is_file():
            parser.error(f"Missing image: {args.image}")
        result = model.predict(source=str(args.image), conf=args.confidence,
                               device=args.device, verbose=False)[0]
        frame = detection_frame(result, args.source_key or f"input/{args.image.name}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_detection_frame(args.output_dir / f"{args.image.stem}_detections.json", frame)
        boxes = [item.box for item in frame.detections if item.role == "vertical_bar"]
        _write_overlay(args.output_dir / f"{args.image.stem}_overlay.png", read_image(args.image), boxes)
        print(json.dumps({"source_key": frame.source_key, "detections": len(frame.detections),
                          "vertical_bars": len(boxes), "output_dir": str(args.output_dir)}))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(model, args.weights, args.data_root, args.split,
                                  args.output_dir, args.limit, args.confidence,
                                  args.batch, args.device), indent=2))
    else:
        print(json.dumps(validate(model, args.weights, args.data_root, args.split,
                                  args.output_dir, args.batch, args.device), indent=2))


if __name__ == "__main__":
    main()
