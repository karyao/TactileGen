"""Zero-shot and PlotQA-trained YOLO baselines for bars on data/pmc_yolo_bars.

Confidence is chosen on the val split (best F1 at IoU 0.5 over a fixed grid);
the test split is predicted and scored once, at that value. Predictions are
made at a low confidence once per split and filtered offline, which gives the
same boxes as predicting at the chosen value because NMS only removes boxes
outscored by a kept one.

Class handling:
  * PlotQA model (15 classes): its ``vertical_bar`` detections are the bar
    predictions; every other class is ignored.
  * COCO model (80 classes, no chart classes): every detection of any class
    counts as a bar. This is the most generous mapping; a strict one would
    predict nothing and score exactly zero.

Usage:
  python -m src.baseline.pmc_bars_baseline --output-dir <dir>
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.baseline.opencv_bars import match_boxes
from src.contracts import CLASS_IDS, Box
from src.dataset.bar_orientation import BAR, orientation, read_label_boxes

IOUS = (0.5, 0.75, 0.9)
CONF_GRID = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8)
PREDICT_CONF = 0.01
NMS_IOU = 0.70
GROUPS = ("all", "vertical", "horizontal", "undecided")

MODELS = {
    "COCO zero-shot (yolov8s.pt)": ("yolov8s.pt", None),
    "PlotQA YOLO, no mosaic": ("models/best_no_mosaic.pt", CLASS_IDS["vertical_bar"]),
}


def load_split(root: Path, split: str):
    """[(image path, orientation, ground-truth bar boxes)]"""
    items = []
    for image in sorted((root / "images" / split).iterdir()):
        with Image.open(image) as img:
            size = img.size
        boxes = read_label_boxes(root / "labels" / split / (image.stem + ".txt"), size)
        items.append((image, orientation(boxes), boxes.get(BAR, [])))
    return items


def predict(model, images, keep_class):
    """{image path: [(confidence, Box)]} at PREDICT_CONF; keep_class None keeps all."""
    out = {}
    for start in range(0, len(images), 16):
        chunk = images[start:start + 16]
        for path, result in zip(chunk, model.predict(
                source=[str(p) for p in chunk], conf=PREDICT_CONF, iou=NMS_IOU, verbose=False)):
            out[path] = [
                (float(c), Box(*map(float, xyxy)))
                for xyxy, c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                      result.boxes.cls.tolist())
                if keep_class is None or int(k) == keep_class]
    return out


def score(items, predictions, conf):
    """{group: {iou: tp/fp/fn, 'images', 'exact'}} at one confidence."""
    stats = {g: {"images": 0, "exact": 0, **{i: Counter() for i in IOUS}} for g in GROUPS}
    for path, orient, actual in items:
        predicted = [box for c, box in predictions[path] if c >= conf]
        for group in ("all", orient):
            s = stats[group]
            s["images"] += 1
            s["exact"] += len(predicted) == len(actual)
            for iou in IOUS:
                tp = match_boxes(predicted, actual, iou)
                s[iou].update(tp=tp, fp=len(predicted) - tp, fn=len(actual) - tp)
    return stats


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    val, test = load_split(args.root, "val"), load_split(args.root, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"orientation_counts": {
        s: dict(Counter(o for _, o, _ in items)) for s, items in (("val", val), ("test", test))},
        "models": {}}
    for name, (weights, keep) in MODELS.items():
        model = YOLO(weights)
        val_pred = predict(model, [p for p, _, _ in val], keep)
        by_conf = {c: prf(score(val, val_pred, c)["all"][0.5])[2] for c in CONF_GRID}
        conf = max(by_conf, key=lambda c: (by_conf[c], c))
        print(f"{name}: val F1@0.5 by confidence {by_conf} -> chose {conf}")
        test_pred = predict(model, [p for p, _, _ in test], keep)  # test is predicted once
        stats = score(test, test_pred, conf)
        results["models"][name] = {
            "weights": weights, "confidence": conf, "val_f1_at_0.5": by_conf,
            "test": {g: {"images": s["images"], "exact_count": s["exact"] / s["images"] if s["images"] else None,
                         "iou": {str(i): dict(s[i], **dict(zip(("precision", "recall", "f1"), prf(s[i]))))
                                 for i in IOUS}}
                     for g, s in stats.items() if s["images"]}}
    (args.output_dir / "pmc_bars_baseline.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["orientation_counts"]))


if __name__ == "__main__":
    main()
