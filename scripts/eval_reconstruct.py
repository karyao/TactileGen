"""Evaluate YOLO + Tesseract + reconstruct_vbar on PlotQA vbar_categorical test charts.

Run from the repository root:
    python -m scripts.eval_reconstruct --n 200 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path

import cv2

from src.contracts import ChartInterpretationError
from src.detection import load_model, run_detection
from src.ocr import run_ocr
from src.reconstruct import (
    COLOR_DISTANCE, MIN_CONFIDENCE, _centre, _dedupe, _inside, reconstruct_vbar,
)
from src.tesseract_ocr import TesseractBackend


def _norm(text) -> str:
    return " ".join(str(text).lower().split())


def _hex_rgb(value: str):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def _match_series(spec, models):
    """Pair each ground-truth model with a predicted series by colour (one-to-one)."""
    if len(spec.series) == 1 and len(models) == 1:
        return [(spec.series[0], models[0])]
    pairs, used = [], set()
    for model in models:
        color = _hex_rgb(model["colors"]) if isinstance(model.get("colors"), str) else None
        best = None
        for k, series in enumerate(spec.series):
            if k in used or color is None or series.source_color is None:
                continue
            d = math.dist(color, _hex_rgb(series.source_color))
            if d <= 2 * COLOR_DISTANCE and (best is None or d < best[0]):
                best = (d, k)
        if best is not None:
            used.add(best[1])
            pairs.append((spec.series[best[1]], model))
    return pairs


def _relative_errors(spec, record):
    """Per-value relative errors, or None if series/categories cannot be aligned."""
    models = record["models"]
    truth_categories = [_norm(c) for c in models[0]["x"]]
    if len(spec.categories) != len(truth_categories):
        return None
    pairs = _match_series(spec, models)
    if len(pairs) != len(models):
        return None
    errors = []
    for series, model in pairs:
        for predicted, truth in zip(series.values, model["y"]):
            if truth is None or truth == 0:
                continue
            errors.append(math.inf if predicted is None
                          else abs(predicted - truth) / abs(truth))
    return errors


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


# --- --diagnose: where do bars get lost in charts flagged category_count_mismatch? ---

STAGES = ("raw", "confidence", "dedupe", "plot_area")
NEAR_ZERO_FRACTION = 0.05   # |value| below 5% of the chart's largest |value|
NEAR_ZERO_PIXELS = 5        # or a bar at most this tall
ADJACENT_RATIO = 0.10       # gap to a neighbour under 10% of bar width
LOW_CONFIDENCE = 0.5
COVER_IOU = 0.5


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(d):
    return (d.box.x1, d.box.y1, d.box.x2, d.box.y2)


def _stage_lists(frame):
    """Detections left after each stage, mirroring reconstruct_vbar's filtering order."""
    raw = [d for d in frame.detections if d.role == "vertical_bar"]
    confident = [d for d in raw if d.confidence >= MIN_CONFIDENCE]
    deduped = _dedupe(confident)
    plot = max((d for d in frame.detections if d.role == "plot_area"
                and d.confidence >= MIN_CONFIDENCE), key=lambda d: d.confidence, default=None)
    inside = [d for d in deduped if plot is None or _inside(_centre(d), plot)]
    x_raw = [d for d in frame.detections if d.role == "x_tick_label"]
    x_deduped = _dedupe(d for d in x_raw if d.confidence >= MIN_CONFIDENCE)
    if plot is not None:
        mx = max(8.0, 0.05 * (plot.box.x2 - plot.box.x1))
        x_deduped = [d for d in x_deduped if plot.box.x1 - mx <= _centre(d)[0] <= plot.box.x2 + mx]
    return {"raw": raw, "confidence": confident, "dedupe": deduped, "plot_area": inside}, x_raw, x_deduped


def _ground_truth_bars(record):
    """Annotation bars as dicts with xyxy box, value and adjacency, in x order."""
    bars = []
    for model in record["models"]:
        for box, value in zip(model["bboxes"], model["y"]):
            # Some annotations carry a negative w/h; normalise to a proper xyxy box.
            xs = sorted((box["x"], box["x"] + box["w"]))
            ys = sorted((box["y"], box["y"] + box["h"]))
            bars.append({"box": (xs[0], ys[0], xs[1], ys[1]), "value": value})
    bars.sort(key=lambda b: b["box"][0])
    scale = max((abs(b["value"]) for b in bars if b["value"] is not None), default=0) or 1
    for i, bar in enumerate(bars):
        x1, y1, x2, y2 = bar["box"]
        width = x2 - x1
        gaps = [bars[j]["box"][0] - x2 for j in (i + 1,) if j < len(bars)]
        gaps += [x1 - bars[j]["box"][2] for j in (i - 1,) if j >= 0]
        bar["adjacent"] = bool(gaps) and min(gaps) < ADJACENT_RATIO * width
        bar["small_value"] = (bar["value"] is not None
                              and abs(bar["value"]) < NEAR_ZERO_FRACTION * scale)
        bar["short"] = (y2 - y1) <= NEAR_ZERO_PIXELS
        bar["height"] = y2 - y1
        bar["near_zero"] = bar["small_value"] or bar["short"]
    return bars


def diagnose_chart(frame, record, spec):
    stages, x_raw, x_kept = _stage_lists(frame)
    truth = _ground_truth_bars(record)
    lost = []
    for bar in truth:
        covered = {name: any(_box_iou(bar["box"], _xyxy(d)) >= COVER_IOU for d in stages[name])
                   for name in STAGES}
        if covered["plot_area"]:
            continue
        # First stage at which no remaining detection covers this bar.
        if not covered["raw"]:
            stage = "not_detected"
        elif not covered["confidence"]:
            stage = "confidence"
        elif not covered["dedupe"]:
            stage = "dedupe"
        else:
            stage = "plot_area"
        best = max(stages["raw"], key=lambda d: _box_iou(bar["box"], _xyxy(d)), default=None)
        best_iou = _box_iou(bar["box"], _xyxy(best)) if best else 0.0
        lost.append({"stage": stage, "near_zero": bar["near_zero"],
                     "small_value": bar["small_value"], "short": bar["short"],
                     "height": round(bar["height"], 1), "adjacent": bar["adjacent"],
                     "best_iou": round(best_iou, 3),
                     "confidence": round(best.confidence, 3) if best and best_iou > 0.1 else None})
    spurious = sum(
        not any(_box_iou(_xyxy(d), b["box"]) >= COVER_IOU for b in truth) for d in stages["plot_area"])
    return {
        "gt_bars": len(truth), "gt_labels": len(record["models"][0]["x"]),
        "bars": {name: len(stages[name]) for name in STAGES},
        "x_labels": {"raw": len(x_raw), "filtered": len(x_kept)},
        "series_bar_counts": [sum(v is not None for v in s.values) for s in spec.series],
        "spurious_bars": spurious, "lost": lost,
        "truth_near_zero": sum(b["near_zero"] for b in truth),
        "truth_adjacent": sum(b["adjacent"] for b in truth),
    }


def print_diagnosis(rows):
    mismatched = [r for r in rows if r.get("diagnosis")]
    print(f"\n=== category_count_mismatch diagnosis: {len(mismatched)} charts ===")
    print("id      gt_bars gt_lbls | raw conf dedupe plot | x_raw x_kept | series_bars | lost spurious")
    for r in mismatched:
        d = r["diagnosis"]
        b, x = d["bars"], d["x_labels"]
        print(f"{r['image_id']:<7} {d['gt_bars']:>7} {d['gt_labels']:>7} | {b['raw']:>3} "
              f"{b['confidence']:>4} {b['dedupe']:>6} {b['plot_area']:>4} | {x['raw']:>5} "
              f"{x['filtered']:>6} | {d['series_bar_counts']} | {len(d['lost']):>4} {d['spurious_bars']:>8}")

    lost = [(r["image_id"], item) for r in mismatched for item in r["diagnosis"]["lost"]]
    total_truth = sum(r["diagnosis"]["gt_bars"] for r in mismatched)
    base_zero = sum(r["diagnosis"]["truth_near_zero"] for r in mismatched)
    base_adjacent = sum(r["diagnosis"]["truth_adjacent"] for r in mismatched)
    no_loss = [r["image_id"] for r in mismatched if not r["diagnosis"]["lost"]]
    print(f"\nlost bars: {len(lost)} of {total_truth} ground-truth bars in these charts; "
          f"{len(mismatched) - len(no_loss)} charts lose at least one; "
          f"{len(no_loss)} lose none (extra bars or label miscount): {no_loss[:3]}")
    print(f"extra (unmatched) bars surviving all stages: "
          f"{sum(r['diagnosis']['spurious_bars'] for r in mismatched)} in "
          f"{sum(bool(r['diagnosis']['spurious_bars']) for r in mismatched)} charts")
    label_off = [r["image_id"] for r in mismatched
                 if r["diagnosis"]["x_labels"]["filtered"] != r["diagnosis"]["gt_labels"]]
    print(f"charts whose filtered x label count differs from GT: {len(label_off)} {label_off[:3]}")

    def pct(k, n):
        return f"{100 * k / n:.0f}%" if n else "-"

    print("\nstage that loses the bar | bars | charts | near-zero | adjacent | low-conf(<0.5) | examples")
    for stage in ("not_detected", "confidence", "dedupe", "plot_area"):
        items = [(i, x) for i, x in lost if x["stage"] == stage]
        ids = list(dict.fromkeys(i for i, _ in items))
        print(f"{stage:<24} | {len(items):>4} | {len(ids):>6} | "
              f"{pct(sum(x['near_zero'] for _, x in items), len(items)):>9} | "
              f"{pct(sum(x['adjacent'] for _, x in items), len(items)):>8} | "
              f"{pct(sum(x['confidence'] is not None and x['confidence'] < LOW_CONFIDENCE for _, x in items), len(items)):>14} | {ids[:3]}")
    print(f"{'ALL LOST':<24} | {len(lost):>4} | {'':>6} | "
          f"{pct(sum(x['near_zero'] for _, x in lost), len(lost)):>9} | "
          f"{pct(sum(x['adjacent'] for _, x in lost), len(lost)):>8} | "
          f"{pct(sum(x['confidence'] is not None and x['confidence'] < LOW_CONFIDENCE for _, x in lost), len(lost)):>14} |")
    print(f"{'baseline: all GT bars':<24} | {total_truth:>4} | {'':>6} | {pct(base_zero, total_truth):>9} | "
          f"{pct(base_adjacent, total_truth):>8} |")
    heights = sorted(x["height"] for _, x in lost)
    if heights:
        print(f"\nlost-bar height px: median {heights[len(heights) // 2]}, max {heights[-1]}; "
              f"value<5% of max: {sum(x['small_value'] for _, x in lost)}, "
              f"height<=5px: {sum(x['short'] for _, x in lost)}, "
              f"both: {sum(x['small_value'] and x['short'] for _, x in lost)}")
    print("\nexamples by attribute of lost bars (image ids):")
    for label, test in (
            ("near-zero", lambda x: x["near_zero"]),
            ("adjacent", lambda x: x["adjacent"]),
            ("low-confidence", lambda x: x["confidence"] is not None and x["confidence"] < LOW_CONFIDENCE),
            ("mislocalised (best raw IoU 0.1-0.5)", lambda x: 0.1 <= x["best_iou"] < COVER_IOU),
            ("none of the above", lambda x: not (x["near_zero"] or x["adjacent"] or (
                x["confidence"] is not None and x["confidence"] < LOW_CONFIDENCE))),
    ):
        ids = list(dict.fromkeys(i for i, x in lost if test(x)))
        print(f"  {label}: {sum(test(x) for _, x in lost)} bars, e.g. {ids[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations_test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/png_test/png"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolo_vbars/best.pt"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--diagnose", action="store_true",
                        help="explain bar loss in category_count_mismatch charts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output/eval_reconstruct.json"))
    args = parser.parse_args()

    records = [r for r in json.loads(args.annotations.read_text())
               if r["type"] == "vbar_categorical"]
    sample = random.Random(args.seed).sample(records, args.n)
    model, backend = load_model(args.weights), TesseractBackend()

    rows = []
    for number, record in enumerate(sample, 1):
        image_id = record["image_index"]
        path = args.images / f"{image_id}.png"
        row = {"image_id": image_id}
        try:
            image = cv2.imread(str(path))
            frame = run_detection(model, path)
            ocr = run_ocr(image, frame, backend, "eng")
            spec = reconstruct_vbar(frame, ocr, image)
            row.update(ok=True, flags=spec.flags,
                       categories_match=[_norm(c) for c in spec.categories]
                       == [_norm(c) for c in record["models"][0]["x"]],
                       count_match="category_count_mismatch" not in spec.flags
                       and len(spec.categories) == len(record["models"][0]["x"]),
                       errors=_relative_errors(spec, record))
            if args.diagnose and "category_count_mismatch" in spec.flags:
                row["diagnosis"] = diagnose_chart(frame, record, spec)
        except ChartInterpretationError as exc:
            row.update(ok=False, error=str(exc), flags=[])
        rows.append(row)
        print(f"[{number}/{args.n}] {image_id} {'ok' if row['ok'] else row['error']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))

    done = [r for r in rows if r["ok"]]
    comparable = [r for r in done if r["errors"]]
    pooled = [e for r in comparable for e in r["errors"]]
    finite = [e for e in pooled if math.isfinite(e)]
    pct = lambda k, n=len(rows): f"{100 * k / n:.1f}% ({k}/{n})"
    print(f"\nN={len(rows)} seed={args.seed} weights={args.weights}")
    print("reconstructed without error:", pct(len(done)))
    print("category count correct     :", pct(sum(r["count_match"] for r in done)))
    print("all categories text match  :", pct(sum(r["categories_match"] for r in done)))
    print("values comparable (counts + series aligned):", pct(len(comparable)))
    if pooled:
        print(f"relative value error over {len(pooled)} values in {len(comparable)} charts "
              f"(missing value counted as inf): "
              f"median {statistics.median(pooled):.4f}, p90 {_percentile(pooled, 0.9):.4f}; "
              f"{len(pooled) - len(finite)} missing")
        for limit in (0.01, 0.05):
            print(f"  values within {limit:.0%}: {sum(e <= limit for e in pooled) / len(pooled):.1%}")
    flags = collections.Counter(f.split(":")[0] for r in done for f in r["flags"])
    print("flag counts (charts):", dict(flags.most_common()))
    failures = collections.Counter(r["error"] for r in rows if not r["ok"])
    print("failure reasons:", dict(failures))

    def chart_error(r):
        finite_errors = [e for e in r["errors"] if math.isfinite(e)]
        return statistics.median(finite_errors) if finite_errors else math.inf

    print("\n10 worst reconstructed charts (median relative error per chart):")
    for r in sorted(comparable, key=chart_error, reverse=True)[:10]:
        print(f"  {r['image_id']}: {chart_error(r):.3f} flags={r['flags']}")
    if args.diagnose:
        print_diagnosis(rows)
    print("\nfailed or not value-comparable (first 10 ids):",
          [r["image_id"] for r in rows if not r["ok"] or not r["errors"]][:10])


if __name__ == "__main__":
    main()
