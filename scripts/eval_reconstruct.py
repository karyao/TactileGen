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
from src.reconstruct import COLOR_DISTANCE, reconstruct_vbar
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations_test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/png_test/png"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolo_vbars/best.pt"))
    parser.add_argument("--n", type=int, default=200)
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
    print("\nfailed or not value-comparable (first 10 ids):",
          [r["image_id"] for r in rows if not r["ok"] or not r["errors"]][:10])


if __name__ == "__main__":
    main()
