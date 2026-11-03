"""Shared protocol for scoring OCR engines on ground-truth text crops (data/pmc_yolo_bars).

Crops come from the plots_detection COCO boxes (the only place the ground-truth
``text`` lives) for images in the val and test splits of data/pmc_yolo_bars.
Only vertical bar charts are scored (src/dataset/bar_orientation.py); a second
"scope" row keeps single-series charts (fewer than two legend patches) with at
most 10 bars.

Every engine (src/baseline/ocr_tesseract.py, ocr_easyocr.py) goes through the
same steps:
  tune     try the engine's settings on a sample of val crops and keep the best
           per class group (numeric: highest correct-parse rate; text: lowest CER)
  test     score the chosen settings once on test (refuses a second run)
  rescore  recompute the test tables from saved reads (metric changes only)
  predict  save reads for val with the chosen settings (no scoring)

Metrics: exact match raw (strip only) and normalized (see normalize), character
error rate on normalized text, and for value-axis labels the share that parse to
the ground-truth number.
"""

import argparse
import json
import math
import random
import re
import shutil
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from src.dataset.bar_orientation import orientation, read_label_boxes
from src.textutil import normalize, parse_number  # noqa: F401  (re-exported for the scoring scripts)

COCO = Path("data/pmc_2022/pmc_coco/plots_detection")
COCO_FILES = {"train": "bar_train.json", "val": "bar_val.json", "test": "bar_test.json"}
CLASSES = {1: "x_title", 2: "y_title", 7: "chart_title", 5: "xlabel", 6: "ylabel",
           11: "legend_label", 12: "legend_title", 15: "value_label"}  # COCO category ids
NUMERIC = {"ylabel", "value_label"}
LEGEND_PATCH_COCO = 10
WHITELIST = "0123456789.,-+%eE−–"


# ---------------------------------------------------------------- text helpers
def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = row
    return previous[-1]


# --------------------------------------------------------------------- crops
def load_crops(root: Path, split: str):
    """[{image, klass, text, box, chart}] for scored text boxes, plus per-chart scope flags."""
    images = {p.name: p for p in (root / "images" / split).iterdir()}
    crops, charts = [], {}
    # The split follows our paper-level re-split, so its images come from all three COCO files.
    for coco_file in COCO_FILES.values():
        coco = json.loads((COCO / coco_file).read_text())
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for im in coco["images"]:
            path = images.get(im["file_name"])
            if path is None:
                continue
            label = root / "labels" / split / (path.stem + ".txt")
            if orientation(read_label_boxes(label, (im["width"], im["height"]))) != "vertical":
                continue
            anns = by_image[im["id"]]
            n_bars = sum(a["category_id"] == 19 for a in anns)
            legend = sum(a["category_id"] == LEGEND_PATCH_COCO for a in anns)
            charts[path.name] = {"scope": legend < 2 and n_bars <= 10}
            for a in anns:
                if a["category_id"] in CLASSES and a.get("text"):
                    crops.append({"image": path, "klass": CLASSES[a["category_id"]], "text": a["text"],
                                  "box": a["bbox"], "chart": path.name})
    return crops, charts


def crop_image(img, box, pad=4):
    x, y, w, h = box
    x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
    x2, y2 = min(img.shape[1], math.ceil(x + w)), min(img.shape[0], math.ceil(y + h))
    crop = img[y1:y2, x1:x2]
    return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def prepare(crop, height, binarize, sharpen=False):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.shape[0] < height:
        scale = height / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if sharpen:  # mild unsharp mask
        gray = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.5, 0)
    if binarize:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if gray.mean() < 127:  # text should be dark on a light page
            gray = 255 - gray
    return gray


def allow_chars(cfg):
    """Characters an engine may output: cfg["allow"] if given, else WHITELIST when cfg["whitelist"]."""
    return cfg.get("allow") or (WHITELIST if cfg.get("whitelist") else None)


# ------------------------------------------------------------------ engines
def read_best(engine, crop, cfg):
    """Best read over the orientations worth trying for this crop shape."""
    h, w = crop.shape[:2]
    variants = [crop]
    if h > 1.5 * w:  # possibly rotated text: try both quarter turns
        variants += [cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    reads = [engine.read(prepare(v, cfg["height"], cfg["binarize"], cfg.get("sharpen", False)), cfg)
             for v in variants]
    return max(reads, key=lambda r: (bool(r[0]), r[1]))


# ------------------------------------------------------------------- scoring
def score_crops(engine, crops, cfg):
    cache = {}

    def one(c):
        path = c["image"]
        if path not in cache:
            cache[path] = cv2.imread(str(path))
        return read_best(engine, crop_image(cache[path], c["box"]), cfg)

    with ThreadPoolExecutor(engine.workers) as pool:
        reads = list(pool.map(one, crops))
    return [dict(c, pred=t, conf=p) for c, (t, p) in zip(crops, reads)]


def summarize(rows, charts=None):
    """{class: metrics} for scored crops; ``charts`` adds the all-labels-correct rate."""
    out = {}
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["klass"]].append(r)
    for klass, rs in sorted(by_class.items()):
        ref = [normalize(r["text"]) for r in rs]
        hyp = [normalize(r["pred"]) for r in rs]
        edits = [edit_distance(h, g) for h, g in zip(hyp, ref)]
        m = {"crops": len(rs),
             "exact_raw": sum(r["pred"].strip() == r["text"].strip() for r in rs) / len(rs),
             "exact_normalized": sum(h == g for h, g in zip(hyp, ref)) / len(rs),
             "cer_micro": sum(edits) / max(1, sum(len(g) for g in ref)),
             "cer_median": statistics.median(e / max(1, len(g)) for e, g in zip(edits, ref)),
             "mean_confidence": statistics.fmean(r["conf"] for r in rs)}
        if klass in NUMERIC:
            gt = [parse_number(r["text"]) for r in rs]
            pr = [parse_number(r["pred"]) for r in rs]
            usable = [(g, p) for g, p in zip(gt, pr) if g is not None]
            m.update(gt_parseable=len(usable),
                     parse_rate=sum(p is not None for _, p in usable) / max(1, len(usable)),
                     correct_parse_rate=sum(p is not None and math.isclose(p, g, rel_tol=1e-9, abs_tol=1e-12)
                                            for g, p in usable) / max(1, len(usable)))
            if charts is not None and klass == "ylabel":
                per_chart = defaultdict(list)
                for r, g, p in zip(rs, gt, pr):
                    if g is not None:
                        per_chart[r["chart"]].append(p is not None and math.isclose(p, g, rel_tol=1e-9, abs_tol=1e-12))
                m["charts_all_ylabels_correct"] = (sum(all(v) for v in per_chart.values()), len(per_chart))
        out[klass] = m
    return out


def tune(engine, args):
    crops, _ = load_crops(args.root, "val")
    rnd = random.Random(0)
    by_class = defaultdict(list)
    for c in crops:
        by_class[c["klass"]].append(c)
    sample = []
    for klass, cs in by_class.items():
        rnd.shuffle(cs)
        sample += cs[:args.per_class]
    print(f"{engine.name}: val vertical crops {len(crops)}; tuning on {len(sample)} ({args.per_class} per class max)")
    results = []
    for cfg0 in engine.grid():
        for wl in (False, True):
            if wl:  # whitelist only for the numeric classes
                numeric = [c for c in sample if c["klass"] in NUMERIC]
                summary = summarize(score_crops(engine, numeric, dict(cfg0, whitelist=True)))
            else:
                summary = summarize(score_crops(engine, sample, dict(cfg0, whitelist=False)))
            results.append((dict(cfg0, whitelist=wl), summary))
            print(cfg0, "whitelist" if wl else "", {k: round(v["correct_parse_rate" if k in NUMERIC else "cer_micro"], 3)
                                                     for k, v in summary.items()}, flush=True)
    chosen = {}
    for klass in {c["klass"] for c in sample}:
        key = (lambda s: -s[klass]["correct_parse_rate"]) if klass in NUMERIC else (lambda s: s[klass]["cer_micro"])
        pool = [(cfg, s) for cfg, s in results if klass in s]
        chosen[klass] = min(pool, key=lambda cs: key(cs[1]))[0]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ocr_tuning.json").write_text(json.dumps({
        "chosen": chosen, "grid": [{"config": c, "val_summary": s} for c, s in results]}, indent=1))
    print("chosen:", json.dumps(chosen))


def report(out, charts_meta):
    print(json.dumps(charts_meta))
    for name in ("all_vertical", "scope"):
        print(name)
        for klass, m in out[name].items():
            print(f"  {klass:<13}{m['crops']:>5}  exact raw {m['exact_raw']:.3f}  norm {m['exact_normalized']:.3f}  "
                  f"CER {m['cer_micro']:.3f} (median {m['cer_median']:.3f})"
                  + (f"  parse {m['parse_rate']:.3f}  correct {m['correct_parse_rate']:.3f}" if klass in NUMERIC else "")
                  + (f"  all-ylabels-correct charts {m['charts_all_ylabels_correct'][0]}/{m['charts_all_ylabels_correct'][1]}"
                     if "charts_all_ylabels_correct" in m else ""))


def _summaries(rows, charts):
    scope = [r for r in rows if charts[r["chart"]]["scope"]]
    return {"all_vertical": summarize(rows, charts), "scope": summarize(scope, charts),
            "charts": {"all_vertical": len(charts), "scope": sum(v["scope"] for v in charts.values())}}


def test(engine, args):
    result = args.out / "ocr_test.json"
    if result.exists():
        raise SystemExit(f"test already scored once: {result}")
    chosen = json.loads((args.out / "ocr_tuning.json").read_text())["chosen"]
    crops, charts = load_crops(args.root, "test")
    rows = []
    for klass in {c["klass"] for c in crops}:
        # A class with no val crops (e.g. legend_title) has no tuned setting; it uses the legend_label one.
        cfg = chosen.get(klass) or chosen["legend_label"]
        rows += score_crops(engine, [c for c in crops if c["klass"] == klass], cfg)
    out = dict(_summaries(rows, charts), chosen=chosen)
    result.write_text(json.dumps(out, indent=1))
    (args.out / "ocr_test_predictions.json").write_text(json.dumps(
        [{k: (str(v) if k == "image" else v) for k, v in r.items()} for r in rows]))
    report(out, out["charts"])


def rescore(engine, args):
    """Recompute the test tables from saved reads (no OCR is run)."""
    rows = json.loads((args.out / "ocr_test_predictions.json").read_text())
    _, charts = load_crops(args.root, "test")
    out = _summaries(rows, charts)
    (args.out / "ocr_test_rescored.json").write_text(json.dumps(out, indent=1))
    report(out, out["charts"])


def predict_split(engine, args):
    """Save reads for a split with the settings chosen by ``tune``; nothing is scored.

    Used to get val reads for choosing thresholds in later steps (test has its own command).
    """
    if args.split == "test":
        raise SystemExit("test reads come from the `test` command, which scores it once")
    chosen = json.loads((args.out / "ocr_tuning.json").read_text())["chosen"]
    crops, _ = load_crops(args.root, args.split)
    rows = []
    for klass in {c["klass"] for c in crops}:
        rows += score_crops(engine, [c for c in crops if c["klass"] == klass],
                            chosen.get(klass) or chosen["legend_label"])
    (args.out / f"ocr_{args.split}_predictions.json").write_text(json.dumps(
        [{k: (str(v) if k == "image" else v) for k, v in r.items()} for r in rows]))
    print(f"{engine.name}: saved {len(rows)} {args.split} reads")


def main(engine):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("tune", "test", "rescore", "predict"))
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--split", default="val", help="split for the predict command (not test)")
    args = ap.parse_args()
    {"tune": tune, "test": test, "rescore": rescore, "predict": predict_split}[args.command](engine, args)
