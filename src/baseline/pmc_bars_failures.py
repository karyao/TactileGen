"""Why does a PMC bar model get the bar count wrong? (diagnosis only)

Predicts bar_test with one checkpoint at a fixed confidence, matches boxes at IoU 0.5, and for
every image whose predicted count differs from the true count says whether bars were missed,
extra boxes were predicted, or both. Failures are broken down by the number of true bars,
orientation (src/dataset/bar_orientation.py) and two proxies for the series structure, and a
montage of failures is saved (green: ground truth, red: predictions).

This reads the test split to understand errors. Nothing here chooses a model, a threshold or
a setting; do not tune against it.

  python -m src.baseline.pmc_bars_failures --output-dir <dir>
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from src.baseline.opencv_bars import intersection_over_union, match_boxes
from src.baseline.pmc_bars_baseline import load_split, predict
from src.dataset.bar_orientation import BAR, read_label_boxes

WEIGHTS = Path("output/pmc_bars_finetune/coco/weights/best.pt")
CONFIDENCE = 0.40
LEGEND_PATCH, XLABEL = 9, 4  # class ids in data/pmc_yolo_bars/data.yaml
BAR_BINS = (("1-3", 3), ("4-6", 6), ("7-10", 10), ("11-20", 20), ("21+", 10 ** 9))


def kind(r):
    return "both" if r["fn"] and r["fp"] else "missed only" if r["fn"] else "extra only"


def bar_bin(n):
    return next(label for label, top in BAR_BINS if n <= top)


def collect(root, weights, confidence):
    from ultralytics import YOLO
    items = load_split(root, "test")
    predictions = predict(YOLO(str(weights)), [p for p, _, _ in items], BAR)
    rows = []
    for path, orient, truth in items:
        with Image.open(path) as im:
            size = im.size
        boxes = read_label_boxes(root / "labels" / "test" / (path.stem + ".txt"), size)
        found = [box for c, box in predictions[path] if c >= confidence]
        tp = match_boxes(found, truth, 0.5)
        rows.append({"path": path, "orientation": orient, "n_true": len(truth), "n_pred": len(found), "fn": len(truth) - tp,
                     "fp": len(found) - tp, "series": "multi" if len(boxes.get(LEGEND_PATCH, [])) >= 2 else "single",
                     "grouped": len(truth) > len(boxes.get(XLABEL, [])) > 0, "truth": truth, "found": found})
    return rows


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for name, rs in groups.items():
        wrong = [r for r in rs if r["n_true"] != r["n_pred"]]
        kinds = Counter(kind(r) for r in wrong)
        out[str(name)] = {"images": len(rs), "wrong": len(wrong), "rate": len(wrong) / len(rs),
                          "missed only": kinds["missed only"], "extra only": kinds["extra only"], "both": kinds["both"]}
    return out


def analyse(rows):
    wrong = [r for r in rows if r["n_true"] != r["n_pred"]]
    diffs = [r["n_pred"] - r["n_true"] for r in wrong]
    zero_fn = partial_fn = zero_fp = partial_fp = 0
    for r in wrong:
        for t in r["truth"]:
            best = max((intersection_over_union(q, t) for q in r["found"]), default=0)
            if best < 0.5:
                zero_fn, partial_fn = zero_fn + (best == 0), partial_fn + (best > 0)
        for q in r["found"]:
            best = max((intersection_over_union(q, t) for t in r["truth"]), default=0)
            if best < 0.5:
                zero_fp, partial_fp = zero_fp + (best == 0), partial_fp + (best > 0)
    series_order = ("single", "multi")
    return {
        "images": len(rows), "wrong": len(wrong), "wrong_rate": len(wrong) / len(rows),
        "failure_type": dict(Counter(kind(r) for r in wrong)),
        "net": {"undercount": sum(d < 0 for d in diffs), "overcount": sum(d > 0 for d in diffs)},
        "count_equal_but_boxes_mismatch": sum(r["n_true"] == r["n_pred"] and r["fn"] > 0 for r in rows),
        "median_pred_minus_true": statistics.median(diffs), "off_by_one": sum(abs(d) == 1 for d in diffs),
        "off_by_five_or_more": sum(abs(d) >= 5 for d in diffs),
        "by_true_bars": breakdown(rows, lambda r: bar_bin(r["n_true"])),
        "by_orientation": breakdown(rows, lambda r: r["orientation"]),
        "by_series_proxy": breakdown(rows, lambda r: r["series"]),
        "by_bars_exceed_xlabels": breakdown(rows, lambda r: "grouped/stacked?" if r["grouped"] else "one bar per label"),
        "by_orientation_and_series": breakdown(rows, lambda r: f"{r['orientation'][:5]}/{r['series']}"),
        "unmatched_in_all_images": {"missed_true_bars": sum(r["fn"] for r in rows), "unmatched_predictions": sum(r["fp"] for r in rows)},
        "unmatched_in_wrong_images": {"missed_true_bars": sum(r["fn"] for r in wrong), "unmatched_predictions": sum(r["fp"] for r in wrong),
                                      "missed_with_no_overlap": zero_fn, "missed_with_partial_overlap": partial_fn,
                                      "extra_with_no_overlap": zero_fp, "extra_with_partial_overlap": partial_fp},
        "median_true_bars": {"wrong": statistics.median(r["n_true"] for r in wrong),
                             "right": statistics.median(r["n_true"] for r in rows if r["n_true"] == r["n_pred"])},
    }


def montage(rows, path, seed=0, per_kind=4):
    rnd = random.Random(seed)
    picks = []
    for k in ("missed only", "extra only", "both"):
        pool = [r for r in rows if r["n_true"] != r["n_pred"] and kind(r) == k]
        rnd.shuffle(pool)
        picks += pool[:per_kind]
    tiles = []
    for r in picks:
        im = Image.open(r["path"]).convert("RGB")
        draw = ImageDraw.Draw(im)
        w = max(2, im.width // 300)
        for t in r["truth"]:
            draw.rectangle((t.x1, t.y1, t.x2, t.y2), outline=(0, 170, 0), width=w)
        for q in r["found"]:
            draw.rectangle((q.x1, q.y1, q.x2, q.y2), outline=(230, 0, 0), width=w)
        im = im.resize((480, int(480 * im.height / im.width)))
        ImageDraw.Draw(im).rectangle((0, 0, 479, 14), fill=(255, 255, 255))
        ImageDraw.Draw(im).text((2, 1), f"{kind(r)} true={r['n_true']} pred={r['n_pred']} {r['orientation'][:5]} {r['path'].stem[:22]}", fill=(0, 0, 0))
        tiles.append(im)
    height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (4 * 480, 3 * height), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 480, (i // 4) * height))
    sheet.save(path)


def print_table(title, table):
    print(f"\n{title}")
    print(f"  {'group':<20}{'images':>7}{'wrong':>7}{'rate':>6}{'missed only':>12}{'extra only':>11}{'both':>6}")
    for name, v in table.items():
        print(f"  {name:<20}{v['images']:>7}{v['wrong']:>7}{v['rate']:>6.0%}{v['missed only']:>12}{v['extra only']:>11}{v['both']:>6}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--confidence", type=float, default=CONFIDENCE)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.root, args.weights, args.confidence)
    summary = analyse(rows)
    montage(rows, args.output_dir / "failure_montage.png")
    (args.output_dir / "failures.json").write_text(json.dumps(summary, indent=1))
    print(f"images {summary['images']}, wrong count {summary['wrong']} ({summary['wrong_rate']:.1%}); type {summary['failure_type']}; net {summary['net']}")
    for key, title in (("by_true_bars", "by number of true bars"), ("by_orientation", "by orientation"),
                       ("by_series_proxy", "by series (two or more legend patches = multi; inferred)"),
                       ("by_bars_exceed_xlabels", "by bars > x-labels (grouped or stacked proxy)"),
                       ("by_orientation_and_series", "orientation x series")):
        print_table(title, summary[key])
    print("\nunmatched in wrong images:", summary["unmatched_in_wrong_images"])


if __name__ == "__main__":
    main()
