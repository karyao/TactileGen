"""Stage 1 of the end-to-end slice: ground-truth boxes and ground-truth text in.

Runs the interpreter (src/interpret_bars.py) on the vertical, single-series charts with
at most 10 bars of a split of data/pmc_yolo_bars (same scope subset as the OCR scoring).
Nothing here is learned or tuned, so errors come from the interpreter's logic, not from
detection or OCR. Reports status counts, flag reasons, pairing and category-name
checks, axis agreement with the labels, and bar-value error where every bar has a
printed number.

  python -m src.baseline.e2e_stage1 --split val --output <json>
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.axis_fit import pair_by_overlap
from src.baseline.ocr_eval import COCO, COCO_FILES, load_crops
from src.dataset.pmc_frames import ground_truth_frames
from src.interpret_bars import interpret_vertical_bars
from src.textutil import clean_text, parse_number

TOL = 0.01  # value error as a fraction of the tick span


def scope_charts(root: Path, split: str):
    """[(image dict, annotations)] for the scope subset of a split."""
    _, charts = load_crops(root, split)
    scope = {n for n, v in charts.items() if v["scope"]}
    out = []
    for coco_file in COCO_FILES.values():
        coco = json.loads((COCO / coco_file).read_text())
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        out += [(im, by_image[im["id"]]) for im in coco["images"] if im["file_name"] in scope]
    return sorted(out, key=lambda t: t[0]["file_name"])


def _ground_truth_checks(image, annotations, chart):
    """Pairing and category-name checks against independent ground truth, for one built chart."""
    cat = defaultdict(list)
    for a in annotations:
        cat[a["category_id"]].append(a)
    bars = sorted(cat[19], key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
    labels = sorted(cat[5], key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
    ticks = sorted(cat[8], key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
    cx = lambda a: a["bbox"][0] + a["bbox"][2] / 2
    out = {"bars_match_categories": len(chart.categories) == len(bars)}
    if len(ticks) == len(bars) == len(labels):
        pitch = statistics.median(cx(b) - cx(a) for a, b in zip(bars, bars[1:])) if len(bars) > 1 else 1
        out["bars_aligned_with_ticks"] = all(abs(cx(t) - cx(b)) <= 0.5 * pitch for t, b in zip(ticks, bars))
        # An independent route: the nearest tick mark to each bar, then the label nearest to that tick
        # once the constant offset of rotated labels (their centre sits left of the tick) is removed.
        offset = statistics.median(cx(l) - cx(t) for l, t in zip(labels, ticks))
        bar_tick = [min(range(len(ticks)), key=lambda t: abs(cx(ticks[t]) - cx(b))) for b in bars]
        tick_label = [min(range(len(labels)), key=lambda l: abs(cx(labels[l]) - offset - cx(t))) for t in ticks]
        via_ticks = [clean_text(labels[tick_label[bar_tick[i]]]["text"]) for i in range(len(bars))]
        out["pairing_agrees"] = list(chart.categories) == via_ticks
    return out


def _value_error(chart, annotations):
    """Per-bar errors against printed numbers; None unless every bar has a numeric label."""
    cat = defaultdict(list)
    for a in annotations:
        cat[a["category_id"]].append(a)
    bars = sorted(cat[19], key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
    labels = cat[15]
    if len(labels) != len(bars) or any(parse_number(a["text"]) is None for a in labels):
        return None
    pairs = pair_by_overlap([a["bbox"] for a in labels], [a["bbox"] for a in bars])
    if len(pairs) != len(bars):
        return None
    ticks = chart.y_tick_values
    span = ticks[-1] - ticks[0]
    errs = []
    for i, value in enumerate(chart.series[0].values):
        printed = parse_number(labels[pairs[i]]["text"])
        errs.append({"printed": printed, "value": value, "error": abs(value - printed), "span_fraction": abs(value - printed) / span,
                     "log_ratio": abs(math.log10(abs(value / printed))) if value and printed else None})
    return errs


def run(root, split):
    results, rows = Counter(), []
    reasons = Counter()
    checks = Counter()
    value_charts = []
    for image, annotations in scope_charts(root, split):
        key = f"{split}/{image['file_name']}"
        det, ocr = ground_truth_frames(key, image, annotations)
        result = interpret_vertical_bars(det, ocr)
        results[result.status] += 1
        for reason in result.reasons:
            reasons[reason] += 1
        for item in result.items:
            if item.flagged:
                reasons[f"item {item.kind}: {item.reason}"] += 1
        row = {"chart": image["file_name"], "status": result.status, "reasons": list(result.reasons),
               "flagged_items": [(i.kind, i.reason) for i in result.items if i.flagged]}
        if result.chart is not None:
            results["chart built"] += 1
            for name, ok in _ground_truth_checks(image, annotations, result.chart).items():
                checks[name + (" true" if ok else " false")] += 1
            errs = _value_error(result.chart, annotations)
            if errs:
                value_charts.append({"chart": image["file_name"], "status": result.status, "bars": errs})
        rows.append(row)
    summary = {"charts": len(rows), "status": {k: v for k, v in results.items()}, "reasons": dict(reasons.most_common()),
               "checks": dict(checks)}
    def value_summary(charts):
        bars = [e for c in charts for e in c["bars"]]
        if not bars:
            return None
        return {"charts": len(charts), "bars": len(bars),
                "charts_all_bars_within_1pct_of_span": sum(all(e["span_fraction"] <= TOL for e in c["bars"]) for c in charts),
                "bars_within_1pct_of_span": sum(e["span_fraction"] <= TOL for e in bars),
                "median_span_fraction": statistics.median(e["span_fraction"] for e in bars),
                "max_span_fraction": max(e["span_fraction"] for e in bars),
                "decade_errors": sum(e["log_ratio"] is not None and e["log_ratio"] > 0.5 for e in bars)}
    summary["value_error_all"] = value_summary(value_charts)
    summary["value_error_accepted"] = value_summary([c for c in value_charts if c["status"] == "accepted"])
    return summary, rows, value_charts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    summary, rows, value_charts = run(args.root, args.split)
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "charts": rows, "value_charts": value_charts}, indent=1))
    for r in rows:
        if r["status"] != "accepted":
            print(r["status"], r["chart"][:40], r["reasons"], r["flagged_items"][:3])


if __name__ == "__main__":
    main()
