"""Independent checks for a wrong value-axis scale (e.g. a dropped decimal point).

The fit in ocr_axis_fit.py is not changed. These checks only add reasons to flag a
chart that the fit accepted:

  a_value_labels  fitted bar values vs the numbers printed on the bars (OCR reads of
                  ``value_label`` boxes paired to bars by horizontal overlap); flagged
                  when the median |log10(printed / fitted)| exceeds a threshold
  b2_decimals     the tick reads do not use one number of decimal places
                  (strict) or the non-integer reads disagree (lenient)
  c_unit_suffix   some tick reads end in %, others do not (a % read as 6 or 8)
  c_title_percent the y-axis title read mentions % or percent but the fitted ticks
                  run past a cap

The nice-step check (fitted step not 1, 2, 2.5 or 5 times a power of ten) was measured
and dropped: it cannot see a pure scale error (10x a nice step is still nice) and caught
nothing. Only c_unit_suffix and a_value_labels are adopted (ADOPTED); b2_decimals and
c_title_percent are kept here as measured, not adopted.

Thresholds are chosen on val only, pooled over both engines: the most sensitive
setting whose false-flag rate on val charts the fit already got right is at most 10%.
Test is then scored once with those choices.

  python -m src.baseline.ocr_axis_checks --output <json>
"""
import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from src.axis_fit import pair_by_overlap
from src.baseline.ocr_axis_fit import analyse, tick_positions
from src.baseline.ocr_eval import COCO, COCO_FILES, normalize, parse_number

MAX_FALSE_FLAG_RATE = 0.10
TOL = 0.01  # a fit "matches" the reference within 1% slope and offset
GRIDS = {"a_value_labels": [0.05, 0.1, 0.2, 0.3, 0.5, 0.9],        # median |log10 ratio|, most sensitive first
         "b2_decimals": ["strict", "lenient"],
         "c_title_percent": [100, 120, 150, 200]}                  # cap on fitted tick values
FIXED = ("c_unit_suffix",)
ADOPTED = ("c_unit_suffix", "a_value_labels")
ENGINES = ("tesseract", "easyocr")


def bars_by_image(root, split):
    """{image: ground-truth bar boxes}"""
    images = {p.name for p in (root / "images" / split).iterdir()}
    out = {}
    for coco_file in COCO_FILES.values():
        coco = json.loads((COCO / coco_file).read_text())
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            if a["category_id"] == 19:
                by_image[a["image_id"]].append(a["bbox"])
        for im in coco["images"]:
            if im["file_name"] in images:
                out[im["file_name"]] = by_image[im["id"]]
    return out


def decimals(read):
    m = re.fullmatch(r"[-+]?\d*[.,](\d+)", normalize(read).replace(" ", "").rstrip("%"))
    return len(m.group(1)) if m else 0


def features(entry, rows_of_chart, bars):
    """Numbers each check needs, from one chart's fit and OCR reads."""
    a, b = entry["fit_a"], entry["fit_b"]
    ys = entry["tick_ys"]
    reads = [r["pred"] for r in rows_of_chart if r["klass"] == "ylabel" and r["pred"].strip()]
    f = {"reads": reads, "top_value": max(abs(a + b * y) for y in ys)}
    titles = [r["pred"] for r in rows_of_chart if r["klass"] == "y_title"]
    f["title_percent"] = any("%" in t or "percent" in t.lower() for t in titles)
    labels = [r for r in rows_of_chart if r["klass"] == "value_label"]
    pairs = pair_by_overlap([r["box"] for r in labels], bars)
    logs = []
    for bi, li in pairs.items():
        printed = parse_number(labels[li]["pred"])
        _, y1, _, h = bars[bi]
        fitted = max((a + b * y1, a + b * (y1 + h)), key=abs)
        if printed and fitted:
            logs.append(abs(math.log10(abs(printed / fitted))))
    f["label_log_error"] = statistics.median(logs) if len(logs) >= 2 else None
    return f


def flags(f, params):
    """{check: True/False/None}; None when the check cannot be evaluated for this chart."""
    out = {}
    reads = f["reads"]
    kinds = {decimals(r) for r in reads if parse_number(r) is not None}
    nonint = {decimals(r) for r in reads if parse_number(r) is not None and decimals(r) > 0}
    out["b2_decimals"] = None if not kinds else (len(kinds) > 1 if params["b2_decimals"] == "strict" else len(nonint) > 1)
    out["c_unit_suffix"] = None if not reads else (any(r.strip().endswith("%") for r in reads)
                                                   and not all(r.strip().endswith("%") for r in reads))
    out["c_title_percent"] = (f["top_value"] > params["c_title_percent"]) if f["title_percent"] else None
    out["a_value_labels"] = None if f["label_log_error"] is None else f["label_log_error"] > params["a_value_labels"]
    return out


def load_charts(engine, split, root):
    rows = json.loads(Path(f"output/ocr/{engine}/ocr_{split}_predictions.json").read_text())
    by_chart = defaultdict(list)
    for r in rows:
        by_chart[r["chart"]].append(r)
    bars = bars_by_image(root, split)
    out = []
    for e in analyse(rows, tick_positions(root, split))["charts"]:
        if "fit_a" not in e or e["flagged"]:
            continue  # only charts the fit accepted can gain a new flag
        matches = e["slope_rel_err"] <= TOL and e["offset_err"] <= TOL
        out.append({"engine": engine, "chart": e["chart"], "correct": matches,
                    "features": features(e, by_chart[e["chart"]], bars.get(e["chart"], []))})
    return out


def evaluate(charts, params):
    names = list(GRIDS) + list(FIXED)
    stats = {n: {"available": 0, "silent_caught": 0, "silent_total": 0, "false_flags": 0, "correct_total": 0} for n in names}
    stats["adopted"] = {"available": len(charts), "silent_caught": 0, "silent_total": 0, "false_flags": 0, "correct_total": 0}
    for c in charts:
        fl = flags(c["features"], params)
        hit = [n for n in ADOPTED if fl[n]]
        for n in names:
            s = stats[n]
            s["available"] += fl[n] is not None
            if c["correct"]:
                s["correct_total"] += 1
                s["false_flags"] += bool(fl[n])
            else:
                s["silent_total"] += 1
                s["silent_caught"] += bool(fl[n])
        s = stats["adopted"]
        if c["correct"]:
            s["correct_total"] += 1
            s["false_flags"] += bool(hit)
        else:
            s["silent_total"] += 1
            s["silent_caught"] += bool(hit)
    return stats


def choose(val_charts):
    """Per tunable check, the most sensitive setting with val false-flag rate <= MAX_FALSE_FLAG_RATE."""
    params = {k: v[-1] for k, v in GRIDS.items()}
    report = {}
    for name, grid in GRIDS.items():
        report[name] = []
        picked = None
        for value in grid:
            s = evaluate(val_charts, dict(params, **{name: value}))[name]
            rate = s["false_flags"] / max(1, s["correct_total"])
            report[name].append({"value": value, "val_false_flag_rate": rate, "val_available": s["available"]})
            if picked is None and rate <= MAX_FALSE_FLAG_RATE:
                picked = value
        params[name] = picked if picked is not None else grid[-1]
        report[name + "_adopted"] = picked is not None
    return params, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    val = [c for e in ENGINES for c in load_charts(e, "val", args.root)]
    params, report = choose(val)
    print("chosen on val:", params)
    for name in GRIDS:
        print(" ", name, [(r["value"], round(r["val_false_flag_rate"], 3)) for r in report[name]], "adopted" if report[name + "_adopted"] else "NOT adoptable (cost over cap at every setting)")
    out = {"params": params, "val_choice": report, "val": {}, "test": {}}
    for split in ("val", "test"):
        for engine in ENGINES:
            charts = [c for c in (val if split == "val" else [c for e in ENGINES for c in load_charts(e, "test", args.root)]) if c["engine"] == engine]
            out[split][engine] = evaluate(charts, params)
            print(f"\n{split} {engine}: fit-accepted charts {len(charts)}")
            print(f"  {'check':<18}{'available':>10}{'silent caught':>15}{'false flags':>13}")
            for n, s in out[split][engine].items():
                print(f"  {n:<18}{s['available']:>10}{s['silent_caught']:>9}/{s['silent_total']:<5}{s['false_flags']:>8}/{s['correct_total']:<4}")
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
