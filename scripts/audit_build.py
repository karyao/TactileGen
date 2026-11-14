"""Build results/failure_audit.csv from the per-chart cache written by scripts/audit_cache.py.

usage: python scripts/audit_build.py CACHE_DIR TREE OUT_CSV
Rates (all over the cached charts):
  (a) emitted without blocking = exit 0
  (b) of emitted charts, share whose content matches ground truth
  (c) share of all charts that are emitted AND correct
"Correct" = series count, category count, category text, series names and every value
(|pred-gt| <= 1% of the GT y-axis maximum) match. Titles and axis titles are not compared.
"""

from __future__ import annotations

import csv
import math
import pickle
import re
import sys
from pathlib import Path

TREE = Path(sys.argv[2])
sys.path.insert(0, str(TREE))
sys.dont_write_bytecode = True
from src.reconstruct import COLOR_DISTANCE, MIN_CONFIDENCE, _centre, _dedupe, _inside  # noqa: E402
from src.tactile.completeness import check_completeness  # noqa: E402

VALUE_TOL = 0.01     # fraction of the GT y-axis maximum
IOU = 0.5

# Priority order for the single primary cause (first match wins).
CAUSE_ORDER = [
    "no_spec_interpretation_error", "detector_missing_bars", "detector_extra_bars",
    "ocr_category_text", "ocr_title_or_axis_title", "y_calibration", "series_false_alarm_single_series",
    "series_separation", "category_pairing", "bar_rescue_wrong", "other_wrong_values",
    "strict_bana_layout", "false_block", "none_correct",
]


def norm(text) -> str:
    return " ".join(str(text).lower().split())


def hex_rgb(value):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def xyxy_det(d):
    return (d.box.x1, d.box.y1, d.box.x2, d.box.y2)


def gt_bars(record):
    bars = []
    for k, model in enumerate(record["models"]):
        for j, (box, value) in enumerate(zip(model["bboxes"], model["y"])):
            xs = sorted((box["x"], box["x"] + box["w"]))
            ys = sorted((box["y"], box["y"] + box["h"]))
            bars.append({"box": (xs[0], ys[0], xs[1], ys[1]), "value": value, "series": k, "cat": j})
    return bars


def surviving_bars(frame):
    """Bars that reach reconstruct's series/category logic (same filters as reconstruct_vbar)."""
    raw = [d for d in frame.detections if d.role == "vertical_bar"]
    kept = _dedupe([d for d in raw if d.confidence >= MIN_CONFIDENCE])
    plot = max((d for d in frame.detections if d.role == "plot_area" and d.confidence >= MIN_CONFIDENCE),
               key=lambda d: d.confidence, default=None)
    return raw, [d for d in kept if plot is None or _inside(_centre(d), plot)]


def match_bars(dets, truth):
    """Greedy one-to-one IoU matching. Returns (det_idx -> gt_idx, unmatched gt idx list)."""
    pairs = sorted(((iou(xyxy_det(d), t["box"]), i, j) for i, d in enumerate(dets)
                    for j, t in enumerate(truth)), reverse=True)
    used_d, used_t, m = set(), set(), {}
    for v, i, j in pairs:
        if v < IOU:
            break
        if i in used_d or j in used_t:
            continue
        used_d.add(i); used_t.add(j); m[i] = j
    return m, [j for j in range(len(truth)) if j not in used_t]


def match_series(spec, models):
    """Pair predicted series with GT models: colour, then name, one-to-one. -> list of (pred_idx, gt_idx)."""
    pairs, used = [], set()
    if len(spec.series) == 1 and len(models) == 1:
        return [(0, 0)]
    for i, s in enumerate(spec.series):
        best = None
        for k, m in enumerate(models):
            if k in used:
                continue
            d = None
            if s.source_color and isinstance(m.get("colors"), str):
                d = math.dist(hex_rgb(s.source_color), hex_rgb(m["colors"]))
            name_eq = s.name is not None and norm(s.name) == norm(m.get("name", ""))
            if (d is not None and d <= 2 * COLOR_DISTANCE) or name_eq:
                score = (0 if name_eq else 1, d if d is not None else 1e9)
                if best is None or score < best[0]:
                    best = (score, k)
        if best:
            used.add(best[1]); pairs.append((i, best[1]))
    return pairs


def block_reasons(stdout: str, exit_code=None):
    tags = re.findall(r"\((\w+)\)", " ".join(re.findall(r"^  - .*$", stdout, re.M)))
    needs = ([re.sub(r"\[\d+\]", "[n]", f) for f in re.findall(r"^  ([\w\[\]]+): ", stdout, re.M)]
             if exit_code == 2 else [])
    tags = tags if exit_code == 3 else []
    bana = sorted(set(re.findall(r"^\s+\[(\w+)\] ", stdout, re.M)) - {"overview_first"})
    return tags, needs, bana


def analyse(r):
    record, spec, frame = r["record"], r.get("spec"), r["frame"]
    models = record["models"]
    row = {"chart_id": r["id"], "exit": r["exit"]}
    row["status"] = "emitted" if r["exit"] == 0 else "blocked"
    tags, needs, bana = block_reasons(r["stdout"], r["exit"])
    row["block_reasons"] = ";".join(sorted(set(tags)) + sorted(set(needs)) + [f"bana:{b}" for b in bana])
    if r["exit"] == 1 and not bana and "--strict" not in r["stderr"]:
        row["block_reasons"] = "error:" + (r["stderr"].strip().splitlines() or ["?"])[-1][:80]
    flags = list(spec.flags) if spec else []
    row["flags"] = ";".join(flags)
    row["flag_names"] = ";".join(sorted({f.split(":")[0] for f in flags}))

    # ground truth
    gt_cats = models[0]["x"]
    # PlotQA repeats the tick list once per series; compare the distinct sorted values
    ticks = sorted({float(t) for t in record["general_figure_info"]["y_axis"]["major_ticks"]["values"]})
    gt_vals = [v for m in models for v in m["y"] if v is not None]
    axis_max = max(ticks) if ticks else (max(abs(v) for v in gt_vals) or 1.0)
    if axis_max <= 0:
        axis_max = max((abs(v) for v in gt_vals), default=1.0) or 1.0
    truth = gt_bars(record)
    row.update(gt_series=len(models), gt_categories=len(gt_cats), gt_bars=len(truth),
               gt_axis_max=axis_max, gt_series_names="|".join(str(m.get("name")) for m in models),
               gt_categories_text="|".join(map(str, gt_cats)))

    # detector stage
    raw, surv = surviving_bars(frame)
    m, missed = match_bars(surv, truth)
    raw_m, raw_missed = match_bars(raw, truth)
    extras = [i for i in range(len(surv)) if i not in m]
    row.update(det_bars_raw=len(raw), det_bars_used=len(surv), det_missing_gt=len(raw_missed) if False else len(missed),
               det_not_detected_raw=len(raw_missed), det_extra=len(extras))
    # why each missed GT bar was lost: never detected / dropped below MIN_CONFIDENCE / dedupe or plot-area filter
    stage_counts = {"not_detected": 0, "low_conf": 0, "dedupe_or_plot": 0}
    near_zero = 0
    for j in missed:
        t = truth[j]
        best = max(raw, key=lambda d: iou(xyxy_det(d), t["box"]), default=None)
        if best is None or iou(xyxy_det(best), t["box"]) < IOU:
            stage_counts["not_detected"] += 1
        elif best.confidence < MIN_CONFIDENCE:
            stage_counts["low_conf"] += 1
        else:
            stage_counts["dedupe_or_plot"] += 1
        near_zero += (t["value"] is not None and abs(t["value"]) < 0.05 * axis_max) or (t["box"][3] - t["box"][1]) <= 5
    row.update(missed_not_detected=stage_counts["not_detected"], missed_low_conf=stage_counts["low_conf"],
               missed_dedupe_or_plot=stage_counts["dedupe_or_plot"], missed_near_zero=near_zero)
    row["extra_conf"] = ";".join(f"{surv[i].confidence:.3f}" for i in extras)
    row["true_conf"] = ";".join(f"{surv[i].confidence:.3f}" for i in m)

    # prediction vs ground truth
    row.update(pred_series="", pred_categories="", pred_bars="", pred_series_names="", pred_categories_text="",
               count_ok="", series_ok="", labels_ok="", names_ok="", values_ok="", max_err_frac="", n_bad_values="",
               rescued_bad=0, ytick_ok="")
    cause_checks = {}
    if spec is not None:
        pb = sum(v is not None for s in spec.series for v in s.values)
        row.update(pred_series=len(spec.series), pred_categories=len(spec.categories), pred_bars=pb,
                   pred_series_names="|".join(str(s.name) for s in spec.series),
                   pred_categories_text="|".join(spec.categories))
        count_ok = len(spec.categories) == len(gt_cats)
        pairs = match_series(spec, models)
        series_ok = len(spec.series) == len(models) and len(pairs) == len(models)
        labels_ok = [norm(c) for c in spec.categories] == [norm(c) for c in gt_cats]
        names_ok = series_ok and all(
            (spec.series[i].name is not None and norm(spec.series[i].name) == norm(models[k].get("name", "")))
            or (spec.series[i].name is None and len(models) == 1) for i, k in pairs)
        errs, bad, rescued_bad = [], 0, 0
        if count_ok:
            for i, k in pairs:
                for j, (p, t) in enumerate(zip(spec.series[i].values, models[k]["y"])):
                    if t is None:
                        continue
                    e = math.inf if p is None else abs(p - t) / axis_max
                    errs.append(e)
                    if e > VALUE_TOL:
                        bad += 1
                        if spec.series[i].value_rescued and spec.series[i].value_rescued[j]:
                            rescued_bad += 1
        values_ok = count_ok and series_ok and bool(errs) and bad == 0
        loose = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())
        row["labels_ok_loose"] = [loose(c) for c in spec.categories] == [loose(c) for c in gt_cats]
        row.update(count_ok=count_ok, series_ok=series_ok, labels_ok=labels_ok, names_ok=names_ok,
                   values_ok=values_ok, n_bad_values=bad if count_ok else "", rescued_bad=rescued_bad,
                   max_err_frac=("inf" if errs and math.isinf(max(errs)) else f"{max(errs):.4f}") if errs else "")
        got = sorted({round(float(t), 6) for t in spec.y_axis.tick_values})
        row["ytick_ok"] = got == [round(t, 6) for t in ticks]
        row["spec_value_errors"] = ";".join("inf" if math.isinf(e) else f"{e:.4f}" for e in errs)
        row["correct"] = bool(count_ok and series_ok and labels_ok and names_ok and values_ok)
        row["correct_values_only"] = bool(values_ok)
        try:
            comp = check_completeness(spec)
            row["completeness"] = comp.status
        except Exception as exc:  # noqa: BLE001
            row["completeness"] = f"error:{exc}"
        cause_checks = dict(count_ok=count_ok, series_ok=series_ok, labels_ok=labels_ok,
                            values_ok=values_ok, rescued_bad=rescued_bad)
    else:
        row.update(correct=False, correct_values_only=False, completeness="no_spec", spec_value_errors="")
        row["block_reasons"] = row["block_reasons"] or "interpretation_error"
        row["interp_error"] = r.get("error", "")[:120]
    row.setdefault("interp_error", "")
    row["emitted_and_correct"] = bool(row["status"] == "emitted" and row["correct"])
    row["false_block"] = bool(row["status"] == "blocked" and spec is not None and row["correct"]
                              and r["exit"] in (1, 3))

    # first stage that went wrong + primary cause (priority order = CAUSE_ORDER)
    fl = {f.split(":")[0] for f in flags}
    cause = "none_correct"
    if row["emitted_and_correct"]:
        cause = "none_correct"
    elif row["false_block"]:
        cause = "false_block"
    elif spec is None:
        cause = "no_spec_interpretation_error"
    elif missed:
        cause = "detector_missing_bars"
    elif extras:
        cause = "detector_extra_bars"
    elif (not cause_checks["labels_ok"] and cause_checks["count_ok"]) or "category_unreadable" in fl \
            or any(n.startswith("category") for n in needs):
        cause = "ocr_category_text"
    elif any(n in ("title", "y_axis_title", "x_axis_title") or n.startswith("series_name") for n in needs):
        cause = "ocr_title_or_axis_title"
    elif fl & {"y_calibration_ambiguous", "y_tick_leading_zero", "y_ticks_not_arithmetic", "value_clipped_at_axis",
               "baseline_outside_plot"} or row["ytick_ok"] is False:
        cause = "y_calibration"
    elif "series_ambiguous" in fl and len(models) == 1:
        cause = "series_false_alarm_single_series"
    elif "series_ambiguous" in fl or not cause_checks["series_ok"]:
        cause = "series_separation"
    elif fl & {"category_count_mismatch", "category_ambiguous"} or not cause_checks["count_ok"]:
        cause = "category_pairing"
    elif cause_checks["rescued_bad"]:
        cause = "bar_rescue_wrong"
    elif not cause_checks["values_ok"]:
        cause = "other_wrong_values"
    elif bana:
        cause = "strict_bana_layout"
    elif not row["correct"]:
        cause = "other_wrong_values"
    row["primary_cause"] = cause
    stage = {"no_spec_interpretation_error": "detector/reconstruct", "detector_missing_bars": "detector",
             "detector_extra_bars": "detector", "ocr_category_text": "OCR", "ocr_title_or_axis_title": "OCR",
             "y_calibration": "calibration", "series_false_alarm_single_series": "series separation",
             "series_separation": "series separation", "category_pairing": "category pairing",
             "bar_rescue_wrong": "rescue", "other_wrong_values": "values (unattributed)",
             "strict_bana_layout": "layout (post-reconstruction)", "false_block": "completeness/BANA gate",
             "none_correct": ""}[cause]
    row["first_bad_stage"] = stage
    return row


def main():
    cache, out = Path(sys.argv[1]), Path(sys.argv[3])
    rows = []
    for f in sorted(cache.glob("*.pkl")):
        rows.append(analyse(pickle.loads(f.read_bytes())))
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    em = sum(r["status"] == "emitted" for r in rows)
    ec = sum(r["emitted_and_correct"] for r in rows)
    print(f"n={n} (a) emitted without blocking {em}/{n}={em / n:.1%}; (b) emitted & correct / emitted {ec}/{em}="
          f"{ec / em:.1%}; (c) emitted & correct / all {ec}/{n}={ec / n:.1%}")


if __name__ == "__main__":
    main()
