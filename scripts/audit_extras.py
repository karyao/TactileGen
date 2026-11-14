"""series_ambiguous breakdown and the extra-bar removal rule sweep, from the audit cache.

usage: python scripts/audit_extras.py CACHE_DIR TREE OUT_DIR
Writes OUT_DIR/series_ambiguous.csv, OUT_DIR/extra_bar_conf.csv, OUT_DIR/extra_bar_rule_sweep.csv.

The rule is evaluated offline by re-running legend matching + reconstruct_vbar from the cached
detections/OCR with the removed bars filtered out (no detector re-run). It is not wired into the
pipeline. A chart "passes" when reconstruct's completeness status is READY and the content matches
ground truth (same `correct` definition as scripts/audit_build.py, but before layout/BANA).
Threshold is picked on the 20% tuning split (fixed seed over sorted chart ids) and reported on the 80%.
"""

from __future__ import annotations

import csv
import math
import pickle
import random
import sys
from dataclasses import replace
from pathlib import Path

CACHE, TREE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.argv = [sys.argv[0], str(CACHE), str(TREE), str(OUT / "unused.csv")]
sys.path.insert(0, str(Path(__file__).parent))
import audit_build as ab  # noqa: E402  (also puts TREE on sys.path)

import cv2  # noqa: E402

from src.legend_matcher import match_legend_to_bars  # noqa: E402
from src.reconstruct import COLOR_DISTANCE, reconstruct_vbar  # noqa: E402
from src.contracts import ChartInterpretationError  # noqa: E402
from src.tactile.completeness import READY, check_completeness  # noqa: E402

PNG = Path.home() / "lsi-code/TactileGen/data/png_test/png"
THRESHOLDS = [round(0.30 + 0.05 * k, 2) for k in range(0, 14)]   # 0.30 (no-op: MIN_CONFIDENCE) .. 0.95
SPLIT_SEED = 0


def split_of(ids):
    ids = sorted(ids)
    tune = set(random.Random(SPLIT_SEED).sample(ids, round(0.2 * len(ids))))
    return {i: ("tune" if i in tune else "report") for i in ids}


def passes(spec, record):
    """READY (reconstruct-level) and content matches ground truth. None if no spec."""
    if spec is None:
        return False
    try:
        ready = check_completeness(spec).status == READY
    except Exception:  # noqa: BLE001
        return False
    models = record["models"]
    gt_cats = models[0]["x"]
    ticks = [float(t) for t in record["general_figure_info"]["y_axis"]["major_ticks"]["values"]]
    gt_vals = [v for m in models for v in m["y"] if v is not None]
    axis_max = max(ticks) if ticks and max(ticks) > 0 else (max((abs(v) for v in gt_vals), default=1.0) or 1.0)
    if len(spec.categories) != len(gt_cats):
        return False
    pairs = ab.match_series(spec, models)
    if len(spec.series) != len(models) or len(pairs) != len(models):
        return False
    if [ab.norm(c) for c in spec.categories] != [ab.norm(c) for c in gt_cats]:
        return False
    for i, k in pairs:
        s = spec.series[i]
        if not ((s.name is not None and ab.norm(s.name) == ab.norm(models[k].get("name", "")))
                or (s.name is None and len(models) == 1)):
            return False
        for p, t in zip(s.values, models[k]["y"]):
            if t is not None and (p is None or abs(p - t) / axis_max > ab.VALUE_TOL):
                return False
    return True


def series_reason(r, row):
    """One primary reason for a series_ambiguous chart (priority order as listed in REASONS)."""
    record, frame, mapping, spec = r["record"], r["frame"], r["legend_mapping"], r["spec"]
    models = record["models"]
    gt_n = len(models)
    legend_items = (record["general_figure_info"].get("legend") or {}).get("items", [])
    swatches = [d for d in frame.detections if d.role == "legend_swatch"]
    labels = [d for d in frame.detections if d.role == "legend_label"]
    status = {}
    for a in mapping.assignments:
        status[a.bar_id] = a.status
    from collections import Counter
    st = Counter(status.values())
    by_swatch = {}
    for a in mapping.assignments:
        if a.status == "matched":
            by_swatch.setdefault(a.swatch_id, set()).add((a.label_id, a.series_name))
    conflicting = sum(len(v) > 1 for v in by_swatch.values())
    colors = [ab.hex_rgb(m["colors"]) for m in models if isinstance(m.get("colors"), str)]
    min_gap = min((math.dist(a, b) for i, a in enumerate(colors) for b in colors[i + 1:]), default=None)
    row.update(gt_series=gt_n, gt_legend_items=len(legend_items), det_swatches=len(swatches), det_legend_labels=len(labels),
               mapping_entries=len(mapping.entries), conflicting_swatches=conflicting,
               status_counts=";".join(f"{k}:{v}" for k, v in st.most_common()),
               min_gt_color_dist=None if min_gap is None else round(min_gap, 1),
               det_extra=row_audit[r["id"]]["det_extra"], det_missing=row_audit[r["id"]]["det_missing"])
    unresolved = {k: v for k, v in st.items() if k != "matched"}
    if gt_n == 1:
        return "single_series_false_alarm"
    if row["det_extra"] > 0:
        return "extra_detected_bars"
    if row["det_missing"] > 0:
        return "missing_detected_bars"
    if conflicting:
        return "conflicting_swatch_identity"
    if not swatches and not labels:
        return "legend_not_detected"
    if len(swatches) != len(legend_items) or len(labels) != len(legend_items):
        return "legend_count_mismatch"
    if min_gap is not None and min_gap < 2 * COLOR_DISTANCE:
        return "similar_series_colors"
    if unresolved:
        top = max(unresolved, key=unresolved.get)
        return f"unresolved_bars:{top}"
    return "other"


row_audit: dict = {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = sorted(CACHE.glob("*.pkl"))
    split = split_of([f.stem for f in cache])
    sa_rows, bar_rows, sweep = [], [], []
    per, memo = {}, {}
    for f in cache:
        r = pickle.loads(f.read_bytes())
        spec, frame, rec = r.get("spec"), r["frame"], r["record"]
        truth = ab.gt_bars(rec)
        raw, surv = ab.surviving_bars(frame)
        m, missed = ab.match_bars(surv, truth)
        row_audit[r["id"]] = {"det_extra": len(surv) - len(m), "det_missing": len(missed)}
        if spec is None:
            per[r["id"]] = None
            continue
        status = {a.bar_id: a.status for a in r["legend_mapping"].assignments}
        for i, d in enumerate(surv):
            bar_rows.append({"chart_id": r["id"], "split": split[r["id"]], "bar_id": d.id, "conf": round(d.confidence, 4),
                             "is_true_bar": i in m, "legend_status": status.get(d.id, "none"),
                             "gt_series": len(rec["models"]), "flags_series_ambiguous": "series_ambiguous" in spec.flags})
        if "series_ambiguous" in spec.flags:
            row = {"chart_id": r["id"], "split": split[r["id"]], "exit": r["exit"]}
            row["reason"] = series_reason(r, row)
            row["pred_series"] = len(spec.series)
            sa_rows.append(row)
        per[r["id"]] = (r, surv, m, status)
    for name, rows in (("series_ambiguous.csv", sa_rows), ("extra_bar_conf.csv", bar_rows)):
        cols = list(dict.fromkeys(k for x in rows for k in x))
        with (OUT / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # rule sweep: remove a bar when conf < tau AND (variant A) its legend assignment is not "matched"
    #                              or (variant B) the chart is series_ambiguous
    for variant in ("A_bar_unmatched_to_legend", "B_chart_series_ambiguous"):
        for tau in THRESHOLDS:
            acc = {s: dict(n=0, affected=0, fixed=0, broke=0, still_wrong=0, true_removed=0, extra_removed=0, fixed_only_false_bars_removed=0, fixed_true_bar_removed=0, fixed_chart_had_extras=0)
                   for s in ("tune", "report")}
            for cid, p in per.items():
                if p is None:
                    continue
                r, surv, m, status = p
                a = acc[split[cid]]
                a["n"] += 1
                remove = set()
                for i, d in enumerate(surv):
                    if d.confidence >= tau:
                        continue
                    cond = (status.get(d.id, "none") != "matched") if variant.startswith("A") else (
                        "series_ambiguous" in r["spec"].flags)
                    if cond:
                        remove.add(i)
                if not remove:
                    continue
                a["affected"] += 1
                a["true_removed"] += sum(i in m for i in remove)
                a["extra_removed"] += sum(i not in m for i in remove)
                only_false = all(i not in m for i in remove)
                had_extra = any(i not in m for i in range(len(surv)))

                def tally(before, after, a=a, only_false=only_false, had_extra=had_extra, r=r):
                    a["fixed"] += (not before) and after
                    a["broke"] += before and (not after)
                    a["still_wrong"] += (not before) and (not after)
                    if (not before) and after:
                        a["fixed_only_false_bars_removed" if only_false else "fixed_true_bar_removed"] += 1
                        a["fixed_chart_had_extras"] += had_extra
                key = (cid, frozenset(remove))
                if key in memo:
                    tally(*memo[key])
                    continue
                ids = {surv[i].id for i in remove}
                new_frame = replace(r["frame"], detections=tuple(d for d in r["frame"].detections if d.id not in ids))
                image = cv2.imread(str(PNG / f"{cid}.png"))
                try:
                    mapping = match_legend_to_bars(image, new_frame, r["ocr"])
                    spec = reconstruct_vbar(new_frame, r["ocr"], image, legend_mapping=mapping)
                except ChartInterpretationError:
                    spec = None
                before, after = passes(r["spec"], r["record"]), passes(spec, r["record"])
                memo[key] = (before, after)
                tally(before, after)
            for s, a in acc.items():
                sweep.append({"variant": variant, "tau": tau, "split": s, **a, "net": a["fixed"] - a["broke"]})
            print(variant, tau, {s: (acc[s]["fixed"], acc[s]["broke"]) for s in acc}, flush=True)
    cols = list(sweep[0])
    with (OUT / "extra_bar_rule_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sweep)


if __name__ == "__main__":
    main()
