"""End-to-end audit: run PlotQA bar charts through the full pipeline and explain each failure.

Ported from Chloe's vertical-bar audit (branch analysis/vbar-failure-audit-v2) to
the current code, and extended to horizontal bars. Two steps:

    # 1. run every chart through src.tactile.pipeline (--yes --strict) and cache it
    python scripts/audit_e2e.py run --annotations data/plotqa/test/annotations.json \
        --images data/plotqa/test/png --cache output/audit/cache --workers 2

    # 2. compare each cached chart with the ground truth, give it one primary
    #    cause, and write a CSV, a Markdown report and example sheets
    python scripts/audit_e2e.py report --cache output/audit/cache --out output/audit/report

``run`` is resumable: charts already in the cache are skipped. Use a new cache
folder for every checkpoint or code version you compare.

Rates (over the audited charts of one orientation):
  (a) built   = pipeline exit 0 (a tactile page was written)
  (b) correct = of built charts, share whose content matches the ground truth
  (c) built AND correct, over all charts
"Correct" = series count, category count, category text, series names and every
value within 1% of the ground-truth value-axis maximum. Titles are not compared.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHART_TYPES = {"vbar_categorical": "vertical", "hbar_categorical": "horizontal"}
BAR_ROLE = {"vertical": "vertical_bar", "horizontal": "horizontal_bar"}
VALUE_TOL = 0.01      # fraction of the ground-truth value-axis maximum
IOU = 0.5
NEAR_ZERO = 0.05      # a bar under 5% of the axis maximum, or <= 5 px thick, is near-zero
THIN_PX = 5

# One primary cause per chart; the first that applies wins (same order as Chloe's audit).
CAUSE_ORDER = [
    "none_correct", "false_block", "no_spec_interpretation_error", "detector_missing_bars",
    "detector_extra_bars", "ocr_category_text", "ocr_title_or_axis_title", "y_calibration",
    "series_false_alarm_single_series", "series_separation", "category_pairing",
    "bar_rescue_wrong", "other_wrong_values", "strict_bana_layout",
]
GROUP_OF_CAUSE = {
    "series_separation": "series_legend_colour_match",
    "series_false_alarm_single_series": "series_legend_colour_match",
    "ocr_category_text": "category_label_ocr",
    "ocr_title_or_axis_title": "title_axis_title_ocr",
    "false_block": "false_block_gate",
    "y_calibration": "y_axis_calibration",
    "other_wrong_values": "bar_values_off",
}
GROUP_WHY = {
    "tiny_bars_undetected": "every missed bar is near-zero; the detector does not find it",
    "real_bars_undetected": "at least one clearly visible bar has no detection",
    "series_legend_colour_match": "bars do not resolve to legend colours, so series are ambiguous",
    "category_label_ocr": "bars are right but a category label is misread or unreadable",
    "title_axis_title_ocr": "the title or an axis title could not be read",
    "false_block_gate": "the content is correct but a BANA/completeness check blocked it",
    "y_axis_calibration": "value-axis ticks dropped, uneven, or a value clipped at the axis",
    "bar_values_off": "bars, labels and series are right but a value is off by more than 1%",
    "other_rare": "pairing, extra bars, rescued values, interpretation errors, layout",
}


# --- ground truth --------------------------------------------------------------

def _box(b) -> tuple[float, float, float, float]:
    xs = sorted((float(b["x"]), float(b["x"]) + float(b["w"])))
    ys = sorted((float(b["y"]), float(b["y"]) + float(b["h"])))
    return xs[0], ys[0], xs[1], ys[1]


def ground_truth(record: dict) -> dict:
    """Categories, series and bars in reading order for either orientation.

    PlotQA keeps categories in ``x`` and values in ``y`` for vertical charts, and
    the other way round for horizontal ones. Categories are reordered to how the
    pipeline reports them: left to right for vertical bars, top to bottom for
    horizontal bars.
    """
    orientation = CHART_TYPES[record["type"]]
    vertical = orientation == "vertical"
    cat_key, val_key = ("x", "y") if vertical else ("y", "x")
    models = record["models"]
    first = [_box(b) for b in models[0]["bboxes"]]
    position = [(b[0] + b[2]) / 2 if vertical else (b[1] + b[3]) / 2 for b in first]
    order = sorted(range(len(first)), key=lambda j: position[j])
    categories = [str(models[0][cat_key][j]) for j in order]
    series = [{"name": m.get("name"), "color": m.get("colors"),
               "values": [m[val_key][j] for j in order]} for m in models]
    bars = []
    for k, m in enumerate(models):
        for rank, j in enumerate(order):
            box = _box(m["bboxes"][j])
            bars.append({"box": box, "value": m[val_key][j], "series": k, "cat": rank,
                         "thickness": box[3] - box[1] if vertical else box[2] - box[0]})
    axis = record["general_figure_info"]["y_axis" if vertical else "x_axis"]
    ticks = sorted({float(t) for t in axis["major_ticks"]["values"]})
    values = [v for s in series for v in s["values"] if v is not None]
    axis_max = max(ticks) if ticks and max(ticks) > 0 else (max((abs(v) for v in values), default=1.0) or 1.0)
    return {"orientation": orientation, "categories": categories, "series": series,
            "bars": bars, "ticks": ticks, "axis_max": axis_max}


# --- step 1: run the pipeline and cache what happened ---------------------------

def _select(annotations: Path, types: list[str], limit: int | None, seed: int) -> list[dict]:
    from src.dataset.prepare_plotqa import iter_annotations
    by_type = defaultdict(list)
    for record in iter_annotations(annotations):
        if record["type"] in types:
            by_type[record["type"]].append(record)
    rng = random.Random(seed)
    chosen = []
    for chart_type in types:
        records = sorted(by_type[chart_type], key=lambda r: r["image_index"])
        if limit is not None and limit < len(records):
            records = sorted(rng.sample(records, limit), key=lambda r: r["image_index"])
        chosen += records
    return chosen


def _worker(job) -> None:
    records, settings, shard = job
    os.chdir(ROOT)
    import src.detection as detection
    from src.tactile import pipeline

    original_load, original_detect = detection.load_model, detection.run_detection
    detection.load_model = lru_cache(maxsize=2)(original_load)
    captured: dict = {}

    def detect(model, image_path, **kwargs):
        kwargs["imgsz"] = settings["imgsz"]
        frame = original_detect(model, image_path, **kwargs)
        captured["raw_frame"] = frame
        return frame

    detection.run_detection = detect
    cache = Path(settings["cache"])
    tmp = Path(tempfile.mkdtemp(prefix=f"audit{shard}_"))

    def analyse(*args, **kwargs):
        result = pipeline.analyse(*args, ocr_engine=settings["ocr"], **kwargs)
        captured["spec"], error, captured["frame"] = result
        captured["error"] = None if error is None else str(error)
        return result

    for record in records:
        target = cache / f"{record['image_index']}.pkl"
        if target.exists():
            continue
        captured.clear()
        image = Path(settings["images"]) / f"{record['image_index']}.png"
        args = pipeline.parse_args(["--image", str(image), "--output", str(tmp / "out.svg"),
                                    "--weights", settings["weights"], "--yes", "--strict",
                                    "--ocr", settings["ocr"]])
        out, err = io.StringIO(), io.StringIO()
        start = time.time()
        try:
            code = pipeline.run(args, analyse_fn=analyse, stdin_is_tty=False, out=out, err=err)
        except Exception:  # noqa: BLE001  a crash counts as an error exit, with its traceback
            code = 1
            err.write(traceback.format_exc())
        row = {"id": str(record["image_index"]), "chart_type": record["type"], "exit": code,
               "seconds": round(time.time() - start, 2), "stdout": out.getvalue(),
               "stderr": err.getvalue(), "record": record, **captured}
        for leftover in tmp.glob("*"):
            leftover.unlink()
        part = target.with_suffix(".tmp")
        part.write_bytes(pickle.dumps(row))
        part.replace(target)


def cmd_run(args) -> None:
    types = [t for t, o in CHART_TYPES.items() if o in args.orientations]
    records = _select(args.annotations, types, args.limit, args.seed)
    args.cache.mkdir(parents=True, exist_ok=True)
    settings = {"cache": str(args.cache.resolve()), "images": str(args.images.resolve()),
                "weights": str(args.weights), "ocr": args.ocr, "imgsz": args.imgsz}
    (args.cache / "run_settings.json").write_text(json.dumps(
        {**settings, "annotations": str(args.annotations), "orientations": args.orientations,
         "limit": args.limit, "seed": args.seed, "charts": len(records)}, indent=2) + "\n",
        encoding="utf-8")
    todo = [r for r in records if not (args.cache / f"{r['image_index']}.pkl").exists()]
    print(f"{len(records)} charts selected, {len(records) - len(todo)} already cached, "
          f"{len(todo)} to run on {args.workers} worker(s)", flush=True)
    if not todo:
        return
    shards = [(todo[k::args.workers], settings, k) for k in range(args.workers)]
    if args.workers == 1:
        _worker(shards[0])
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.map(_worker, shards, chunksize=1)


# --- step 2: compare with the ground truth and explain --------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(d) -> tuple[float, float, float, float]:
    return d.box.x1, d.box.y1, d.box.x2, d.box.y2


def _norm(text) -> str:
    return " ".join(str(text).lower().split())


def _loose(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _hex_rgb(value: str):
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def surviving_bars(frame, orientation: str):
    """Bars that reach the series/category logic: the reconstruct filters."""
    from src.reconstruction.vertical import MIN_CONFIDENCE, _centre, _dedupe, _inside
    raw = [d for d in frame.detections if d.role == BAR_ROLE[orientation]]
    kept = _dedupe([d for d in raw if d.confidence >= MIN_CONFIDENCE])
    plot = max((d for d in frame.detections if d.role == "plot_area" and d.confidence >= MIN_CONFIDENCE),
               key=lambda d: d.confidence, default=None)
    return raw, [d for d in kept if plot is None or _inside(_centre(d), plot)]


def match_bars(detections, truth):
    """Greedy one-to-one IoU matching -> ({det index: gt index}, unmatched gt indices)."""
    pairs = sorted(((_iou(_xyxy(d), t["box"]), i, j) for i, d in enumerate(detections)
                    for j, t in enumerate(truth)), reverse=True)
    used_d, used_t, matched = set(), set(), {}
    for overlap, i, j in pairs:
        if overlap < IOU:
            break
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        matched[i] = j
    return matched, [j for j in range(len(truth)) if j not in used_t]


def match_series(spec, series):
    """Pair predicted with true series by name, then colour, one-to-one."""
    from src.reconstruction.vertical import COLOR_DISTANCE
    if len(spec.series) == 1 and len(series) == 1:
        return [(0, 0)]
    pairs, used = [], set()
    for i, s in enumerate(spec.series):
        best = None
        for k, t in enumerate(series):
            if k in used:
                continue
            distance = None
            if s.source_color and isinstance(t["color"], str):
                distance = math.dist(_hex_rgb(s.source_color), _hex_rgb(t["color"]))
            same_name = s.name is not None and _norm(s.name) == _norm(t["name"] or "")
            if (distance is not None and distance <= 2 * COLOR_DISTANCE) or same_name:
                score = (0 if same_name else 1, distance if distance is not None else 1e9)
                if best is None or score < best[0]:
                    best = (score, k)
        if best:
            used.add(best[1])
            pairs.append((i, best[1]))
    return pairs


def block_reasons(stdout: str, exit_code: int):
    """(blocking flags, missing text fields, failed BANA rules) from the pipeline output."""
    flags = re.findall(r"\((\w+)\)", " ".join(re.findall(r"^  - .*$", stdout, re.M))) if exit_code == 3 else []
    needs = ([re.sub(r"\[\d+\]", "[n]", f) for f in re.findall(r"^  ([\w\[\]]+): ", stdout, re.M)]
             if exit_code == 2 else [])
    bana = sorted(set(re.findall(r"^\s+\[(\w+)\] ", stdout, re.M)) - {"overview_first"})
    return flags, needs, bana


def analyse(r: dict) -> dict:
    record, spec = r["record"], r.get("spec")
    frame = r.get("frame") or r.get("raw_frame")
    gt = ground_truth(record)
    orientation, truth, series, axis_max = gt["orientation"], gt["bars"], gt["series"], gt["axis_max"]
    row = {"chart_id": r["id"], "orientation": orientation, "exit": r["exit"],
           "status": "built" if r["exit"] == 0 else "blocked", "seconds": r.get("seconds")}
    flags_blocking, needs, bana = block_reasons(r["stdout"], r["exit"])
    if r["exit"] == 3 and spec is not None:
        # Some blockers explain themselves without a "(flag)" tag; read the flags directly.
        from src.tactile.completeness import BLOCKING_FLAGS
        flags_blocking = [f.split(":")[0] for f in spec.flags if f.split(":")[0] in BLOCKING_FLAGS]
        if not spec.categories:
            flags_blocking.append("no_categories")
        if any(v is None for s in spec.series if not s.no_visible_bars for v in s.values):
            flags_blocking.append("values_unread")
    row["block_reasons"] = ";".join(sorted(set(flags_blocking)) + sorted(set(needs))
                                    + [f"bana:{b}" for b in bana])
    if r["exit"] == 1 and not bana and "--strict" not in r["stderr"]:
        row["block_reasons"] = "error:" + (r["stderr"].strip().splitlines() or ["?"])[-1][:80]
    flags = list(spec.flags) if spec else []
    row["flags"] = ";".join(flags)
    row.update(gt_series=len(series), gt_categories=len(gt["categories"]), gt_bars=len(truth),
               gt_axis_max=axis_max, gt_categories_text="|".join(gt["categories"]),
               gt_series_names="|".join(str(s["name"]) for s in series))

    # detector stage
    missed, extras = list(range(len(truth))), []
    stage = Counter()
    near_zero = thin = 0
    from src.reconstruction.vertical import MIN_CONFIDENCE
    if frame is not None:
        raw, kept = surviving_bars(frame, orientation)
        matched, missed = match_bars(kept, truth)
        extras = [i for i in range(len(kept)) if i not in matched]
        row.update(det_bars_raw=len(raw), det_bars_used=len(kept))
        row["extra_conf"] = ";".join(f"{kept[i].confidence:.3f}" for i in extras)
        for j in missed:
            t = truth[j]
            best = max(raw, key=lambda d: _iou(_xyxy(d), t["box"]), default=None)
            if best is None or _iou(_xyxy(best), t["box"]) < IOU:
                stage["not_detected"] += 1
            elif best.confidence < MIN_CONFIDENCE:
                stage["low_conf"] += 1
            else:
                stage["dedupe_or_plot"] += 1
    for j in missed:
        t = truth[j]
        tiny = (t["value"] is not None and abs(t["value"]) < NEAR_ZERO * axis_max) or t["thickness"] <= THIN_PX
        near_zero += tiny
        thin += t["thickness"] <= 4
    row.update(det_missing=len(missed), det_extra=len(extras), missed_near_zero=near_zero,
               missed_under_4px=thin, missed_not_detected=stage["not_detected"],
               missed_low_conf=stage["low_conf"], missed_dedupe_or_plot=stage["dedupe_or_plot"])

    # prediction vs ground truth
    checks = {}
    if spec is not None:
        count_ok = len(spec.categories) == len(gt["categories"])
        pairs = match_series(spec, series)
        series_ok = len(spec.series) == len(series) and len(pairs) == len(series)
        labels_ok = [_norm(c) for c in spec.categories] == [_norm(c) for c in gt["categories"]]
        names_ok = series_ok and all(
            (spec.series[i].name is not None and _norm(spec.series[i].name) == _norm(series[k]["name"] or ""))
            or (spec.series[i].name is None and len(series) == 1) for i, k in pairs)
        errors, bad, rescued_bad = [], 0, 0
        if count_ok:
            for i, k in pairs:
                rescued = spec.series[i].value_rescued
                for j, (p, t) in enumerate(zip(spec.series[i].values, series[k]["values"])):
                    if t is None:
                        continue
                    e = math.inf if p is None else abs(p - t) / axis_max
                    errors.append(e)
                    if e > VALUE_TOL:
                        bad += 1
                        rescued_bad += bool(rescued and j < len(rescued) and rescued[j])
        values_ok = count_ok and series_ok and bool(errors) and bad == 0
        ticks = sorted({round(float(t), 6) for t in spec.y_axis.tick_values})
        row.update(pred_series=len(spec.series), pred_categories=len(spec.categories),
                   pred_categories_text="|".join(spec.categories),
                   pred_series_names="|".join(str(s.name) for s in spec.series),
                   count_ok=count_ok, series_ok=series_ok, labels_ok=labels_ok,
                   labels_ok_loose=[_loose(c) for c in spec.categories] == [_loose(c) for c in gt["categories"]],
                   names_ok=names_ok, values_ok=values_ok, n_bad_values=bad if count_ok else "",
                   max_err_frac=("inf" if errors and math.isinf(max(errors)) else f"{max(errors):.4f}")
                   if errors else "",
                   ticks_ok=ticks == [round(t, 6) for t in gt["ticks"]])
        row["correct"] = bool(count_ok and series_ok and labels_ok and names_ok and values_ok)
        checks = dict(count_ok=count_ok, series_ok=series_ok, labels_ok=labels_ok,
                      values_ok=values_ok, rescued_bad=rescued_bad, ticks_ok=row["ticks_ok"])
    else:
        row["correct"] = False
        row["block_reasons"] = row["block_reasons"] or "interpretation_error"
        row["interp_error"] = (r.get("error") or "")[:120]
    row["built_and_correct"] = row["status"] == "built" and row["correct"]
    row["false_block"] = (row["status"] == "blocked" and spec is not None and row["correct"]
                          and r["exit"] in (1, 3))

    names = {f.split(":")[0] for f in flags}
    if row["built_and_correct"]:
        cause = "none_correct"
    elif row["false_block"]:
        cause = "false_block"
    elif spec is None:
        cause = "no_spec_interpretation_error"
    elif missed:
        cause = "detector_missing_bars"
    elif extras:
        cause = "detector_extra_bars"
    elif ((not checks["labels_ok"] and checks["count_ok"]) or "category_unreadable" in names
          or any(n.startswith("category") for n in needs)):
        cause = "ocr_category_text"
    elif any(n in ("title", "y_axis_title", "x_axis_title") or n.startswith("series_name") for n in needs):
        cause = "ocr_title_or_axis_title"
    elif names & {"y_calibration_ambiguous", "y_tick_leading_zero", "y_ticks_not_arithmetic",
                  "value_clipped_at_axis", "baseline_outside_plot"} or checks["ticks_ok"] is False:
        cause = "y_calibration"
    elif "series_ambiguous" in names and len(series) == 1:
        cause = "series_false_alarm_single_series"
    elif "series_ambiguous" in names or not checks["series_ok"]:
        cause = "series_separation"
    elif names & {"category_count_mismatch", "category_ambiguous"} or not checks["count_ok"]:
        cause = "category_pairing"
    elif checks["rescued_bad"]:
        cause = "bar_rescue_wrong"
    elif not checks["values_ok"]:
        cause = "other_wrong_values"
    elif bana:
        cause = "strict_bana_layout"
    else:
        cause = "other_wrong_values"
    row["primary_cause"] = cause
    row["group"] = failure_group(cause, len(missed), near_zero)
    return row


def failure_group(cause: str, missing: int, near_zero: int) -> str:
    if cause == "none_correct":
        return ""
    if cause == "detector_missing_bars":
        return "tiny_bars_undetected" if near_zero == missing else "real_bars_undetected"
    return GROUP_OF_CAUSE.get(cause, "other_rare")


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def summarise(rows: list[dict], orientation: str) -> list[str]:
    rows = [r for r in rows if r["orientation"] == orientation]
    n = len(rows)
    if not n:
        return []
    built = sum(r["status"] == "built" for r in rows)
    good = sum(r["built_and_correct"] for r in rows)
    failures = [r for r in rows if not r["built_and_correct"]]
    lines = [f"## {orientation.capitalize()} bars ({n} charts)", "",
             f"- (a) built (exit 0): {built}/{n} = **{_pct(built, n)}**",
             f"- (b) of built, content correct: {good}/{built} = {_pct(good, built)}",
             f"- (c) built AND correct: {good}/{n} = **{_pct(good, n)}**",
             f"- built but wrong: {built - good}; blocked: {n - built}", ""]
    lines += ["### Failure groups", "",
              "| group | charts | blocked | built wrong | % of failures | % of all | why | examples |",
              "|---|---|---|---|---|---|---|---|"]
    groups = Counter(r["group"] for r in failures)
    for group, count in groups.most_common():
        members = [r for r in failures if r["group"] == group]
        blocked = sum(r["status"] == "blocked" for r in members)
        examples = ", ".join(r["chart_id"] for r in members[:3])
        lines.append(f"| {group} | {count} | {blocked} | {count - blocked} | {_pct(count, len(failures))} "
                     f"| {_pct(count, n)} | {GROUP_WHY[group]} | {examples} |")
    lines += ["", "### Primary causes", "", "| cause | charts | % of all |", "|---|---|---|"]
    for cause, count in Counter(r["primary_cause"] for r in failures).most_common():
        lines.append(f"| {cause} | {count} | {_pct(count, n)} |")
    lines += ["", "### Most common block reasons", "", "| block reasons | charts |", "|---|---|"]
    for reasons, count in Counter(r["block_reasons"] for r in rows if r["status"] == "blocked").most_common(12):
        lines.append(f"| {reasons or '(none)'} | {count} |")
    missed = sum(r["det_missing"] for r in rows)
    lines += ["", "### Missed bars", "",
              f"- true bars: {sum(r['gt_bars'] for r in rows)}, missed after filtering: {missed}",
              f"- of the missed: near-zero {sum(r['missed_near_zero'] for r in rows)}, "
              f"under 4 px {sum(r['missed_under_4px'] for r in rows)}",
              f"- why missed: never detected {sum(r['missed_not_detected'] for r in rows)}, "
              f"below confidence {sum(r['missed_low_conf'] for r in rows)}, "
              f"removed as duplicate/outside plot {sum(r['missed_dedupe_or_plot'] for r in rows)}",
              f"- charts with extra bars: {sum(r['det_extra'] > 0 for r in rows)}", ""]
    return lines


def contact_sheet(cache: Path, images: Path, rows: list[dict], path: Path, size: int = 440) -> None:
    """Up to 6 charts side by side: green = true bars, red = bars the pipeline used."""
    import cv2
    import numpy as np
    tiles = []
    for r in rows[:6]:
        data = pickle.loads((cache / f"{r['chart_id']}.pkl").read_bytes())
        image = cv2.imread(str(images / f"{r['chart_id']}.png"))
        if image is None:
            continue
        gt = ground_truth(data["record"])
        # Shrink first, then draw, so thin outlines survive the resize.
        scale = size / max(image.shape[:2])
        tile = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)

        def draw(box, colour, width):
            x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
            cv2.rectangle(tile, (x1, y1), (max(x2, x1 + 1), max(y2, y1 + 1)), colour, width)

        for bar in gt["bars"]:
            draw(bar["box"], (0, 170, 0), 3)
        frame = data.get("frame") or data.get("raw_frame")
        if frame is not None:
            for d in surviving_bars(frame, gt["orientation"])[1]:
                draw(_xyxy(d), (0, 0, 230), 1)
        canvas = np.full((size + 24, size, 3), 255, np.uint8)
        canvas[24:24 + tile.shape[0], :tile.shape[1]] = tile
        cv2.putText(canvas, r["chart_id"], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(canvas)
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def cmd_report(args) -> None:
    rows = [analyse(pickle.loads(f.read_bytes())) for f in sorted(args.cache.glob("*.pkl"))]
    if not rows:
        raise SystemExit(f"No cached charts in {args.cache}")
    args.out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    settings = args.cache / "run_settings.json"
    header = ["# End-to-end audit", ""]
    if settings.exists():
        s = json.loads(settings.read_text(encoding="utf-8"))
        header += [f"Checkpoint `{s['weights']}`, OCR `{s['ocr']}`, image size {s['imgsz']}, "
                   f"annotations `{s['annotations']}`, `--yes --strict`.", ""]
    lines = header + [line for o in ("vertical", "horizontal") for line in summarise(rows, o)]
    images = Path(json.loads(settings.read_text(encoding="utf-8"))["images"]) if settings.exists() else None
    if images is not None and args.sheets:
        lines += ["## Example sheets", "", "Green = true bars, red = bars the pipeline used.", ""]
        for orientation in ("vertical", "horizontal"):
            failures = [r for r in rows if r["orientation"] == orientation and not r["built_and_correct"]]
            for group in sorted({r["group"] for r in failures}):
                members = [r for r in failures if r["group"] == group]
                path = args.out / "sheets" / f"{orientation}_{group}.png"
                contact_sheet(args.cache, images, members, path)
                if path.exists():
                    lines.append(f"- {orientation} / {group}: `sheets/{path.name}`")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for orientation in ("vertical", "horizontal"):
        part = [r for r in rows if r["orientation"] == orientation]
        if part:
            built = sum(r["status"] == "built" for r in part)
            good = sum(r["built_and_correct"] for r in part)
            print(f"{orientation}: {len(part)} charts, built {_pct(built, len(part))}, "
                  f"built and correct {_pct(good, len(part))}")
    print(f"Wrote {args.out / 'report.md'} and {args.out / 'audit.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline on PlotQA charts and cache the results")
    run.add_argument("--annotations", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True, help="one folder per checkpoint/code version")
    run.add_argument("--weights", default="models/yolo_bars_hv/best.pt")
    run.add_argument("--orientations", nargs="+", choices=("vertical", "horizontal"),
                     default=["vertical", "horizontal"])
    run.add_argument("--limit", type=int, help="charts per orientation (default: all)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--ocr", choices=("rapidocr", "tesseract"), default="rapidocr")
    run.add_argument("--imgsz", type=int, default=640)
    run.add_argument("--workers", type=int, default=1)
    report = sub.add_parser("report", help="explain every cached chart and write the report")
    report.add_argument("--cache", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--no-sheets", dest="sheets", action="store_false",
                        help="skip the example image sheets")
    args = parser.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
