"""Stage 2 of the end-to-end slice: ground-truth boxes, OCR reads in place of ground-truth text.

Same scope charts as Stage 1 (src/baseline/e2e_stage1.py). The reference for each chart is the
Stage 1 result on ground-truth text; only charts Stage 1 accepted have a trusted reference.
Errors here come from the OCR reads.

A chart's output is *wrong* when no chart was built, a category name differs from the
reference (normalized), or any bar value differs by more than 1% of the reference axis
span. A chart is *flagged* when its status is not ``accepted``. Flag precision is the share of
flagged reference charts that are wrong, recall the share of wrong charts that are flagged.

The confidence below which a read is flagged is chosen per engine on val (best F1 of the
flags over a fixed grid; ties go to the higher threshold) and test is scored once.

  python -m src.baseline.e2e_stage2 --out output/e2e_stage2
"""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from src.baseline.e2e_stage1 import _value_error, scope_charts
from src.contracts import OCRFrame, RecognizedText
from src.dataset.pmc_frames import ground_truth_frames
from src.interpret_bars import interpret_vertical_bars
from src.textutil import normalize

ENGINES = ("tesseract", "easyocr")
GRID = (None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TOL = 0.01
COCO_CATEGORY = {"x_title": 1, "y_title": 2, "chart_title": 7, "xlabel": 5, "ylabel": 6,
                 "legend_label": 11, "value_label": 15}


def load_reads(engine, split):
    rows = json.loads(Path(f"output/ocr/{engine}/ocr_{split}_predictions.json").read_text())
    return {(r["chart"], COCO_CATEGORY[r["klass"]], tuple(r["box"])): r for r in rows if r["klass"] in COCO_CATEGORY}


def ocr_frame(detections, annotations, name, reads):
    ids = {d.id for d in detections.detections}
    regions = []
    for a in annotations:
        did, row = f"gt{a['id']}", reads.get((name, a["category_id"], tuple(a["bbox"])))
        if did in ids and row is not None:
            regions.append(RecognizedText(did, row["pred"], row["conf"]))
    return OCRFrame(detections.source_key, tuple(regions))


def prepare(root, split):
    """[(image, annotations, ground-truth frame, Stage 1 result)] for the scope charts of a split."""
    out = []
    for image, annotations in scope_charts(root, split):
        det, ocr = ground_truth_frames(f"{split}/{image['file_name']}", image, annotations)
        out.append((image, annotations, det, interpret_vertical_bars(det, ocr)))
    return out


def compare(reference, result):
    """How a Stage 2 result differs from the Stage 1 chart: category and value mistakes."""
    ref = reference.chart
    if result.chart is None:
        return {"built": False, "categories_wrong": len(ref.categories), "values_wrong": len(ref.categories), "decade": 0}
    cats = sum(normalize(a) != normalize(b) for a, b in zip(result.chart.categories, ref.categories))
    span = ref.y_tick_values[-1] - ref.y_tick_values[0]
    values = decade = 0
    for v, r in zip(result.chart.series[0].values, ref.series[0].values):
        values += abs(v - r) > TOL * span
        decade += bool(v and r and abs(math.log10(abs(v / r))) > 0.5)
    return {"built": True, "categories_wrong": cats, "values_wrong": values, "decade": decade}


def run(charts, reads, threshold):
    records = []
    for image, annotations, det, ref in charts:
        ocr = ocr_frame(det, annotations, image["file_name"], reads)
        result = interpret_vertical_bars(det, ocr, min_confidence=threshold)
        rec = {"chart": image["file_name"], "status": result.status, "flagged": result.status != "accepted",
               "reference_ok": ref.status == "accepted", "reasons": list(result.reasons),
               "flagged_items": [(i.kind, i.text, i.reason) for i in result.items if i.flagged]}
        if rec["reference_ok"]:
            rec.update(compare(ref, result))
            rec["wrong_values"] = (not rec["built"]) or rec["values_wrong"] > 0
            rec["wrong"] = rec["wrong_values"] or rec["categories_wrong"] > 0
            if result.chart is not None:
                errs = _value_error(result.chart, annotations)
                if errs:
                    rec["printed_value_error"] = [e["span_fraction"] for e in errs]
        rec["chart_obj"] = result.chart
        records.append(rec)
    return records


def flag_scores(records, key="wrong"):
    ref = [r for r in records if r["reference_ok"]]
    wrong = [r for r in ref if r[key]]
    flagged = [r for r in ref if r["flagged"]]
    caught = [r for r in wrong if r["flagged"]]
    precision = len(caught) / len(flagged) if flagged else 0.0
    recall = len(caught) / len(wrong) if wrong else 1.0
    return {"reference_charts": len(ref), "wrong": len(wrong), "flagged": len(flagged), "caught": len(caught),
            "silent_errors": len(wrong) - len(caught), "false_flags": len(flagged) - len(caught),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def summarize(records):
    ref = [r for r in records if r["reference_ok"]]
    built = [r for r in ref if r["built"]]
    cats = sum(len(r["chart_obj"].categories) - r["categories_wrong"] for r in built)
    total = sum(len(r["chart_obj"].categories) for r in built)
    printed = [e for r in records if r["status"] == "accepted" for e in r.get("printed_value_error", [])]
    return {
        "status": dict(Counter(r["status"] for r in records)),
        "scope_charts": len(records), "reference_charts": len(ref),
        "chart_built_of_reference": len(built),
        "bar_count_correct": sum(r["built"] and len(r["chart_obj"].categories) == len(r["chart_obj"].series[0].values) for r in built),
        "category_exact": f"{cats}/{total}" if total else None,
        "charts_all_categories_right": sum(r["built"] and r["categories_wrong"] == 0 for r in ref),
        "charts_all_values_within_1pct": sum(r["built"] and r["values_wrong"] == 0 for r in ref),
        "factor_of_ten_errors": {"charts": sum(r.get("decade", 0) > 0 for r in ref),
                                 "flagged": sum(r.get("decade", 0) > 0 and r["flagged"] for r in ref)},
        "printed_value_error_accepted": ({"bars": len(printed), "within_1pct": sum(e <= TOL for e in printed),
                                          "max_span_fraction": max(printed)} if printed else None),
        "non_reference_charts_flagged": f"{sum(r['flagged'] for r in records if not r['reference_ok'])}/{sum(not r['reference_ok'] for r in records)}",
        "flags": flag_scores(records),
        "flags_values_only": flag_scores(records, "wrong_values")}


def slim(records):
    """Per-chart records without the chart object, for the saved report."""
    return [{k: v for k, v in r.items() if k != "chart_obj"} for r in records]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true",
                    help="recompute from the thresholds saved by the first run (same reads, nothing re-chosen)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result_file = args.out / "stage2_test.json"
    if result_file.exists() and not args.reuse:
        raise SystemExit(f"test already scored once: {result_file}; use --reuse to recompute with its thresholds")
    val, test = prepare(args.root, "val"), prepare(args.root, "test")
    report = {"val": {}, "test": {}, "thresholds": {}, "sweep": {}, "records": {"val": {}, "test": {}}}
    saved = json.loads(result_file.read_text()) if args.reuse else None
    for engine in ENGINES:
        reads = load_reads(engine, "val")
        if saved:
            best, report["sweep"][engine] = saved["thresholds"][engine], saved["sweep"][engine]
        else:
            sweep = {("off" if t is None else t): flag_scores(run(val, reads, t)) for t in GRID}
            best = max(GRID, key=lambda t: (round(sweep["off" if t is None else t]["f1"], 6), t if t is not None else 2))
            report["sweep"][engine] = sweep
            print(f"\n{engine}: val sweep of the confidence threshold (flag scores on {sweep['off']['reference_charts']} reference charts)")
            for t, s in sweep.items():
                print(f"  {t!s:<5} flagged {s['flagged']:>2}  wrong {s['wrong']:>2}  caught {s['caught']:>2}  silent {s['silent_errors']:>2}  precision {s['precision']:.2f}  recall {s['recall']:.2f}  F1 {s['f1']:.2f}")
            print(f"  chosen on val: {best}")
        report["thresholds"][engine] = best
        for split, charts in (("val", val), ("test", test)):
            records = run(charts, load_reads(engine, split), best)
            report[split][engine] = summarize(records)
            report["records"][split][engine] = slim(records)
    for split in ("val", "test"):
        for engine in ENGINES:
            print(f"\n{split.upper()} {engine} (threshold {report['thresholds'][engine]})")
            print(json.dumps(report[split][engine], indent=1))
    if not args.reuse:
        result_file.write_text(json.dumps(report, indent=1, default=str))
    else:
        (args.out / "stage2_test_with_records.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
