"""Can the value axis be recovered from noisy tick reads? (ground-truth tick boxes + OCR reads)

For each vertical test chart, every ``ylabel`` box is paired with the nearest
ground-truth ``y_tick`` box (by vertical centre). The pixel position of the tick
and the number parsed from the OCR read are the points of a line value = a + b * y.
The fit does not assume the reads are right: every pair of points proposes a
line, the line supported by the most points wins (inliers lie within 2% of the
implied value span), and the winner is refit on its inliers. Lines that rise
with y are rejected because pixel y grows downward and the value axis of a
vertical bar chart should grow upward.

The reference line is a least-squares fit to the ground-truth label text on the
same ticks. A fit "matches" when its slope is within TOL (relative) and its
offset within TOL of the reference value span. A chart is flagged for a human
when fewer than 3 points parse, fewer than 3 are inliers, inliers are under 60%
of the ticks, or the best line is not unique.

  python -m src.baseline.ocr_axis_fit --predictions <ocr_test_predictions.json>
"""
import argparse
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.baseline.ocr_eval import COCO, COCO_FILES, load_crops, parse_number
from src.dataset.bar_orientation import orientation, read_label_boxes

YTICK = 9  # COCO category id
INLIER_TOL = 0.02
MIN_INLIERS = 3
MIN_SUPPORT = 0.6
TOLS = (0.005, 0.01, 0.02, 0.05)
LINEAR_RMS = 0.01  # reference labels must themselves lie on a line (else log or irregular axis)


def tick_positions(root, split="test"):
    """{(image name, ylabel box tuple): tick centre y} from nearest unique ground-truth y_ticks."""
    images = {p.name for p in (root / "images" / split).iterdir()}
    out = {}
    for coco_file in COCO_FILES.values():
        coco = json.loads((COCO / coco_file).read_text())
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for im in coco["images"]:
            if im["file_name"] not in images:
                continue
            labels = [a["bbox"] for a in by_image[im["id"]] if a["category_id"] == 6]
            ticks = [a["bbox"] for a in by_image[im["id"]] if a["category_id"] == YTICK]
            pairs = sorted((abs(l[1] + l[3] / 2 - (t[1] + t[3] / 2)), i, j)
                           for i, l in enumerate(labels) for j, t in enumerate(ticks))
            used_l, used_t = set(), set()
            for _, i, j in pairs:
                if i not in used_l and j not in used_t:
                    used_l.add(i)
                    used_t.add(j)
                    out[(im["file_name"], tuple(labels[i]))] = ticks[j][1] + ticks[j][3] / 2
    return out


def ols(points):
    """(a, b) of v = a + b*y; None when y does not vary."""
    n = len(points)
    my = sum(y for y, _ in points) / n
    mv = sum(v for _, v in points) / n
    sxx = sum((y - my) ** 2 for y, _ in points)
    if sxx == 0:
        return None
    b = sum((y - my) * (v - mv) for y, v in points) / sxx
    return mv - b * my, b


def robust_fit(points, strict_ambiguity=False):
    """-> (a, b, inliers, unique) or None when no admissible line exists.

    ``unique`` is False when two lines with the most inliers are supported by
    different sets of points. (strict_ambiguity=True is the first version of this
    rule, which also counted lines with the same inliers but slightly different
    slopes; it over-flags and is kept only to report that comparison.)
    """
    span = max(y for y, _ in points) - min(y for y, _ in points)
    best, ties = None, []
    for (y1, v1), (y2, v2) in itertools.combinations(points, 2):
        if abs(y2 - y1) < 1:
            continue
        b = (v2 - v1) / (y2 - y1)
        if b >= 0:
            continue
        a = v1 - b * y1
        tol = INLIER_TOL * abs(b) * span
        inliers = [(y, v) for y, v in points if abs(v - (a + b * y)) <= tol]
        score = (len(inliers), -sum((v - (a + b * y)) ** 2 for y, v in inliers) / max(tol, 1e-12) ** 2)
        if best is None or score[0] > best[0][0]:
            best, ties = (score, inliers, b), [(b, frozenset(inliers))]
        elif score[0] == best[0][0]:
            ties.append((b, frozenset(inliers)))
            if score[1] > best[0][1]:
                best = (score, inliers, b)
    if best is None:
        return None
    fit = ols(best[1]) if len(best[1]) >= 2 else None
    if fit is None:
        return None
    if strict_ambiguity:
        unique = all(abs(b - best[2]) <= 0.02 * abs(best[2]) for b, _ in ties)
    else:
        unique = len({inl for _, inl in ties}) == 1
    return fit[0], fit[1], best[1], unique


def analyse(rows, ticks, strict_ambiguity=False):
    charts = defaultdict(list)
    for r in rows:
        if r["klass"] == "ylabel" and (r["chart"], tuple(r["box"])) in ticks:
            charts[r["chart"]].append((ticks[(r["chart"], tuple(r["box"]))], parse_number(r["text"]), parse_number(r["pred"])))
    result = {"too_few_ticks": 0, "nonlinear_reference": 0, "charts": []}
    for name, pts in sorted(charts.items()):
        ref_pts = [(y, g) for y, g, _ in pts if g is not None]
        if len(ref_pts) < 3 or ols(ref_pts) is None:
            result["too_few_ticks"] += 1
            continue
        ref_a, ref_b = ols(ref_pts)
        pix_span = max(y for y, _ in ref_pts) - min(y for y, _ in ref_pts)
        val_span = abs(ref_b) * pix_span
        rms = (statistics.fmean((g - (ref_a + ref_b * y)) ** 2 for y, g in ref_pts)) ** 0.5
        if val_span == 0 or rms / val_span > LINEAR_RMS:
            result["nonlinear_reference"] += 1
            continue
        parsed = [(y, p) for y, _, p in pts if p is not None]
        all_correct = all(p is not None and abs(p - g) <= 1e-9 * max(1, abs(g)) for _, g, p in pts if g is not None)
        fit = robust_fit(parsed, strict_ambiguity) if len(parsed) >= 3 else None
        reasons = []
        if len(parsed) < 3:
            reasons.append("fewer than 3 parsed reads")
        elif fit is None:
            reasons.append("no admissible line")
        else:
            a, b, inliers, unique = fit
            if len(inliers) < MIN_INLIERS:
                reasons.append("fewer than 3 inliers")
            if len(inliers) < MIN_SUPPORT * len(pts):
                reasons.append("inliers under 60% of ticks")
            if not unique:
                reasons.append("ambiguous best line")
        entry = {"chart": name, "ticks": len(pts), "parsed": len(parsed), "all_reads_correct": all_correct,
                 "flagged": bool(reasons), "reasons": reasons}
        entry.update(tick_ys=[y for y, _, _ in pts], reads=[p for _, _, p in pts])
        if fit is not None:
            entry.update(slope_rel_err=abs(fit[1] - ref_b) / abs(ref_b),
                         offset_err=abs(fit[0] - ref_a) / val_span, inliers=len(fit[2]),
                         fit_a=fit[0], fit_b=fit[1])
        result["charts"].append(entry)
    return result


def summarize(result):
    charts = result["charts"]
    n = len(charts)
    out = {"charts_scored": n, "excluded_too_few_ticks": result["too_few_ticks"],
           "excluded_nonlinear_reference": result["nonlinear_reference"],
           "all_reads_correct": sum(c["all_reads_correct"] for c in charts),
           "flagged": sum(c["flagged"] for c in charts), "by_tolerance": {}}
    for tol in TOLS:
        def ok(c):
            return "slope_rel_err" in c and c["slope_rel_err"] <= tol and c["offset_err"] <= tol
        accepted = [c for c in charts if not c["flagged"]]
        out["by_tolerance"][str(tol)] = {
            "fit_matches_reference": sum(ok(c) for c in charts),
            "accepted_and_matches": sum(ok(c) for c in accepted),
            "accepted_but_wrong": sum(not ok(c) for c in accepted),
            "flagged_but_would_match": sum(ok(c) for c in charts if c["flagged"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rows = json.loads(args.predictions.read_text())
    ticks = tick_positions(args.root, args.split)
    out = {}
    for label, strict in (("ambiguity by inlier set", False), ("first version: ambiguity by slope", True)):
        result = analyse(rows, ticks, strict)
        summary = summarize(result)
        reasons = Counter(r for c in result["charts"] if c["flagged"] for r in c["reasons"])
        print(label, json.dumps({k: v for k, v in summary.items() if k != "by_tolerance"}),
              "flag reasons", dict(reasons))
        for tol, v in summary["by_tolerance"].items():
            print(f"   tol {tol}: {v}")
        out[label] = {"summary": summary, "charts": result["charts"]}
    if args.output:
        args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
