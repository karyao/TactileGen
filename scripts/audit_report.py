"""Tables, failure groups, upper bounds and contact sheets from results/failure_audit.csv + results/extras/.

usage: python scripts/audit_report.py RESULTS_DIR CACHE_DIR TREE
Run with an interpreter that has pandas + matplotlib (the cache pickles only need src.contracts).
Everything descriptive is over all cached charts; the extra-bar threshold is chosen on the 20% tuning split
and reported on the 80% report split (same split as scripts/audit_extras.py).
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, CACHE, TREE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(TREE))
sys.dont_write_bytecode = True
PNG = Path.home() / "lsi-code/TactileGen/data/png_test/png"

# group name -> (description, fix class, heuristic_fixable)
GROUPS = {
    "tiny_bars_undetected": ("Every missed bar is near-zero (<5% of axis max or <=5 px tall); YOLO does not find it and the rescue "
                             "step either guesses 0 or the count mismatch/series flag blocks the chart.",
                             "detector retrain (small-bar data) or pixel heuristic", True),
    "real_bars_undetected": ("At least one clearly visible bar (not near-zero) has no surviving detection.",
                             "detector retrain", False),
    "series_legend_colour_match": ("Detected legend swatches do not resolve to bar colours (weak_match/ambiguous), so series are unnamed/ambiguous.",
                                   "pixel/heuristic fix (colour matching) - hypothesis", True),
    "category_label_ocr": ("Bars are right but a category label is misread, hyphen-dropped or unreadable.",
                           "needs VLM (or better OCR) - hypothesis", False),
    "title_axis_title_ocr": ("Title / axis title / series name text could not be read, so the pipeline asks for input.",
                             "needs VLM (or better OCR) - hypothesis", False),
    "false_block_gate": ("Pipeline's answer is correct but a BANA/completeness gate blocked it.",
                         "pixel/heuristic fix (rule change)", True),
    "y_axis_calibration": ("Y ticks dropped / not arithmetic / leading zero / value clipped at axis end.",
                           "pixel/heuristic fix - hypothesis", True),
    "other_rare": ("Pairing ambiguity, extra bars, interpretation errors, layout errors, residual wrong values.",
                   "mixed; long tail", False),
}
VLM_GROUPS = {"category_label_ocr", "title_axis_title_ocr"}  # hypothesis: image alone could resolve these


def assign_group(r):
    c = r["primary_cause"]
    if c == "detector_missing_bars":
        return "tiny_bars_undetected" if r["missed_near_zero"] >= r["det_missing_gt"] else "real_bars_undetected"
    return {"series_separation": "series_legend_colour_match", "series_false_alarm_single_series": "series_legend_colour_match",
            "ocr_category_text": "category_label_ocr", "ocr_title_or_axis_title": "title_axis_title_ocr",
            "false_block": "false_block_gate",
            "y_calibration": "y_axis_calibration"}.get(c, "other_rare")


def indicators(r):
    """Independent (priority-free) cause indicators, used only to count charts with a single cause."""
    flags = set(str(r["flag_names"]).split(";")) - {""}
    br = str(r["block_reasons"])
    out = set()
    if r["det_missing_gt"] > 0:
        out.add("tiny_bars_undetected" if r["missed_near_zero"] >= r["det_missing_gt"] else "real_bars_undetected")
    if "series_ambiguous" in flags or str(r["series_ok"]) == "False":
        out.add("series_legend_colour_match")
    if str(r["labels_ok"]) == "False" or "category_unreadable" in flags or "category[n]" in br:
        out.add("category_label_ocr")
    if any(k in br for k in ("title", "series_name[n]")):
        out.add("title_axis_title_ocr")
    if flags & {"y_calibration_ambiguous", "y_tick_leading_zero", "y_ticks_not_arithmetic", "value_clipped_at_axis",
                "baseline_outside_plot"}:
        out.add("y_axis_calibration")
    if r["det_extra"] > 0 or r["status"] == "blocked" and r["spec_value_errors"] == "" and r["pred_series"] == "":
        out.add("other_rare")
    if flags & {"category_count_mismatch", "category_ambiguous"} and "series_legend_colour_match" not in out \
            and not out & {"tiny_bars_undetected", "real_bars_undetected"}:
        out.add("other_rare")
    return out


EFFORT = {"false_block_gate": (1, "S: rule change (key-runover fix exists on a branch)"),
          "y_axis_calibration": (2, "S-M: tick/clip heuristics"),
          "series_legend_colour_match": (3, "M: colour matching / drop weak bars then rescue"),
          "tiny_bars_undetected": (4, "M-L: detector retrain or pixel scan for near-zero bars"),
          "title_axis_title_ocr": (4, "M: OCR tuning or VLM text read"),
          "category_label_ocr": (4, "M-L: OCR tuning or VLM text read"),
          "real_bars_undetected": (5, "L: detector retrain + data"),
          "other_rare": (5, "L: long tail")}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def severity(r):
    n = pd.to_numeric(r["n_bad_values"], errors="coerce")
    base = (0 if pd.isna(n) else n) + r["det_missing_gt"] + r["det_extra"]
    return base + 0.01 * len([f for f in str(r["flags"]).split(";") if f])


def contact_sheet(group, ids, df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ids), figsize=(7 * len(ids), 7.5))
    axes = np.atleast_1d(axes)
    for ax, (role, cid) in zip(axes, ids):
        r = df[df.chart_id == cid].iloc[0]
        img = plt.imread(PNG / f"{cid}.png")
        ax.imshow(img); ax.axis("off")
        c = pickle.loads((CACHE / f"{cid}.pkl").read_bytes())
        for m in c["record"]["models"]:
            for b in m["bboxes"]:
                ax.add_patch(patches.Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, ec="#2ca02c", lw=1.2, ls="--"))
        for d in c["frame"].detections:
            if d.role == "vertical_bar":
                ax.add_patch(patches.Rectangle((d.box.x1, d.box.y1), d.box.x2 - d.box.x1, d.box.y2 - d.box.y1, fill=False,
                                               ec="#d62728", lw=1.0, alpha=min(1, 0.3 + d.confidence)))
        txt = (f"{role}: chart {cid} | exit {r['exit']} | {r['status']}\n"
               f"blocks: {r['block_reasons'] or '-'}\nflags: {r['flag_names'] or '-'}\n"
               f"GT   series={r['gt_series']} cats={r['gt_categories']} bars={r['gt_bars']}\n"
               f"PRED series={r['pred_series']} cats={r['pred_categories']} bars={r['pred_bars']}\n"
               f"max |err|/axis max: {r['max_err_frac'] or 'n/a'}  bad values: {r['n_bad_values']}\n"
               f"GT cats:   {str(r['gt_categories_text'])[:70]}\nPRED cats: {str(r['pred_categories_text'])[:70]}\n"
               f"GT series:   {str(r['gt_series_names'])[:70]}\nPRED series: {str(r['pred_series_names'])[:70]}")
        ax.set_title(txt, fontsize=7.5, loc="left", family="monospace")
    fig.tight_layout()
    fig.savefig(outdir / "contact_sheet.png", dpi=110)
    plt.close(fig)


def main():
    a = pd.read_csv(RES / "failure_audit.csv", dtype={"chart_id": str, "n_bad_values": object, "max_err_frac": object}).fillna("")
    for col in ("emitted_and_correct", "correct", "false_block"):
        a[col] = a[col].astype(str) == "True"
    a["group"] = ""
    fail = a[~a.emitted_and_correct].copy()
    a.loc[fail.index, "group"] = fail.apply(assign_group, axis=1)
    fail["group"] = a.loc[fail.index, "group"]
    a.to_csv(RES / "failure_audit.csv", index=False)
    N, F = len(a), len(fail)
    em = a[a.status == "emitted"]
    out = ["# TactileGen vertical-bar failure audit", "",
           f"Population: all {N} PlotQA test vbar_categorical charts, cache of commit 2412bb2e (`--yes --strict`).", ""]

    # --- Step 0 rates
    ec = int(a.emitted_and_correct.sum())
    out += ["## Rates", "",
            f"- (a) emitted without blocking: {len(em)}/{N} = {len(em) / N:.1%}  <- the 76.3% baseline",
            f"- (b) of emitted, content matches ground truth: {ec}/{len(em)} = {ec / len(em):.1%}",
            f"- (c) emitted AND correct, of all charts: {ec}/{N} = {ec / N:.1%}",
            f"- emitted but wrong: {len(em) - ec} ({(len(em) - ec) / N:.1%} of all); blocked: {N - len(em)} ({(N - len(em)) / N:.1%});"
            f" total not (emitted AND correct): {F} ({F / N:.1%})",
            "- 'correct' = series count, category count, category text (normalised case/whitespace), series names and every value "
            f"within 1% of the GT y-axis maximum. Titles/axis titles are not compared. Also lenient variants below.", ""]
    lenient_vals = int((em.values_ok.astype(str) == "True").sum())
    loose = int(((em.values_ok.astype(str) == "True") & (em.series_ok.astype(str) == "True") & (em.labels_ok_loose.astype(str) == "True")).sum())
    out += [f"- lenient: emitted with values within 1% & series aligned (text ignored): {lenient_vals}/{len(em)} = {lenient_vals / len(em):.1%}",
            f"- lenient: as above with category text compared alphanumerics-only (ignores hyphen/space/case): {loose}/{len(em)} = {loose / len(em):.1%}", ""]

    # --- Step 1 cause table
    cause = fail.groupby(["primary_cause"]).agg(charts=("chart_id", "count"), blocked=("status", lambda s: (s == "blocked").sum()),
                                                emitted_wrong=("status", lambda s: (s == "emitted").sum())).sort_values("charts", ascending=False)
    cause["% of failures"] = (100 * cause.charts / F).round(1)
    cause["% of all charts"] = (100 * cause.charts / N).round(1)
    cause = cause.reset_index()
    cause.to_csv(RES / "cause_table.csv", index=False)
    out += ["## Failure causes (one primary cause per chart)", "",
            "Priority order (first match wins): " + " > ".join(
                ["no_spec_interpretation_error", "detector_missing_bars", "detector_extra_bars", "ocr_category_text",
                 "ocr_title_or_axis_title", "y_calibration", "series_false_alarm_single_series", "series_separation",
                 "category_pairing", "bar_rescue_wrong", "other_wrong_values", "strict_bana_layout"]) +
            ". `false_block` = blocked although content is correct (exit 1/3 only).", "", md_table(cause), "",
            "Because detector causes outrank the others, a chart blocked by `series_ambiguous` that also lost a bar is counted under "
            "`detector_missing_bars`. The block-reason view below is not subject to that ordering.", ""]
    blocked = a[a.status == "blocked"]
    br = blocked.block_reasons.value_counts().head(15).rename_axis("block reasons (all raised, ';'-joined)").reset_index(name="charts")
    out += ["### Blocked charts by exact set of block reasons (multi-flag overlaps)", "", md_table(br), ""]
    ov = a[a.flag_names != ""].flag_names.value_counts().head(12).rename_axis("flag set (spec flags, any status)").reset_index(name="charts")
    out += ["### Most common spec flag sets", "", md_table(ov), ""]

    # --- series_ambiguous
    s = pd.read_csv(RES / "extras/series_ambiguous.csv", dtype={"chart_id": str}).fillna("")
    n_tag = int(blocked.block_reasons.str.contains("series_ambiguous").sum())
    ms = a[a.gt_series > 1]
    a["sa"] = a.chart_id.isin(s.chart_id)
    sm = s.merge(a[["chart_id", "correct", "values_ok", "series_ok", "names_ok", "labels_ok"]], on="chart_id")
    out += ["## series_ambiguous", "",
            f"- Verified: {n_tag} charts are blocked with the series_ambiguous tag (summary.md said 1577); the flag is on {len(s)} specs; "
            f"all are exit 3.",
            f"- Rate: {a[a.gt_series > 1].sa.mean():.1%} of multi-series charts ({int(a[a.gt_series > 1].sa.sum())}/{len(ms)}), "
            f"{a[a.gt_series == 1].sa.mean():.2%} of single-series ({int(a[a.gt_series == 1].sa.sum())}/{(a.gt_series == 1).sum()}).",
            f"- False alarms (GT is a single series): {int((s.gt_series == 1).sum())}. Flagged charts whose reconstructed content is "
            f"nevertheless correct: {int((sm.correct.astype(str) == 'True').sum())}/{len(sm)}.", ""]
    mech = s.apply(lambda r: "conflicting_swatch" if r.conflicting_swatches > 0 else
                   "unresolved:" + "+".join(sorted(k.split(":")[0] for k in r.status_counts.split(";") if k and not k.startswith("matched"))), axis=1)
    out += ["Mechanism that raised the flag (measured, from legend-mapping statuses of the detected bars):", "",
            md_table(mech.value_counts().rename_axis("mechanism").reset_index(name="charts")), "",
            "Primary reason by priority (single_series > extra bars > missing bars > conflicting swatch > legend not detected > "
            "legend count mismatch > similar GT colours > unresolved status):", "",
            md_table(s.reason.value_counts().rename_axis("reason").reset_index(name="charts").assign(
                pct=lambda d: (100 * d.charts / len(s)).round(1))), ""]
    cw = a[a.chart_id.isin(s.chart_id)].group.replace("", "(correct but blocked)").value_counts().rename_axis("group (primary cause)").reset_index(name="series_ambiguous charts")
    out += ["Where the 1577 series_ambiguous charts land in the primary-cause groups (the priority order puts bar-detection first):", "",
            md_table(cw), ""]
    cm = pd.to_numeric(s.min_gt_color_dist, errors="coerce")
    base = a[(a.gt_series > 1) & (~a.sa)]
    out += ["Co-factors (each over all flagged charts; not exclusive; association, not proven cause):", "",
            f"- detected legend swatches == GT legend entries: {int((s.det_swatches == s.gt_legend_items).sum())}/{len(s)}",
            f"- any GT bar not detected: {(s.det_missing > 0).mean():.1%} of flagged vs {(base.det_missing_gt > 0).mean():.1%} of non-flagged multi-series charts",
            f"- any extra detected bar: {(s.det_extra > 0).mean():.1%} vs {(base.det_extra > 0).mean():.1%}",
            f"- closest pair of GT series colours < 80 RGB distance: {(cm < 80).mean():.1%} of flagged (n={cm.notna().sum()}); "
            f"overall multi-series baseline not computed (unverified)", ""]

    # --- extra-bar rule
    bars = pd.read_csv(RES / "extras/extra_bar_conf.csv", dtype={"chart_id": str})
    ex_charts = set(a[a.det_extra > 0].chart_id)
    inex = bars[bars.chart_id.isin(ex_charts)]
    q = lambda x: x.quantile([.1, .5, .9]).round(3).tolist()
    out += ["## Extra-bar detections", "",
            f"Charts with more surviving bars than GT bars (unmatched at IoU 0.5): {len(ex_charts)}/{N}; false bars: {int((~inex.is_true_bar).sum())}.",
            f"- confidence of false bars (p10/p50/p90): {q(inex[~inex.is_true_bar].conf)} (n={int((~inex.is_true_bar).sum())})",
            f"- confidence of true bars in those charts: {q(inex[inex.is_true_bar].conf)} (n={int(inex.is_true_bar.sum())})",
            f"- confidence of true bars in all charts: {q(bars[bars.is_true_bar].conf)} (n={int(bars.is_true_bar.sum())})", ""]
    sw = pd.read_csv(RES / "extras/extra_bar_rule_sweep.csv")
    ns = sw[(sw.variant == sw.variant.iloc[0]) & (sw.tau == sw.tau.iloc[0])].set_index("split").n
    out += [f"Split (seed 0 over sorted chart ids): tuning n={int(ns['tune'])}, report n={int(ns['report'])} (charts with a spec).", ""]
    picks = []
    for v, g in sw.groupby("variant"):
        t = g[g.split == "tune"]
        best = t.sort_values(["net", "tau"], ascending=[False, True]).iloc[0]
        picks.append((v, best.tau))
    out += ["Rule: remove a surviving bar when `conf < tau` AND (A: its legend assignment is not `matched` / B: the chart is flagged "
            "series_ambiguous); legend matching + reconstruct are re-run from cached detections (no detector re-run) and a chart "
            "'passes' when completeness is READY and content matches GT (pre-layout). fixed = was not passing, now passes; "
            "broke = was passing, now not. **Not enabled in the pipeline.**", "",
            "Selected on tuning split (max net = fixed - broke): " + ", ".join(f"{v}: tau={t}" for v, t in picks), ""]
    tab = sw[sw.split == "report"][["variant", "tau", "affected", "fixed", "broke", "net", "fixed_only_false_bars_removed", "fixed_true_bar_removed",
                                    "fixed_chart_had_extras", "true_removed", "extra_removed"]]
    tab_tune = sw[sw.split == "tune"][["variant", "tau", "fixed", "broke", "net"]].rename(columns={"fixed": "tune_fixed", "broke": "tune_broke", "net": "tune_net"})
    tab = tab.merge(tab_tune, on=["variant", "tau"])
    tab.to_csv(RES / "extra_bar_rule_report_split.csv", index=False)
    out += ["Reading the sweep: most fixes remove true bars (`fixed_true_bar_removed`), not false ones. Dropping unresolved bars clears "
            "`series_ambiguous`, and the rescue step re-measures the removed bars from pixels. Extra bars exist in only "
            f"{len(ex_charts)} charts and have high confidence, so a confidence threshold cannot target them.", "",
            "All thresholds, 80% report split (columns tune_* = 20% tuning split):", "", md_table(tab), ""]

    # --- Step 2 groups
    gt = fail.groupby("group").agg(charts=("chart_id", "count"), blocked=("status", lambda s: (s == "blocked").sum()),
                                   emitted_wrong=("status", lambda s: (s == "emitted").sum())).reindex(list(GROUPS)).fillna(0).astype(int)
    gt["% of failures"] = (100 * gt.charts / F).round(1)
    rows = []
    gdir = RES / "failure_groups"
    ex_rows = []
    for name, (desc, fix, heur) in GROUPS.items():
        g = fail[fail.group == name].copy()
        if g.empty:
            continue
        g["sev"] = g.apply(severity, axis=1)
        gs = g.sort_values(["sev", "chart_id"])
        typical = gs.iloc[len(gs) // 2].chart_id
        borderline = gs[gs.sev > 0.5].iloc[0].chart_id if (gs.sev > 0.5).any() else gs.iloc[0].chart_id
        extreme = gs.iloc[-1].chart_id
        ids = [("typical", typical), ("borderline", borderline), ("extreme", extreme)]
        contact_sheet(name, ids, a, gdir / name)
        for role, cid in ids:
            ex_rows.append({"group": name, "role": role, "chart_id": cid, "image": str(PNG / f"{cid}.png")})
        rows.append({"group": name, "charts": len(g), "% of failures": round(100 * len(g) / F, 1),
                     "why it fails": desc, "estimated fix": fix,
                     "examples (typical/borderline/extreme)": f"{typical}/{borderline}/{extreme}"})
    pd.DataFrame(ex_rows).to_csv(RES / "failure_groups/examples.csv", index=False)
    out += ["## Failure groups", "", md_table(pd.DataFrame(rows)), "",
            "Image paths: `data/png_test/png/<id>.png`; contact sheets in `results/failure_groups/<group>/contact_sheet.png` "
            "(green dashed = GT bar boxes, red = surviving YOLO bars). Group assignment follows the primary-cause priority order; "
            "'estimated fix' is a hypothesis, not tested.", ""]

    # --- Step 3 upper bounds
    ind = fail.apply(indicators, axis=1)
    ub = []
    for name, (desc, fix, heur) in GROUPS.items():
        n = int((fail.group == name).sum())
        sole = int(sum(1 for gname, i in zip(fail.group, ind) if gname == name and i <= {name}))
        ub.append({"group": name, "charts": n, "sole-cause charts (strict floor)": sole,
                   "gain if fixed alone, upper (pts)": round(100 * n / N, 2),
                   "gain, sole-cause floor (pts)": round(100 * sole / N, 2),
                   "effort (my estimate)": EFFORT[name][1],
                   "upper gain per effort-step": round(100 * n / N / EFFORT[name][0], 2),
                   "(c) after": f"{100 * (ec + n) / N:.1f}%", "heuristic-fixable?": "yes" if heur else "no",
                   "VLM-plausible?": "yes (hypothesis)" if name in VLM_GROUPS else "no/unknown"})
    ubd = pd.DataFrame(ub).sort_values("upper gain per effort-step", ascending=False)
    heur_n = int(fail.group.isin([k for k, v in GROUPS.items() if v[2]]).sum())
    vlm_n = int(fail.group.isin(VLM_GROUPS).sum())
    fb = int(a.false_block.sum())
    fb_bana = int((a.false_block & a.block_reasons.str.contains("bana:label_owner_distance")).sum())
    nb = len(blocked)
    out += ["## Upper bounds (no fix implemented; assumes each group is fully fixed and the chart then emits correctly)", "", md_table(ubd), "",
            f"- All heuristic-fixable groups fixed: +{100 * heur_n / N:.1f} pts -> (c) {100 * (ec + heur_n) / N:.1f}% ({heur_n} charts).",
            f"- All groups fixed would reach 100% by construction; the long-tail group is `other_rare`.",
            f"- Needs-VLM groups (hypothesis): {vlm_n} charts = {100 * vlm_n / F:.1f}% of failures = +{100 * vlm_n / N:.1f} pts of (c).",
            f"- False blocks: {fb}/{nb} blocked charts ({100 * fb / nb:.1f}%) have correct content, = +{100 * fb / N:.1f} pts of (c); "
            f"{fb_bana} of them are BANA `label_owner_distance` (verified: with the fix commit 189c31c8 applied, all 203 exit 0, both on 2412bb2e + that commit and on the branch tip).",
            f"- Completeness-blocker (exit 3) false blocks: {int((a.false_block & (a.exit == 3)).sum())}/{int((a.exit == 3).sum())}.", ""]
    (RES / "failure_report.md").write_text("\n".join(out))
    cat = fail[fail.group == "category_label_ocr"]
    out.append(f"- Of {len(cat)} category_label_ocr charts, {int((cat.labels_ok_loose.astype(str) == 'True').sum())} differ only by "
               "punctuation/space/case (labels_ok_loose) and {} are real misreads or unreadable.".format(
                   int((cat.labels_ok_loose.astype(str) != 'True').sum())))
    (RES / "failure_report.md").write_text("\n".join(out))
    ubd.to_csv(RES / "upper_bounds.csv", index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
