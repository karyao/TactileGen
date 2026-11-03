"""Why do value-axis reads lose decimal points, and does a preprocessing change fix it? (val only)

  diagnose  on val ``ylabel`` and ``value_label`` crops with the already-saved reads: how
            often the ground truth has a decimal point (or comma) and the read does not,
            how often the read invents one, crop size for each case, and a montage of
            failing crops
  fixes     re-read the same val crops with one change at a time (larger upscale, no
            binarization, mild sharpening, a narrower allow-list), then the changes that
            helped together; picks the best setting for the numeric classes

Best setting, fixed before running: the highest correct-parse rate pooled over both classes;
within 0.3 points, the lower decimal-drop rate. Test is never read here.

  python -m src.baseline.ocr_decimal_diagnosis diagnose --engine tesseract --out output/ocr/tesseract
  ~/.venvs/tactilegen-easyocr/bin/python -m src.baseline.ocr_decimal_diagnosis fixes --engine easyocr --out output/ocr/easyocr
"""
import argparse
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import cv2

from src.baseline.ocr_eval import NUMERIC, load_crops, score_crops, summarize
from src.textutil import normalize, parse_number

ALLOW_NARROW = "0123456789.,-%"
VARIANTS = {
    "base": {},
    "height 96": {"height": 96},
    "height 128": {"height": 128},
    "no binarization": {"binarize": False},
    "mild sharpening": {"sharpen": True},
    "allow only 0-9 . , - %": {"allow": ALLOW_NARROW},
}
TIE = 0.003


def has_decimal(text: str) -> bool:
    """A decimal point, or a decimal comma (not a thousands comma such as 1,000)."""
    t = normalize(text).replace(" ", "")
    if re.search(r"\d\.\d", t):
        return True
    return bool(re.search(r"\d,\d", t)) and not re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?%?", t)


def engine_by_name(name):
    if name == "tesseract":
        from src.baseline.ocr_tesseract import Tesseract
        return Tesseract()
    from src.baseline.ocr_easyocr import EasyOCR
    return EasyOCR()


def classify(row):
    gt, pr = has_decimal(row["text"]), has_decimal(row["pred"])
    return "retained" if gt and pr else "dropped" if gt else "invented" if pr else "no decimal"


def box_stats(rows):
    if not rows:
        return None
    h, w = [r["box"][3] for r in rows], [r["box"][2] for r in rows]
    q = lambda v: statistics.quantiles(v, n=4)[1:] if len(v) >= 4 else [min(v), max(v)]
    return {"n": len(rows), "median_h": statistics.median(h), "median_w": statistics.median(w),
            "h_q1_q3": [min(q(h)), max(q(h))], "w_q1_q3": [min(q(w)), max(q(w))]}


def diagnose(args):
    rows = [r for r in json.loads((args.out / "ocr_val_predictions.json").read_text()) if r["klass"] in NUMERIC]
    out = {"engine": args.engine, "classes": {}}
    for klass in sorted(NUMERIC):
        rs = [r for r in rows if r["klass"] == klass]
        groups = {k: [r for r in rs if classify(r) == k] for k in ("retained", "dropped", "invented", "no decimal")}
        gt_dec = len(groups["retained"]) + len(groups["dropped"])
        dropped = groups["dropped"]
        digits = lambda t: re.sub(r"\D", "", normalize(t))
        out["classes"][klass] = {
            "crops": len(rs), "gt_with_decimal": gt_dec,
            "dropped": len(dropped), "drop_rate": len(dropped) / max(1, gt_dec),
            "dropped_digits_right_point_missing": sum(digits(r["pred"]) == digits(r["text"]) for r in dropped),
            "dropped_empty_read": sum(not r["pred"].strip() for r in dropped),
            "gt_without_decimal": len(groups["invented"]) + len(groups["no decimal"]),
            "invented": len(groups["invented"]),
            "invent_rate": len(groups["invented"]) / max(1, len(groups["invented"]) + len(groups["no decimal"])),
            "comma_for_point": sum("," in r["pred"] and "." in r["text"] for r in groups["retained"]),
            "box": {k: box_stats(v) for k, v in groups.items()},
            "drop_rate_by_height": {
                label: [sum(1 for r in dropped if lo <= r["box"][3] < hi),
                        sum(1 for r in groups["retained"] + dropped if lo <= r["box"][3] < hi)]
                for label, lo, hi in (("<10 px", 0, 10), ("10-13 px", 10, 14), ("14-19 px", 14, 20), (">=20 px", 20, 10 ** 6))}}
    # montage of failing crops
    rnd = random.Random(0)
    fails = [r for r in rows if classify(r) in ("dropped", "invented")]
    rnd.shuffle(fails)
    tiles = []
    for r in fails[:12]:
        img = cv2.imread(r["image"])
        x, y, w, h = r["box"]
        crop = img[max(0, int(y)):int(y + h) + 1, max(0, int(x)):int(x + w) + 1]
        scale = 56 / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tile = cv2.copyMakeBorder(crop, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tile = cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        cv2.putText(tile, f"GT {r['text'][:14]}", (4, 56 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1)
        cv2.putText(tile, f"read {r['pred'].strip()[:12] or '(empty)'}", (4, 56 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        tiles.append(tile)
    if tiles:
        cell_w = max(t.shape[1] for t in tiles) + 6
        cols = 4
        sheet = 255 * __import__("numpy").ones(((tiles[0].shape[0] + 6) * ((len(tiles) + cols - 1) // cols), cols * cell_w, 3), "uint8")
        for i, t in enumerate(tiles):
            r0, c0 = (i // cols) * (tiles[0].shape[0] + 6), (i % cols) * cell_w
            sheet[r0:r0 + t.shape[0], c0:c0 + t.shape[1]] = t
        cv2.imwrite(str(args.out / "decimal_failures_montage.png"), sheet)
    (args.out / "decimal_diagnosis.json").write_text(json.dumps(out, indent=1))
    for klass, c in out["classes"].items():
        print(f"{args.engine} {klass}: {c['crops']} crops; GT has decimal {c['gt_with_decimal']}, dropped {c['dropped']} "
              f"({c['drop_rate']:.1%}; digits right but point missing {c['dropped_digits_right_point_missing']}, empty {c['dropped_empty_read']}); "
              f"GT without {c['gt_without_decimal']}, invented {c['invented']} ({c['invent_rate']:.1%}); comma for point {c['comma_for_point']}")
        for k, b in c["box"].items():
            if b:
                print(f"    {k:<11} n={b['n']:<4} median h {b['median_h']:.0f} w {b['median_w']:.0f}  h IQR {b['h_q1_q3']}  w IQR {b['w_q1_q3']}")
        print("    dropped / decimal crops by height:", c["drop_rate_by_height"])


def decimal_stats(rows):
    cls = Counter(classify(r) for r in rows)
    gt = cls["retained"] + cls["dropped"]
    plain = cls["invented"] + cls["no decimal"]
    return {"drop_rate": cls["dropped"] / max(1, gt), "invent_rate": cls["invented"] / max(1, plain)}


def evaluate(engine, crops, tuned, changes):
    """Score the numeric crops with each class's tuned setting plus ``changes``."""
    rows = []
    for klass in NUMERIC:
        cfg = dict(tuned[klass], **changes)
        rows += score_crops(engine, [c for c in crops if c["klass"] == klass], cfg)
    summary = summarize(rows)
    usable = sum(summary[k]["gt_parseable"] for k in NUMERIC)
    correct = sum(summary[k]["correct_parse_rate"] * summary[k]["gt_parseable"] for k in NUMERIC)
    return {"pooled_correct_parse": correct / usable,
            "ylabel_correct_parse": summary["ylabel"]["correct_parse_rate"],
            "value_label_correct_parse": summary["value_label"]["correct_parse_rate"],
            **decimal_stats(rows)}


def fixes(args):
    engine = engine_by_name(args.engine)
    tuned = json.loads((args.out / "ocr_tuning.json").read_text())["chosen"]
    crops = [c for c in load_crops(args.root, "val")[0] if c["klass"] in NUMERIC]
    results = {}
    for name, changes in VARIANTS.items():
        results[name] = dict(evaluate(engine, crops, tuned, changes), changes=changes)
        print(f"{args.engine}: {name:<24}", {k: round(v, 4) for k, v in results[name].items() if k != "changes"}, flush=True)
    base = results["base"]["pooled_correct_parse"]
    helped = [n for n in VARIANTS if n != "base" and results[n]["pooled_correct_parse"] > base]
    heights = [n for n in helped if n.startswith("height")]
    combo = {}
    if heights:
        combo.update(VARIANTS[max(heights, key=lambda n: results[n]["pooled_correct_parse"])])
    for n in helped:
        if not n.startswith("height"):
            combo.update(VARIANTS[n])
    if len(helped) > 1:
        results["combined (those that helped)"] = dict(evaluate(engine, crops, tuned, combo), changes=combo)
        print(f"{args.engine}: combined {helped}", {k: round(v, 4) for k, v in results["combined (those that helped)"].items() if k != "changes"})
    top = max(r["pooled_correct_parse"] for r in results.values())
    near = [n for n, r in results.items() if top - r["pooled_correct_parse"] <= TIE]
    best = min(near, key=lambda n: results[n]["drop_rate"])
    print(f"{args.engine}: best by val = {best}")
    (args.out / "decimal_fixes_val.json").write_text(json.dumps(
        {"results": results, "best": best, "best_config": {k: dict(tuned[k], **results[best]["changes"]) for k in NUMERIC}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("diagnose", "fixes"))
    ap.add_argument("--engine", choices=("tesseract", "easyocr"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    args = ap.parse_args()
    {"diagnose": diagnose, "fixes": fixes}[args.command](args)


if __name__ == "__main__":
    main()
