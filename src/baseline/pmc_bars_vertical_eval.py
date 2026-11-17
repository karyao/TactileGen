"""Compare the vertical-only fine-tune with variant A on vertical images.

Confidence for each model is chosen on the 150 vertical val images (best F1 at
IoU 0.5). Variant A is also shown at its original 0.40, chosen on all of val.
The 139 vertical test images are reporting only; nothing is chosen on them.

  python -m src.baseline.pmc_bars_vertical_eval --output-dir <dir>
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from src.baseline.pmc_bars_baseline import CONF_GRID, IOUS, load_split, predict, prf, score
from src.dataset.bar_orientation import BAR

MODELS = {
    "A (all charts)": "output/pmc_bars_finetune/coco/weights/best.pt",
    "V (vertical only)": "output/pmc_bars_vertical/coco/weights/best.pt",
}
A_ORIGINAL_CONF = 0.40


def table(items, predictions, conf):
    s = score(items, predictions, conf)["vertical"]
    return {"images": s["images"], "exact_count": s["exact"] / s["images"],
            **{str(i): dict(zip(("precision", "recall", "f1"), prf(s[i]))) for i in IOUS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    split = {s: [i for i in load_split(args.root, s) if i[1] == "vertical"] for s in ("val", "test")}
    out = {}
    for name, weights in MODELS.items():
        model = YOLO(weights)
        pred = {s: predict(model, [p for p, _, _ in items], BAR) for s, items in split.items()}
        by_conf = {c: prf(score(split["val"], pred["val"], c)["vertical"][0.5])[2] for c in CONF_GRID}
        tuned = max(by_conf, key=lambda c: (by_conf[c], c))
        confs = {"tuned on vertical val": tuned}
        if name.startswith("A"):
            confs["original (all val)"] = A_ORIGINAL_CONF
        out[name] = {"val_f1_by_conf": by_conf, "rows": {
            label: {"confidence": c, "val": table(split["val"], pred["val"], c),
                    "test_reporting_only": table(split["test"], pred["test"], c)}
            for label, c in confs.items()}}
    (args.output_dir).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vertical_eval.json").write_text(json.dumps(out, indent=1))
    for name, m in out.items():
        for label, r in m["rows"].items():
            for part in ("val", "test_reporting_only"):
                t = r[part]
                print(f"{name:<18}{label:<22}conf {r['confidence']:<5}{part:<20}imgs {t['images']:>3} exact {t['exact_count']:.3f} | "
                      + " | ".join(f"IoU{i}: P{t[i]['precision']:.3f} R{t[i]['recall']:.3f} F1 {t[i]['f1']:.3f}" for i in ("0.5", "0.75", "0.9")))


if __name__ == "__main__":
    main()
