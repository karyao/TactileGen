"""Fine-tune YOLO on data/pmc_yolo_bars (19 classes) and compare variants on val.

Subcommands:
  train --variant {coco,plotqa}   one training run; records wall time and hardware
  select                          score every trained variant on val and name a winner
  test --variant NAME             score the winner on test, ONCE (refuses a second run)

Variants share epochs, image size, batch, seed and augmentation (mosaic off, as in
the best PlotQA run, models/best_no_mosaic.pt) and differ only in initial weights:
  coco    yolov8s.pt, COCO-pretrained
  plotqa  models/best_no_mosaic.pt, trained on synthetic PlotQA charts
A PlotQA+PMC mix needs dataset_yolo_vbars_clean, which is not in the repository.
Training output goes under --runs (default output/pmc_bars_finetune, gitignored).
"""

import argparse
import json
import platform
import time
from pathlib import Path

from src.baseline.pmc_bars_baseline import (
    CONF_GRID, GROUPS, IOUS, load_split, predict, prf, score,
)
from src.dataset.bar_orientation import BAR

VARIANTS = {"coco": "yolov8s.pt", "plotqa": "models/best_no_mosaic.pt"}
# Same settings as the PlotQA no-mosaic run (its checkpoint's train_args).
TRAIN_ARGS = dict(epochs=50, imgsz=640, batch=16, seed=0, mosaic=0, close_mosaic=10,
                  patience=100, optimizer="MuSGD", lr0=0.01, deterministic=True, workers=8)


def hardware() -> dict:
    import torch
    import ultralytics
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
    }


def train(args):
    from ultralytics import YOLO
    runs = args.runs.resolve()
    start = time.perf_counter()
    model = YOLO(VARIANTS[args.variant])
    model.train(data=str((args.root / "data.yaml").resolve()), project=str(runs),
                name=args.variant, exist_ok=False, **TRAIN_ARGS)
    seconds = time.perf_counter() - start
    info = {"variant": args.variant, "init_weights": VARIANTS[args.variant],
            "seconds": round(seconds, 1), "train_args": TRAIN_ARGS, "hardware": hardware()}
    (runs / args.variant / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"{args.variant}: {seconds / 60:.1f} min")


def _score_weights(weights, items):
    from ultralytics import YOLO
    return predict(YOLO(str(weights)), [p for p, _, _ in items], BAR)


def _table(items, predictions, conf):
    stats = score(items, predictions, conf)
    return {g: {"images": s["images"], "exact_count": s["exact"] / s["images"],
                "iou": {str(i): dict(zip(("precision", "recall", "f1"), prf(s[i]))) for i in IOUS}}
            for g, s in stats.items() if s["images"]}


def select(args):
    val = load_split(args.root, "val")
    out = {}
    for name in VARIANTS:
        weights = args.runs / name / "weights" / "best.pt"
        if not weights.is_file():
            print(f"skipping {name}: no {weights}")
            continue
        predictions = _score_weights(weights, val)
        by_conf = {c: prf(score(val, predictions, c)["all"][0.5])[2] for c in CONF_GRID}
        conf = max(by_conf, key=lambda c: (by_conf[c], c))
        out[name] = {"confidence": conf, "val_f1_at_0.5_by_conf": by_conf,
                     "val": _table(val, predictions, conf)}
    for name, r in out.items():
        a = r["val"]["all"]
        print(f"{name:<7} conf {r['confidence']}  F1@0.5 {a['iou']['0.5']['f1']:.3f}  "
              f"F1@0.9 {a['iou']['0.9']['f1']:.3f}  exact {a['exact_count']:.3f}")
    (args.runs / "val_selection.json").write_text(json.dumps(out, indent=1))


def test(args):
    result = args.runs / f"test_{args.variant}.json"
    if result.exists():
        raise SystemExit(f"test already scored once for {args.variant}: {result}")
    chosen = json.loads((args.runs / "val_selection.json").read_text())[args.variant]
    items = load_split(args.root, "test")
    predictions = _score_weights(args.runs / args.variant / "weights" / "best.pt", items)
    table = _table(items, predictions, chosen["confidence"])
    result.write_text(json.dumps({"variant": args.variant, "confidence": chosen["confidence"],
                                  "test": table}, indent=1))
    for g in GROUPS:
        if g in table:
            t = table[g]
            print(f"{g:<10} imgs {t['images']:>4} exact {t['exact_count']:.3f} " + " | ".join(
                f"IoU{i}: P{v['precision']:.3f} R{v['recall']:.3f} F1 {v['f1']:.3f}"
                for i, v in t["iou"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "select", "test"))
    ap.add_argument("--variant", choices=tuple(VARIANTS))
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--runs", type=Path, default=Path("output/pmc_bars_finetune"))
    args = ap.parse_args()
    if args.command in ("train", "test") and not args.variant:
        ap.error(f"{args.command} needs --variant")
    {"train": train, "select": select, "test": test}[args.command](args)


if __name__ == "__main__":
    main()
