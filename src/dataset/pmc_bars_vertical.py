#!/usr/bin/env python3
"""Make a vertical-bars-only copy of data/pmc_yolo_bars for train and val.

Train and val keep only images that bar_orientation.orientation() calls
"vertical"; horizontal and undecided charts are dropped. The test split is copied
whole and unchanged. Images are symlinked and label files copied.

Usage:
  python -m src.dataset.pmc_bars_vertical --src data/pmc_yolo_bars --out data/pmc_yolo_bars_vertical
"""
import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from src.dataset.bar_orientation import orientation, read_label_boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--out", type=Path, default=Path("data/pmc_yolo_bars_vertical"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new --out")
    counts = {}
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        seen = Counter()
        for image in sorted((args.src / "images" / split).iterdir()):
            label = args.src / "labels" / split / (image.stem + ".txt")
            with Image.open(image) as img:
                kind = orientation(read_label_boxes(label, img.size))
            seen[kind] += 1
            if split != "test" and kind != "vertical":
                continue
            os.symlink(image.resolve(), args.out / "images" / split / image.name)
            shutil.copy2(label, args.out / "labels" / split / label.name)
        counts[split] = seen
    names = (args.src / "data.yaml").read_text().split("names:")[1]
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:{names}")
    for split, seen in counts.items():
        kept = seen["vertical"] if split != "test" else sum(seen.values())
        print(f"{split}: kept {kept} of {sum(seen.values())}  " + ", ".join(f"{k} {v}" for k, v in sorted(seen.items())))


if __name__ == "__main__":
    main()
