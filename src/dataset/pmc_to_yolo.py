#!/usr/bin/env python3
"""Convert ChartDete PMC element_detection (COCO) to YOLO format, re-splitting train+val BY PAPER.

Why: the official train/val split puts figures from the same paper (same PMC id) on both sides,
which inflates val scores. Here train.json + val.json are merged and split so every paper lands
wholly in train or wholly in val. split3_test is converted as-is (it shares no papers with either).

Usage:
  python -m src.dataset.pmc_to_yolo \
      --root data/pmc_2022/pmc_coco/element_detection \
      --out  data/pmc_yolo \
      --val-frac 0.1 --seed 0

Optional class mapping (after you verify what xlabel/ylabel vs x_tick/y_tick mean):
  --class-map map.json     {"xlabel": "tick_label_x", "tick_grouping": null, ...}
  null drops a class. Unlisted classes keep their original name.

Images are symlinked by default (use --copy to copy). Output is gitignore-worthy: CC BY-NC-SA data.
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

PAPER_RX = re.compile(r"(PMC\d+)")


def load_coco(json_path: Path, img_dir: Path):
    d = json.load(open(json_path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    items = []
    for im in d["images"]:
        name = os.path.basename(im["file_name"])
        items.append({
            "name": name,
            "path": img_dir / name,
            "w": im["width"], "h": im["height"],
            "anns": anns.get(im["id"], []),
            "cats": cats,
        })
    return items


def paper_id(name: str) -> str:
    m = PAPER_RX.search(name)
    return m.group(1) if m else name  # unmatched files become their own group


def write_split(items, split, out: Path, names_to_idx, class_map, copy, stats):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for it in items:
        if not it["path"].exists():
            stats["missing_image"] += 1
            continue
        lines = []
        for a in it["anns"]:
            cname = it["cats"][a["category_id"]]
            cname = class_map.get(cname, cname)
            if cname is None:
                stats["dropped_class"] += 1
                continue
            x, y, w, h = a["bbox"]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(it["w"]), x + w), min(float(it["h"]), y + h)
            if x1 - x0 < 1 or y1 - y0 < 1:
                stats["degenerate_box"] += 1
                continue
            cx, cy = (x0 + x1) / 2 / it["w"], (y0 + y1) / 2 / it["h"]
            bw, bh = (x1 - x0) / it["w"], (y1 - y0) / it["h"]
            lines.append(f"{names_to_idx[cname]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["boxes_" + split] += 1
        dst = out / "images" / split / it["name"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(it["path"], dst)
        else:
            os.symlink(it["path"].resolve(), dst)
        (out / "labels" / split / (Path(it["name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images_" + split] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="element_detection folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of PAPERS sent to val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-map", type=Path)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    root = args.root
    trainval = load_coco(root / "train.json", root / "train") + load_coco(root / "val.json", root / "val")
    test = load_coco(root / "split3_test.json", root / "split3_test")
    dup = [n for n, c in Counter(i["name"] for i in trainval + test).items() if c > 1]
    assert not dup, f"duplicate file names across splits, e.g. {dup[:3]}"
    class_map = json.load(open(args.class_map)) if args.class_map else {}

    # class list: original names after mapping, stable order by original COCO id
    all_cats = {}
    for it in trainval + test:
        all_cats.update(it["cats"])
    names = []
    for _, n in sorted(all_cats.items()):
        n = class_map.get(n, n)
        if n is not None and n not in names:
            names.append(n)
    names_to_idx = {n: i for i, n in enumerate(names)}

    # split by paper
    papers = sorted({paper_id(it["name"]) for it in trainval})
    random.Random(args.seed).shuffle(papers)
    n_val = max(1, round(len(papers) * args.val_frac))
    val_papers = set(papers[:n_val])
    train_items = [i for i in trainval if paper_id(i["name"]) not in val_papers]
    val_items = [i for i in trainval if paper_id(i["name"]) in val_papers]

    # sanity: no paper on both sides, and none shared with test
    tr_p = {paper_id(i["name"]) for i in train_items}
    va_p = {paper_id(i["name"]) for i in val_items}
    te_p = {paper_id(i["name"]) for i in test}
    assert not (tr_p & va_p), "paper in both train and val"
    assert not ((tr_p | va_p) & te_p), "paper shared with test"

    stats = Counter()
    write_split(train_items, "train", args.out, names_to_idx, class_map, args.copy, stats)
    write_split(val_items, "val", args.out, names_to_idx, class_map, args.copy, stats)
    write_split(test, "test", args.out, names_to_idx, class_map, args.copy, stats)

    out = args.out.resolve()
    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    (args.out / "val_papers.json").write_text(json.dumps(sorted(val_papers)))

    print(f"papers: train {len(tr_p)}, val {len(va_p)}, test {len(te_p)}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"classes ({len(names)}): {names}")
    print(f"wrote {args.out}/data.yaml")


if __name__ == "__main__":
    main()
