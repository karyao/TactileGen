#!/usr/bin/env python3
"""Convert ChartDete PMC plots_detection (COCO, 18 element classes + bar) to YOLO.

Every image here has bar labels, and train/val/test contain only such images
(element_detection images without bar labels may hold unlabeled bars).

Papers get ONE split, shared with the element_detection conversion:
  1. paper in element_detection split3_test, or in plots_detection bar_test -> test
  2. paper in --val-papers (written by pmc_to_yolo.py)                       -> val
  3. everything else                                                         -> train
The official plots_detection train/val split is ignored.

Usage:
  python -m src.dataset.pmc_bars_to_yolo \
      --root data/pmc_2022/pmc_coco \
      --val-papers data/pmc_yolo/val_papers.json \
      --out data/pmc_yolo_bars

Images are symlinked by default (use --copy to copy). CC BY-NC-SA data: keep gitignored.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.dataset.pmc_to_yolo import load_coco, paper_id, write_split


def assign_papers(test_papers: set[str], val_papers: set[str], items) -> dict[str, str]:
    """{paper: split} by the shared rule; raises if a paper is claimed twice."""
    clash = test_papers & val_papers
    if clash:
        raise ValueError(f"{len(clash)} papers are both test and val, e.g. {sorted(clash)[:3]}")
    split = {}
    for item in items:
        paper = paper_id(item["name"])
        split[paper] = "test" if paper in test_papers else "val" if paper in val_papers else "train"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="pmc_coco folder (holds element_detection and plots_detection)")
    ap.add_argument("--val-papers", required=True, type=Path, help="val_papers.json from pmc_to_yolo.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    plots = args.root / "plots_detection"
    items = (load_coco(plots / "bar_train.json", plots / "bar_images")
             + load_coco(plots / "bar_val.json", plots / "bar_images"))
    test_items = load_coco(plots / "bar_test.json", plots / "bar_test")
    elements = json.load(open(args.root / "element_detection" / "split3_test.json"))
    test_papers = ({paper_id(i["file_name"]) for i in elements["images"]}
                   | {paper_id(i["name"]) for i in test_items})
    val_papers = set(json.load(open(args.val_papers)))

    pool = items + test_items
    names_seen = Counter(i["name"] for i in pool)
    dup = [n for n, c in names_seen.items() if c > 1]
    assert not dup, f"duplicate file names across plots_detection splits, e.g. {dup[:3]}"
    bar_id = {c: n for it in pool for c, n in it["cats"].items()}
    names = [n for _, n in sorted(bar_id.items())]
    assert names[-1] == "bar" and len(names) == 19, names
    unlabeled = [i["name"] for i in pool if not any(it_a["category_id"] == 19 for it_a in i["anns"])]
    assert not unlabeled, f"images without bar labels, e.g. {unlabeled[:3]}"

    split_of = assign_papers(test_papers, val_papers, pool)
    by_split = {s: [i for i in pool if split_of[paper_id(i["name"])] == s]
                for s in ("train", "val", "test")}
    papers = {s: {paper_id(i["name"]) for i in v} for s, v in by_split.items()}
    assert not (papers["train"] & papers["val"] or papers["train"] & papers["test"]
                or papers["val"] & papers["test"]), "paper in several splits"
    assert not ((papers["train"] | papers["val"]) & test_papers), "test paper in train or val"

    names_to_idx = {n: i for i, n in enumerate(names)}
    stats = Counter()
    for s, v in by_split.items():
        write_split(v, s, args.out, names_to_idx, {}, args.copy, stats)

    out = args.out.resolve()
    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    (args.out / "papers.json").write_text(json.dumps(
        {s: sorted(p) for s, p in papers.items()}, indent=1))

    print("papers: " + ", ".join(f"{s} {len(p)}" for s, p in papers.items()))
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"classes ({len(names)}): {names}")
    print(f"wrote {args.out}/data.yaml")


if __name__ == "__main__":
    main()
