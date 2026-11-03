#!/usr/bin/env python3
"""Check that dataset splits do not leak into each other.

Checks, in order:
  1. Same file name in more than one split
  2. Exact duplicate images (SHA-256 of file bytes)
  3. Near-duplicate images (64-bit dHash, Hamming distance <= --phash-dist)
  4. Same source group in more than one split (optional, --group-regex),
     e.g. figures from the same PMC paper.

Usage:
  python -m src.dataset.check_splits \
      --split train=data/plotqa/train/png \
      --split val=data/plotqa/val/png \
      --split test=data/plotqa/test/png

  # PMC: group by paper id taken from the file name (adjust the regex to your names)
  python -m src.dataset.check_splits --split train=... --split test=... \
      --group-regex '(PMC\\d+)'

Exit code is 0 if clean, 1 if any leak is found.
Run with --phash-warn-only to report near-duplicates without failing.

The checks are plain functions that return results (see run_checks), so the
CLI below and tests/test_split_leakage.py share one implementation.
"""
import argparse
import hashlib
import itertools
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash64(path: Path) -> int:
    img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    px = list(img.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicates(hashes_by_split, max_dist):
    """Cross-split pairs with Hamming distance <= max_dist.

    Pigeonhole: split the 64 bits into (max_dist + 1) chunks; two hashes within
    max_dist bits must be identical in at least one chunk, so only hashes that
    share a chunk value need comparing. Avoids all-pairs on large splits.
    """
    n_chunks = max_dist + 1
    bounds = [round(i * 64 / n_chunks) for i in range(n_chunks + 1)]
    found = []
    for (sa, ha), (sb, hb) in itertools.combinations(hashes_by_split.items(), 2):
        buckets = defaultdict(list)
        for path, h in hb:
            for c in range(n_chunks):
                lo, hi = bounds[c], bounds[c + 1]
                buckets[(c, (h >> lo) & ((1 << (hi - lo)) - 1))].append((path, h))
        seen = set()
        for path_a, h_a in ha:
            for c in range(n_chunks):
                lo, hi = bounds[c], bounds[c + 1]
                key = (c, (h_a >> lo) & ((1 << (hi - lo)) - 1))
                for path_b, h_b in buckets.get(key, ()):
                    if (path_a, path_b) in seen:
                        continue
                    seen.add((path_a, path_b))
                    d = hamming(h_a, h_b)
                    if d <= max_dist:
                        found.append((sa, path_a, sb, path_b, d))
    return found


def find_shared_names(splits):
    """(split_a, split_b, file name) for every name present in two splits."""
    names = {k: {p.name for p in v} for k, v in splits.items()}
    return [(a, b, n) for a, b in itertools.combinations(names, 2)
            for n in sorted(names[a] & names[b])]


def hash_splits(splits):
    """{split: {sha256: [paths]}}"""
    sha = {k: defaultdict(list) for k in splits}
    for k, files in splits.items():
        for p in files:
            sha[k][sha256(p)].append(p)
    return sha


def find_exact_duplicates(sha):
    """(split_a, path_a, split_b, path_b) for byte-identical images across splits."""
    return [(a, sha[a][h][0], b, sha[b][h][0])
            for a, b in itertools.combinations(sha, 2)
            for h in sha[a].keys() & sha[b].keys()]


def count_internal_duplicates(sha):
    """{split: extra copies inside that split}; not a leak, only wasteful."""
    counts = {k: sum(len(v) - 1 for v in d.values() if len(v) > 1) for k, d in sha.items()}
    return {k: n for k, n in counts.items() if n}


def find_near_duplicate_pairs(splits, sha, max_dist):
    """dHash near-duplicates across splits, excluding byte-identical pairs."""
    hashes = {k: [(p, dhash64(p)) for p in v] for k, v in splits.items()}
    exact_pairs = {(pa, pb) for a, b in itertools.combinations(sha, 2)
                   for h in sha[a].keys() & sha[b].keys()
                   for pa in sha[a][h] for pb in sha[b][h]}
    return [n for n in near_duplicates(hashes, max_dist) if (n[1], n[3]) not in exact_pairs]


def find_shared_groups(splits, group_regex):
    """Source groups (e.g. PMC papers) in several splits.

    Returns (shared {group: splits}, total groups, files that did not match).
    """
    rx = re.compile(group_regex)
    groups = defaultdict(set)
    unmatched = 0
    for k, files in splits.items():
        for p in files:
            m = rx.search(p.name)
            if m:
                groups[m.group(1)].add(k)
            else:
                unmatched += 1
    shared = {g: s for g, s in groups.items() if len(s) > 1}
    return shared, len(groups), unmatched


@dataclass
class SplitReport:
    counts: dict
    shared_names: list
    exact_duplicates: list
    internal_duplicates: dict
    near_duplicates: list | None = None      # None: check skipped
    shared_groups: dict | None = None        # None: check skipped
    n_groups: int = 0
    unmatched_group_files: int = 0
    phash_dist: int = 3
    failures: list = field(default_factory=list)  # names of failed hard checks


def run_checks(splits, phash_dist=3, group_regex=None, do_phash=True):
    """Run every check on {split: [image paths]}; pure, prints nothing."""
    sha = hash_splits(splits)
    report = SplitReport(
        counts={k: len(v) for k, v in splits.items()},
        shared_names=find_shared_names(splits),
        exact_duplicates=find_exact_duplicates(sha),
        internal_duplicates=count_internal_duplicates(sha),
        phash_dist=phash_dist,
    )
    if do_phash:
        report.near_duplicates = find_near_duplicate_pairs(splits, sha, phash_dist)
    if group_regex:
        (report.shared_groups, report.n_groups,
         report.unmatched_group_files) = find_shared_groups(splits, group_regex)
    if report.shared_names:
        report.failures.append("shared file names")
    if report.exact_duplicates:
        report.failures.append("exact duplicate images")
    if report.shared_groups:
        report.failures.append("shared source groups")
    return report


def load_splits(split_dirs):
    """{name: directory} -> {name: sorted image paths}; raises ValueError if a split is empty."""
    splits = {}
    for name, d in split_dirs.items():
        files = list_images(Path(d))
        if not files:
            raise ValueError(f"No images found for split '{name}' in {d}")
        splits[name] = files
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=DIR",
                    help="split name and image directory; repeat for each split")
    ap.add_argument("--phash-dist", type=int, default=3, help="max Hamming distance for near-duplicates (default 3)")
    ap.add_argument("--phash-warn-only", action="store_true", help="report near-duplicates but do not fail")
    ap.add_argument("--no-phash", action="store_true", help="skip the near-duplicate check")
    ap.add_argument("--group-regex", help="regex with one group extracting a source id from the file name")
    ap.add_argument("--show", type=int, default=10, help="max examples to print per check")
    args = ap.parse_args()

    split_dirs = {}
    for s in args.split:
        name, _, d = s.partition("=")
        split_dirs[name] = d
    try:
        splits = load_splits(split_dirs)
    except ValueError as exc:
        sys.exit(str(exc))
    for name, files in splits.items():
        print(f"{name}: {len(files)} images")
    if len(splits) < 2:
        sys.exit("Need at least two splits")

    report = run_checks(splits, args.phash_dist, args.group_regex, not args.no_phash)
    failed = bool(report.failures)

    print(f"\n[1] shared file names: {len(report.shared_names)}")
    for a, b, n in report.shared_names[: args.show]:
        print(f"    {a} & {b}: {n}")

    print(f"[2] exact duplicate images across splits: {len(report.exact_duplicates)}")
    for a, pa, b, pb in report.exact_duplicates[: args.show]:
        print(f"    {a}:{pa.name}  ==  {b}:{pb.name}")
    for k, within in report.internal_duplicates.items():
        print(f"    note: {k} has {within} duplicate images inside itself (not a leak, but wasteful)")

    if report.near_duplicates is not None:
        near = report.near_duplicates
        print(f"[3] near-duplicates, not byte-identical (dHash distance <= {args.phash_dist}): {len(near)}")
        for sa, pa, sb, pb, d in near[: args.show]:
            print(f"    {sa}:{pa.name}  ~  {sb}:{pb.name}  (dist {d})")
        if near:
            print("    note: simple synthetic charts can collide at this resolution; open a few pairs to confirm.")
            failed |= not args.phash_warn_only
    else:
        print("[3] near-duplicate check skipped")

    if report.shared_groups is not None:
        bad = report.shared_groups
        unmatched = report.unmatched_group_files
        print(f"[4] source groups spanning several splits: {len(bad)} of {report.n_groups}"
              + (f"  ({unmatched} files did not match the regex)" if unmatched else ""))
        for g, s in list(bad.items())[: args.show]:
            print(f"    {g}: {sorted(s)}")
    else:
        print("[4] source-group check skipped (no --group-regex)")

    print("\nRESULT:", "LEAK FOUND" if failed else "clean")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
