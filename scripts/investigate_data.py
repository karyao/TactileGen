"""
Read-only investigation of a YOLO-format dataset. Changes nothing.

Usage:
    python investigate_data.py /var/tmp/$USER/data/charts
    python investigate_data.py /var/tmp/$USER/data/charts --check-images   # also opens every image (slower)

Expected layout: images/{train,val}/ and labels/{train,val}/ under the root.
"""
import sys
import pathlib
import statistics
from collections import Counter, defaultdict

root = pathlib.Path(sys.argv[1])
check_images = "--check-images" in sys.argv
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

report = []


def say(msg=""):
    print(msg)
    report.append(msg)


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "n/a"


stems_by_split = {}
class_counts = {}
sizes_by_class = defaultdict(lambda: {"w": [], "h": []})
problems = Counter()
examples = defaultdict(list)  # problem -> a few example locations


def note(kind, where):
    problems[kind] += 1
    if len(examples[kind]) < 5:
        examples[kind].append(where)


for split in ("train", "val"):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        say(f"[{split}] missing folder: {img_dir} or {lbl_dir}")
        continue

    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    stems_by_split[split] = set(imgs)

    say(f"=== {split} ===")
    say(f"images: {len(imgs)}   label files: {len(lbls)}")
    say(f"images without a label file: {len(set(imgs) - set(lbls))}")
    say(f"label files without an image: {len(set(lbls) - set(imgs))}")
    for s in list(set(imgs) - set(lbls))[:3]:
        examples["image_without_label"].append(f"{split}/{s}")
    for s in list(set(lbls) - set(imgs))[:3]:
        examples["label_without_image"].append(f"{split}/{s}")

    counts = Counter()
    empty_files = 0
    boxes_per_file = []

    for stem, f in lbls.items():
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if not lines:
            empty_files += 1
            note("empty_label_file", f"{split}/{f.name}")
        seen = set()
        n_boxes = 0
        for i, line in enumerate(lines, 1):
            where = f"{split}/{f.name}:{i}"
            parts = line.split()
            if len(parts) != 5:
                note("bad_format (not 5 fields; may be polygon labels)", where)
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                note("non_numeric_value", where)
                continue
            n_boxes += 1
            counts[c] += 1
            sizes_by_class[c]["w"].append(w)
            sizes_by_class[c]["h"].append(h)

            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                note("coordinate_outside_0_1", where)
            else:
                if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                    note("box_extends_past_image_edge", where)
            if w <= 0 or h <= 0:
                note("zero_or_negative_size", where)
            elif w < 0.002 or h < 0.002:
                note("very_thin_box (<0.2% of image; normal for gridlines)", where)
            key = (c, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
            if key in seen:
                note("duplicate_box_in_file", where)
            seen.add(key)
        boxes_per_file.append(n_boxes)

    class_counts[split] = counts
    total = sum(counts.values())
    say(f"empty label files: {empty_files}")
    if boxes_per_file:
        say(f"boxes per image: min {min(boxes_per_file)}, median {statistics.median(boxes_per_file)}, "
            f"max {max(boxes_per_file)}")
    say("boxes per class id:")
    for c in sorted(counts):
        say(f"  class {c}: {counts[c]}  ({pct(counts[c], total)})")
    say()

# Class ids vs data.yaml
yaml_path = root / "data.yaml"
say("=== data.yaml ===")
if yaml_path.exists():
    say(yaml_path.read_text().strip())
    seen_ids = sorted(set().union(*[set(c) for c in class_counts.values()])) if class_counts else []
    say(f"class ids seen in labels: {seen_ids}")
else:
    say("data.yaml not found")
say()

# Box sizes per class
say("=== box size per class (fraction of image) ===")
for c in sorted(sizes_by_class):
    ws, hs = sizes_by_class[c]["w"], sizes_by_class[c]["h"]
    say(f"class {c}: width median {statistics.median(ws):.3f} (min {min(ws):.3f}, max {max(ws):.3f}); "
        f"height median {statistics.median(hs):.3f} (min {min(hs):.3f}, max {max(hs):.3f})")
say()

# Leakage between splits
if "train" in stems_by_split and "val" in stems_by_split:
    overlap = stems_by_split["train"] & stems_by_split["val"]
    say("=== train/val leakage ===")
    say(f"file names present in both train and val: {len(overlap)}")
    if overlap:
        say(f"  examples: {sorted(overlap)[:5]}")
    say("(Same file name does not always mean same chart; different charts could share a name.")
    say(" Different file names could still be the same chart. This only catches the obvious case.)")
    say()

# Optional: open every image
if check_images:
    try:
        from PIL import Image
    except ImportError:
        say("Pillow not installed; skipping image check (pip install pillow)")
    else:
        say("=== opening every image ===")
        dims = Counter()
        for split in ("train", "val"):
            d = root / "images" / split
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() not in IMG_EXT:
                    continue
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        dims[im.size] += 1
                except Exception as e:
                    note("corrupt_or_unreadable_image", f"{split}/{p.name} ({type(e).__name__})")
        say("most common image sizes (w, h): " + ", ".join(f"{k}: {v}" for k, v in dims.most_common(5)))
        say()

# Summary of problems
say("=== problems found ===")
if not problems:
    say("none")
for kind, n in problems.most_common():
    say(f"{kind}: {n}")
    for ex in examples[kind]:
        say(f"    e.g. {ex}")
for kind in ("image_without_label", "label_without_image"):
    if examples[kind] and kind not in problems:
        say(f"{kind} examples: {examples[kind]}")

out = pathlib.Path("data_report.txt")
out.write_text("\n".join(report) + "\n")
print(f"\nSaved report to {out.resolve()}")