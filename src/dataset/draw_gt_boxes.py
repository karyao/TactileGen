"""Draw YOLO ground-truth boxes with class names on random images.

Picks images greedily so that together they show every class (when the
dataset has one), then prints, per class, the box count and median box
width/height in pixels over the whole dataset. Used to check what a class such
as xlabel vs x_tick actually marks; it makes no assumption about meaning.

Example:
  python -m src.dataset.draw_gt_boxes --root data/pmc_yolo --output-dir <scratch dir>
"""

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.dataset.check_splits import list_images


def read_class_names(root: Path) -> dict[int, str]:
    """Parse the `names:` block of data.yaml ('  0: x_title')."""
    names = {}
    in_names = False
    for line in (root / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
        elif in_names and line.startswith(" ") and ":" in line:
            key, _, value = line.strip().partition(":")
            names[int(key)] = value.strip()
        elif in_names and line.strip():
            break
    return names


def read_boxes(label: Path, width: int, height: int):
    """[(class_id, x1, y1, x2, y2)] in pixels from a YOLO label file."""
    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append((cls, (cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def collect(root: Path):
    """{image path: (width, height, boxes)} over every split."""
    data = {}
    for image in list_images(root / "images"):
        label = root / "labels" / image.parent.name / (image.stem + ".txt")
        if not label.exists():
            continue
        with Image.open(image) as img:
            width, height = img.size
        data[image] = (width, height, read_boxes(label, width, height))
    return data


def pick_images(data, n, rng):
    """Greedy cover: each pick adds the most classes not yet shown (random ties)."""
    classes = {c: {p for p, (_, _, boxes) in data.items() if any(b[0] == c for b in boxes)}
               for c in {b[0] for _, _, boxes in data.values() for b in boxes}}
    shown, picked = set(), []
    candidates = list(data)
    for _ in range(n):
        rng.shuffle(candidates)
        best = max(candidates, key=lambda p: len({b[0] for b in data[p][2]} - shown))
        gain = {b[0] for b in data[best][2]} - shown
        if not gain and picked:
            # Everything is covered; fill the rest at random.
            best = rng.choice([p for p in candidates if p not in picked])
        picked.append(best)
        shown |= {b[0] for b in data[best][2]}
        candidates.remove(best)
    return picked, set(classes) - shown


def draw(image: Path, boxes, names, out: Path):
    img = Image.open(image).convert("RGB")
    canvas = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, img.height // 60))
    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (170, 110, 40), (128, 128, 0), (0, 128, 128),
               (255, 0, 0), (0, 0, 0), (128, 0, 0), (0, 0, 128), (250, 190, 190),
               (128, 128, 128), (100, 100, 255), (255, 170, 0)]
    for cls, x1, y1, x2, y2 in boxes:
        color = palette[cls % len(palette)]
        canvas.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"{cls}:{names.get(cls, '?')}"
        tx, ty = x1, max(0, y1 - font.size - 2)
        canvas.rectangle(canvas.textbbox((tx, ty), label, font=font), fill=color)
        canvas.text((tx, ty), label, fill=(255, 255, 255), font=font)
    img.save(out)


def print_stats(data, names):
    sizes = defaultdict(list)
    for width, height, boxes in data.values():
        for cls, x1, y1, x2, y2 in boxes:
            sizes[cls].append((x2 - x1, y2 - y1))
    print(f"{'id':>2}  {'class':<14}{'boxes':>8}{'med w px':>10}{'med h px':>10}")
    for cls in sorted(names):
        s = sizes.get(cls, [])
        if s:
            print(f"{cls:>2}  {names[cls]:<14}{len(s):>8}"
                  f"{statistics.median(w for w, _ in s):>10.1f}"
                  f"{statistics.median(h for _, h in s):>10.1f}")
        else:
            print(f"{cls:>2}  {names[cls]:<14}{0:>8}{'-':>10}{'-':>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = read_class_names(args.root)
    data = collect(args.root)
    picked, uncovered = pick_images(data, args.count, random.Random(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(picked):
        _, _, boxes = data[image]
        out = args.output_dir / f"{i:02d}_{image.stem}.png"
        draw(image, boxes, names, out)
        present = sorted({names[b[0]] for b in boxes})
        print(f"{out}  [{image.parent.name}]  {len(boxes)} boxes: {', '.join(present)}")
    if uncovered:
        print("classes not covered by these images:", ", ".join(names[c] for c in sorted(uncovered)))
    print()
    print_stats(data, names)


if __name__ == "__main__":
    main()
