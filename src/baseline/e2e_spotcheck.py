"""Visual spot checks of the bar-to-label pairing for charts with rotated category labels.

For each chosen chart the image shows the bars with red numbers and the category label
boxes with blue numbers: label k is the one the interpreter paired with bar k (left to right).
A panel lists the ground-truth text and each engine's read for the same pairs. A pairing is
right when blue k sits under, or at the foot of the slanted text of, red k.

  python -m src.baseline.e2e_spotcheck --output-dir <dir>
"""
import argparse
import random
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.baseline.e2e_stage2 import ENGINES, load_reads, prepare
from src.textutil import clean_text

PANEL_WIDTH = 520


def rotated(det_frame):
    """True when the category labels look rotated: tall boxes, or centres far left of their ticks."""
    labels = sorted((d for d in det_frame.detections if d.role == "x_tick_label"), key=lambda d: d.box.x1 + d.box.x2)
    ticks = sorted((d for d in det_frame.detections if d.role == "x_tick"), key=lambda d: d.box.x1 + d.box.x2)
    bars = sorted((d for d in det_frame.detections if d.role == "vertical_bar"), key=lambda d: d.box.x1 + d.box.x2)
    if len(labels) != len(bars) or len(bars) < 3:
        return False
    tall = statistics.median((d.box.y2 - d.box.y1) / (d.box.x2 - d.box.x1) for d in labels) > 1.0
    pitch = statistics.median((b.box.x1 + b.box.x2) / 2 - (a.box.x1 + a.box.x2) / 2 for a, b in zip(bars, bars[1:]))
    shifted = len(ticks) == len(bars) and abs(statistics.median(
        (l.box.x1 + l.box.x2) / 2 - (t.box.x1 + t.box.x2) / 2 for l, t in zip(labels, ticks))) > 0.4 * pitch
    return tall or shifted


def draw_chart(root, split, image, det, annotations, reads_by_engine, path):
    img = Image.open(root / "images" / split / image["file_name"]).convert("RGB")
    scale = max(1.0, 760 / img.width)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    font = ImageFont.load_default(size=15)
    draw = ImageDraw.Draw(img)
    bars = sorted((d for d in det.detections if d.role == "vertical_bar"), key=lambda d: d.box.x1 + d.box.x2)
    labels = sorted((d for d in det.detections if d.role == "x_tick_label"), key=lambda d: d.box.x1 + d.box.x2)
    for k, (bar, label) in enumerate(zip(bars, labels), 1):
        for d, colour in ((bar, (220, 0, 0)), (label, (0, 70, 220))):
            b = d.box
            draw.rectangle([b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale], outline=colour, width=2)
            draw.text(((b.x1 + b.x2) / 2 * scale - 5, b.y1 * scale - 17 if d is bar else b.y2 * scale + 1), str(k), fill=colour, font=font)
    by_id = {f"gt{a['id']}": a for a in annotations}
    lines = ["#   truth  |  " + "  |  ".join(ENGINES)]
    for k, label in enumerate(labels, 1):
        a = by_id[label.id]
        reads = []
        for engine in ENGINES:
            row = reads_by_engine[engine].get((image["file_name"], a["category_id"], tuple(a["bbox"])))
            reads.append(clean_text(row["pred"])[:22] if row else "-")
        lines.append(f"{k:<3} {clean_text(a['text'])[:24]:<24} | " + " | ".join(f"{r:<22}" for r in reads))
    panel = Image.new("RGB", (img.width, 22 * (len(lines) + 1)), "white")
    pd = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        pd.text((6, 4 + 22 * i), line, fill=(0, 0, 0), font=ImageFont.load_default(size=14))
    sheet = Image.new("RGB", (img.width, img.height + panel.height + 24), "white")
    ImageDraw.Draw(sheet).text((6, 4), f"{split}/{image['file_name']}", fill=(0, 0, 0), font=font)
    sheet.paste(img, (0, 24))
    sheet.paste(panel, (0, img.height + 24))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--per-split", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)
    for split in ("val", "test"):
        charts = [c for c in prepare(args.root, split) if c[3].chart is not None and rotated(c[2])]
        print(f"{split}: {len(charts)} built charts with rotated labels")
        rnd.shuffle(charts)
        reads = {e: load_reads(e, split) for e in ENGINES}
        for n, (image, annotations, det, ref) in enumerate(charts[:args.per_split], 1):
            out = args.output_dir / f"{split}_{n}_{image['file_name'].rsplit('.', 1)[0]}.png"
            draw_chart(args.root, split, image, det, annotations, reads, out)
            print(" ", out.name)


if __name__ == "__main__":
    main()
