#!/usr/bin/env python3
"""Make a blank sheet for reading bar values off charts by eye (truth for value error).

Picks scope charts (vertical, single series, at most 10 bars) from val and test that have
the same number of bars and category labels and no printed number on every bar (those
already give truth), shuffles them under neutral ids C01.., draws numbered tags on the
bars, and writes:

  handread_sheet.html      self-contained page (images embedded) with the instructions
  handread_template.csv    one row per bar to fill in
  handread_key.json        which image each id is; keep it away from the reader

Output contains PMC images (CC BY-NC-SA): keep it under output/ and out of commits.

  python -m src.dataset.handread_sheet --output-dir output/handread
"""
import argparse
import base64
import csv
import html
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.baseline.e2e_stage1 import scope_charts
from src.textutil import parse_number

INSTRUCTIONS = [
    "Each chart is a bar chart with one series. The bars carry red numbers (1, 2, 3, ... from left to right) so you can say which bar you mean. The red outline and number are only labels, not part of the chart.",
    "For each numbered bar, read its value from the vertical axis: find where the top edge of the bar meets the scale and estimate the number to the precision the axis allows (a bar halfway between the ticks 40 and 60 is 50). If a number printed on or next to the bar is that bar's own value, copy it and write \"printed\" in the notes. Ignore labels that are not values, such as sample sizes (n=12), p-values and letters.",
    "Bars that hang below zero have negative values. If the axis does not start at zero, still read the value at the top of the bar.",
    "Unit: the unit of the value axis (for example %, mg/L, count) as written on the chart; leave it blank if the chart does not say. Scale: write linear if equal steps on the axis are equal differences, or log if equal steps are equal factors (1, 10, 100, ...).",
    "If you cannot tell which bar is which, the chart is not a simple one-series bar chart (for example stacked or grouped bars), or the axis cannot be read, write a short reason in the problem column and leave the values blank. Do not guess.",
    "Use only the chart in front of you. Do not look anything up and do not try to match values to a source.",
    "Fill in the CSV, not this page. Put the unit and scale on the first row of each chart. One row per bar: chart_id, bar_number, value, unit, scale, problem, notes. Use a dot for decimals and no thousands separators. Expect about 2 to 4 minutes per chart.",
]


def pick(root: Path, per_split: dict[str, int], seed: int):
    rnd = random.Random(seed)
    chosen = []
    for split, count in per_split.items():
        candidates = []
        for image, annotations in scope_charts(root, split):
            bars = [a for a in annotations if a["category_id"] == 19]
            labels = [a for a in annotations if a["category_id"] == 5]
            printed = [a for a in annotations if a["category_id"] == 15]
            all_printed = len(printed) == len(bars) and all(parse_number(a["text"]) is not None for a in printed)
            if len(bars) == len(labels) and not all_printed:
                candidates.append((split, image, bars))
        rnd.shuffle(candidates)
        chosen += candidates[:count]
    rnd.shuffle(chosen)
    return chosen


def tagged_image(path: Path, bars: list[dict]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, img.height // 28))
    for number, bar in enumerate(sorted(bars, key=lambda a: a["bbox"][0]), 1):
        x, y, w, h = bar["bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=(220, 0, 0), width=1)
        tag = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        tx = x + w / 2 - (right - left) / 2
        ty = y - (bottom - top) - 6 if y > bottom - top + 8 else y + 2
        draw.rectangle((tx - 2, ty - 1, tx + right - left + 2, ty + bottom - top + 3), fill=(255, 255, 255), outline=(220, 0, 0))
        draw.text((tx, ty), tag, fill=(220, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--test", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    chosen = pick(args.root, {"val": args.val, "test": args.test}, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key, rows, sections = {}, [], []
    for n, (split, image, bars) in enumerate(chosen, 1):
        chart_id = f"C{n:02d}"
        key[chart_id] = {"split": split, "file": image["file_name"], "bars": len(bars)}
        img = tagged_image(args.root / "images" / split / image["file_name"], bars)
        buffer = io.BytesIO()
        img.save(buffer, "PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        table = "".join(f"<tr><td>{i}</td><td></td></tr>" for i in range(1, len(bars) + 1))
        sections.append(
            f'<section><h2>{chart_id} ({len(bars)} bars)</h2><img src="data:image/png;base64,{data}" alt="{chart_id}">'
            f"<table><tr><th>Bar</th><th>Value</th></tr>{table}</table>"
            "<p>Unit: ________ &nbsp; Scale (linear / log): ________ &nbsp; Problem: ______________________</p></section>")
        rows += [{"chart_id": chart_id, "bar_number": i, "value": "", "unit": "", "scale": "", "problem": "", "notes": ""}
                 for i in range(1, len(bars) + 1)]
    with (args.output_dir / "handread_template.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["chart_id", "bar_number", "value", "unit", "scale", "problem", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "handread_key.json").write_text(json.dumps(key, indent=1))
    instructions = "".join(f"<li>{html.escape(t)}</li>" for t in INSTRUCTIONS)
    (args.output_dir / "handread_sheet.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Reading bar values</title>"
        "<style>body{font:15px sans-serif;max-width:900px;margin:2em auto;padding:0 1em}"
        "img{max-width:100%;border:1px solid #888}table{border-collapse:collapse;margin:.5em 0}"
        "td,th{border:1px solid #888;padding:4px 14px;min-width:90px}section{page-break-inside:avoid;margin:2em 0}</style>"
        f"<h1>Reading bar values</h1><p>{len(chosen)} charts. Please read the instructions first.</p><ol>{instructions}</ol>"
        + "".join(sections))
    print(f"{len(chosen)} charts, {len(rows)} bars: {args.output_dir}")
    print("per split:", {s: sum(v['split'] == s for v in key.values()) for s in ('val', 'test')})


if __name__ == "__main__":
    main()
