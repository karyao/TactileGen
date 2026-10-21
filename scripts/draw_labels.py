import sys
import random
import getpass
import pathlib
from PIL import Image, ImageDraw

root = pathlib.Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
specific = sys.argv[3] if len(sys.argv) > 3 else None
split = "train"

img_dir = root / "images" / split
lbl_dir = root / "labels" / split
names = (root / "classes.txt").read_text().split()
out = pathlib.Path("/var/tmp") / getpass.getuser() / "label_check"
out.mkdir(parents=True, exist_ok=True)

palette = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown",
    "olive", "teal", "navy", "maroon", "lime", "gold", "deeppink",
]

if specific:
    stems = [specific]
else:
    stems = [p.stem for p in lbl_dir.glob("*.txt")]
    random.shuffle(stems)
    stems = stems[:n]

for stem in stems:
    img_path = next((p for p in img_dir.glob(stem + ".*")), None)
    if img_path is None:
        print(f"no image for {stem}")
        continue
    im = Image.open(img_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    counts = {}
    for line in (lbl_dir / f"{stem}.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:])
        box = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
        d.rectangle(box, outline=palette[c % len(palette)], width=1)
        nm = names[c] if c < len(names) else str(c)
        counts[nm] = counts.get(nm, 0) + 1
    dest = out / f"{stem}_labeled.png"
    im.save(dest)
    print(f"{dest}  size={W}x{H}  boxes: {counts}")

print("\nColour key (class: colour):")
for i, nm in enumerate(names):
    print(f"  {i:>2} {nm}: {palette[i % len(palette)]}")
