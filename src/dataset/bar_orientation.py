"""Guess whether a PMC bar chart is vertical or horizontal from its labels.

plots_detection has a single ``bar`` class and no orientation field. Two
signals from the ground-truth boxes, in pixels:

1. ``ylabel`` boxes mark the *value* axis whatever the orientation (the
   dataset names axes by role, not by side). They sit left of ``plot_area`` in
   a vertical chart and below it in a horizontal one.
2. Fallback when that is missing or tied: bars in a vertical chart share a
   baseline (bottom edges line up) and bars in a horizontal chart share a left
   edge. The smaller spread wins when it is under half the other.

Returns ``"vertical"``, ``"horizontal"`` or ``"undecided"``.
"""

import statistics

from src.contracts import Box

PLOT_AREA, YLABEL, BAR = 2, 5, 18  # class ids in data/pmc_yolo_bars/data.yaml


def read_label_boxes(path, size) -> dict[int, list[Box]]:
    """{class id: boxes in pixels} from a YOLO label file."""
    width, height = size
    boxes: dict[int, list[Box]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.setdefault(cls, []).append(Box(
            (cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def orientation(boxes: dict[int, list[Box]]) -> str:
    plot = boxes.get(PLOT_AREA)
    labels = boxes.get(YLABEL)
    if plot and labels:
        area = plot[0]
        below = left = 0
        for box in labels:
            if (box.y1 + box.y2) / 2 > area.y2:
                below += 1
            elif (box.x1 + box.x2) / 2 < area.x1:
                left += 1
        if below != left:
            return "horizontal" if below > left else "vertical"
    bars = boxes.get(BAR, [])
    if len(bars) >= 2:
        bottom_spread = statistics.pstdev(b.y2 for b in bars)
        left_spread = statistics.pstdev(b.x1 for b in bars)
        if bottom_spread < left_spread * 0.5:
            return "vertical"
        if left_spread < bottom_spread * 0.5:
            return "horizontal"
    return "undecided"
