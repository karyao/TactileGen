"""Interpret a vertical, single-series bar chart from detections and OCR reads.

Input is a DetectionFrame plus an OCRFrame (ground truth or model output); output is
an InterpretationResult: a VerticalBarChart, or the reasons a person has to look first.

Steps: scope check (1 to MAX_BARS bars, one series), pair bars with category labels by
x position, fit tick pixel position to tick value (src/axis_fit.py), read each bar's
value off the fit, add the axis fields, then run the checks that flag a doubtful chart.
Nothing is invented: a missing or unreadable text leaves the chart unbuilt or the field
None, and the item is flagged.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from src.axis_fit import pair_by_overlap, robust_fit
from src.contracts import (
    BarSeries,
    Box,
    DetectionFrame,
    InterpretationResult,
    OCRFrame,
    ReviewItem,
    VerticalBarChart,
    join_text_regions,
)
from src.textutil import clean_text, parse_number

MAX_BARS = 10
MIN_INLIERS = 3
MIN_SUPPORT = 0.6                 # inliers over ticks, as in src/baseline/ocr_axis_fit.py
VALUE_LABEL_LOG_THRESHOLD = 0.05  # median |log10(printed / fitted)|, chosen on val (docs/ocr_baselines.md)
PAIRING_TOLERANCE = 0.5           # label or tick centre within this many bar pitches of its bar


def _cx(box: Box) -> float:
    return (box.x1 + box.x2) / 2


def _cy(box: Box) -> float:
    return (box.y1 + box.y2) / 2


def _xywh(box: Box) -> tuple[float, float, float, float]:
    return box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1


@dataclass
class _Run:
    """State shared by the steps while one chart is interpreted."""

    frame: DetectionFrame
    reads: dict
    min_confidence: float | None
    items: list
    reasons: list

    def read(self, detection):
        return self.reads.get(detection.id)

    def text_item(self, kind, detection, index=None, flagged=False, reason=""):
        read = self.read(detection)
        text = read.text if read else None
        confidence = read.confidence if read else None
        if read is None and not flagged:
            flagged, reason = True, "no OCR read"
        elif read is not None and not read.text.strip() and not flagged:
            flagged, reason = True, "empty OCR read"
        elif (self.min_confidence is not None and confidence is not None and confidence < self.min_confidence
              and not flagged):
            flagged, reason = True, f"low OCR confidence ({confidence:.2f})"
        item = ReviewItem(kind, index, detection.id, text, confidence, detection.box, flagged=flagged, reason=reason)
        self.items.append(item)
        return item


def interpret_vertical_bars(detections: DetectionFrame, ocr: OCRFrame,
                            min_confidence: float | None = None) -> InterpretationResult:
    reads = {d.id: r for d, r in join_text_regions(detections, ocr)}
    run = _Run(detections, reads, min_confidence, [], [])
    by_role: dict[str, list] = {}
    for d in detections.detections:
        by_role.setdefault(d.role, []).append(d)
    key = detections.source_key

    def result(status, chart=None):
        return InterpretationResult(key, status, chart, tuple(run.items), tuple(run.reasons))

    bars = sorted(by_role.get("vertical_bar", []), key=lambda d: _cx(d.box))
    if not bars:
        run.reasons.append("out of scope: no vertical bars detected")
        return result("rejected")
    if len(bars) > MAX_BARS:
        run.reasons.append(f"out of scope: {len(bars)} bars (limit {MAX_BARS})")
        return result("rejected")
    if len(by_role.get("legend_swatch", [])) >= 2:
        run.reasons.append("out of scope: several legend entries (more than one series)")
        return result("rejected")

    categories = _categories(run, bars, by_role)
    axis = _axis(run, by_role)
    values = _bar_values(run, bars, axis) if axis else None
    _value_label_check(run, bars, by_role, axis)
    for i, bar in enumerate(bars):
        run.items.append(ReviewItem("bar", i, bar.id, box=bar.box,
                                    value=values[i][0] if values else None,
                                    uncertainty=values[i][1] if values else None))
    titles = {role: _title(run, by_role.get(role, []), role) for role in ("chart_title", "x_axis_title", "y_axis_title")}
    if categories is None or axis is None or values is None:
        return result("needs_review")
    try:
        chart = VerticalBarChart(
            key, titles["chart_title"], titles["x_axis_title"], titles["y_axis_title"], tuple(categories),
            (BarSeries(_series_name(run, by_role), tuple(v for v, _ in values)),),
            axis["y_min"], axis["y_max"], axis["tick_values"])
    except ValueError as exc:
        run.reasons.append(f"chart could not be built: {exc}")
        return result("needs_review")
    flagged = any(item.flagged for item in run.items) or run.reasons
    return result("needs_review" if flagged else "accepted", chart)


def _categories(run, bars, by_role):
    labels = sorted(by_role.get("x_tick_label", []), key=lambda d: _cx(d.box))
    for i, label in enumerate(labels):
        run.text_item("category", label, i)
    if len(labels) != len(bars):
        run.reasons.append(f"{len(bars)} bars but {len(labels)} category labels")
        return None
    pitch = (statistics.median(_cx(b.box) - _cx(a.box) for a, b in zip(bars, bars[1:])) if len(bars) > 1
             else bars[0].box.x2 - bars[0].box.x1)
    ticks = sorted(by_role.get("x_tick", []), key=lambda d: _cx(d.box))
    anchors = ticks if len(ticks) == len(bars) else labels  # tick marks stay put when labels rotate
    if any(abs(_cx(a.box) - _cx(b.box)) > PAIRING_TOLERANCE * pitch for a, b in zip(anchors, bars)):
        run.reasons.append("category labels do not line up with the bars")
    texts = [clean_text(run.read(label).text) if run.read(label) else "" for label in labels]
    if any(not t for t in texts):
        run.reasons.append("a category label has no readable text")
        return None
    return texts


def _axis(run, by_role):
    labels = sorted(by_role.get("y_tick_label", []), key=lambda d: _cy(d.box))
    marks = by_role.get("y_tick", [])
    positions = _tick_positions(run, labels, marks)
    points, owners = [], []
    for label, y in zip(labels, positions):
        read = run.read(label)
        value = parse_number(read.text) if read else None
        if value is None:
            run.text_item("y_tick_label", label, flagged=True, reason="not a readable number")
        else:
            points.append((y, value))
            owners.append(label)
    if len(points) < 3:
        run.reasons.append("fewer than 3 tick labels parse as numbers")
        return None
    fit = robust_fit(points)
    if fit is None:
        run.reasons.append("no admissible line through the tick labels")
        return None
    a, b, inliers, unique = fit
    inlier_set = set(inliers)
    for label, point in zip(owners, points):
        if point in inlier_set:
            run.text_item("y_tick_label", label)
        else:
            run.text_item("y_tick_label", label, flagged=True, reason="read disagrees with the fitted axis")
    if len(inliers) < MIN_INLIERS:
        run.reasons.append("fewer than 3 tick labels agree on one axis")
    if len(inliers) < MIN_SUPPORT * len(labels):
        run.reasons.append("under 60% of tick labels agree on one axis")
    if not unique:
        run.reasons.append("two different axes fit the tick labels equally well")
    reads = [run.read(d).text.strip() for d in labels if run.read(d) and run.read(d).text.strip()]
    if any(r.endswith("%") for r in reads) and not all(r.endswith("%") for r in reads):
        run.reasons.append("some tick labels end in % and others do not (a misread % or unit)")
    ys = [y for y, _ in points]
    plot = by_role.get("plot_area", [])
    y_min = y_max = None
    tick_values = sorted({v for _, v in inliers})
    resolution = _decimals(b)
    if plot:
        top, bottom = a + b * plot[0].box.y1, a + b * plot[0].box.y2
        y_max, y_min = max(top, bottom, tick_values[-1]), min(top, bottom, tick_values[0])
        y_min, y_max = round(y_min, resolution), round(y_max, resolution)
        y_min, y_max = min(y_min, tick_values[0]), max(y_max, tick_values[-1])
        if y_min >= y_max:
            y_min = y_max = None
    return {"a": a, "b": b, "ys": ys, "y_min": y_min, "y_max": y_max, "tick_values": tuple(tick_values),
            "decimals": resolution}


def _decimals(slope: float) -> int:
    """Decimal places that match the pixel resolution of the fit (value per pixel)."""
    return max(0, -math.floor(math.log10(abs(slope)))) if slope else 0


def _tick_positions(run, labels, marks):
    """Pixel y of each label's tick: the nearest unused tick mark, else the label centre."""
    centres = [_cy(label.box) for label in labels]
    if len(marks) != len(labels):
        if marks or labels:
            run.reasons.append("tick marks missing or not one per label; using label centres")
        return centres
    order = sorted(((abs(_cy(m.box) - c), li, mi) for li, c in enumerate(centres) for mi, m in enumerate(marks)))
    used_l, used_m, out = set(), set(), list(centres)
    for _, li, mi in order:
        if li not in used_l and mi not in used_m:
            used_l.add(li)
            used_m.add(mi)
            out[li] = _cy(marks[mi].box)
    return out


def _bar_values(run, bars, axis):
    a, b, ys = axis["a"], axis["b"], axis["ys"]
    zero = -a / b if b else None
    pad = statistics.median(y2 - y1 for y1, y2 in zip(sorted(ys), sorted(ys)[1:])) if len(ys) > 1 else 0
    zero_in_range = zero is not None and min(ys) - pad <= zero <= max(ys) + pad
    out, baselines = [], []
    for bar in bars:
        y1, y2 = bar.box.y1, bar.box.y2
        top_value, bottom_value = a + b * y1, a + b * y2
        negative = zero_in_range and abs(y1 - zero) < abs(y2 - zero)
        value = bottom_value if negative else top_value
        baselines.append(y1 if negative else y2)
        out.append((round(value, axis["decimals"]), 0.5 * abs(b)))
    if max(baselines) - min(baselines) > max(3.0, 0.02 * run.frame.height):
        run.reasons.append("bars do not share a baseline (stacked, floating or mixed bars)")
    return out


def _value_label_check(run, bars, by_role, axis):
    labels = by_role.get("value_label", [])
    for i, label in enumerate(sorted(labels, key=lambda d: _cx(d.box))):
        run.text_item("value_label", label, i)
    if not axis or not labels:
        return
    a, b = axis["a"], axis["b"]
    pairs = pair_by_overlap([_xywh(d.box) for d in labels], [_xywh(d.box) for d in bars])
    logs = []
    for bi, li in pairs.items():
        read = run.read(labels[li])
        printed = parse_number(read.text) if read else None
        fitted = max((a + b * bars[bi].box.y1, a + b * bars[bi].box.y2), key=abs)
        if printed and fitted:
            logs.append(abs(math.log10(abs(printed / fitted))))
    if len(logs) >= 2 and statistics.median(logs) > VALUE_LABEL_LOG_THRESHOLD:
        run.reasons.append("numbers printed on the bars disagree with the fitted axis")


def _title(run, candidates, kind):
    if not candidates:
        return None
    best = max(candidates, key=lambda d: d.confidence)
    run.text_item(kind, best)
    text = clean_text(run.read(best).text) if run.read(best) else ""
    return text or None


def _series_name(run, by_role):
    swatches, labels = by_role.get("legend_swatch", []), by_role.get("legend_label", [])
    if len(swatches) == 1 and len(labels) == 1:
        run.text_item("legend", labels[0])
        text = clean_text(run.read(labels[0]).text) if run.read(labels[0]) else ""
        return text or None
    return None
