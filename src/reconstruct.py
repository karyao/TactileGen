"""Rebuild a vertical bar chart spec from YOLO detections and OCR text.

``image`` is a BGR array as returned by ``cv2.imread``; it is only used to
sample bar and legend colours. All geometry comes from the ``DetectionFrame``.
"""

from __future__ import annotations

import itertools
import math
import re
import statistics
from collections.abc import Iterable, Mapping

import numpy as np

from src.contracts import (
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    OCRFrame,
    RecognizedText,
    SeriesSpec,
    VerticalBarSpec,
    YAxisSpec,
)

MIN_CONFIDENCE = 0.3
DUPLICATE_IOU = 0.5
COLOR_DISTANCE = 40.0
RESCUE_CONFIDENCE = 0.2
RESCUE_STRIP_FRACTION = 0.5   # central share of the bar width that is probed
RESCUE_ROW_FRACTION = 0.6     # share of strip pixels that must match for a row to count
_LUMA = np.array([0.299, 0.587, 0.114])
_BORDER_LUMINANCE = 100.0
RESCUE_AXIS_ROWS = 2          # non-matching rows tolerated at the zero line (axis stroke)
SEED_MERGE_DISTANCE = 10.0  # swatches this close are one legend entry (duplicate detections)
DARK_LUMINANCE = 60.0
TITLE_EDGE_PX = 2
ADJACENT_GAP_RATIO = 0.10
ARITHMETIC_TOLERANCE = 0.03

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


# --- OCR and detection plumbing ------------------------------------------------

def _text_table(ocr_texts) -> dict[str, tuple[str, float]]:
    """Normalise OCR input to ``{detection_id: (text, confidence)}``."""
    if isinstance(ocr_texts, OCRFrame):
        ocr_texts = ocr_texts.regions
    if isinstance(ocr_texts, Mapping):
        items = ocr_texts.items()
    else:
        items = ((region.detection_id, region) for region in ocr_texts)
    table = {}
    for key, value in items:
        if isinstance(value, RecognizedText):
            table[key] = (value.text.strip(), value.confidence)
        elif isinstance(value, str):
            table[key] = (value.strip(), 1.0)
        else:
            text, confidence = value
            table[key] = (text.strip(), float(confidence))
    return table


def _centre(detection: Detection) -> tuple[float, float]:
    box = detection.box
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box.x1, a.box.y1, a.box.x2, a.box.y2
    bx1, by1, bx2, by2 = b.box.x1, b.box.y1, b.box.x2, b.box.y2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(detections: Iterable[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.confidence):
        if all(_iou(detection, other) <= DUPLICATE_IOU for other in kept):
            kept.append(detection)
    return kept


def _group_roles(frame: DetectionFrame, min_confidence: float) -> dict[str, list[Detection]]:
    roles: dict[str, list[Detection]] = {}
    for detection in frame.detections:
        if detection.confidence >= min_confidence:
            roles.setdefault(detection.role, []).append(detection)
    return {role: _dedupe(items) for role, items in roles.items()}


def _best(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda d: d.confidence, default=None)


def _inside(point: tuple[float, float], box: Detection, margin_x=0.0, margin_y=0.0) -> bool:
    b = box.box
    return (b.x1 - margin_x <= point[0] <= b.x2 + margin_x
            and b.y1 - margin_y <= point[1] <= b.y2 + margin_y)


# --- y calibration -------------------------------------------------------------

def parse_tick_value(text: str) -> tuple[float, str | None] | None:
    """Parse an axis label to ``(value, unit)``; ``None`` if it is not a number.

    K/M/B suffixes are expanded into the value (``1.5M`` -> 1500000.0); only
    ``%`` is kept as a unit.
    """
    cleaned = (text.strip().replace("−", "-").replace("–", "-")
               .replace(",", "").replace(" ", "").lstrip("$"))
    unit = None
    if cleaned.endswith("%"):
        unit, cleaned = "%", cleaned[:-1]
    scale = 1.0
    if cleaned and cleaned[-1].upper() in _SUFFIX and not cleaned[-1].isdigit():
        scale = _SUFFIX[cleaned[-1].upper()]
        cleaned = cleaned[:-1]
    if not _NUMBER.fullmatch(cleaned):
        return None
    value = float(cleaned) * scale
    return (value, unit) if math.isfinite(value) else None


def _calibrate(ticks: list[tuple[float, float]]):
    """Fit ``value = a * y + b`` robustly.

    Every pair of ticks proposes a line; the line with the most ticks within a
    pixel tolerance wins (ties go to the smaller residual) and is refit by
    least squares on its inliers. Returns ``(a, b, inlier_indexes, ambiguous)``.
    """
    ys = sorted(y for y, _ in ticks)
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0]
    tolerance = max(3.0, 0.2 * statistics.median(gaps)) if gaps else 3.0
    best = None
    tied_sets: set[frozenset[int]] = set()
    for i, j in itertools.combinations(range(len(ticks)), 2):
        (yi, vi), (yj, vj) = ticks[i], ticks[j]
        if yi == yj or vi == vj:
            continue
        a = (vj - vi) / (yj - yi)
        b = vi - a * yi
        residuals = [abs(v - (a * y + b)) / abs(a) for y, v in ticks]
        inliers = frozenset(k for k, r in enumerate(residuals) if r <= tolerance)
        score = (len(inliers), -sum(residuals[k] ** 2 for k in inliers))
        if best is None or score > best[0]:
            best, tied_sets = (score, inliers), {inliers}
        elif score[0] == best[0][0]:
            tied_sets.add(inliers)
    if best is None or len(best[1]) < 2:
        raise ChartInterpretationError("y-axis calibration needs at least 2 consistent ticks")
    inliers = sorted(best[1])
    a, b = np.polyfit([ticks[k][0] for k in inliers], [ticks[k][1] for k in inliers], 1)
    return float(a), float(b), inliers, len(tied_sets) > 1


def _is_arithmetic(values: list[float]) -> bool:
    values = sorted(values)
    diffs = [b - a for a, b in zip(values, values[1:])]
    if len(diffs) < 2:
        return True
    step = statistics.median(diffs)
    return all(abs(d - step) <= ARITHMETIC_TOLERANCE * abs(step) + 1e-12 for d in diffs)


# --- colour --------------------------------------------------------------------

def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image[:, :, :3][:, :, ::-1]


def _sample_rgb(rgb: np.ndarray, detection: Detection) -> tuple[int, int, int] | None:
    """Median colour of the central 60% of a box, skipping dark border pixels."""
    box = detection.box
    width, height = box.x2 - box.x1, box.y2 - box.y1
    left, right = round(box.x1 + 0.2 * width), round(box.x2 - 0.2 * width)
    top, bottom = round(box.y1 + 0.2 * height), round(box.y2 - 0.2 * height)
    right, bottom = max(right, left + 1), max(bottom, top + 1)
    pixels = rgb[top:bottom, left:right].reshape(-1, 3).astype(float)
    if not len(pixels):
        return None
    light = pixels[pixels @ np.array([0.299, 0.587, 0.114]) >= DARK_LUMINANCE]
    if len(light):
        pixels = light
    return tuple(int(v) for v in np.median(pixels, axis=0))


def _hex(color: tuple[int, int, int] | None) -> str | None:
    return None if color is None else "#%02x%02x%02x" % color


def _distance(a, b) -> float:
    return math.dist(a, b)


def _cluster_colors(colors: list[tuple[int, int, int] | None],
                    seeds: list[tuple[int, int, int]] = ()):
    """Group bar indexes by colour.

    Each seed (a legend swatch colour) owns a cluster; bars go to the nearest
    seed within ``COLOR_DISTANCE``. Remaining bars are grouped greedily, and
    bars without a colour stay alone. Returns ``(clusters, seed_of_cluster)``
    ordered by first bar, where ``seed_of_cluster`` is the seed index or None.
    """
    clusters: list[list[int]] = [[] for _ in seeds]
    owner: list[int | None] = list(range(len(seeds)))
    centres: list[tuple[float, ...]] = [tuple(float(v) for v in seed) for seed in seeds]
    for index, color in enumerate(colors):
        if color is None:
            clusters.append([index])
            owner.append(None)
            centres.append((-1e9,) * 3)
            continue
        seeded = [c for c in range(len(centres)) if owner[c] is not None]
        nearest = min(seeded, key=lambda c: _distance(centres[c], color), default=None)
        if nearest is not None and _distance(centres[nearest], color) <= COLOR_DISTANCE:
            clusters[nearest].append(index)
            continue
        free = [c for c in range(len(centres)) if owner[c] is None]
        nearest = min(free, key=lambda c: _distance(centres[c], color), default=None)
        if nearest is not None and _distance(centres[nearest], color) <= COLOR_DISTANCE:
            clusters[nearest].append(index)
            members = [colors[k] for k in clusters[nearest]]
            centres[nearest] = tuple(statistics.median(m[ch] for m in members) for ch in range(3))
        else:
            clusters.append([index])
            owner.append(None)
            centres.append(tuple(float(v) for v in color))
    order = sorted((c for c in range(len(clusters)) if clusters[c]), key=lambda c: clusters[c][0])
    return [clusters[c] for c in order], [owner[c] for c in order]


def _legend_seeds(swatch_colors):
    """Distinct swatch colours as seeds: ``(seed colours, seed -> swatch index)``."""
    seeds, swatch_of_seed = [], []
    for index, color in enumerate(swatch_colors):
        if color is None or any(_distance(color, seed) < SEED_MERGE_DISTANCE for seed in seeds):
            continue
        seeds.append(color)
        swatch_of_seed.append(index)
    return seeds, swatch_of_seed


# --- bar rescue ---------------------------------------------------------------

def _run_length(rgb: np.ndarray, left: int, right: int, start: int, step: int,
                color: tuple[float, ...]) -> int:
    """Contiguous rows of ``color`` from ``start`` in direction ``step`` (+1 down, -1 up).

    A few non-matching rows right at the zero line are skipped so an axis
    stroke does not hide a short bar.
    """
    height = rgb.shape[0]
    color_luma = float(np.dot(color, _LUMA))
    run = skipped = 0
    y = start
    while 0 <= y < height:
        pixels = rgb[y, left:right].astype(float)
        match = np.mean(np.linalg.norm(pixels - np.array(color), axis=1) <= COLOR_DISTANCE)
        if match >= RESCUE_ROW_FRACTION:
            run += 1
        elif run == 0 and skipped < RESCUE_AXIS_ROWS:
            skipped += 1
        else:
            # Bars carry a dark outline: count it when it caps a run of fill.
            if run and np.mean(pixels @ _LUMA) < min(_BORDER_LUMINANCE, 0.8 * color_luma):
                run += 1
            break
        y += step
    return run


def _rescue_value(rgb: np.ndarray, centre_x: float, width: float, baseline: float,
                  slope: float, color: tuple[float, ...]) -> float:
    """Value of the bar at ``centre_x`` measured from pixels, 0.0 if none are found."""
    half = RESCUE_STRIP_FRACTION * width / 2
    left = max(0, int(round(centre_x - half)))
    right = min(rgb.shape[1], max(left + 1, int(round(centre_x + half))))
    if left >= right:
        return 0.0
    up = _run_length(rgb, left, right, int(round(baseline)) - 1, -1, color)
    down = _run_length(rgb, left, right, int(round(baseline)), 1, color)
    if max(up, down) < 1:
        return 0.0
    # value = slope * y + intercept with zero at the baseline: y moves by the run.
    return -slope * up if up >= down else slope * down


# --- legend --------------------------------------------------------------------

def _pair_legend(swatches: list[Detection], labels: list[Detection]):
    """Pair each swatch with its nearest label (one-to-one, same-row preferred)."""
    costs = []
    for s, swatch in enumerate(swatches):
        sx, sy = swatch.box.x2, _centre(swatch)[1]
        for l, label in enumerate(labels):
            lx, ly = label.box.x1, _centre(label)[1]
            costs.append((math.hypot(lx - sx, 3 * (ly - sy)), s, l))
    pairs, used_s, used_l = {}, set(), set()
    for _, s, l in sorted(costs):
        if s not in used_s and l not in used_l:
            pairs[s] = l
            used_s.add(s)
            used_l.add(l)
    return pairs


# --- main ----------------------------------------------------------------------

def _title(role: str, roles, texts, flags, confidence, key: str, missing_flag: str):
    detection = _best(roles.get(role, []))
    text, ocr_conf = texts.get(detection.id, ("", 0.0)) if detection else ("", 0.0)
    if detection is None or not text:
        flags.append(missing_flag)
        return None, detection
    confidence[key] = round(detection.confidence * ocr_conf, 4)
    return text, detection


def reconstruct_vbar(frame: DetectionFrame, ocr_texts, image: np.ndarray,
                     min_confidence: float = MIN_CONFIDENCE,
                     rescue_bars: bool = True) -> VerticalBarSpec:
    """Turn detections plus OCR into a ``VerticalBarSpec``.

    Raises ``ChartInterpretationError`` when there are no bars, no x tick
    labels, or fewer than two consistent y ticks. Everything else unresolved is
    reported through ``flags``.
    """
    texts = _text_table(ocr_texts)
    roles = _group_roles(frame, min_confidence)
    flags: list[str] = []
    confidence: dict[str, float] = {}

    plot = _best(roles.get("plot_area", []))
    if plot is None:
        flags.append("plot_area_missing")
    legend = _best(roles.get("legend", []))

    # Bars must sit inside the plot; labels sit outside it, so they are only
    # kept if they line up with the plot along the axis they annotate.
    bars = [d for d in roles.get("vertical_bar", []) if plot is None or _inside(_centre(d), plot)]
    if not bars:
        raise ChartInterpretationError("no vertical bars detected inside the plot area")
    bars.sort(key=lambda d: _centre(d)[0])

    my = mx = 0.0
    if plot is not None:
        my = max(8.0, 0.05 * (plot.box.y2 - plot.box.y1))
        mx = max(8.0, 0.05 * (plot.box.x2 - plot.box.x1))
    y_labels = [d for d in roles.get("y_tick_label", [])
                if plot is None or plot.box.y1 - my <= _centre(d)[1] <= plot.box.y2 + my]
    x_labels = sorted((d for d in roles.get("x_tick_label", [])
                       if plot is None or plot.box.x1 - mx <= _centre(d)[0] <= plot.box.x2 + mx),
                      key=lambda d: _centre(d)[0])
    if not x_labels:
        raise ChartInterpretationError("no x tick labels detected")

    # y calibration
    parsed = []
    for detection in y_labels:
        value = parse_tick_value(texts.get(detection.id, ("", 0.0))[0])
        if value is not None:
            parsed.append((detection, *value))
    if len(parsed) < 2:
        raise ChartInterpretationError("fewer than 2 readable y tick labels")
    a, b, inliers, ambiguous = _calibrate([(_centre(d)[1], v) for d, v, _ in parsed])
    good = [parsed[k] for k in inliers]
    dropped = len(y_labels) - len(good)
    if dropped:
        flags.append(f"y_ticks_dropped:{dropped}")
    if ambiguous:
        flags.append("y_calibration_ambiguous")
    tick_values = [v for _, v, _ in sorted(good, key=lambda t: t[1])]
    if not _is_arithmetic(tick_values):
        flags.append("y_ticks_not_arithmetic")
    units = {u for _, _, u in good if u}
    y_axis = YAxisSpec(min(tick_values), max(tick_values), tick_values,
                       units.pop() if len(units) == 1 else None)
    confidence["y_axis"] = round(statistics.mean(
        d.confidence * texts[d.id][1] for d, _, _ in good), 4)

    # bar values: the free end is the edge farther from the calibrated zero line
    baseline = -b / a
    tick_ys = [_centre(d)[1] for d, _, _ in good]
    lo, hi = (plot.box.y1 - my, plot.box.y2 + my) if plot else (min(tick_ys), max(tick_ys))
    baseline_inside = lo <= baseline <= hi
    if not baseline_inside:
        flags.append("baseline_outside_plot")
    values = []
    free_ends = []
    offsets = []
    for bar in bars:
        box = bar.box
        if baseline_inside and abs(box.y2 - baseline) > abs(box.y1 - baseline):
            free, anchor = box.y2, box.y1
        elif baseline_inside:
            free, anchor = box.y1, box.y2
        else:
            free, anchor = box.y1, box.y2
        values.append(a * free + b)
        free_ends.append(free)
        offsets.append(abs(anchor - baseline))
    if baseline_inside and plot and statistics.median(offsets) > 0.03 * (plot.box.y2 - plot.box.y1):
        flags.append("bar_baseline_offset")
    if plot is not None:
        edges = [a * plot.box.y1 + b, a * plot.box.y2 + b]
        y_axis = YAxisSpec(min(y_axis.min, *edges), max(y_axis.max, *edges),
                           y_axis.tick_values, y_axis.unit)
    # A bar reaching or passing the plot edge beyond the last tick is probably clipped.
    if plot is not None and any(
            (free <= plot.box.y1 + 2 or free >= plot.box.y2 - 2)
            and not min(tick_values) <= v <= max(tick_values)
            for free, v in zip(free_ends, values)):
        flags.append("value_clipped_at_axis")
    confidence["values"] = round(statistics.mean(d.confidence for d in bars), 4)

    # categories
    categories = [texts.get(d.id, ("", 0.0))[0] for d in x_labels]
    if any(not c for c in categories):
        flags.append("category_unreadable")
    confidence["categories"] = round(statistics.mean(
        d.confidence * texts.get(d.id, ("", 0.0))[1] for d in x_labels), 4)
    label_x = [_centre(d)[0] for d in x_labels]
    bar_label = [min(range(len(label_x)), key=lambda k: abs(label_x[k] - _centre(bar)[0]))
                 for bar in bars]

    # series by colour
    rgb = _to_rgb(image)
    bar_colors = [_sample_rgb(rgb, bar) for bar in bars]

    swatches = [d for d in roles.get("legend_swatch", [])
                if legend is None or _inside(_centre(d), legend, 4, 4)]
    legend_labels = [d for d in roles.get("legend_label", [])
                     if legend is None or _inside(_centre(d), legend, 4, 4)]
    swatch_colors = [_sample_rgb(rgb, d) for d in swatches]
    swatch_name = {s: (texts.get(legend_labels[l].id, ("", 0.0))[0] or None)
                   for s, l in _pair_legend(swatches, legend_labels).items()}
    seeds, swatch_of_seed = _legend_seeds(swatch_colors)
    clusters, seed_of_cluster = _cluster_colors(bar_colors, seeds)
    cluster_centre = [
        tuple(statistics.median(bar_colors[k][ch] for k in members) for ch in range(3))
        if bar_colors[members[0]] is not None else None
        for members in clusters
    ]
    # one-to-one nearest-colour match between clusters and legend swatches
    matches: dict[int, int] = {c: swatch_of_seed[seed] for c, seed in enumerate(seed_of_cluster)
                               if seed is not None}
    candidates = sorted(
        (_distance(cluster_centre[c], swatch_colors[s]), c, s)
        for c in range(len(clusters)) for s in range(len(swatches))
        if cluster_centre[c] is not None and swatch_colors[s] is not None
        and c not in matches and s not in swatch_of_seed)
    for distance, c, s in candidates:
        if distance > COLOR_DISTANCE:
            break
        if c not in matches and s not in matches.values():
            matches[c] = s

    ambiguous_series = False
    if swatches and len(matches) < max(len(clusters), len(seeds)):
        ambiguous_series = True
    if not swatches and len(clusters) > 1:
        ambiguous_series = True
    if any(bar_colors[members[0]] is None for members in clusters):
        ambiguous_series = True

    bar_width = statistics.median(d.box.x2 - d.box.x1 for d in bars)
    can_rescue = rescue_bars and baseline_inside
    series = []
    count_mismatch = category_ambiguous = False
    rescued_total = rescued_zero = 0
    for c, members in enumerate(clusters):
        row: list[float | None] = [None] * len(categories)
        kept: dict[int, int] = {}
        best_gap: dict[int, float] = {}
        for k in members:
            slot = bar_label[k]
            gap = abs(label_x[slot] - _centre(bars[k])[0])
            if slot in best_gap:
                category_ambiguous = True
                if gap >= best_gap[slot]:
                    continue
            best_gap[slot] = gap
            kept[slot] = k
            row[slot] = values[k]
        rescued = [False] * len(categories)
        value_conf: list[float | None] = [
            bars[kept[j]].confidence if j in kept else None for j in range(len(categories))]
        centre = cluster_centre[c]
        n_rescued = 0
        if can_rescue and centre is not None and kept:
            # Groups sit at a fixed offset from the tick that depends on the
            # series count, so measure it from this series' detected bars.
            offset = statistics.median(_centre(bars[k])[0] - label_x[j] for j, k in kept.items())
            for j in range(len(categories)):
                if row[j] is not None:
                    continue
                row[j] = _rescue_value(rgb, label_x[j] + offset, bar_width, baseline, a, centre)
                rescued[j] = True
                value_conf[j] = RESCUE_CONFIDENCE
                n_rescued += 1
                rescued_zero += row[j] == 0.0
        rescued_total += n_rescued
        if len(members) + n_rescued != len(categories):
            count_mismatch = True
        name = swatch_name.get(matches[c]) if c in matches else None
        series.append(SeriesSpec(
            name, _hex(None if centre is None else tuple(int(v) for v in centre)), row,
            value_rescued=rescued, value_confidence=value_conf))
    if rescued_total:
        flags.append(f"bars_rescued:{rescued_total}")
    if rescued_zero:
        flags.append("value_inferred_zero")
    if count_mismatch:
        flags.append("category_count_mismatch")
    if category_ambiguous:
        flags.append("category_ambiguous")
    if ambiguous_series:
        flags.append("series_ambiguous")
    if swatches:
        pair_conf = [d.confidence for d in swatches]
        confidence["series"] = round(statistics.mean(pair_conf), 4)

    # titles
    title, title_box = _title("chart_title", roles, texts, flags, confidence,
                              "title", "title_missing")
    x_title, _ = _title("x_axis_title", roles, texts, flags, confidence,
                        "x_axis_title", "x_axis_title_missing")
    y_title, _ = _title("y_axis_title", roles, texts, flags, confidence,
                        "y_axis_title", "y_axis_title_missing")
    if title_box is not None and title_box.box.x2 >= frame.width - TITLE_EDGE_PX:
        flags.append("title_touches_right_edge")

    # adjacency: every neighbouring gap is under 10% of the bar width
    gaps = [(n.box.x1 - p.box.x2) / ((n.box.x2 - n.box.x1 + p.box.x2 - p.box.x1) / 2)
            for p, n in zip(bars, bars[1:])]
    adjacent = bool(gaps) and max(gaps) < ADJACENT_GAP_RATIO

    return VerticalBarSpec(
        source_key=frame.source_key,
        title=title, x_axis_title=x_title, y_axis_title=y_title,
        y_axis=y_axis, categories=categories, series=series,
        bars_adjacent=adjacent, flags=flags, confidence=confidence,
    )
