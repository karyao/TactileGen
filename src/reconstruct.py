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

from src.diagnostics import AnalysisDiagnostics

from src.contracts import (
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
    SeriesSpec,
    VerticalBarSpec,
    YAxisSpec,
    join_legend_assignments,
)
from src.legend_matcher import match_legend_to_bars
from src.text_provenance import consumer_text, text_evidence

MIN_CONFIDENCE = 0.3
DUPLICATE_IOU = 0.5
COLOR_DISTANCE = 40.0
UNREADABLE_CATEGORY_CONFIDENCE = 0.1
MIN_DETECTED_BAR_FRACTION = 0.6  # below this share of detected slots, most values are guesses
BAR_EDGE_TOLERANCE_PX = 5     # a pixel-measured top edge this far from the box edge wins
SHORT_BAR_MIN_PX = 6          # detected bars shorter than this are re-measured from pixels
SHORT_BAR_PLOT_FRACTION = 0.03  # ... or shorter than this share of the plot height
RESCUE_CONFIDENCE = 0.2
# A legend entry no bar matched becomes a series with no visible bars only when its
# swatch box, label box and OCR text are all read at least this confidently (GUESS).
LEGEND_ENTRY_MIN_CONFIDENCE = 0.5
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
_LEADING_ZERO = re.compile(r"(-?)0(\d+)")  # "04": the decimal point of "0.4" was probably lost
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


# --- OCR and detection plumbing ------------------------------------------------

def _text_table(ocr_texts, *, diagnostics=None) -> dict[str, tuple[str, float]]:
    """Normalise OCR input to ``{detection_id: (text, confidence)}``."""
    if isinstance(ocr_texts, OCRFrame):
        ocr_texts = ocr_texts.regions
    if isinstance(ocr_texts, Mapping):
        items = ocr_texts.items()
    else:
        items = ((region.detection_id, region) for region in ocr_texts)
    table = {}
    for key, value in items:
        convert_confidence = False
        if isinstance(value, RecognizedText):
            original, confidence = value.text, value.confidence
        elif isinstance(value, str):
            original, confidence = value, 1.0
        else:
            original, confidence = value
            convert_confidence = True
        trimmed = original.strip()
        table[key] = (trimmed, float(confidence) if convert_confidence else confidence)
        consumer_text(diagnostics, "reconstruction", key, original, table[key][0], table[key][1])
    return table


def _legend_ocr_frame(
    frame: DetectionFrame, texts: Mapping[str, tuple[str, float]]
) -> OCRFrame:
    """Adapt legacy reconstruction OCR input for the legend matcher."""
    return OCRFrame(frame.source_key, tuple(
        RecognizedText(detection.id, *texts[detection.id])
        for detection in frame.detections
        if detection.role == "legend_label" and detection.id in texts
    ))


def _centre(detection: Detection) -> tuple[float, float]:
    box = detection.box
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box.x1, a.box.y1, a.box.x2, a.box.y2
    bx1, by1, bx2, by2 = b.box.x1, b.box.y1, b.box.x2, b.box.y2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(detections: Iterable[Detection], *, diagnostics=None) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.confidence):
        blocker = None
        overlap = None
        for other in kept:
            overlap = _iou(detection, other)
            if not overlap <= DUPLICATE_IOU:
                blocker = other
                break
        if blocker is None:
            kept.append(detection)
        if diagnostics is not None:
            diagnostics.event("reconstruction", "detection_deduplication",
                detection_id=detection.id, role=detection.role,
                outcome="duplicate" if blocker else "retained",
                retained_detection_id=blocker.id if blocker else detection.id,
                duplicate_iou=DUPLICATE_IOU, measured_iou=overlap if blocker else None)
    return kept


def _group_roles(frame: DetectionFrame, min_confidence: float, *, diagnostics=None) -> dict[str, list[Detection]]:
    roles: dict[str, list[Detection]] = {}
    for detection in frame.detections:
        accepted = detection.confidence >= min_confidence
        if diagnostics is not None:
            diagnostics.event("reconstruction", "detection_confidence",
                detection_id=detection.id, role=detection.role, confidence=detection.confidence,
                threshold=min_confidence, outcome="retained" if accepted else "below_confidence")
        if accepted:
            roles.setdefault(detection.role, []).append(detection)
    return {role: _dedupe(items, diagnostics=diagnostics) for role, items in roles.items()}


def _best(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda d: d.confidence, default=None)


def _inside(point: tuple[float, float], box: Detection, margin_x=0.0, margin_y=0.0) -> bool:
    b = box.box
    return (b.x1 - margin_x <= point[0] <= b.x2 + margin_x
            and b.y1 - margin_y <= point[1] <= b.y2 + margin_y)


# --- y calibration -------------------------------------------------------------

def _normalized_tick_text(text: str, *, steps=None) -> str:
    operations = (
        ("strip", lambda value: value.strip()),
        ("unicode_minus", lambda value: value.replace("−", "-")),
        ("en_dash", lambda value: value.replace("–", "-")),
        ("remove_commas", lambda value: value.replace(",", "")),
        ("remove_spaces", lambda value: value.replace(" ", "")),
        ("leading_dollar_signs", lambda value: value.lstrip("$")),
    )
    for name, operation in operations:
        before, text = text, operation(text)
        if steps is not None:
            steps.append({"operation": name, "before": before, "after": text})
    return text


def parse_tick_value(text: str, *, diagnostics=None, detection_id=None) -> tuple[float, str | None] | None:
    """Parse an axis label to (value, unit), preserving existing numeric rules."""
    steps = [] if diagnostics is not None else None
    cleaned = _normalized_tick_text(text, steps=steps)
    normalized = cleaned
    unit = None
    if cleaned.endswith("%"):
        unit, cleaned = "%", cleaned[:-1]
        if steps is not None:
            steps.append({"operation": "percent_unit", "before": cleaned + "%", "after": cleaned})
    scale = 1.0
    if cleaned and cleaned[-1].upper() in _SUFFIX and not cleaned[-1].isdigit():
        scale = _SUFFIX[cleaned[-1].upper()]
        before, cleaned = cleaned, cleaned[:-1]
        if steps is not None:
            steps.append({"operation": "suffix_expansion", "before": before,
                          "after": cleaned, "multiplier": scale})
    matched = bool(_NUMBER.fullmatch(cleaned))
    value = float(cleaned) * scale if matched else None
    result = (value, unit) if matched and math.isfinite(value) else None
    if diagnostics is not None:
        diagnostics.event("reconstruction", "numeric_parse", detection_id=detection_id,
            consumer_text=text, normalized_text=normalized, numeric_text=cleaned,
            transformations=steps, multiplier=scale, unit=unit, converted_value=value,
            result=result, text_evidence=text_evidence(diagnostics, detection_id),
            reason="parsed" if result else "nonfinite" if matched else "invalid_number")
    return result


def _repair_lost_decimal(text: str) -> str | None:
    """``"04"`` -> ``"0.4"``: a tick label that is a zero followed by digits lost its decimal point."""
    cleaned = text.strip().replace("−", "-").replace("–", "-").replace(",", "").replace(" ", "").lstrip("$")
    match = _LEADING_ZERO.fullmatch(cleaned)
    return f"{match.group(1)}0.{match.group(2)}" if match else None


def _calibrate(ticks: list[tuple[float, float]], *, diagnostics: AnalysisDiagnostics | None = None):
    """Fit ``value = a * y + b`` robustly.

    Every pair of ticks proposes a line; the line with the most ticks within a
    pixel tolerance wins (ties go to the smaller residual) and is refit by
    least squares on its inliers. Returns ``(a, b, inlier_indexes, ambiguous)``.
    """
    ys = sorted(y for y, _ in ticks)
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0]
    tolerance = max(3.0, 0.2 * statistics.median(gaps)) if gaps else 3.0
    if diagnostics is not None:
        diagnostics.event("reconstruction", "calibration_inputs", ticks=ticks,
            sorted_y=ys, positive_gaps=gaps, tolerance_pixels=tolerance,
            tolerance_floor=3.0, median_gap_fraction=0.2)
    best = None
    tied_sets: set[frozenset[int]] = set()
    for i, j in itertools.combinations(range(len(ticks)), 2):
        (yi, vi), (yj, vj) = ticks[i], ticks[j]
        if yi == yj or vi == vj:
            if diagnostics is not None:
                diagnostics.event("reconstruction", "calibration_candidate", pair=[i, j],
                    outcome="skipped", reason="equal_y" if yi == yj else "equal_value")
            continue
        a = (vj - vi) / (yj - yi)
        b = vi - a * yi
        residuals = [abs(v - (a * y + b)) / abs(a) for y, v in ticks]
        inliers = frozenset(k for k, r in enumerate(residuals) if r <= tolerance)
        score = (len(inliers), -sum(residuals[k] ** 2 for k in inliers))
        if diagnostics is not None:
            diagnostics.event("reconstruction", "calibration_candidate", pair=[i, j],
                a=a, b=b, residuals_pixels=residuals, inlier_indexes=sorted(inliers), score=score,
                selection="new_best" if best is None or score > best[0] else
                          "same_inlier_count" if score[0] == best[0][0] else "not_selected")
        if best is None or score > best[0]:
            best, tied_sets = (score, inliers, a, b, residuals), {inliers}
        elif score[0] == best[0][0]:
            tied_sets.add(inliers)
    if diagnostics is not None:
        diagnostics.event("reconstruction", "calibration_selection",
            outcome="rejected" if best is None or len(best[1]) < 2 else "selected",
            selected_score=best[0] if best else None,
            inlier_indexes=sorted(best[1]) if best else [],
            competing_inlier_sets=sorted([sorted(group) for group in tied_sets]),
            ambiguous=len(tied_sets) > 1,
            flag="chart_interpretation_blocked" if best is None or len(best[1]) < 2 else
                 "y_calibration_ambiguous" if len(tied_sets) > 1 else None)
    if best is None or len(best[1]) < 2:
        raise ChartInterpretationError("y-axis calibration needs at least 2 consistent ticks")
    inliers = sorted(best[1])
    a, b = np.polyfit([ticks[k][0] for k in inliers], [ticks[k][1] for k in inliers], 1)
    if diagnostics is not None:
        diagnostics.event("reconstruction", "y_calibration", a=float(a), b=float(b),
            formula="value = a * y + b", tolerance_pixels=tolerance,
            inlier_indexes=inliers, ambiguous=len(tied_sets) > 1,
            selection_a=best[2], selection_b=best[3],
            selection_residuals_pixels=best[4],
            residuals_pixels=[float(abs(v - (a * y + b)) / abs(a)) for y, v in ticks])
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


def _unreadable_category(text: str) -> bool:
    """Empty or punctuation-only text, including a lone symbol like ``|``.

    OCR confidence is not used: wrong reads and correct reads overlap (0.7-0.96).
    """
    return not any(ch.isalnum() for ch in text)


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


# --- main ----------------------------------------------------------------------

def _title(role: str, roles, texts, flags, confidence, key: str, missing_flag: str,
           *, diagnostics=None, frame=None):
    detection = _best(roles.get(role, []))
    text, ocr_conf = texts.get(detection.id, ("", 0.0)) if detection else ("", 0.0)
    if diagnostics is not None:
        candidates = [d.id for d in frame.detections if d.role == role] if frame else []
        reason = ("no_detection" if not candidates else "filtered_out") if detection is None else (
            "absent_ocr" if detection.id not in texts else "empty_text" if not text else "read")
        evidence = text_evidence(diagnostics, detection.id) if detection else {}
        event_id = diagnostics.event("reconstruction", "title_selection", role=role, field=key,
            candidate_ids=candidates, retained_ids=[d.id for d in roles.get(role, [])],
            detection_id=detection.id if detection else None, text=text, ocr_confidence=ocr_conf,
            reason=reason, text_evidence=evidence,
            filter_evidence_ids=[e["event_id"] for e in diagnostics.record["events"]
                                 if e.get("detection_id") in candidates
                                 and e["code"] in ("detection_confidence", "detection_deduplication")],
            flag=missing_flag if detection is None or not text else None)
        diagnostics.field_source(key, [detection.id] if detection else [], [event_id])
    if detection is None or not text:
        flags.append(missing_flag)
        return None, detection
    confidence[key] = round(detection.confidence * ocr_conf, 4)
    return text, detection


def _bar_less_series(pending, entries, frame, texts, rgb, legend, taken_names, n_categories,
                     *, diagnostics=None):
    """Series with no visible bars for legend entries no bar matched.

    Returns ``(added, skipped)``: ``added`` is ``(legend position, SeriesSpec)`` pairs and
    ``skipped`` has one ``legend_entry_skipped:<reason>:<text>`` flag per entry that failed a
    guard: the text was unreadable, the swatch, label or text confidence was below
    ``LEGEND_ENTRY_MIN_CONFIDENCE``, the swatch lay outside the legend box, the name repeated
    another series, or the swatch colour could not be sampled.
    """
    by_id = {detection.id: detection for detection in frame.detections}
    position = {(entry.swatch_id, entry.label_id): index for index, entry in enumerate(entries)}
    names = {name.casefold() for name in taken_names if name}
    added, skipped = [], []
    for entry in pending:
        swatch, label = by_id[entry.swatch_id], by_id[entry.label_id]
        text, text_confidence = texts.get(entry.label_id, ("", 0.0))
        colour = None if not text else _sample_rgb(rgb, swatch)
        if not text:
            reason = "unreadable_text"
        elif min(swatch.confidence, label.confidence, text_confidence) < LEGEND_ENTRY_MIN_CONFIDENCE:
            reason = "low_confidence"
        elif legend is not None and not _inside(_centre(swatch), legend, 4, 4):
            reason = "outside_legend"
        elif (name_key := text.casefold()) in names:
            reason = "duplicate_name"
        elif colour is None:
            reason = "unreadable_swatch"
        else:
            reason = None
        if diagnostics is not None:
            diagnostics.event("reconstruction", "legend_entry_decision",
                label_id=label.id, swatch_id=swatch.id, text=text,
                text_evidence=text_evidence(diagnostics, label.id),
                swatch_confidence=swatch.confidence, label_confidence=label.confidence,
                ocr_confidence=text_confidence, confidence_threshold=LEGEND_ENTRY_MIN_CONFIDENCE,
                first_failing_guard=reason, outcome="skipped" if reason else "added",
                duplicate_name_key=name_key if reason in (None, "duplicate_name") else None,
                existing_name_keys=sorted(names) if reason in (None, "duplicate_name") else None,
                flag=f"legend_entry_skipped:{reason}:{text or entry.label_id}" if reason else None)
        if reason is not None:
            skipped.append(f"legend_entry_skipped:{reason}:{text or entry.label_id}")
            continue
        names.add(text.casefold())
        added.append((position[(entry.swatch_id, entry.label_id)],
                      SeriesSpec(text, _hex(colour), [None] * n_categories, no_visible_bars=True)))
    return added, skipped


def _in_legend_order(series, identities, entries, added):
    """Merge bar-less series into ``series`` so the key follows the legend order.

    Series with bars keep their left-to-right bar order. When that order matches their legend
    order, every series is returned in legend order. When it does not, the bar order is kept
    and the bar-less series follow in legend order; the second result is then True.
    """
    position = {(entry.swatch_id, entry.label_id): index for index, entry in enumerate(entries)}
    visible = [position.get(identity[:2]) if identity else None for identity in identities]
    if all(p is not None for p in visible) and visible == sorted(visible):
        merged = sorted([*zip(visible, series), *added], key=lambda item: item[0])
        return [item for _, item in merged], False
    return [*series, *(item for _, item in sorted(added, key=lambda pair: pair[0]))], True


def reconstruct_vbar(frame: DetectionFrame, ocr_texts, image: np.ndarray,
                     min_confidence: float = MIN_CONFIDENCE,
                     rescue_bars: bool = True,
                     category_guard: bool = True, *,
                     legend_mapping: LegendMappingFrame | None = None,
                     diagnostics: AnalysisDiagnostics | None = None
                     ) -> VerticalBarSpec:
    """Turn detections plus OCR into a ``VerticalBarSpec``.

    ``legend_mapping`` supplies validated series names from the shared matcher.
    When omitted, the mapping is computed from ``frame``, ``ocr_texts``, and
    ``image`` so legacy callers use the same matching implementation.

    Raises ``ChartInterpretationError`` when there are no bars, no x tick
    labels, or fewer than two consistent y ticks. Everything else unresolved is
    reported through ``flags``.
    """
    if diagnostics is not None:
        diagnostics.record["spec_sources"] = {}
    texts = _text_table(ocr_texts, diagnostics=diagnostics)

    def trace(code, **evidence):
        if diagnostics is not None:
            return diagnostics.event("reconstruction", code, **evidence)

    if legend_mapping is None:
        legend_ocr = (
            ocr_texts
            if isinstance(ocr_texts, OCRFrame)
            else _legend_ocr_frame(frame, texts)
        )
        legend_mapping = match_legend_to_bars(image, frame, legend_ocr,
            **({"diagnostics": diagnostics} if diagnostics is not None else {}))
    assignment_by_bar = {
        bar.id: assignment
        for bar, assignment in join_legend_assignments(frame, legend_mapping)
    }
    roles = _group_roles(frame, min_confidence, diagnostics=diagnostics)
    trace("detection_filter", min_confidence=min_confidence, duplicate_iou=DUPLICATE_IOU,
          retained_ids=[d.id for items in roles.values() for d in items],
          excluded=[{"id": d.id, "reason": "below_confidence" if d.confidence < min_confidence
                     else "duplicate"} for d in frame.detections
                    if d not in roles.get(d.role, [])])
    flags: list[str] = []
    confidence: dict[str, float] = {}

    plot = _best(roles.get("plot_area", []))
    if plot is None:
        flags.append("plot_area_missing")
    legend = _best(roles.get("legend", []))

    # Bars must sit inside the plot; labels sit outside it, so they are only
    # kept if they line up with the plot along the axis they annotate.
    bars = [d for d in roles.get("vertical_bar", []) if plot is None or _inside(_centre(d), plot)]
    trace("plot_selection", plot_id=plot.id if plot else None,
          retained_bar_ids=[d.id for d in bars],
          excluded_bar_ids=[d.id for d in roles.get("vertical_bar", []) if d not in bars])
    if not bars:
        trace("reconstruction_blocked", reason="no_bars_inside_plot",
              flag="chart_interpretation_blocked")
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
    trace("tick_selection", x_label_ids=[d.id for d in x_labels],
          y_label_ids=[d.id for d in y_labels], margin_x=mx, margin_y=my)
    for role, labels, lower, upper, axis in (
        ("x_tick_label", x_labels, plot.box.x1 - mx if plot else None,
         plot.box.x2 + mx if plot else None, 0),
        ("y_tick_label", y_labels, plot.box.y1 - my if plot else None,
         plot.box.y2 + my if plot else None, 1),
    ):
        if diagnostics is not None:
            trace("tick_spatial_selection", role=role,
                candidate_ids=[d.id for d in roles.get(role, [])],
                accepted_bounds=[lower, upper],
                decisions=[{"detection_id": d.id, "coordinate": _centre(d)[axis],
                            "outcome": "retained" if d in labels else "outside_plot_band"}
                           for d in roles.get(role, [])])
    if not x_labels:
        trace("reconstruction_blocked", reason="no_x_labels", role="x_tick_label",
              detected_ids=[d.id for d in frame.detections if d.role == "x_tick_label"],
              flag="chart_interpretation_blocked")
        raise ChartInterpretationError("no x tick labels detected")

    # y calibration
    parsed = []
    repaired_ids: set[str] = set()
    for detection in y_labels:
        text = texts.get(detection.id, ("", 0.0))[0]
        repaired = _repair_lost_decimal(text)
        value = parse_tick_value(repaired if repaired else text,
                                 diagnostics=diagnostics, detection_id=detection.id)
        if diagnostics is not None:
            consumed = text_evidence(diagnostics, detection.id)
            numeric = next(e for e in reversed(diagnostics.record["events"])
                           if e["code"] == "numeric_parse" and e["detection_id"] == detection.id)
            trace("y_tick_parse", detection_id=detection.id,
                  raw_text=consumed["input_text"], consumer_text=texts.get(detection.id, ("", 0))[0],
                  input_presence=consumed["input_presence"], normalized_text=numeric["normalized_text"],
                  evidence_ids=[numeric["event_id"], *consumed["evidence_ids"]],
                  ocr_confidence=texts.get(detection.id, ("", 0))[1],
                  y=_centre(detection)[1], parsed_value=value[0] if value else None,
                  reason="numeric_candidate" if value else "unparseable")
        if value is not None:
            parsed.append((detection, *value))
            if repaired:
                repaired_ids.add(detection.id)
    if diagnostics is not None:
        source_ids = [d.id for d, _, _ in parsed]
        link = trace("calibration_tick_sources", ticks=[
            {"index": k, "detection_id": d.id, "y": _centre(d)[1], "value": v, "unit": unit,
             "parse_evidence_ids": [e["event_id"] for e in diagnostics.record["events"]
                                    if e["code"] == "numeric_parse" and e["detection_id"] == d.id]}
            for k, (d, v, unit) in enumerate(parsed)])
        diagnostics.field_source("y_axis.ticks", source_ids, [link])
    if len(parsed) < 2:
        trace("reconstruction_blocked", reason="insufficient_readable_y_ticks",
              readable_ids=[d.id for d, _, _ in parsed], required_count=2,
              flag="chart_interpretation_blocked")
        raise ChartInterpretationError("fewer than 2 readable y tick labels")
    a, b, inliers, ambiguous = _calibrate(
        [(_centre(d)[1], v) for d, v, _ in parsed], diagnostics=diagnostics)
    trace("y_tick_decisions", ticks=[{
        "detection_id": d.id, "value": v, "y": _centre(d)[1],
        "reason": "retained" if k in inliers else "numeric_outlier"}
        for k, (d, v, _) in enumerate(parsed)])
    good = [parsed[k] for k in inliers]
    if diagnostics is not None:
        diagnostics.field_source("y_axis.ticks", [d.id for d, _, _ in sorted(good, key=lambda t: t[1])],
            [link, *diagnostics.evidence_for("y_calibration"),
             *diagnostics.evidence_for("calibration_selection")])
    dropped = len(y_labels) - len(good)
    if dropped:
        parsed_ids = {d.id for d, _, _ in parsed}
        trace("y_ticks_dropped", count=dropped, flag=f"y_ticks_dropped:{dropped}",
              unparseable_ids=[d.id for d in y_labels if d.id not in parsed_ids],
              numeric_outlier_ids=[d.id for k, (d, _, _) in enumerate(parsed) if k not in inliers])
    if dropped:
        flags.append(f"y_ticks_dropped:{dropped}")
    if ambiguous:
        flags.append("y_calibration_ambiguous")
    negative_read = any(v < 0 for _, v, _ in parsed)
    if negative_read:
        # A minus sign was read on some tick label, even if the fit dropped it. OCR loses
        # minus signs, so an axis whose surviving ticks are all >= 0 may still be negative.
        trace("y_tick_negative_read", detection_ids=[d.id for d, v, _ in parsed if v < 0],
              flag="y_tick_negative_read")
        flags.append("y_tick_negative_read")
    repaired_good = [d for d, _, _ in good if d.id in repaired_ids]
    if repaired_good:
        trace("y_tick_decimal_repaired", detection_ids=[d.id for d in repaired_good],
              ambiguous_calibration=bool(ambiguous),
              flag=f"y_tick_decimal_repaired:{len(repaired_good)}")
        flags.append(f"y_tick_decimal_repaired:{len(repaired_good)}")
        if ambiguous:   # two scales fit equally well, so the repair cannot be trusted
            trace("leading_zero_check", detection_ids=[d.id for d in repaired_good],
                  matched=True, flag="y_tick_leading_zero")
            flags.append("y_tick_leading_zero")
    tick_values = [v for _, v, _ in sorted(good, key=lambda t: t[1])]
    trace("y_ticks_not_arithmetic" if not _is_arithmetic(tick_values) else "y_tick_spacing",
          tick_values=tick_values, differences=[v - u for u, v in zip(tick_values, tick_values[1:])],
          relative_tolerance=ARITHMETIC_TOLERANCE, absolute_tolerance=1e-12)
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
    trace("baseline_outside_plot" if not baseline_inside else "baseline",
          zero_y=baseline, accepted_bounds=[lo, hi], a=a, b=b, margin_y=my)
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
        trace("bar_value", bar_id=bar.id, free_end_y=free, anchor_y=anchor,
              anchor_offset=offsets[-1], value=values[-1])
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
    label_confidence = []
    for j, d in enumerate(x_labels):
        text, ocr_conf = texts.get(d.id, ("", 0.0))
        unreadable = _unreadable_category(text) if category_guard else not text
        if diagnostics is not None:
            evidence = text_evidence(diagnostics, d.id)
            event_id = trace("category_text_check", detection_id=d.id, category_index=j,
                text=text, ocr_confidence=ocr_conf, category_guard=category_guard,
                rule="no_alphanumeric" if category_guard else "empty_text",
                unreadable=unreadable, unreadable_confidence=UNREADABLE_CATEGORY_CONFIDENCE,
                text_evidence=evidence, flag="category_unreadable" if unreadable else None)
            diagnostics.field_source(f"category[{j}]", [d.id], [event_id])
        label_confidence.append(
            UNREADABLE_CATEGORY_CONFIDENCE if unreadable else d.confidence * ocr_conf)
        if unreadable:
            confidence[f"category[{j}]"] = UNREADABLE_CATEGORY_CONFIDENCE
    if any(key.startswith("category[") for key in confidence):
        flags.append("category_unreadable")
    confidence["categories"] = round(statistics.mean(label_confidence), 4)
    label_x = [_centre(d)[0] for d in x_labels]
    bar_label = [min(range(len(label_x)), key=lambda k: abs(label_x[k] - _centre(bar)[0]))
                 for bar in bars]

    # Matched legend identities are authoritative series groups. Colour is
    # only a fallback for bars whose semantic assignment was unresolved.
    rgb = _to_rgb(image)
    bar_colors = [_sample_rgb(rgb, bar) for bar in bars]

    swatches = [d for d in roles.get("legend_swatch", [])
                if legend is None or _inside(_centre(d), legend, 4, 4)]
    swatch_colors = [_sample_rgb(rgb, d) for d in swatches]
    seeds, _ = _legend_seeds(swatch_colors)

    def group_bars(skip: set[int]):
        matched_groups: dict[tuple[str, str, str], list[int]] = {}
        identities_by_swatch: dict[str, set[tuple[str, str]]] = {}
        unresolved = []
        for index, bar in enumerate(bars):
            if index in skip:
                continue
            assignment = assignment_by_bar.get(bar.id)
            if (
                assignment is None
                or assignment.status != "matched"
                or assignment.swatch_id is None
                or assignment.label_id is None
                or assignment.series_name is None
            ):
                unresolved.append(index)
                continue
            identity = (
                assignment.swatch_id,
                assignment.label_id,
                assignment.series_name,
            )
            matched_groups.setdefault(identity, []).append(index)
            identities_by_swatch.setdefault(assignment.swatch_id, set()).add(
                (assignment.label_id, assignment.series_name)
            )

        conflicting_swatches = {
            swatch_id
            for swatch_id, identities in identities_by_swatch.items()
            if len(identities) > 1
        }
        grouped: list[tuple[list[int], str | None, tuple[str, str, str] | None]] = [
            (members, None if identity[0] in conflicting_swatches else identity[2], identity)
            for identity, members in matched_groups.items()
        ]
        if unresolved:
            fallback_clusters, _ = _cluster_colors(
                [bar_colors[index] for index in unresolved], seeds
            )
            grouped.extend(
                ([unresolved[local_index] for local_index in members], None, None)
                for members in fallback_clusters
            )
        return matched_groups, unresolved, conflicting_swatches, grouped

    # Very short boxes carry mostly outline and antialiasing pixels, so their
    # colour is unreliable and their height is a few pixels of box noise: leave
    # them out of grouping and let the rescue re-measure their slots.
    plot_height = plot.box.y2 - plot.box.y1 if plot is not None else 0.0
    short_limit = max(SHORT_BAR_MIN_PX, SHORT_BAR_PLOT_FRACTION * plot_height)
    short = {k for k, bar in enumerate(bars) if bar.box.y2 - bar.box.y1 < short_limit}
    if not (rescue_bars and baseline_inside) or len(short) == len(bars):
        short = set()
    matched_groups, unresolved, conflicting_swatches, grouped = group_bars(short)
    # Rescue is safe when group membership is settled. With a mixture of
    # semantic and fallback groups (or conflicting semantic identities), the
    # same pixels could otherwise be inferred into more than one series.
    grouping_is_settled = not (
        (matched_groups and unresolved) or conflicting_swatches
    )
    can_rescue = rescue_bars and baseline_inside and grouping_is_settled
    if short and not can_rescue:
        short = set()   # nothing will re-measure them, so keep them in their groups
        matched_groups, unresolved, conflicting_swatches, grouped = group_bars(short)
        grouping_is_settled = not (
            (matched_groups and unresolved) or conflicting_swatches
        )
        can_rescue = rescue_bars and baseline_inside and grouping_is_settled
    short_bars = len(short)
    grouped.sort(key=lambda item: min(item[0]))
    clusters = [members for members, _, _ in grouped]
    cluster_names = [name for _, name, _ in grouped]
    cluster_identities = [identity for _, _, identity in grouped]
    cluster_centre = [
        tuple(statistics.median(bar_colors[k][ch] for k in members) for ch in range(3))
        if bar_colors[members[0]] is not None else None
        for members in clusters
    ]
    has_detected_swatches = any(
        detection.role == "legend_swatch" for detection in frame.detections
    )
    ambiguous_series = bool(conflicting_swatches)
    if unresolved and (has_detected_swatches or len(clusters) > 1):
        ambiguous_series = True
    if any(bar_colors[members[0]] is None for members in clusters):
        ambiguous_series = True

    bar_width = statistics.median(d.box.x2 - d.box.x1 for d in bars)
    # The detector sometimes puts a bar's top edge several pixels off while the
    # box still overlaps the bar well. Check positive bars against the pixels.
    edges_corrected = 0
    if can_rescue:
        for c, members in enumerate(clusters):
            if cluster_centre[c] is None:
                continue
            for k in members:
                box = bars[k].box
                if abs(box.y2 - baseline) > abs(box.y1 - baseline):
                    continue
                half = RESCUE_STRIP_FRACTION * (box.x2 - box.x1) / 2
                left = max(0, int(round(_centre(bars[k])[0] - half)))
                right = min(rgb.shape[1], max(left + 1, int(round(_centre(bars[k])[0] + half))))
                up = _run_length(rgb, left, right, int(round(baseline)) - 1, -1, cluster_centre[c])
                if up >= 1 and abs(box.y1 - (round(baseline) - up)) >= BAR_EDGE_TOLERANCE_PX:
                    values[k] = -a * up
                    edges_corrected += 1
    trace("series_ambiguous" if ambiguous_series else "series_grouping",
          unresolved_bar_ids=[bars[k].id for k in unresolved],
          conflicting_swatches=sorted(conflicting_swatches),
          unreadable_color_bar_ids=[bars[k].id for k, color in enumerate(bar_colors) if color is None],
          has_detected_swatches=has_detected_swatches,
          clusters=[{"bar_ids": [bars[k].id for k in members], "name": name,
                     "identity": identity, "color": centre}
                    for members, name, identity, centre in zip(
                        clusters, cluster_names, cluster_identities, cluster_centre)],
          rescue_enabled=can_rescue, rescue_requested=rescue_bars,
          baseline_inside=baseline_inside, grouping_is_settled=grouping_is_settled)
    series = []
    detected_slots = 0
    # Record immediately so failures retain evidence. Keep the original cluster
    # index and recorded event reference for legend-only series that shift indexes.
    series_events = []
    series_sources = {}

    def trace_series(code, **evidence):
        if diagnostics is not None:
            trace(code, **evidence)
            series_events.append((evidence["series_index"], diagnostics.record["events"][-1]))

    count_mismatch = category_ambiguous = False
    rescued_total = rescued_zero = 0
    for c, members in enumerate(clusters):
        row: list[float | None] = [None] * len(categories)
        kept: dict[int, int] = {}
        best_gap: dict[int, float] = {}
        for k in members:
            slot = bar_label[k]
            gap = abs(label_x[slot] - _centre(bars[k])[0])
            trace_series("category_assignment", series_index=c, bar_id=bars[k].id,
                  category_index=slot, label_id=x_labels[slot].id,
                  label=categories[slot], label_x=label_x[slot], bar_x=_centre(bars[k])[0], gap=gap)
            if slot in best_gap:
                trace_series("category_ambiguous", series_index=c, category_index=slot,
                      competing_bar_ids=[bars[kept[slot]].id, bars[k].id],
                      gaps=[best_gap[slot], gap],
                      selected_bar_id=bars[kept[slot] if gap >= best_gap[slot] else k].id)
                category_ambiguous = True
                if gap >= best_gap[slot]:
                    continue
            best_gap[slot] = gap
            kept[slot] = k
            row[slot] = values[k]
        detected_slots += len(kept)
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
                trace_series("pixel_rescue", series_index=c, category_index=j,
                      center_x=label_x[j] + offset, width=bar_width, baseline=baseline,
                      value=row[j], inferred_zero=row[j] == 0.0)
                n_rescued += 1
                rescued_zero += row[j] == 0.0
        rescued_total += n_rescued
        trace_series("category_count_mismatch" if len(members) + n_rescued != len(categories)
              else "category_counts", series_index=c, detected_bars=len(members),
              rescued_bars=n_rescued, category_count=len(categories),
              missing_slots=[j for j, v in enumerate(row) if v is None])
        if len(members) + n_rescued != len(categories):
            count_mismatch = True
        name = cluster_names[c]
        series.append(SeriesSpec(
            name, _hex(None if centre is None else tuple(int(v) for v in centre)), row,
            value_rescued=rescued, value_confidence=value_conf))
        if diagnostics is not None:
            identity = cluster_identities[c]
            event_id = trace("series_text_source", text=name, name_source="legend_mapping",
                         text_evidence=text_evidence(diagnostics, identity[1]) if identity else None,
                         label_id=identity[1] if identity else None,
                         swatch_id=identity[0] if identity else None,
                         bar_ids=[bars[k].id for k in members],
                         reason="matched_label" if name else "unresolved_name")
            source = diagnostics.field_source(f"series_name[{c}]",
                [identity[1]] if identity and name else [], [event_id])
            series_sources[id(series[-1])] = source
    # Share of category x series slots (for series with at least one detected
    # bar) that had a detected bar; very short bars were detected, just re-measured.
    slots = len(categories) * len(clusters)
    if slots and (detected_slots + short_bars) / slots < MIN_DETECTED_BAR_FRACTION:
        flags.append(f"bars_mostly_missed:{round(100 * (1 - (detected_slots + short_bars) / slots))}")
    if edges_corrected:
        flags.append(f"bar_edge_corrected:{edges_corrected}")
    if short_bars:
        flags.append(f"short_bars_remeasured:{short_bars}")
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

    # Legend entries that no bar matched are series with no visible bars. They are
    # added only when every bar in the chart was matched to an entry, so an entry
    # whose bars were merely hard to match can never become a phantom series.
    used_swatches = {identity[0] for identity in matched_groups}
    used_labels = {identity[1] for identity in matched_groups}
    pending = [entry for entry in legend_mapping.entries
               if entry.swatch_id not in used_swatches and entry.label_id not in used_labels]
    if pending:
        settled = (bool(matched_groups) and not unresolved
                   and not conflicting_swatches and not ambiguous_series)
        added: list[tuple[int, SeriesSpec]] = []
        if settled:
            added, skipped = _bar_less_series(
                pending, legend_mapping.entries, frame, texts, rgb, legend,
                [s.name for s in series], len(categories), diagnostics=diagnostics)
        else:
            skipped = [f"legend_entry_skipped:bars_not_settled:{entry.name or entry.label_id}"
                       for entry in pending]
        flags.extend(skipped)
        if diagnostics is not None and not settled:
            for entry, flag in zip(pending, skipped):
                trace("legend_entry_decision", label_id=entry.label_id, swatch_id=entry.swatch_id,
                      first_failing_guard="bars_not_settled", outcome="skipped", flag=flag)
        if diagnostics is not None:
            for position, item in added:
                entry = legend_mapping.entries[position]
                event_id = trace("series_text_source", label_id=entry.label_id,
                                 swatch_id=entry.swatch_id, text=item.name,
                                 text_evidence=text_evidence(diagnostics, entry.label_id),
                                 name_source="consumer_text", no_visible_bars=True)
                series_sources[id(item)] = {"detection_ids": [entry.label_id],
                    "evidence_ids": [event_id], "availability": "available"}
        if added:
            visible_series = series
            series, reordered = _in_legend_order(
                series, cluster_identities, legend_mapping.entries, added)
            final_indexes = {id(item): index for index, item in enumerate(series)}
            original_to_final = [final_indexes[id(item)] for item in visible_series]
            for original_index, event in series_events:
                event["series_index"] = original_to_final[original_index]
            if reordered:
                flags.append("legend_order_differs")

    if diagnostics is not None:
        sources = diagnostics.record.setdefault("spec_sources", {})
        for key in list(sources):
            if key.startswith("series_name["):
                del sources[key]
        for index, item in enumerate(series):
            sources[f"series_name[{index}]"] = series_sources[id(item)]

    # titles
    title, title_box = _title("chart_title", roles, texts, flags, confidence,
                              "title", "title_missing", diagnostics=diagnostics, frame=frame)
    x_title, _ = _title("x_axis_title", roles, texts, flags, confidence,
                        "x_axis_title", "x_axis_title_missing", diagnostics=diagnostics, frame=frame)
    y_title, _ = _title("y_axis_title", roles, texts, flags, confidence,
                        "y_axis_title", "y_axis_title_missing", diagnostics=diagnostics, frame=frame)
    if title_box is not None and title_box.box.x2 >= frame.width - TITLE_EDGE_PX:
        trace("title_edge_check", detection_id=title_box.id, x2=title_box.box.x2,
              accepted_maximum=frame.width - TITLE_EDGE_PX, threshold_pixels=TITLE_EDGE_PX,
              flag="title_touches_right_edge")
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
