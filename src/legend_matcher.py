"""In-memory OpenCV matching between legend swatches and vertical bars."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.contracts import (
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendEntry,
    LegendMappingFrame,
    OCRFrame,
    validate_unit_interval,
)


@dataclass(frozen=True)
class _LegendCandidate:
    swatch: Detection
    label: Detection
    name: str | None


@dataclass(frozen=True)
class _BarScore:
    bar_id: str
    candidate: _LegendCandidate | None = None
    best_error: float | None = None
    runner_up_error: float | None = None
    unavailable_status: str | None = None


@dataclass(frozen=True)
class _LegendScores:
    source_key: str
    bars: tuple[_BarScore, ...]
    entries: tuple[LegendEntry, ...] = ()


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: tuple[float, float, int, int]


def _center(detection: Detection) -> tuple[float, float]:
    box = detection.box
    return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)


def _sort_key(detection: Detection) -> tuple[float, float, str]:
    x, y = _center(detection)
    return (y, x, detection.id)


def _contains_center(parent: Detection, child: Detection) -> bool:
    x, y = _center(child)
    box = parent.box
    return box.x1 <= x < box.x2 and box.y1 <= y < box.y2


def _parent_for(child: Detection, parents: list[Detection]) -> str | None:
    containing = [parent for parent in parents if _contains_center(parent, child)]
    if not containing:
        return None
    return min(
        containing,
        key=lambda item: (
            (item.box.x2 - item.box.x1) * (item.box.y2 - item.box.y1),
            *_sort_key(item),
        ),
    ).id


def _pair_geometry(swatch: Detection, label: Detection) -> tuple[float, float] | None:
    swatch_x, swatch_y = _center(swatch)
    label_x, label_y = _center(label)
    if label_x <= swatch_x:
        return None
    vertical_overlap = min(swatch.box.y2, label.box.y2) - max(
        swatch.box.y1, label.box.y1
    )
    if vertical_overlap <= 0:
        return None
    horizontal_gap = max(0.0, label.box.x1 - swatch.box.x2)
    larger_height = max(
        swatch.box.y2 - swatch.box.y1,
        label.box.y2 - label.box.y1,
    )
    if horizontal_gap > 3 * larger_height:
        return None
    return horizontal_gap, abs(label_y - swatch_y)


def _add_cost(
    left: tuple[float, float, int, int],
    right: tuple[float, float, int, int],
) -> tuple[float, float, int, int]:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    )


def _negate_cost(
    value: tuple[float, float, int, int],
) -> tuple[float, float, int, int]:
    return (-value[0], -value[1], -value[2], -value[3])


def _minimum_cost_pairs(
    swatches: list[Detection], labels: list[Detection]
) -> list[tuple[Detection, Detection]]:
    """Return a maximum-cardinality, minimum-geometry-cost pairing."""
    if not swatches or not labels:
        return []
    source = 0
    swatch_offset = 1
    label_offset = swatch_offset + len(swatches)
    sink = label_offset + len(labels)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    pair_edges: list[tuple[_FlowEdge, Detection, Detection]] = []
    zero = (0.0, 0.0, 0, 0)

    def add_edge(
        start: int, end: int, capacity: int,
        cost: tuple[float, float, int, int],
    ) -> _FlowEdge:
        forward = _FlowEdge(end, len(graph[end]), capacity, cost)
        backward = _FlowEdge(start, len(graph[start]), 0, _negate_cost(cost))
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    for index in range(len(swatches)):
        add_edge(source, swatch_offset + index, 1, zero)
    for index in range(len(labels)):
        add_edge(label_offset + index, sink, 1, zero)
    for swatch_index, swatch in enumerate(swatches):
        for label_index, label in enumerate(labels):
            geometry = _pair_geometry(swatch, label)
            if geometry is None:
                continue
            edge = add_edge(
                swatch_offset + swatch_index,
                label_offset + label_index,
                1,
                (*geometry, swatch_index, label_index),
            )
            pair_edges.append((edge, swatch, label))

    while True:
        distances: list[tuple[float, float, int, int] | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = zero
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                current = distances[node]
                if current is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity == 0:
                        continue
                    candidate = _add_cost(current, edge.cost)
                    target = distances[edge.to]
                    if target is None or candidate < target:
                        distances[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            step = previous[node]
            if step is None:
                raise RuntimeError("Broken legend-pairing augmenting path")
            prior, edge_index = step
            edge = graph[prior][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior

    return [
        (swatch, label)
        for edge, swatch, label in pair_edges
        if edge.capacity == 0
    ]


def _pair_legend_items(
    detections: DetectionFrame,
) -> tuple[list[tuple[Detection, Detection]], str | None]:
    parents = sorted(
        (item for item in detections.detections if item.role == "legend"),
        key=_sort_key,
    )
    swatches = sorted(
        (item for item in detections.detections if item.role == "legend_swatch"),
        key=_sort_key,
    )
    labels = sorted(
        (item for item in detections.detections if item.role == "legend_label"),
        key=_sort_key,
    )
    if not swatches and not labels:
        return [], "no_legend"
    if not swatches:
        return [], "missing_swatch"
    if not labels:
        return [], "missing_label"

    group_order: list[str | None] = [parent.id for parent in parents]
    if not parents:
        group_order.append(None)
    swatches_by_parent: dict[str | None, list[Detection]] = {
        group_id: [] for group_id in group_order
    }
    labels_by_parent: dict[str | None, list[Detection]] = {
        group_id: [] for group_id in group_order
    }

    def group_for(item: Detection) -> str | None:
        group_id = _parent_for(item, parents)
        if group_id is None and len(parents) == 1:
            group_id = parents[0].id
        return group_id

    for item in swatches:
        swatches_by_parent.setdefault(group_for(item), []).append(item)
    for item in labels:
        labels_by_parent.setdefault(group_for(item), []).append(item)

    # Detector parent boxes can be tighter than their detected children. Keep
    # children with the sole detected legend; retain ambiguous multi-parent
    # orphans in a fallback group rather than silently discarding them.
    if parents and (None in swatches_by_parent or None in labels_by_parent):
        group_order.append(None)

    pairs: list[tuple[Detection, Detection]] = []
    for group_id in group_order:
        group_swatches = swatches_by_parent.get(group_id, [])
        group_labels = labels_by_parent.get(group_id, [])
        pairs.extend(_minimum_cost_pairs(group_swatches, group_labels))
    pairs.sort(key=lambda pair: (*_sort_key(pair[0]), *_sort_key(pair[1])))
    return pairs, None if pairs else "incomplete_legend"


def _crop(image, detection: Detection):
    height, width = image.shape[:2]
    box = detection.box
    left = max(0, min(width, math.floor(box.x1)))
    top = max(0, min(height, math.floor(box.y1)))
    right = max(0, min(width, math.ceil(box.x2)))
    bottom = max(0, min(height, math.ceil(box.y2)))
    inset = max(1, math.ceil(min(right - left, bottom - top) * 0.15))
    left += inset
    top += inset
    right -= inset
    bottom -= inset
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _normalized_crop(crop):
    import numpy as np

    return crop.astype(np.float32) / 255.0


def _candidate_error(bar, template) -> float:
    import cv2
    import numpy as np

    bar_height, bar_width = bar.shape[:2]
    template_height, template_width = template.shape[:2]
    scale = min(1.0, bar_width / template_width, bar_height / template_height)
    if scale < 1.0:
        resized_width = max(1, math.floor(template_width * scale))
        resized_height = max(1, math.floor(template_height * scale))
        template = cv2.resize(
            template, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    response = cv2.matchTemplate(bar, template, cv2.TM_SQDIFF)
    error = float(np.median(response / template.size))
    return min(1.0, max(0.0, error))


def _unavailable_scores(
    source_key: str, bars: list[Detection], status: str
) -> _LegendScores:
    return _LegendScores(source_key, tuple(
        _BarScore(bar.id, unavailable_status=status) for bar in bars
    ))


def _score_legend_to_bars(
    image, detections: DetectionFrame, ocr: OCRFrame | None
) -> _LegendScores:
    """Compute reusable color errors without applying acceptance thresholds."""
    import numpy as np

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an H x W x 3 uint8 BGR array")
    if image.shape[:2] != (detections.height, detections.width):
        raise ValueError("image dimensions must match the detection frame")
    if ocr is not None and ocr.source_key != detections.source_key:
        raise ValueError("OCR and detections refer to different source images")

    bars = [item for item in detections.detections if item.role == "vertical_bar"]
    if not bars:
        return _LegendScores(detections.source_key, ())
    if ocr is None:
        return _unavailable_scores(detections.source_key, bars, "ocr_unavailable")

    pairs, unavailable_status = _pair_legend_items(detections)
    if unavailable_status is not None:
        return _unavailable_scores(detections.source_key, bars, unavailable_status)

    text_by_id = {region.detection_id: region.text.strip() for region in ocr.regions}
    candidates = [
        _LegendCandidate(swatch, label, text_by_id.get(label.id) or None)
        for swatch, label in pairs
    ]
    entries = tuple(
        LegendEntry(candidate.swatch.id, candidate.label.id, candidate.name)
        for candidate in candidates
    )
    candidate_crops = [
        _crop(image, candidate.swatch) for candidate in candidates
    ]
    if any(crop is None for crop in candidate_crops):
        return _unavailable_scores(detections.source_key, bars, "unusable_crop")
    normalized_candidates = [
        (candidate, _normalized_crop(crop))
        for candidate, crop in zip(candidates, candidate_crops, strict=True)
    ]

    scores = []
    for bar in bars:
        bar_crop = _crop(image, bar)
        if bar_crop is None:
            scores.append(_BarScore(bar.id, unavailable_status="unusable_crop"))
            continue
        normalized_bar = _normalized_crop(bar_crop)
        ranked = sorted(
            (
                _candidate_error(normalized_bar, crop),
                _sort_key(candidate.swatch),
                candidate,
            )
            for candidate, crop in normalized_candidates
        )
        best_error, _, best = ranked[0]
        scores.append(_BarScore(
            bar.id,
            candidate=best,
            best_error=best_error,
            runner_up_error=ranked[1][0] if len(ranked) > 1 else None,
        ))
    return _LegendScores(detections.source_key, tuple(scores), entries)


def _classify_legend_scores(
    scores: _LegendScores, *, max_error: float, min_margin: float
) -> LegendMappingFrame:
    """Apply acceptance thresholds to reusable color scores."""
    max_error = validate_unit_interval(max_error, "max_error")
    min_margin = validate_unit_interval(min_margin, "min_margin")
    assignments = []
    for scored in scores.bars:
        if scored.unavailable_status is not None:
            assignments.append(LegendBarAssignment(
                scored.bar_id, status=scored.unavailable_status
            ))
            continue
        candidate = scored.candidate
        if candidate is None or scored.best_error is None:
            assignments.append(LegendBarAssignment(
                scored.bar_id, status="unusable_crop"
            ))
            continue
        if candidate.name is None:
            assignments.append(LegendBarAssignment(
                scored.bar_id, status="unreadable_label"
            ))
            continue
        if scored.best_error > max_error:
            status = "weak_match"
        elif (
            scored.runner_up_error is not None
            and scored.runner_up_error - scored.best_error < min_margin
        ):
            status = "ambiguous"
        else:
            status = "matched"
        assignments.append(LegendBarAssignment(
            scored.bar_id,
            swatch_id=candidate.swatch.id,
            label_id=candidate.label.id,
            series_name=candidate.name if status == "matched" else None,
            match_score=1.0 - scored.best_error,
            status=status,
        ))
    return LegendMappingFrame(
        scores.source_key, tuple(assignments), scores.entries
    )


def match_legend_to_bars(
    image,
    detections: DetectionFrame,
    ocr: OCRFrame | None,
    *,
    max_error: float = 0.03,
    min_margin: float = 0.01,
) -> LegendMappingFrame:
    """Return one deterministic, in-memory legend assignment per vertical bar."""
    return _classify_legend_scores(
        _score_legend_to_bars(image, detections, ocr),
        max_error=max_error,
        min_margin=min_margin,
    )
