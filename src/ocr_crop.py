"""Shared crop selection for YOLO-guided OCR backends."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.contracts import Detection


_MIN_PADDING = 2
_MAX_PADDING = 8
# PlotQA's chart_title boxes stop short of the drawn text, and YOLO inherits
# that bias, so titles are read from a full-width strip. Set to False to crop
# titles like every other role.
_TITLE_FULL_WIDTH_STRIP = True
_TITLE_ROLE = "chart_title"
_STRIP_NEIGHBOR_MARGIN = 2
# A legend label sits ~35 px right of its swatch and frame, neither of which is
# a detected text region, so keep clear of them.
_STRIP_LEGEND_MARGIN = 40


def _padding_for_detection(detection: Detection) -> int:
    box = detection.box
    return max(
        _MIN_PADDING,
        min(_MAX_PADDING, math.ceil(0.1 * min(box.x2 - box.x1, box.y2 - box.y1))),
    )


def _crop_with_padding(image: np.ndarray, detection: Detection) -> np.ndarray:
    height, width = image.shape[:2]
    box = detection.box
    padding = _padding_for_detection(detection)
    left = max(0, math.floor(box.x1) - padding)
    top = max(0, math.floor(box.y1) - padding)
    right = min(width, math.ceil(box.x2) + padding)
    bottom = min(height, math.ceil(box.y2) + padding)
    return image[top:bottom, left:right].copy()


def _title_strip_bounds(
    detection: Detection, neighbors: Sequence[Detection], width: int
) -> tuple[int, int]:
    """Horizontal extent of the title strip, stopping at other text in its band."""
    box = detection.box
    padding = max(
        _MIN_PADDING, min(_MAX_PADDING, math.ceil(0.1 * (box.y2 - box.y1)))
    )
    top, bottom = box.y1 - padding, box.y2 + padding
    left, right = 0, width
    for other in neighbors:
        other_box = other.box
        if other.id == detection.id or other.role == _TITLE_ROLE:
            continue
        if other_box.y2 <= top or other_box.y1 >= bottom:
            continue
        margin = (
            _STRIP_LEGEND_MARGIN
            if other.role == "legend_label"
            else _STRIP_NEIGHBOR_MARGIN
        )
        # Text overlapping the title box itself is ambiguous; leave it alone.
        if other_box.x2 <= box.x1:
            left = max(left, math.ceil(other_box.x2) + margin)
        elif other_box.x1 >= box.x2:
            right = min(right, math.floor(other_box.x1) - margin)
    # Never let a neighbour cut into the detected title box itself.
    return min(left, math.floor(box.x1)), max(right, math.ceil(box.x2))


def _title_strip(
    image: np.ndarray, detection: Detection, neighbors: Sequence[Detection]
) -> np.ndarray:
    height, width = image.shape[:2]
    box = detection.box
    padding = max(
        _MIN_PADDING, min(_MAX_PADDING, math.ceil(0.1 * (box.y2 - box.y1)))
    )
    top = max(0, math.floor(box.y1) - padding)
    bottom = min(height, math.ceil(box.y2) + padding)
    left, right = _title_strip_bounds(detection, neighbors, width)
    return image[top:bottom, left:right].copy()


def crop_text_region(
    image: np.ndarray,
    detection: Detection,
    neighbors: Sequence[Detection] = (),
) -> np.ndarray:
    """Return the engine-neutral crop for one detected text region.

    Chart titles use the neighbor-aware horizontal strip that compensates for
    PlotQA's clipped title boxes. Every other role uses its padded detection box.
    The returned array is always an independent copy of the source pixels.
    """
    if _TITLE_FULL_WIDTH_STRIP and detection.role == _TITLE_ROLE:
        return _title_strip(image, detection, neighbors)
    return _crop_with_padding(image, detection)
