"""Horizontal bar charts: tell which way the bars run, then rebuild the chart.

A horizontal bar chart is a vertical bar chart on its side. Instead of a
second copy of the reconstruction logic, the detections and the image are
transposed (x <-> y) so the bars stand upright, the axis roles are swapped,
and ``reconstruct_vbar`` does the work. The result is mapped back:

* categories run top to bottom, as printed;
* ``y_axis`` is the value axis, which is printed along the bottom;
* ``x_axis_title`` / ``y_axis_title`` keep their printed meaning (bottom /
  left), so the value heading is ``x_axis_title``.

Legend matching compares bar and swatch colours, which do not depend on the
bar direction, so it runs on the untransposed image.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from src.contracts import (
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
    VerticalBarSpec,
)
from src.legend_matcher import match_legend_to_bars
from src.reconstruct import (
    MIN_CONFIDENCE,
    TITLE_EDGE_PX,
    _best,
    _group_roles,
    _legend_ocr_frame,
    _text_table,
    reconstruct_vbar,
)

# Roles that change meaning when the chart is turned on its side.
ROLE_SWAP = {
    "horizontal_bar": "vertical_bar", "vertical_bar": "horizontal_bar",
    "x_axis_title": "y_axis_title", "y_axis_title": "x_axis_title",
    "x_tick_label": "y_tick_label", "y_tick_label": "x_tick_label",
    "x_axis_rule": "y_axis_rule", "y_axis_rule": "x_axis_rule",
}
# ``reconstruct_vbar`` errors, reworded for a horizontal chart.
_ERRORS = {
    "no vertical bars detected inside the plot area":
        "no horizontal bars detected inside the plot area",
    "no x tick labels detected": "no category (y tick) labels detected",
    "fewer than 2 readable y tick labels": "fewer than 2 readable x tick labels",
    "y-axis calibration needs at least 2 consistent ticks":
        "x-axis calibration needs at least 2 consistent ticks",
}
_TITLE_FLAGS = {"x_axis_title_missing": "y_axis_title_missing",
                "y_axis_title_missing": "x_axis_title_missing"}
_TITLE_KEYS = {"x_axis_title": "y_axis_title", "y_axis_title": "x_axis_title"}


def bar_orientation(frame: DetectionFrame, min_confidence: float = MIN_CONFIDENCE) -> str:
    """``"horizontal"`` when most confident bar detections are horizontal.

    Ties, including a frame with no bars, give ``"vertical"`` so charts the
    vertical pipeline already handled keep their behaviour and messages.
    """
    counts = Counter(d.role for d in frame.detections if d.confidence >= min_confidence)
    return "horizontal" if counts["horizontal_bar"] > counts["vertical_bar"] else "vertical"


def transpose_frame(frame: DetectionFrame) -> DetectionFrame:
    """Swap x and y of every box and the axis roles; bars become vertical."""
    return DetectionFrame(frame.source_key, frame.height, frame.width, tuple(
        Detection(d.id, ROLE_SWAP.get(d.role, d.role),
                  Box(d.box.y1, d.box.x1, d.box.y2, d.box.x2), d.confidence)
        for d in frame.detections))


def transpose_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.swapaxes(image, 0, 1))


def bars_as_vertical(frame: DetectionFrame) -> DetectionFrame:
    """Same boxes, with horizontal bars labelled as the bars the legend matcher reads."""
    swap = {"horizontal_bar": "vertical_bar", "vertical_bar": "horizontal_bar"}
    return replace(frame, detections=tuple(
        replace(d, role=swap.get(d.role, d.role)) for d in frame.detections))


def match_legend_to_hbars(image: np.ndarray, frame: DetectionFrame,
                          ocr: OCRFrame | None) -> LegendMappingFrame:
    """Legend assignments for the horizontal bars, one per bar in detection order."""
    return match_legend_to_bars(image, bars_as_vertical(frame), ocr)


def reconstruct_hbar(frame: DetectionFrame, ocr_texts, image: np.ndarray,
                     min_confidence: float = MIN_CONFIDENCE,
                     rescue_bars: bool = True,
                     category_guard: bool = True, *,
                     legend_mapping: LegendMappingFrame | None = None
                     ) -> VerticalBarSpec:
    """Turn detections plus OCR of a horizontal bar chart into a spec.

    Same arguments and errors as ``reconstruct_vbar``; the returned spec has
    ``orientation="horizontal"`` (see the module docstring for its axes).
    """
    if legend_mapping is None:
        legend_ocr = (ocr_texts if isinstance(ocr_texts, OCRFrame)
                      else _legend_ocr_frame(frame, _text_table(ocr_texts)))
        legend_mapping = match_legend_to_hbars(image, frame, legend_ocr)
    try:
        spec = reconstruct_vbar(transpose_frame(frame), ocr_texts, transpose_image(image),
                                min_confidence, rescue_bars, category_guard,
                                legend_mapping=legend_mapping)
    except ChartInterpretationError as error:
        raise ChartInterpretationError(_ERRORS.get(str(error), str(error))) from error

    # The title-edge check used the transposed width; redo it on the image.
    flags = [_TITLE_FLAGS.get(f, f) for f in spec.flags if f != "title_touches_right_edge"]
    title = _best(_group_roles(frame, min_confidence).get("chart_title", []))
    if title is not None and title.box.x2 >= frame.width - TITLE_EDGE_PX:
        flags.append("title_touches_right_edge")
    confidence = {_TITLE_KEYS.get(k, k): v for k, v in spec.confidence.items()}
    return replace(spec, x_axis_title=spec.y_axis_title, y_axis_title=spec.x_axis_title,
                   flags=flags, confidence=confidence, orientation="horizontal")
