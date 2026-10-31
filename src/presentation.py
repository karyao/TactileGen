"""Framework-independent presentation helpers for detection results."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.borealis import BOREALIS_COLORS, hex_to_bgr
from src.contracts import CLASS_NAMES, DetectionFrame


# OpenCV uses BGR channel order.
_SOFT_GLOW = hex_to_bgr(BOREALIS_COLORS["glow"])
_AURORA_BLUE = hex_to_bgr(BOREALIS_COLORS["aurora"])
_DEEP_SKY = hex_to_bgr(BOREALIS_COLORS["deep_sky"])
_SNOW_WHITE = hex_to_bgr(BOREALIS_COLORS["text"])
_PANEL_NAVY = hex_to_bgr(BOREALIS_COLORS["panel"])
_HALO = _SNOW_WHITE

_MARK_ROLES = frozenset({"horizontal_bar", "vertical_bar", "dot", "line_segment"})
_TEXT_ROLES = frozenset({
    "chart_title", "x_axis_title", "y_axis_title",
    "x_tick_label", "y_tick_label",
})
_STRUCTURE_ROLES = frozenset({"x_axis_rule", "y_axis_rule", "plot_area"})
_LEGEND_ROLES = frozenset({"legend", "legend_swatch", "legend_label"})


def _display_role(role: str) -> str:
    return role.replace("_", " ")


def summarize_detections(frame: DetectionFrame) -> str:
    """Return a plain-language summary of one detection frame."""
    total = len(frame.detections)
    counts = Counter(detection.role for detection in frame.detections)
    parts = [f"Detected {total} chart element{'s' if total != 1 else ''}."]
    if counts:
        grouped = "; ".join(
            f"{_display_role(role)}: {counts[role]}"
            for role in CLASS_NAMES
            if role in counts
        )
        confidences = [detection.confidence for detection in frame.detections]
        parts.append(f"By role: {grouped}.")
        parts.append(
            f"Confidence range: {min(confidences):.1%} to {max(confidences):.1%}."
        )
    else:
        parts.append("No roles were detected.")
    parts.append(f"Vertical bars: {counts['vertical_bar']}.")
    parts.append("Chart text and numeric values have not been read.")
    return " ".join(parts)


def detection_rows(frame: DetectionFrame) -> list[dict[str, Any]]:
    """Return table-ready rows in the detection frame's original order."""
    return [
        {
            "Detection ID": detection.id,
            "Role": _display_role(detection.role),
            "Confidence": detection.confidence,
            "X1": detection.box.x1,
            "Y1": detection.box.y1,
            "X2": detection.box.x2,
            "Y2": detection.box.y2,
        }
        for detection in frame.detections
    ]


def _style(role: str) -> tuple[tuple[int, int, int], bool]:
    if role in _MARK_ROLES:
        return _SOFT_GLOW, False
    if role in _TEXT_ROLES:
        return _AURORA_BLUE, False
    if role in _STRUCTURE_ROLES:
        return _DEEP_SKY, True
    if role in _LEGEND_ROLES:
        return _SNOW_WHITE, False
    raise ValueError(f"Unknown detection role: {role}")


def _rectangle_points(box, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    # Detection boxes use exclusive right and bottom edges, while OpenCV's
    # rectangle endpoints are inclusive.
    left = max(0, min(width - 1, int(box.x1)))
    top = max(0, min(height - 1, int(box.y1)))
    right = max(left, min(width - 1, int(box.x2 + 0.999999) - 1))
    bottom = max(top, min(height - 1, int(box.y2 + 0.999999) - 1))
    return (left, top), (right, bottom)


def _draw_dashed_rectangle(image, top_left, bottom_right, color, thickness: int) -> None:
    import cv2

    left, top = top_left
    right, bottom = bottom_right
    # Keep the same cadence for the halo and foreground so each colored dash
    # stays centered inside its white outline. The gap is wide enough to remain
    # visible under the thicker halo stroke.
    dash = 8
    gap = 8
    for start in range(left, right + 1, dash + gap):
        cv2.line(image, (start, top), (min(start + dash, right), top), color, thickness)
        cv2.line(image, (start, bottom), (min(start + dash, right), bottom), color, thickness)
    for start in range(top, bottom + 1, dash + gap):
        cv2.line(image, (left, start), (left, min(start + dash, bottom)), color, thickness)
        cv2.line(image, (right, start), (right, min(start + dash, bottom)), color, thickness)


def _draw_box(image, top_left, bottom_right, color, dashed: bool) -> None:
    import cv2

    if dashed:
        _draw_dashed_rectangle(image, top_left, bottom_right, _HALO, 5)
        _draw_dashed_rectangle(image, top_left, bottom_right, color, 2)
    else:
        cv2.rectangle(image, top_left, bottom_right, _HALO, 5)
        cv2.rectangle(image, top_left, bottom_right, color, 2)


def _draw_label(image, top_left, bottom_right, label: str) -> None:
    import cv2

    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.55, max(0.35, width / 1200))
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    padding_x = 4
    padding_y = 3
    chip_width = min(width, text_width + padding_x * 2)
    chip_height = min(height, text_height + baseline + padding_y * 2)
    left = min(top_left[0], max(0, width - chip_width))
    if top_left[1] >= chip_height:
        top = top_left[1] - chip_height
    else:
        top = min(bottom_right[1] + 1, max(0, height - chip_height))
    right = min(width - 1, left + chip_width - 1)
    bottom = min(height - 1, top + chip_height - 1)
    cv2.rectangle(image, (left, top), (right, bottom), _PANEL_NAVY, -1)
    text_x = left + padding_x
    text_y = min(bottom - padding_y - baseline, top + padding_y + text_height)
    cv2.putText(image, label, (text_x, text_y), font, scale, _SNOW_WHITE,
                thickness, cv2.LINE_AA)


def render_detection_overlay(image, frame: DetectionFrame):
    """Draw every detection on a copy of an OpenCV BGR image."""
    import numpy as np

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an H x W x 3 uint8 BGR array")
    if image.shape[:2] != (frame.height, frame.width):
        raise ValueError("image dimensions must match the detection frame")

    overlay = image.copy()
    for detection in frame.detections:
        color, dashed = _style(detection.role)
        top_left, bottom_right = _rectangle_points(
            detection.box, frame.width, frame.height
        )
        _draw_box(overlay, top_left, bottom_right, color, dashed)
        _draw_label(
            overlay, top_left, bottom_right,
            f"{_display_role(detection.role)} {detection.confidence:.1%}",
        )
    return overlay
