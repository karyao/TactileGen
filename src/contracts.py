"""Shared, dependency-free handoffs between chart pipeline stages.

Boxes use pixel coordinates with a top-left origin and exclusive right/bottom
edges. ``source_key`` is a split-qualified relative image name such as
``val/22410.png``; PlotQA reuses numeric IDs across splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Keep this order aligned with existing YOLO class IDs and classes.txt files.
CLASS_NAMES = (
    "horizontal_bar",
    "vertical_bar",
    "dot",
    "line_segment",
    "chart_title",
    "x_axis_title",
    "y_axis_title",
    "x_tick_label",
    "y_tick_label",
    "x_axis_rule",
    "y_axis_rule",
    "plot_area",
    "legend",
    "legend_swatch",
    "legend_label",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
CHART_MARK_ROLE = {
    "hbar_categorical": "horizontal_bar",
    "vbar_categorical": "vertical_bar",
    "dot_line": "dot",
    "line": "line_segment",
}
TEXT_ROLES = frozenset({
    "chart_title", "x_axis_title", "y_axis_title",
    "x_tick_label", "y_tick_label", "legend_label",
})


def validate_source_key(value: str) -> None:
    """Validate a relative, namespaced image identifier."""
    parts = value.split("/") if isinstance(value, str) else []
    if len(parts) < 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("source_key must be a relative, namespaced image path")


def _confidence(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a number from 0 to 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a number from 0 to 1")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    id: str
    role: str
    box: Box
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("detection ID must be nonempty")
        if self.role not in CLASS_IDS:
            raise ValueError(f"Unknown detection role: {self.role}")
        if not isinstance(self.box, Box):
            raise ValueError("detection box must be a Box")
        _confidence(self.confidence)


@dataclass(frozen=True)
class DetectionFrame:
    source_key: str
    width: int
    height: int
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        if (isinstance(self.width, bool) or isinstance(self.height, bool)
                or not isinstance(self.width, int) or not isinstance(self.height, int)
                or self.width <= 0 or self.height <= 0):
            raise ValueError("image dimensions must be positive integers")
        ids: set[str] = set()
        for detection in self.detections:
            if not isinstance(detection, Detection):
                raise ValueError("detections must contain Detection objects")
            if detection.id in ids:
                raise ValueError(f"Duplicate detection ID: {detection.id}")
            ids.add(detection.id)
            x1, y1, x2, y2 = (
                detection.box.x1, detection.box.y1,
                detection.box.x2, detection.box.y2,
            )
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError(f"Non-finite box for {detection.id}")
            if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
                raise ValueError(f"Box outside image or without area: {detection.id}")


@dataclass(frozen=True)
class RecognizedText:
    detection_id: str
    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, str) or not self.detection_id:
            raise ValueError("OCR detection_id must be nonempty")
        if not isinstance(self.text, str):
            raise ValueError("OCR text must be a string")
        _confidence(self.confidence)


@dataclass(frozen=True)
class OCRFrame:
    source_key: str
    regions: tuple[RecognizedText, ...]

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        ids: set[str] = set()
        for region in self.regions:
            if not isinstance(region, RecognizedText):
                raise ValueError("regions must contain RecognizedText objects")
            if region.detection_id in ids:
                raise ValueError(f"Duplicate OCR detection ID: {region.detection_id}")
            ids.add(region.detection_id)


def join_text_regions(
    detections: DetectionFrame, ocr: OCRFrame
) -> tuple[tuple[Detection, RecognizedText], ...]:
    """Resolve OCR records to text detections in one image, in OCR order."""
    if detections.source_key != ocr.source_key:
        raise ValueError("OCR and detections refer to different source images")
    by_id = {detection.id: detection for detection in detections.detections}
    joined = []
    for region in ocr.regions:
        detection = by_id.get(region.detection_id)
        if detection is None:
            raise ValueError(f"OCR references unknown detection: {region.detection_id}")
        if detection.role not in TEXT_ROLES:
            raise ValueError(f"OCR references non-text role: {detection.role}")
        joined.append((detection, region))
    return tuple(joined)


@dataclass(frozen=True)
class BarSeries:
    name: str | None
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("series name must be nonempty or absent")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in self.values):
            raise ValueError("series values must be finite numbers")


@dataclass(frozen=True)
class VerticalBarChart:
    source_key: str
    title: str
    x_axis_title: str
    y_axis_title: str
    categories: tuple[str, ...]
    series: tuple[BarSeries, ...]

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        for label in (self.title, self.x_axis_title, self.y_axis_title):
            if not isinstance(label, str) or not label:
                raise ValueError("chart and axis titles must be nonempty")
        if not self.categories or any(not isinstance(item, str) or not item
                                      for item in self.categories):
            raise ValueError("categories must be nonempty strings")
        if not self.series or any(not isinstance(item, BarSeries) for item in self.series):
            raise ValueError("chart must contain at least one BarSeries")
        if any(len(item.values) != len(self.categories) for item in self.series):
            raise ValueError("series values must match category count")


class ChartInterpretationError(ValueError):
    """A complete chart cannot be reconstructed from detections and OCR."""
