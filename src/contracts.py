"""Shared, dependency-free handoffs between chart pipeline stages.

Boxes use pixel coordinates with a top-left origin and exclusive right/bottom
edges. ``source_key`` is a split-qualified relative image name such as
``val/22410.png``; PlotQA reuses numeric IDs across splits.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


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
LEGEND_MAPPING_STATUSES = frozenset({
    "matched", "weak_match", "ambiguous", "no_legend",
    "missing_swatch", "missing_label", "unreadable_label",
    "incomplete_legend", "unusable_crop", "ocr_unavailable",
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


@dataclass(frozen=True)
class LegendBarAssignment:
    """One in-memory legend interpretation for a detected vertical bar."""

    bar_id: str
    swatch_id: str | None = None
    label_id: str | None = None
    series_name: str | None = None
    match_score: float | None = None
    status: str = "no_legend"

    def __post_init__(self) -> None:
        for name, value in (
            ("bar_id", self.bar_id),
            ("swatch_id", self.swatch_id),
            ("label_id", self.label_id),
        ):
            if not isinstance(value, str) or not value:
                if value is not None or name == "bar_id":
                    raise ValueError(f"{name} must be nonempty or absent")
        if self.status not in LEGEND_MAPPING_STATUSES:
            raise ValueError(f"Unknown legend mapping status: {self.status}")
        if self.series_name is not None and (
            not isinstance(self.series_name, str) or not self.series_name.strip()
        ):
            raise ValueError("series_name must be nonblank or absent")
        if self.match_score is not None:
            _confidence(self.match_score)

        candidate = self.status in {"matched", "weak_match", "ambiguous"}
        if candidate and (
            self.swatch_id is None or self.label_id is None
            or self.match_score is None
        ):
            raise ValueError(f"{self.status} requires candidate IDs and a score")
        if self.status == "matched" and self.series_name is None:
            raise ValueError("matched assignments require a series name")
        if self.status in {"weak_match", "ambiguous"} and self.series_name is not None:
            raise ValueError(f"{self.status} assignments must not assign a series")
        if not candidate and any(value is not None for value in (
            self.swatch_id, self.label_id, self.series_name, self.match_score
        )):
            raise ValueError(f"{self.status} assignments cannot contain a candidate")


@dataclass(frozen=True)
class LegendMappingFrame:
    """Validated, ephemeral legend-to-bar assignments for one source image."""

    source_key: str
    assignments: tuple[LegendBarAssignment, ...]

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        ids: set[str] = set()
        for assignment in self.assignments:
            if not isinstance(assignment, LegendBarAssignment):
                raise ValueError("assignments must contain LegendBarAssignment objects")
            if assignment.bar_id in ids:
                raise ValueError(f"Duplicate legend mapping bar ID: {assignment.bar_id}")
            ids.add(assignment.bar_id)


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


def join_legend_assignments(
    detections: DetectionFrame, mapping: LegendMappingFrame
) -> tuple[tuple[Detection, LegendBarAssignment], ...]:
    """Validate and join one legend assignment to every vertical bar."""
    if detections.source_key != mapping.source_key:
        raise ValueError("Legend mapping and detections refer to different source images")
    by_id = {detection.id: detection for detection in detections.detections}
    bars = [detection for detection in detections.detections
            if detection.role == "vertical_bar"]
    expected = [bar.id for bar in bars]
    actual = [assignment.bar_id for assignment in mapping.assignments]
    if actual != expected:
        raise ValueError(
            "Legend assignments must cover vertical bars in detection order"
        )
    joined = []
    for bar, assignment in zip(bars, mapping.assignments, strict=True):
        for referenced_id, role in (
            (assignment.swatch_id, "legend_swatch"),
            (assignment.label_id, "legend_label"),
        ):
            if referenced_id is None:
                continue
            referenced = by_id.get(referenced_id)
            if referenced is None:
                raise ValueError(f"Legend mapping references unknown detection: {referenced_id}")
            if referenced.role != role:
                raise ValueError(
                    f"Legend mapping references {referenced.role}, expected {role}"
                )
        joined.append((bar, assignment))
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


_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class YAxisSpec:
    """Calibrated value axis. ``min``/``max`` span the surviving tick values."""

    min: float
    max: float
    tick_values: list[float]
    unit: str | None = None


@dataclass(frozen=True)
class SeriesSpec:
    """One bar series; ``values`` align with ``VerticalBarSpec.categories``.

    ``value_rescued[i]`` is True when ``values[i]`` was measured from pixels
    because no bar was detected (a rescued 0.0 means no bar pixels were found).
    ``value_confidence[i]`` is the detector confidence, or a low fixed value
    for rescued values. Both are empty when not provided.
    """

    name: str | None
    source_color: str | None
    values: list[float | None]
    value_rescued: list[bool] = field(default_factory=list)
    value_confidence: list[float | None] = field(default_factory=list)


@dataclass(frozen=True)
class VerticalBarSpec:
    """Reconstructed vertical bar chart (schema 0.2).

    Unlike ``VerticalBarChart`` it tolerates missing titles and values and
    records what the interpreter could not resolve in ``flags``.
    """

    source_key: str
    title: str | None
    x_axis_title: str | None
    y_axis_title: str | None
    y_axis: YAxisSpec
    categories: list[str]
    series: list[SeriesSpec]
    bars_adjacent: bool
    flags: list[str]
    confidence: dict[str, float]
    chart_type: str = "bar"
    orientation: str = "vertical"
    schema_version: str = "0.2"

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        if self.schema_version != "0.2":
            raise ValueError("VerticalBarSpec schema_version must be 0.2")
        if self.chart_type != "bar" or self.orientation != "vertical":
            raise ValueError("VerticalBarSpec is a vertical bar chart")
        for label in (self.title, self.x_axis_title, self.y_axis_title):
            if label is not None and not isinstance(label, str):
                raise ValueError("titles must be strings or None")
        if not all(math.isfinite(v) for v in (self.y_axis.min, self.y_axis.max,
                                              *self.y_axis.tick_values)):
            raise ValueError("y-axis values must be finite")
        if not self.series:
            raise ValueError("spec must contain at least one series")
        for item in self.series:
            if item.source_color is not None and not _HEX_COLOR.fullmatch(item.source_color):
                raise ValueError("source_color must be #rrggbb or None")
            if len(item.values) != len(self.categories):
                raise ValueError("series values must match category count")
            if any(v is not None and not math.isfinite(v) for v in item.values):
                raise ValueError("series values must be finite or None")
            for extra in (item.value_rescued, item.value_confidence):
                if extra and len(extra) != len(item.values):
                    raise ValueError("per-value lists must match series values")
        for value in self.confidence.values():
            _confidence(value)
