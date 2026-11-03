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
# Roles beyond the 15 PlotQA classes. Kept out of CLASS_NAMES so the PlotQA YOLO class
# order, classes.txt files and model checks stay valid; a Detection may use any of ROLES.
EXTRA_ROLES = ("x_tick", "y_tick", "value_label")
ROLES = CLASS_NAMES + EXTRA_ROLES
CHART_MARK_ROLE = {
    "hbar_categorical": "horizontal_bar",
    "vbar_categorical": "vertical_bar",
    "dot_line": "dot",
    "line": "line_segment",
}
TEXT_ROLES = frozenset({
    "chart_title", "x_axis_title", "y_axis_title",
    "x_tick_label", "y_tick_label", "legend_label", "value_label",
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
        if self.role not in ROLES:
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


AXIS_SCALES = ("linear", "log")


def _finite_number(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


@dataclass(frozen=True)
class VerticalBarChart:
    """A vertical bar chart ready for rendering.

    ``title`` and the axis titles are ``None`` when the source chart has none;
    they are never filled in. The ``y_*`` fields describe the printed value
    axis and are optional: ``y_min``/``y_max`` are the axis ends (not the data
    range), ``y_tick_values`` the numbers printed at the ticks in ascending
    order, and ``y_scale`` is ``"linear"`` or ``"log"``.
    """

    source_key: str
    title: str | None
    x_axis_title: str | None
    y_axis_title: str | None
    categories: tuple[str, ...]
    series: tuple[BarSeries, ...]
    y_min: float | None = None
    y_max: float | None = None
    y_tick_values: tuple[float, ...] = ()
    y_scale: str = "linear"

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        for label in (self.title, self.x_axis_title, self.y_axis_title):
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError("chart and axis titles must be nonempty or absent")
        if not self.categories or any(not isinstance(item, str) or not item
                                      for item in self.categories):
            raise ValueError("categories must be nonempty strings")
        if not self.series or any(not isinstance(item, BarSeries) for item in self.series):
            raise ValueError("chart must contain at least one BarSeries")
        if any(len(item.values) != len(self.categories) for item in self.series):
            raise ValueError("series values must match category count")
        self._validate_axis()

    def _validate_axis(self) -> None:
        if self.y_scale not in AXIS_SCALES:
            raise ValueError(f"y_scale must be one of {', '.join(AXIS_SCALES)}")
        for name, value in (("y_min", self.y_min), ("y_max", self.y_max)):
            if value is not None and not _finite_number(value):
                raise ValueError(f"{name} must be a finite number or absent")
        if not isinstance(self.y_tick_values, tuple) or not all(
                _finite_number(value) for value in self.y_tick_values):
            raise ValueError("y_tick_values must be a tuple of finite numbers")
        if self.y_min is not None and self.y_max is not None and self.y_min >= self.y_max:
            raise ValueError("y_min must be less than y_max")
        ticks = self.y_tick_values
        if any(a >= b for a, b in zip(ticks, ticks[1:])):
            raise ValueError("y_tick_values must be strictly increasing")
        if ticks:
            if self.y_min is not None and ticks[0] < self.y_min:
                raise ValueError("y_tick_values must lie within y_min and y_max")
            if self.y_max is not None and ticks[-1] > self.y_max:
                raise ValueError("y_tick_values must lie within y_min and y_max")
        if self.y_scale == "log":
            positive = [v for v in (self.y_min, self.y_max, *ticks,
                                    *(v for s in self.series for v in s.values))
                        if v is not None]
            if any(v <= 0 for v in positive):
                raise ValueError("log axis requires positive values, limits and ticks")


REVIEW_KINDS = frozenset({
    "category", "y_tick_label", "chart_title", "x_axis_title", "y_axis_title",
    "value_label", "bar", "axis", "legend",
})
RESULT_STATUSES = frozenset({"accepted", "needs_review", "rejected"})


@dataclass(frozen=True)
class ReviewItem:
    """One thing a person may need to check or correct.

    ``text`` is the raw OCR read for a text element (``None`` when there was no
    read); ``value`` and ``uncertainty`` describe a computed number such as a bar
    value. ``flagged`` items carry the reason they need attention.
    """

    kind: str
    index: int | None = None          # position within its kind, e.g. category 0, 1, 2 left to right
    detection_id: str | None = None
    text: str | None = None
    confidence: float | None = None
    box: Box | None = None
    value: float | None = None
    uncertainty: float | None = None
    flagged: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REVIEW_KINDS:
            raise ValueError(f"Unknown review item kind: {self.kind}")
        if self.confidence is not None:
            _confidence(self.confidence)
        if self.flagged and not self.reason:
            raise ValueError("a flagged review item needs a reason")
        for number in (self.value, self.uncertainty):
            if number is not None and not _finite_number(number):
                raise ValueError("review item numbers must be finite")


@dataclass(frozen=True)
class InterpretationResult:
    """What the interpreter hands on: a chart, or why a person has to look first.

    ``accepted``: ``chart`` is set and nothing is flagged. ``needs_review``: something is
    flagged; ``chart`` is a provisional chart when one could be built and ``None`` when
    not (for example the bar and label counts differ). ``rejected``: out of scope, no
    chart. ``items`` lists the elements a person may check, flagged or not; ``reasons``
    are chart-level reasons.
    """

    source_key: str
    status: str
    chart: VerticalBarChart | None
    items: tuple[ReviewItem, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_source_key(self.source_key)
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"status must be one of {', '.join(sorted(RESULT_STATUSES))}")
        if self.chart is not None and not isinstance(self.chart, VerticalBarChart):
            raise ValueError("chart must be a VerticalBarChart or None")
        if any(not isinstance(item, ReviewItem) for item in self.items):
            raise ValueError("items must contain ReviewItem objects")
        flagged = any(item.flagged for item in self.items) or bool(self.reasons)
        if self.status == "accepted" and (self.chart is None or flagged):
            raise ValueError("an accepted result needs a chart and nothing flagged")
        if self.status == "rejected" and (self.chart is not None or not self.reasons):
            raise ValueError("a rejected result has no chart and at least one reason")
        if self.status == "needs_review" and not flagged:
            raise ValueError("a needs_review result needs a flagged item or a reason")


class ChartInterpretationError(ValueError):
    """A complete chart cannot be reconstructed from detections and OCR."""
