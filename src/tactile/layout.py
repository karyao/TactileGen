"""Lay out a vertical bar chart as a tactile graphic page.

All geometry is in millimetres with a top-left origin. The layout is a flat
list of elements (lines, bars, braille labels, key samples) that the SVG
writer draws and the rule checker inspects; neither needs the source chart.

Placement follows BANA *Guidelines and Standards for Tactile Graphics, 2022*
(section numbers in comments). Every measurement lives in
``src/tactile/constants.py`` with its source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from src.contracts import VerticalBarChart
from src.tactile.braille import Cell, to_braille
from src.tactile.constants import (  # noqa: F401  (re-exported for callers and tests)
    AXIS_STRENGTH,
    AXIS_WIDTH,
    BAR_MAX_WIDTH,
    BAR_MIN_WIDTH,
    BAR_GAP,
    BAR_OUTLINE_WIDTH,
    BAR_STRENGTH,
    CELL_PITCH,
    CELLS_PER_LINE,
    DOT_DIAMETER,
    DOT_PITCH,
    GRID_MIN_SPACING,
    GRID_STRENGTH,
    GRID_WIDTH,
    GROUP_GAP,
    KEY_RUNOVER_CELL,
    KEY_SAMPLE_HEIGHT,
    KEY_SAMPLE_LINES,
    KEY_SAMPLE_WIDTH,
    KEY_TEXT_CELL,
    LABEL_GAP,
    LABEL_HEIGHT,
    LABEL_MAX_GAP,
    LABEL_MIN_GAP,
    LINE_PITCH,
    LINES_PER_PAGE,
    MAX_AREA_TEXTURES,
    MAX_GRID_INTERVALS,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    TEXTURES,
    Y_HEADING_MAX_CELLS,
)


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    def gap(self, other: "Rect") -> float:
        """Shortest distance between two rectangles; 0 when they touch or overlap."""
        dx = max(0.0, other.x1 - self.x2, self.x1 - other.x2)
        dy = max(0.0, other.y1 - self.y2, self.y1 - other.y2)
        return math.hypot(dx, dy)

    def inside(self, other: "Rect", tolerance: float = 1e-6) -> bool:
        return (self.x1 >= other.x1 - tolerance and self.y1 >= other.y1 - tolerance
                and self.x2 <= other.x2 + tolerance and self.y2 <= other.y2 + tolerance)


@dataclass(frozen=True)
class Line:
    id: str
    kind: str          # "axis", "grid", "tick" or "lead"
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    strength: int

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def bounds(self) -> Rect:
        half = self.width / 2
        return Rect(min(self.x1, self.x2) - half, min(self.y1, self.y2) - half,
                    max(self.x1, self.x2) + half, max(self.y1, self.y2) + half)


@dataclass(frozen=True)
class Bar:
    id: str
    x: float
    y: float
    width: float
    height: float
    texture: str
    series_index: int
    category_index: int
    value: float

    @property
    def bounds(self) -> Rect:
        half = BAR_OUTLINE_WIDTH / 2
        return Rect(self.x - half, self.y - half,
                    self.x + self.width + half, self.y + self.height + half)


@dataclass(frozen=True)
class KeySample:
    id: str
    x: float
    y: float
    width: float
    height: float
    texture: str
    series_index: int

    @property
    def bounds(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class BrailleLabel:
    """A braille label; ``x``/``y`` are the top-left edge of the first dot."""

    id: str
    role: str          # "title", "key", "y_heading", "y_value", "category", "x_heading"
    text: str
    cells: tuple[Cell, ...]
    x: float
    y: float
    owner_id: str | None = None   # component the label belongs to (§5.10.5)
    numeric: bool = False
    category_index: int | None = None

    @property
    def width(self) -> float:
        return label_width(len(self.cells))

    @property
    def bounds(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.width, self.y + LABEL_HEIGHT)

    def dot_centres(self):
        """Centres of every raised dot, in page millimetres."""
        radius = DOT_DIAMETER / 2
        for index, cell in enumerate(self.cells):
            left = self.x + index * CELL_PITCH + radius
            for dot in cell:
                column, row = (0, dot - 1) if dot <= 3 else (1, dot - 4)
                yield left + column * DOT_PITCH, self.y + radius + row * DOT_PITCH


@dataclass(frozen=True)
class TactileLayout:
    source_key: str
    page_width: float
    page_height: float
    usable: Rect
    plot: Rect
    lines: tuple[Line, ...]
    bars: tuple[Bar, ...]
    labels: tuple[BrailleLabel, ...]
    key_samples: tuple[KeySample, ...]
    category_count: int
    series_textures: tuple[str, ...]
    categories_keyed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def elements(self):
        yield from self.lines
        yield from self.bars
        yield from self.labels
        yield from self.key_samples

    def by_id(self, element_id: str):
        return next((e for e in self.elements() if e.id == element_id), None)


def label_width(cell_count: int) -> float:
    """Width of the raised dots of ``cell_count`` cells (not the trailing gap)."""
    return (cell_count - 1) * CELL_PITCH + DOT_PITCH + DOT_DIAMETER


def wrap_cells(text: str, max_cells: int) -> list[str]:
    """Split text at spaces so each line's braille fits ``max_cells``."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and len(to_braille(candidate)) > max_cells:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for line in lines:
        if len(to_braille(line)) > max_cells:
            raise ValueError(f"word too long for {max_cells} braille cells: {line!r}")
    return lines


def set_width(bar_width: float, n_series: int) -> float:
    """Raised width of one set of bars, outer edge to outer edge."""
    return n_series * (bar_width + BAR_OUTLINE_WIDTH) + (n_series - 1) * BAR_GAP


def bar_width_for(available: float, n_categories: int, n_series: int) -> float:
    """Bar width (inside the outline) that fills ``available`` millimetres.

    ``available`` holds ``n_categories`` sets of ``n_series`` bars with
    BAR_GAP between bars in a set and GROUP_GAP before, between and after the
    sets. The result is capped at the BANA maximum; it can fall below the
    minimum, which the rule checker then reports.
    """
    per_set = (available - (n_categories + 1) * GROUP_GAP) / n_categories
    width = (per_set - (n_series - 1) * BAR_GAP) / n_series - BAR_OUTLINE_WIDTH
    return min(BAR_MAX_WIDTH, width)


def _nice_step(maximum: float, max_intervals: int) -> tuple[float, int]:
    """Smallest 1/2/5 x 10^k step covering ``maximum`` in at most ``max_intervals``."""
    if max_intervals < 1:
        raise ValueError("not enough height for a value axis")
    if maximum <= 0:
        return 1.0, 1
    exponent = math.floor(math.log10(maximum / max_intervals))
    for power in range(exponent, exponent + 3):
        for mantissa in (1, 2, 5):
            step = mantissa * 10 ** power
            intervals = math.ceil(maximum / step - 1e-9)
            if intervals <= max_intervals:
                return float(step), max(intervals, 1)
    raise ValueError("could not choose a value-axis step")


def _format_value(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    text = f"{value:.{decimals}f}"
    return "0" if float(text) == 0 else text


def _label(label_id, role, text, x, y, **extra) -> BrailleLabel:
    return BrailleLabel(label_id, role, text, to_braille(text), x, y, **extra)


def layout_vertical_bar_chart(chart: VerticalBarChart, *,
                              key_categories: bool | None = None) -> TactileLayout:
    """Build a one-page tactile layout for ``chart``.

    The layout does its best and never silently drops content: if the chart
    does not fit the BANA limits (for example bars narrower than 3/8 inch),
    the layout is still produced and the rule checker reports the problem.
    ``key_categories`` forces (True) or forbids (False) a letter key for
    category labels; by default the key is used only when labels do not fit.
    """
    if key_categories is None:
        layout = _layout(chart, keyed=False)
        if _categories_fit(layout):
            return layout
        try:
            return _layout(chart, keyed=True)
        except ValueError as exc:
            # The key itself does not fit; keep the unkeyed layout so the
            # rule checker reports the overlapping labels instead.
            note = f"a category key would not fit on one page ({exc})"
            return replace(layout, notes=layout.notes + (note,))
    return _layout(chart, keyed=key_categories)


def _categories_fit(layout: TactileLayout) -> bool:
    labels = sorted((l for l in layout.labels if l.role == "category"), key=lambda l: l.x)
    if any(l.bounds.x1 < layout.usable.x1 or l.bounds.x2 > layout.usable.x2 for l in labels):
        return False
    return all(b.bounds.x1 - a.bounds.x2 >= LABEL_MIN_GAP for a, b in zip(labels, labels[1:]))


def _layout(chart: VerticalBarChart, keyed: bool) -> TactileLayout:
    values = [v for series in chart.series for v in series.values]
    if any(v < 0 for v in values):
        raise ValueError("negative bar values are not supported yet")
    n_series, n_cat = len(chart.series), len(chart.categories)
    if n_series > MAX_AREA_TEXTURES:
        raise ValueError(f"BANA §3.2.2 allows at most {MAX_AREA_TEXTURES} area textures; "
                         "this chart needs an alphabetic key or several graphics")
    textures = TEXTURES[:n_series]

    usable = Rect((PAGE_WIDTH - CELLS_PER_LINE * CELL_PITCH) / 2,
                  (PAGE_HEIGHT - LINES_PER_PAGE * LINE_PITCH) / 2,
                  (PAGE_WIDTH + CELLS_PER_LINE * CELL_PITCH) / 2,
                  (PAGE_HEIGHT + LINES_PER_PAGE * LINE_PITCH) / 2)
    labels: list[BrailleLabel] = []
    lines: list[Line] = []
    samples: list[KeySample] = []
    notes: list[str] = []
    y = usable.y1

    # Title: centred heading lines, then a blank line.
    for k, text in enumerate(wrap_cells(chart.title, CELLS_PER_LINE)):
        width = label_width(len(to_braille(text)))
        labels.append(_label(f"title-{k}", "title", text,
                             (usable.x1 + usable.x2 - width) / 2, y))
        y += LINE_PITCH
    y += LINE_PITCH

    # Key before the graph (§6.6.6.4), in left-to-right bar order.
    if n_series > 1:
        for s, series in enumerate(chart.series):
            name = series.name or f"series {s + 1}"
            if series.name is None:
                notes.append(f"series {s + 1} has no name in the source; keyed as {name!r}")
            sample = KeySample(f"key-sample-{s}", usable.x1, y, KEY_SAMPLE_WIDTH,
                               KEY_SAMPLE_HEIGHT, textures[s], s)
            samples.append(sample)
            # Explanation in cell 6, top of its cell level with the sample top
            # (§5.8.4.3, §5.8.4.4); runovers in cell 8.
            label_x = usable.x1 + (KEY_TEXT_CELL - 1) * CELL_PITCH
            runover_x = usable.x1 + (KEY_RUNOVER_CELL - 1) * CELL_PITCH
            name_lines = wrap_cells(name, CELLS_PER_LINE - KEY_RUNOVER_CELL + 1)
            for k, text in enumerate(name_lines):
                labels.append(_label(f"key-{s}" + (f"-{k}" if k else ""), "key", text,
                                     runover_x if k else label_x, y + k * LINE_PITCH,
                                     owner_id=sample.id))
            # A sample needs two braille lines (§5.9.1.2); long names need more.
            y += max(KEY_SAMPLE_LINES, len(name_lines) + 1) * LINE_PITCH
    letters = "abcdefghijklmnopqrstuvwxyz"
    if keyed:
        if n_cat > len(letters):
            raise ValueError("too many categories to key with letters")
        for j, name in enumerate(chart.categories):
            for k, text in enumerate(wrap_cells(f"{letters[j]} {name}", CELLS_PER_LINE)):
                # Runover lines are indented two cells.
                labels.append(_label(f"key-category-{j}" + (f"-{k}" if k else ""), "key", text,
                                     usable.x1 + (2 * CELL_PITCH if k else 0), y,
                                     category_index=j))
                y += LINE_PITCH
    if n_series > 1 or keyed:
        y += LINE_PITCH

    # Reserve the bottom: category labels, then the horizontal heading.
    x_heading = wrap_cells(chart.x_axis_title, CELLS_PER_LINE)
    axis_y = (usable.y2 - LABEL_HEIGHT - LINE_PITCH * len(x_heading)
              - LABEL_GAP - AXIS_WIDTH / 2)

    # Vertical heading lines (at most 20 cells each, §6.6.6.6); placed below.
    y_heading = wrap_cells(chart.y_axis_title, Y_HEADING_MAX_CELLS)
    heading_y = y
    y += LINE_PITCH * len(y_heading)
    plot_top = y + DOT_DIAMETER / 2 + DOT_PITCH   # top value label starts at ``y``

    plot_height = axis_y - plot_top
    max_intervals = min(MAX_GRID_INTERVALS, int(plot_height // GRID_MIN_SPACING))
    step, intervals = _nice_step(max(values), max_intervals)
    top_value = step * intervals
    spacing = plot_height / intervals

    def value_y(value: float) -> float:
        return axis_y - value / top_value * plot_height

    tick_texts = [_format_value(step * k, step) for k in range(intervals + 1)]
    value_width = max(label_width(len(to_braille(t))) for t in tick_texts)
    axis_x = usable.x1 + value_width + LABEL_GAP + AXIS_WIDTH / 2

    # Bars: ``n_series`` bars per set with BAR_GAP between them, GROUP_GAP
    # before, between and after the sets. Gaps are between raised outer edges
    # (outline included). Leave room for both half-widths of the axis line.
    plot_left = axis_x + AXIS_WIDTH / 2
    available = usable.x2 - AXIS_WIDTH / 2 - plot_left
    bar_width = bar_width_for(available, n_cat, n_series)
    if bar_width <= 0:
        raise ValueError("too many bars to draw on one page")
    group_width = set_width(bar_width, n_series)
    plot_right = plot_left + GROUP_GAP + n_cat * (group_width + GROUP_GAP)

    axis_id = "axis-x"
    lines.append(Line("axis-y", "axis", axis_x, plot_top, axis_x, axis_y,
                      AXIS_WIDTH, AXIS_STRENGTH))
    lines.append(Line(axis_id, "axis", axis_x, axis_y, plot_right, axis_y,
                      AXIS_WIDTH, AXIS_STRENGTH))
    # Grid lines perpendicular to the bars; tick marks omitted (§6.6.6.3).
    for k, text in enumerate(tick_texts):
        gy = value_y(step * k)
        if k:
            lines.append(Line(f"grid-{k}", "grid", axis_x + AXIS_WIDTH / 2, gy, plot_right, gy,
                              GRID_WIDTH, GRID_STRENGTH))
        width = label_width(len(to_braille(text)))
        # Right-justified, dots 2-5 level with the grid line (§6.6.6.6).
        labels.append(_label(f"y-value-{k}", "y_value", text,
                             axis_x - AXIS_WIDTH / 2 - LABEL_GAP - width,
                             gy - DOT_DIAMETER / 2 - DOT_PITCH,
                             owner_id="axis-y", numeric=True))

    # Vertical heading aligned with the first cell of the vertical values.
    for k, text in enumerate(y_heading):
        labels.append(_label(f"y-heading-{k}", "y_heading", text, usable.x1,
                             heading_y + k * LINE_PITCH))

    bars: list[Bar] = []
    category_y = axis_y + AXIS_WIDTH / 2 + LABEL_GAP
    first_category_x = None
    for j, name in enumerate(chart.categories):
        group_x = plot_left + GROUP_GAP + j * (group_width + GROUP_GAP)
        for s, series in enumerate(chart.series):
            value = float(series.values[j])
            top = value_y(value)
            bar_x = (group_x + BAR_OUTLINE_WIDTH / 2
                     + s * (bar_width + BAR_OUTLINE_WIDTH + BAR_GAP))
            bars.append(Bar(f"bar-{s}-{j}", bar_x, top, bar_width,
                            axis_y - AXIS_WIDTH / 2 - top, textures[s], s, j, value))
        text = letters[j] if keyed else name
        width = label_width(len(to_braille(text)))
        # Centred with the set of bars (§6.6.6.6).
        x = group_x + (group_width - width) / 2
        labels.append(_label(f"category-{j}", "category", text, x, category_y,
                             owner_id=axis_id, category_index=j))
        first_category_x = x if first_category_x is None else first_category_x

    # Horizontal heading below the values, left-justified with the first one (§6.6.6.6).
    heading_x = max(usable.x1, first_category_x)
    fitted = wrap_cells(chart.x_axis_title, int((usable.x2 - heading_x + CELL_PITCH
                                                 - label_width(1)) // CELL_PITCH))
    if len(fitted) <= len(x_heading):
        x_heading = fitted
    else:
        heading_x = usable.x1
        notes.append("horizontal heading starts at the left margin to fit its reserved lines")
    for k, text in enumerate(x_heading):
        labels.append(_label(f"x-heading-{k}", "x_heading", text, heading_x,
                             category_y + (k + 1) * LINE_PITCH))

    if any(b.height <= 0 for b in bars):
        notes.append("zero-height bars have no raised area; consider a transcriber's note")
    if keyed:
        notes.append("category labels did not fit under their bars and are keyed a, b, c")
    return TactileLayout(
        source_key=chart.source_key, page_width=PAGE_WIDTH, page_height=PAGE_HEIGHT,
        usable=usable, plot=Rect(axis_x, plot_top, plot_right, axis_y),
        lines=tuple(lines), bars=tuple(bars), labels=tuple(labels),
        key_samples=tuple(samples), category_count=n_cat,
        series_textures=textures, categories_keyed=keyed, notes=tuple(notes),
    )
