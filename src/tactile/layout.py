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

from src.contracts import BarSeries, VerticalBarChart
from src.tactile.braille import BLANK, TN_CLOSE, TN_OPEN, Cell, to_braille
from src.tactile.keys import abbreviation_keys, numeric_keys
from src.tactile.value_labels import plan_value_labels
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
    LEAD_LINE_LENGTH,
    LEAD_STRENGTH,
    LETTER_KEY_RUNOVER_CELL,
    LINE_PITCH,
    LINES_PER_PAGE,
    MAX_AREA_TEXTURES,
    MAX_GRID_INTERVALS,
    MIN_PLOT_HEIGHT,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    TEXTURES,
    TN_FIRST_CELL,
    TN_RUNOVER_CELL,
    Y_HEADING_CLEARANCE,
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
    x_label_mode: str   # one of X_LABEL_MODES
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Reasons a person must review the page before it is produced.
    needs_review: tuple[str, ...] = field(default_factory=tuple)
    # (part, of) when the graph is split into sections (BANA §3.9), else None.
    section: tuple[int, int] | None = None
    # "graph", or "key" for a key page placed before divided graph pages.
    page_kind: str = "graph"
    # True when the series key is on a key page before this graph (§6.6.6.4).
    series_key_elsewhere: bool = False

    @property
    def categories_keyed(self) -> bool:
        return self.x_label_mode != "full"

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


# Overflow order for category (x) labels, tried in turn:
#   full         - the category names under their bars
#   abbreviation - BANA §5.8.1 alphabetic keys under the bars, names in the key
#   number       - BANA §5.8.1.6 numeric keys under the bars, names in the key
# If none fits, the last layout that could be built is returned with
# ``needs_review`` set.
X_LABEL_MODES = ("full", "abbreviation", "number")


def layout_vertical_bar_chart(chart: VerticalBarChart, *,
                              x_labels: str = "auto",
                              key_on_page: bool = False,
                              value_labels: bool = True) -> TactileLayout:
    """Build a one-page tactile layout for ``chart``.

    The layout never silently drops content: if the chart breaks a BANA
    limit (for example bars narrower than 3/8 inch), the page is still
    produced and the rule checker reports the problem. ``x_labels`` is
    ``"auto"`` (follow the overflow order) or one mode from ``X_LABEL_MODES``.
    ``key_on_page`` leaves the title, notes and series key off the page, for a
    graph page that follows a key page. ``value_labels=False`` draws no bar values.
    """
    extra = {"show_title": False, "show_series_key": False} if key_on_page else {}
    extra["value_labels"] = value_labels
    if x_labels != "auto":
        if x_labels not in X_LABEL_MODES:
            raise ValueError(f"x_labels must be 'auto' or one of {X_LABEL_MODES}")
        return _layout(chart, x_labels, **extra)

    built: TactileLayout | None = None
    tried: list[str] = []
    for mode in X_LABEL_MODES:
        try:
            layout = _layout(chart, mode, **extra)
        except ValueError as exc:
            tried.append(f"{mode}: {exc}")
            continue
        if _categories_fit(layout):
            return layout
        tried.append(f"{mode}: labels do not fit under their bars")
        built = layout
    if built is None:
        raise ValueError("no x-label mode could be laid out: " + "; ".join(tried))
    reason = "x labels do not fit in any overflow mode (" + "; ".join(tried) + ")"
    return replace(built, needs_review=built.needs_review + (reason,))


def layout_vertical_bar_chart_pages(chart: VerticalBarChart, *,
                                    x_labels: str = "auto",
                                    value_labels: bool = True) -> list[TactileLayout]:
    """One page when it passes every rule; otherwise sections (BANA §3.9).

    A series key that would leave the plot under ``MIN_PLOT_HEIGHT`` goes on a
    key page before a single graph page instead.

    The categories are divided into the fewest consecutive sections whose
    pages each pass every rule except ``overview_first``, trying the x-label
    modes in overflow order for each section count. Every section uses the
    same value maximum and vertical layout, so bars are on the same scale.
    The first page keeps the title (§5.3.3) and has a transcriber's note
    giving the type of division and the number of parts (§3.9.2); each page
    has a section title in transcriber's note indicators (§3.9.4). Keys are
    chosen across the whole graph so they stay consistent between pages.
    If no split passes, the one-page layout is returned with its problems.
    """
    from src.tactile.rules import check_layout   # rules imports this module

    single = layout_vertical_bar_chart(chart, x_labels=x_labels, value_labels=value_labels)
    if _has_series_key(chart) and _plot_height(single) < MIN_PLOT_HEIGHT:
        # The key, notes and title leave too little plot: move them to a key
        # page before the graph page (§6.6.6.4), as for a divided graph.
        graph = layout_vertical_bar_chart(chart, x_labels=x_labels, key_on_page=True,
                                          value_labels=value_labels)
        if _plot_height(graph) > _plot_height(single) and not check_layout(graph):
            if _plot_height(graph) < MIN_PLOT_HEIGHT:
                graph = replace(graph, needs_review=graph.needs_review + (
                    f"the plot is {_plot_height(graph):.0f} mm high even with the key on its "
                    f"own page (minimum {MIN_PLOT_HEIGHT:.0f} mm)",))
            return [_key_page(chart, ()), graph]
    if not check_layout(single):
        return [single]
    modes = X_LABEL_MODES if x_labels == "auto" else (x_labels,)
    n_cat = len(chart.categories)
    value_max = max(v for series in chart.series for v in series.values)
    for parts in range(2, n_cat + 1):
        chunks = _split(n_cat, parts)
        for mode in modes:
            try:
                texts = _category_texts(chart.categories, mode)
                pages = _sections(chart, mode, texts, chunks, value_max, value_labels)
            except ValueError:
                continue
            if all(v.rule.id == "overview_first" for page in pages for v in check_layout(page)):
                return pages
    return [single]


def _plot_height(layout: TactileLayout) -> float:
    return layout.plot.y2 - layout.plot.y1


def _no_bars_note(chart: VerticalBarChart) -> str | None:
    """Transcriber's note naming legend entries with no visible bars (not "zero")."""
    names = [series.name for series in chart.series if not series.has_bars]
    return f"No bars are visible for: {', '.join(names)}." if names else None


def _has_series_key(chart: VerticalBarChart) -> bool:
    """Several series always need a key; a single series only when the source
    chart had a legend naming it."""
    return len(chart.series) > 1 or chart.series[0].name is not None


def _split(count: int, parts: int) -> list[range]:
    """``parts`` consecutive ranges of near-equal size covering ``count`` items."""
    base, extra = divmod(count, parts)
    ranges, start = [], 0
    for k in range(parts):
        size = base + (1 if k < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


def _sections(chart: VerticalBarChart, mode: str, texts: list[str],
              chunks: list[range], value_max: float,
              value_labels: bool = True) -> list[TactileLayout]:
    n = len(chunks)
    names = chart.categories

    def section_chart(r: range) -> VerticalBarChart:
        return VerticalBarChart(
            chart.source_key, chart.title, chart.x_axis_title, chart.y_axis_title,
            tuple(names[i] for i in r),
            tuple(BarSeries(s.name, tuple(s.values[i] for i in r) if s.has_bars else (),
                            s.has_bars, tuple(s.rescued[i] for i in r) if s.rescued else ())
                  for s in chart.series))

    division = (f"This graph is divided by category into {n} parts with the same "
                "scale on each part.")
    # With a series key it goes on its own page before the graph pages
    # (§6.6.6.4), together with the title and the division note, so that the
    # graph pages keep their height for the plot.
    key_page = _has_series_key(chart)

    def notes(k: int, r: range) -> tuple[str, ...]:
        span = names[r[0]] if len(r) == 1 else f"{names[r[0]]} to {names[r[-1]]}"
        part = f"Part {k} of {n}: {span}."
        return (division, part) if k == 1 and not key_page else (part,)

    def build(heading_y_min):
        return [_layout(section_chart(r), mode, category_texts=[texts[i] for i in r],
                        show_title=k == 1 and not key_page, transcriber_notes=notes(k, r),
                        show_series_key=not key_page,
                        heading_y_min=heading_y_min, value_max=value_max,
                        value_labels=value_labels)
                for k, r in enumerate(chunks, 1)]

    first = build(None)
    heading_y = max(min(l.y for l in page.labels if l.role == "y_heading") for page in first)
    pages = [replace(page, section=(k, n)) for k, page in enumerate(build(heading_y), 1)]
    return ([_key_page(chart, (division,))] if key_page else []) + pages


def _categories_fit(layout: TactileLayout) -> bool:
    labels = sorted((l for l in layout.labels if l.role == "category"), key=lambda l: l.x)
    if any(l.bounds.x1 < layout.usable.x1 or l.bounds.x2 > layout.usable.x2 for l in labels):
        return False
    return all(b.bounds.x1 - a.bounds.x2 >= LABEL_MIN_GAP for a, b in zip(labels, labels[1:]))


def _category_texts(names: tuple[str, ...], mode: str) -> list[str]:
    if mode == "full":
        return list(names)
    if mode == "abbreviation":
        return abbreviation_keys(names)
    return numeric_keys(len(names))


def _key_entry_lines(symbol: str, name: str) -> list[tuple[str, int]]:
    """Lines of one letter/number key entry as ``(text, start cell)``.

    The symbol starts in cell 1 followed by one blank cell, so the
    explanation starts in cell 3, 4 or 5 for a 1-, 2- or 3-cell symbol;
    runovers start in cell 3 (§5.8.4.5).
    """
    first_cells = CELLS_PER_LINE
    words = name.split()
    first = symbol
    while words and len(to_braille(f"{first} {words[0]}")) <= first_cells:
        first = f"{first} {words.pop(0)}"
    lines = [(first, 1)]
    if words:
        runover_cells = CELLS_PER_LINE - LETTER_KEY_RUNOVER_CELL + 1
        lines += [(text, LETTER_KEY_RUNOVER_CELL)
                  for text in wrap_cells(" ".join(words), runover_cells)]
    return lines


def _note_labels(note_id: str, text: str, x0: float, y: float) -> list[BrailleLabel]:
    """A transcriber's note: opening indicator in cell 7, runovers in cell 5,
    closing indicator after the last word (§5.6.2; indicators UEB Rule 3.27)."""
    words = text.split()
    tokens = [to_braille(word) for word in words]
    tokens[0] = TN_OPEN + tokens[0]
    tokens[-1] = tokens[-1] + TN_CLOSE
    out: list[BrailleLabel] = []
    start_cell, line_cells, line_words = TN_FIRST_CELL, [], []
    for word, cells in zip(words, tokens):
        capacity = CELLS_PER_LINE - start_cell + 1
        extra = len(cells) + (1 if line_cells else 0)
        if line_cells and len(line_cells) + extra > capacity:
            out.append(BrailleLabel(f"{note_id}-{len(out)}", "note", " ".join(line_words),
                                    tuple(line_cells), x0 + (start_cell - 1) * CELL_PITCH,
                                    y + len(out) * LINE_PITCH))
            start_cell, line_cells, line_words = TN_RUNOVER_CELL, [], []
        if len(cells) > CELLS_PER_LINE - start_cell + 1:
            raise ValueError(f"word too long for a transcriber's note line: {word!r}")
        line_cells += ([BLANK] if line_cells else []) + list(cells)
        line_words.append(word)
    out.append(BrailleLabel(f"{note_id}-{len(out)}", "note", " ".join(line_words),
                            tuple(line_cells), x0 + (start_cell - 1) * CELL_PITCH,
                            y + len(out) * LINE_PITCH))
    return out


def _usable_area() -> Rect:
    return Rect((PAGE_WIDTH - CELLS_PER_LINE * CELL_PITCH) / 2,
                (PAGE_HEIGHT - LINES_PER_PAGE * LINE_PITCH) / 2,
                (PAGE_WIDTH + CELLS_PER_LINE * CELL_PITCH) / 2,
                (PAGE_HEIGHT + LINES_PER_PAGE * LINE_PITCH) / 2)


def _header(chart: VerticalBarChart, usable: Rect, y: float, labels: list, samples: list,
            notes: list, *, show_title: bool, transcriber_notes: tuple[str, ...],
            series_key: bool) -> float:
    """Title, transcriber's notes and the series key; returns the next free y."""
    # Title: centred heading lines, then a blank line (§5.3.1). A split
    # graph shows it on its first page only (§5.3.3).
    if show_title:
        for k, text in enumerate(wrap_cells(chart.title, CELLS_PER_LINE)):
            width = label_width(len(to_braille(text)))
            labels.append(_label(f"title-{k}", "title", text,
                                 (usable.x1 + usable.x2 - width) / 2, y))
            y += LINE_PITCH
        y += LINE_PITCH

    # Transcriber's notes come before the key (§5.6.1), then a blank line.
    for n, text in enumerate(transcriber_notes):
        note = _note_labels(f"note-{n}", text, usable.x1, y)
        labels.extend(note)
        y += LINE_PITCH * len(note)
    if transcriber_notes:
        y += LINE_PITCH

    # Series key before the graph (§6.6.6.4), in left-to-right bar order.
    if series_key:
        textures = TEXTURES[:len(chart.series)]
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
    return y


def _key_page(chart: VerticalBarChart, transcriber_notes: tuple[str, ...]) -> TactileLayout:
    """Title, transcriber's notes and series key on a page before the graph
    pages, used when the key cannot share a page with a divided graph (§6.6.6.4)."""
    usable = _usable_area()
    labels: list[BrailleLabel] = []
    samples: list[KeySample] = []
    notes: list[str] = []
    no_bars = _no_bars_note(chart)
    y = _header(chart, usable, usable.y1, labels, samples, notes, show_title=True,
                transcriber_notes=(*transcriber_notes, no_bars) if no_bars else transcriber_notes,
                series_key=_has_series_key(chart))
    if y > usable.y2:
        raise ValueError("the key page does not fit on one page")
    return TactileLayout(
        source_key=chart.source_key, page_width=PAGE_WIDTH, page_height=PAGE_HEIGHT,
        usable=usable, plot=Rect(usable.x1, usable.y2, usable.x2, usable.y2),
        lines=(), bars=(), labels=tuple(labels), key_samples=tuple(samples),
        category_count=0, series_textures=TEXTURES[:len(chart.series)], x_label_mode="full",
        notes=tuple(notes), page_kind="key")


# Room above the plot for the value label of the tallest bar: the label, the 1/8 inch
# clearance and the bar outline; with a lead line, its length and half width too.
VALUE_HEADROOM = LABEL_HEIGHT + LABEL_MIN_GAP + BAR_OUTLINE_WIDTH / 2 + 0.02
VALUE_HEADROOM_RAISED = VALUE_HEADROOM + LEAD_LINE_LENGTH + GRID_WIDTH / 2


def _layout(chart: VerticalBarChart, mode: str, *, value_labels: bool = True,
            **page) -> TactileLayout:
    """One page. Bar value labels change the header (their notes) and so the plot, which
    changes how many labels fit; so the page is built again, with the omitted bars
    forced, until the note naming them matches what was omitted. Labels that need a
    lead line need more headroom, so that is tried only when some bar had no room."""
    if not value_labels or not any(series.has_bars for series in chart.series):
        return _layout_pass(chart, mode, **page)[0]
    best = None
    for headroom in (VALUE_HEADROOM, VALUE_HEADROOM_RAISED):
        forced, estimate = frozenset(), True
        try:
            for _ in range(8):
                result = _layout_pass(chart, mode, value_labels=True, value_headroom=headroom,
                                      forced_omit=forced, estimate_note=estimate, **page)
                omitted, drawn = frozenset(result[1]), result[2]
                if omitted == forced and (drawn > 0) == estimate:
                    break
                forced, estimate = forced | omitted, drawn > 0
            else:
                result = _layout_pass(chart, mode, value_labels=True, value_headroom=headroom,
                                      forced_omit=forced | omitted, estimate_note=True, **page)
        except ValueError:
            break                       # no room for the labels' headroom and notes
        if best is None:
            best = result
        elif len(result[1]) < len(best[1]) and (
                result[0].plot.y2 - result[0].plot.y1 >= MIN_PLOT_HEIGHT):
            best = result               # a lead line freed bars, without squashing the plot
        if not result[1]:
            break
    if best is None:
        # The chart fits without value labels (it did before they existed): keep it so.
        layout = _layout_pass(chart, mode, **page)[0]
        return replace(layout, notes=layout.notes + (
            "bar values are not labelled: no room on the page for them",))
    return best[0]


def _bar_names(chart: VerticalBarChart, bar_ids) -> str:
    """"2004, 2005 (Kenya)" style list of bars for a note; long lists are cut."""
    many = len(chart.series) > 1
    names = []
    for bar_id in sorted(bar_ids, key=lambda b: tuple(map(int, b.split("-")[1:]))[::-1]):
        s, j = map(int, bar_id.split("-")[1:])
        label = chart.categories[j]
        if many:
            label += f" ({chart.series[s].name or f'series {s + 1}'})"
        names.append(label)
    return ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")


def _value_notes(chart, rescued, omitted, estimate) -> tuple[str, ...]:
    notes = []
    if estimate:
        notes.append("Bar values are estimated from the print.")
    if rescued:
        notes.append("No value is shown for bars whose height was measured without a "
                     f"detected bar: {_bar_names(chart, rescued)}.")
    if omitted:
        notes.append(f"No value is shown for bars too narrow for it: {_bar_names(chart, omitted)}.")
    return tuple(notes)


def _layout_pass(chart: VerticalBarChart, mode: str, *,
                 category_texts: list[str] | None = None,
                 show_title: bool = True,
                 transcriber_notes: tuple[str, ...] = (),
                 show_series_key: bool = True,
                 heading_y_min: float | None = None,
                 value_max: float | None = None,
                 value_labels: bool = False,
                 value_headroom: float = 0.0,
                 forced_omit: frozenset = frozenset(),
                 estimate_note: bool = False):
    """One page, as ``(layout, ids of bars with no room for a value, labels drawn)``.
    The keyword arguments are used for sections of a split graph:
    keys chosen across the whole graph, notes, the series key moved to a key
    page, and a shared vertical layout and value maximum so every section has
    the same scale."""
    keyed = mode != "full"
    if category_texts is None:
        category_texts = _category_texts(chart.categories, mode)
    values = [v for series in chart.series for v in series.values]
    if any(v < 0 for v in values):
        raise ValueError("negative bar values are not supported yet")
    # Every series is keyed and has a texture; only series with bars take bar slots.
    n_keyed, n_cat = len(chart.series), len(chart.categories)
    n_series = sum(series.has_bars for series in chart.series)
    if n_keyed > MAX_AREA_TEXTURES:
        raise ValueError(f"BANA §3.2.2 allows at most {MAX_AREA_TEXTURES} area textures; "
                         "this chart needs an alphabetic key or several graphics")
    textures = TEXTURES[:n_keyed]
    has_key = _has_series_key(chart)
    no_bars = _no_bars_note(chart)
    if no_bars and show_series_key:
        transcriber_notes = (*transcriber_notes, no_bars)
    rescued_ids = frozenset(f"bar-{s}-{j}" for s, series in enumerate(chart.series)
                            for j, flag in enumerate(series.rescued) if flag)
    if value_labels:
        transcriber_notes = (*transcriber_notes,
                             *_value_notes(chart, rescued_ids, forced_omit, estimate_note))

    usable = _usable_area()
    labels: list[BrailleLabel] = []
    lines: list[Line] = []
    samples: list[KeySample] = []
    notes: list[str] = []
    y = _header(chart, usable, usable.y1, labels, samples, notes,
                show_title=show_title, transcriber_notes=transcriber_notes,
                series_key=show_series_key and has_key)
    if keyed:
        # Keyed labels in the order they appear, left to right (§5.7.3).
        for j, (symbol, name) in enumerate(zip(category_texts, chart.categories)):
            for k, (text, cell) in enumerate(_key_entry_lines(symbol, name)):
                labels.append(_label(f"key-category-{j}" + (f"-{k}" if k else ""), "key",
                                     text, usable.x1 + (cell - 1) * CELL_PITCH, y,
                                     category_index=j))
                y += LINE_PITCH
    if (show_series_key and has_key) or keyed:
        y += LINE_PITCH

    # Reserve the bottom: category labels, then the horizontal heading.
    x_heading = wrap_cells(chart.x_axis_title, CELLS_PER_LINE)
    axis_y = (usable.y2 - LABEL_HEIGHT - LINE_PITCH * len(x_heading)
              - LABEL_GAP - AXIS_WIDTH / 2)

    # Vertical heading lines (at most 20 cells each, §6.6.6.6); placed below.
    y_heading = wrap_cells(chart.y_axis_title, Y_HEADING_MAX_CELLS)
    if heading_y_min is not None:
        y = max(y, heading_y_min)
    heading_y = y
    heading_bottom = heading_y + (len(y_heading) - 1) * LINE_PITCH + LABEL_HEIGHT
    # The top value label starts on the next braille line, and never closer
    # than Y_HEADING_CLEARANCE below the heading.
    top_label_y = max(y + LINE_PITCH * len(y_heading), heading_bottom + Y_HEADING_CLEARANCE)
    # Its dots 2-5 sit on the top grid line (§6.6.6.6).
    plot_top = top_label_y + DOT_DIAMETER / 2 + DOT_PITCH + value_headroom

    plot_height = axis_y - plot_top
    max_intervals = min(MAX_GRID_INTERVALS, int(plot_height // GRID_MIN_SPACING))
    step, intervals = _nice_step(max(values) if value_max is None else value_max,
                                 max_intervals)
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
        for slot, (s, series) in enumerate(
                (s, series) for s, series in enumerate(chart.series) if series.has_bars):
            value = float(series.values[j])
            top = value_y(value)
            bar_x = (group_x + BAR_OUTLINE_WIDTH / 2
                     + slot * (bar_width + BAR_OUTLINE_WIDTH + BAR_GAP))
            bars.append(Bar(f"bar-{s}-{j}", bar_x, top, bar_width,
                            axis_y - AXIS_WIDTH / 2 - top, textures[s], s, j, value))
        text = category_texts[j]
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

    omitted: tuple = ()
    drawn = 0
    if value_labels and bars:
        plan = plan_value_labels(
            bars, rescued_ids, forced_omit, labels, [l for l in lines if l.kind != "grid"],
            samples, usable, step, tick_texts,
            lambda lid, text, x, y, owner: _label(lid, "value", text, x, y, owner_id=owner,
                                                  numeric=True,
                                                  category_index=int(lid.split("-")[-1])),
            lambda lid, x, y1, y2: Line(lid, "lead", x, y1, x, y2, GRID_WIDTH, LEAD_STRENGTH))
        labels += plan.labels
        lines += plan.leads
        lines = _cut_grid(lines, plan.labels)
        omitted, drawn = plan.omitted, len(plan.labels)

    if any(b.height <= 0 for b in bars):
        notes.append("zero-height bars have no raised area; consider a transcriber's note")
    if keyed:
        notes.append(f"category labels are keyed ({mode} keys, BANA §5.8)")
    layout = TactileLayout(
        source_key=chart.source_key, page_width=PAGE_WIDTH, page_height=PAGE_HEIGHT,
        usable=usable, plot=Rect(axis_x, plot_top, plot_right, axis_y),
        lines=tuple(lines), bars=tuple(bars), labels=tuple(labels),
        key_samples=tuple(samples), category_count=n_cat,
        series_textures=textures, x_label_mode=mode, notes=tuple(notes),
        series_key_elsewhere=not show_series_key and has_key,
    )
    return layout, omitted, drawn


def _cut_grid(lines: list[Line], labels) -> list[Line]:
    """Remove the stretch of each grid line that passes within the label clearance of a
    value label (grid lines are least distinct, §6.6.6.5, and labels need 1/8 inch, §3.4.3.12)."""
    out: list[Line] = []
    for line in lines:
        if line.kind != "grid":
            out.append(line)
            continue
        reach = LABEL_MIN_GAP + line.width / 2 + 0.02
        cuts = sorted((l.bounds.x1 - reach, l.bounds.x2 + reach) for l in labels
                      if l.bounds.y1 - reach < line.y1 < l.bounds.y2 + reach)
        start, pieces = line.x1, []
        for x1, x2 in cuts:
            if x1 > start:
                pieces.append((start, x1))
            start = max(start, x2)
        if start < line.x2:
            pieces.append((start, line.x2))
        if not cuts:
            pieces = [(line.x1, line.x2)]
        for k, (x1, x2) in enumerate(pieces):
            if x2 - x1 >= LABEL_MIN_GAP:
                out.append(replace(line, id=line.id if k == 0 else f"{line.id}-{k}", x1=x1, x2=x2))
    return out
