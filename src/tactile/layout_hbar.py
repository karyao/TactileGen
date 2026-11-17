"""Lay out a horizontal bar chart as a tactile graphic page.

Same page, braille and element types as ``layout.py`` (millimetres, top-left
origin); only the placement differs. BANA *Guidelines and Standards for
Tactile Graphics, 2022*:

* bars run left to right from the vertical axis; grid lines are vertical,
  perpendicular to the bars, and replace the tick marks (§6.6.6.3);
* category labels on the left are left-aligned and centred on their bar or
  set of bars, with one blank cell and at least two unspaced guide dots
  (dot 5) before the vertical axis; runovers are left-justified with the
  line above (§6.6.6.6, Vertical Scale);
* dots 4-5-6 of the first cell of each value sit on its grid line, below
  the horizontal axis (§6.6.6.6, Horizontal Scale);
* the horizontal heading goes below the values, left-justified with the
  first value; the vertical heading is aligned with the first cell of the
  category labels, at most 20 cells a line (§6.6.6.6);
* the key lists the bars from top to bottom (§6.6.6.4).

Not done yet: staggering values onto extra lines with lead lines (BANA
allows it; here the grid step grows until the values fit on one line),
dividing a graph into sections, and place-value alignment for numeric
category labels.
"""

from __future__ import annotations

from dataclasses import replace

from src.contracts import HorizontalBarChart
from src.tactile.braille import BLANK, GUIDE_DOT, to_braille
from src.tactile.constants import (
    AXIS_STRENGTH,
    AXIS_WIDTH,
    BAR_GAP,
    BAR_OUTLINE_WIDTH,
    CATEGORY_COLUMN_MAX_CELLS,
    CELL_PITCH,
    DOT_DIAMETER,
    DOT_PITCH,
    GRID_MIN_SPACING,
    GRID_STRENGTH,
    GRID_WIDTH,
    GROUP_GAP,
    GUIDE_DOTS_MIN,
    LABEL_GAP,
    LABEL_HEIGHT,
    LABEL_MIN_GAP,
    LINE_PITCH,
    MAX_AREA_TEXTURES,
    MAX_GRID_INTERVALS,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    TEXTURES,
    Y_HEADING_CLEARANCE,
    Y_HEADING_MAX_CELLS,
)
from src.tactile.layout import (
    X_LABEL_MODES,
    Bar,
    BrailleLabel,
    Line,
    Rect,
    TactileLayout,
    _category_key,
    _category_texts,
    _format_value,
    _has_series_key,
    _header,
    _label,
    _nice_step,
    _usable_area,
    bar_width_for,
    label_width,
    set_width,
    wrap_cells,
)

# From a label's left edge to the centre of dots 4-5-6 of its first cell.
FIRST_CELL_RIGHT_COLUMN = DOT_DIAMETER / 2 + DOT_PITCH


def layout_horizontal_bar_chart(chart: HorizontalBarChart, *,
                                labels: str = "auto") -> TactileLayout:
    """Build a one-page tactile layout for a horizontal bar chart.

    ``labels`` is ``"auto"`` (full names, then abbreviation keys, then
    numeric keys, BANA §5.8) or one mode from ``X_LABEL_MODES``. Like the
    vertical layout it never drops content: when nothing fits, the last page
    that could be built is returned with ``needs_review`` set.
    """
    if labels != "auto":
        if labels not in X_LABEL_MODES:
            raise ValueError(f"labels must be 'auto' or one of {X_LABEL_MODES}")
        return _layout(chart, labels)
    built: TactileLayout | None = None
    tried: list[str] = []
    for mode in X_LABEL_MODES:
        try:
            layout = _layout(chart, mode)
        except ValueError as exc:
            tried.append(f"{mode}: {exc}")
            continue
        if _categories_fit(layout):
            return layout
        tried.append(f"{mode}: labels do not fit beside their bars")
        built = layout
    if built is None:
        raise ValueError("no category-label mode could be laid out: " + "; ".join(tried))
    reason = "category labels do not fit in any overflow mode (" + "; ".join(tried) + ")"
    return replace(built, needs_review=built.needs_review + (reason,))


def layout_horizontal_bar_chart_pages(chart: HorizontalBarChart, *,
                                      labels: str = "auto") -> list[TactileLayout]:
    """Pages for the chart. Dividing into sections (§3.9) is not done yet, so
    this is always one page; the rule checker reports anything that breaks."""
    return [layout_horizontal_bar_chart(chart, labels=labels)]


def _categories_fit(layout: TactileLayout) -> bool:
    labels = [l for l in layout.labels if l.role == "category"]
    if any(not l.bounds.inside(layout.usable) for l in labels):
        return False
    return all(a.bounds.gap(b.bounds) >= LABEL_MIN_GAP
               for i, a in enumerate(labels) for b in labels[i + 1:]
               if a.category_index != b.category_index)


def _value_scale(maximum: float, axis_x: float, right_limit: float):
    """Grid step for the horizontal values: the most intervals (at most
    MAX_GRID_INTERVALS) whose grid lines are 3/8 inch apart and whose values
    fit on one line 1/8 inch apart. Returns ``(step, intervals, texts, right)``,
    where ``right`` is the x of the last grid line."""
    for cap in range(MAX_GRID_INTERVALS, 0, -1):
        step, intervals = _nice_step(maximum, cap)
        texts = [_format_value(step * k, step) for k in range(intervals + 1)]
        widths = [label_width(len(to_braille(t))) for t in texts]
        # The last value hangs past its grid line; the axis line needs half its width.
        right = min(right_limit - AXIS_WIDTH / 2,
                    right_limit - (widths[-1] - FIRST_CELL_RIGHT_COLUMN))
        spacing = (right - axis_x) / intervals
        if spacing >= GRID_MIN_SPACING and spacing >= max(widths[:-1]) + LABEL_MIN_GAP:
            return step, intervals, texts, right
    raise ValueError("the values do not fit along the horizontal axis")


def _layout(chart: HorizontalBarChart, mode: str) -> TactileLayout:
    keyed = mode != "full"
    category_texts = _category_texts(chart.categories, mode)
    values = [v for series in chart.series for v in series.values]
    if any(v < 0 for v in values):
        raise ValueError("negative bar values are not supported yet")
    n_series, n_cat = len(chart.series), len(chart.categories)
    if n_series > MAX_AREA_TEXTURES:
        raise ValueError(f"BANA §3.2.2 allows at most {MAX_AREA_TEXTURES} area textures; "
                         "this chart needs an alphabetic key or several graphics")
    textures = TEXTURES[:n_series]
    has_key = _has_series_key(chart)

    usable = _usable_area()
    labels: list[BrailleLabel] = []
    lines: list[Line] = []
    notes: list[str] = []
    samples: list = []
    y = _header(chart, usable, usable.y1, labels, samples, notes, show_title=True,
                transcriber_notes=(), series_key=has_key)
    if keyed:
        y = _category_key(chart.categories, category_texts, usable, y, labels)
    if has_key or keyed:
        y += LINE_PITCH

    # Category column: label, one blank cell, then guide dots up to the column edge.
    wrapped = [wrap_cells(t, CATEGORY_COLUMN_MAX_CELLS - 1 - GUIDE_DOTS_MIN)
               for t in category_texts]
    column_cells = max(len(to_braille(t)) for rows in wrapped for t in rows) + 1 + GUIDE_DOTS_MIN
    axis_x = usable.x1 + label_width(column_cells) + LABEL_GAP + AXIS_WIDTH / 2
    first_value_x = axis_x - FIRST_CELL_RIGHT_COLUMN

    # Bottom: values on the line under the axis, then the horizontal heading.
    heading_cells = int((usable.x2 - first_value_x + CELL_PITCH - label_width(1)) // CELL_PITCH)
    x_heading = wrap_cells(chart.x_axis_title, heading_cells)
    axis_y = (usable.y2 - LABEL_HEIGHT - LINE_PITCH * len(x_heading)
              - LABEL_GAP - AXIS_WIDTH / 2)
    value_label_y = axis_y + AXIS_WIDTH / 2 + LABEL_GAP

    step, intervals, tick_texts, plot_right = _value_scale(max(values), axis_x, usable.x2)
    top_value = step * intervals

    def value_x(value: float) -> float:
        return axis_x + value / top_value * (plot_right - axis_x)

    # Bars: thickness fills the height left under the vertical heading; sets
    # are also far enough apart for their label blocks to be 1/8 inch apart.
    y_heading = wrap_cells(chart.y_axis_title, Y_HEADING_MAX_CELLS)
    heading_height = (len(y_heading) - 1) * LINE_PITCH + LABEL_HEIGHT
    bars_bottom = axis_y - AXIS_WIDTH / 2
    available = bars_bottom - (y + heading_height + Y_HEADING_CLEARANCE)
    thickness = bar_width_for(available, n_cat, n_series)
    if thickness <= 0:
        raise ValueError("too many bars to draw on one page")
    group = set_width(thickness, n_series)
    blocks = [(len(rows) - 1) * LINE_PITCH + LABEL_HEIGHT for rows in wrapped]
    pitch = max(group + GROUP_GAP, max(blocks) + LABEL_MIN_GAP)
    plot_top = bars_bottom - (2 * GROUP_GAP + group + (n_cat - 1) * pitch)
    centres = [plot_top + GROUP_GAP + group / 2 + j * pitch for j in range(n_cat)]
    top_extent = min(plot_top, centres[0] - blocks[0] / 2)
    heading_y = top_extent - Y_HEADING_CLEARANCE - heading_height
    if heading_y < y - 1e-6:
        raise ValueError("the category labels need more height than the page has")
    # Bars at their 1 inch maximum leave spare height: close it up so the
    # graph follows the headings and the spare space is at the page bottom.
    shift = heading_y - y
    heading_y, plot_top, bars_bottom = y, plot_top - shift, bars_bottom - shift
    axis_y, value_label_y = axis_y - shift, value_label_y - shift
    centres = [c - shift for c in centres]

    lines.append(Line("axis-y", "axis", axis_x, plot_top, axis_x, axis_y,
                      AXIS_WIDTH, AXIS_STRENGTH))
    lines.append(Line("axis-x", "axis", axis_x, axis_y, plot_right, axis_y,
                      AXIS_WIDTH, AXIS_STRENGTH))
    # Grid lines perpendicular to the bars; tick marks omitted (§6.6.6.3).
    for k, text in enumerate(tick_texts):
        gx = value_x(step * k)
        if k:
            lines.append(Line(f"grid-{k}", "grid", gx, plot_top, gx, bars_bottom,
                              GRID_WIDTH, GRID_STRENGTH))
        # Dots 4-5-6 of the first cell on the grid line (§6.6.6.6).
        labels.append(_label(f"x-value-{k}", "x_value", text, gx - FIRST_CELL_RIGHT_COLUMN,
                             value_label_y, owner_id="axis-x", numeric=True))

    for k, text in enumerate(y_heading):
        labels.append(_label(f"y-heading-{k}", "y_heading", text, usable.x1,
                             heading_y + k * LINE_PITCH))

    plot_left = axis_x + AXIS_WIDTH / 2
    bars: list[Bar] = []
    for j, rows in enumerate(wrapped):
        set_top = centres[j] - group / 2
        for s, series in enumerate(chart.series):
            value = float(series.values[j])
            bar_y = set_top + BAR_OUTLINE_WIDTH / 2 + s * (thickness + BAR_OUTLINE_WIDTH + BAR_GAP)
            bars.append(Bar(f"bar-{s}-{j}", plot_left, bar_y, value_x(value) - plot_left,
                            thickness, textures[s], s, j, value))
        # Block centred on the set; guide dots on its last line lead to the axis.
        block_top = centres[j] - blocks[j] / 2
        for k, text in enumerate(rows):
            cells = to_braille(text)
            last = k == len(rows) - 1
            if last:
                cells += (BLANK,) + (GUIDE_DOT,) * (column_cells - len(cells) - 1)
            labels.append(BrailleLabel(
                f"category-{j}" + (f"-{k}" if k else ""), "category", text, cells,
                usable.x1, block_top + k * LINE_PITCH,
                owner_id="axis-y" if last else None, category_index=j))

    for k, text in enumerate(x_heading):
        labels.append(_label(f"x-heading-{k}", "x_heading", text, first_value_x,
                             value_label_y + (k + 1) * LINE_PITCH))

    if any(b.width <= 0 for b in bars):
        notes.append("zero-length bars have no raised area; consider a transcriber's note")
    if keyed:
        notes.append(f"category labels are keyed ({mode} keys, BANA §5.8)")
    return TactileLayout(
        source_key=chart.source_key, page_width=PAGE_WIDTH, page_height=PAGE_HEIGHT,
        usable=usable, plot=Rect(axis_x, plot_top, plot_right, axis_y),
        lines=tuple(lines), bars=tuple(bars), labels=tuple(labels),
        key_samples=tuple(samples), category_count=n_cat, series_textures=textures,
        x_label_mode=mode, notes=tuple(notes), orientation="horizontal")
