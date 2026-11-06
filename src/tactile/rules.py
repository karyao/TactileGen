"""Check a tactile layout against BANA tactile-graphics rules.

Each rule cites *Guidelines and Standards for Tactile Graphics, 2022*. Rules
only read the layout, so they also catch problems in layouts built by hand
or by a future renderer. ``check_layout`` returns every violation; an empty
list means every implemented rule passed, not that the graphic is fully
BANA compliant (many guidelines need human judgement).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from src.tactile.braille import has_numeric_indicator
from src.tactile.constants import (
    BAR_GAP,
    BAR_MAX_WIDTH,
    BAR_MIN_WIDTH,
    CELL_PITCH,
    GRID_MIN_SPACING,
    GROUP_GAP,
    KEY_SAMPLE_HEIGHT,
    KEY_SAMPLE_WIDTH,
    KEY_TEXT_CELL,
    LABEL_MAX_GAP,
    LABEL_MIN_GAP,
    MAX_AREA_TEXTURES,
    PRIMARY_LINE_MIN_LENGTH,
    TICK_HALF_LENGTH,
    Y_HEADING_MAX_CELLS,
)
from src.tactile.keys import is_valid_abbreviation_key, is_valid_numeric_key
from src.tactile.layout import TactileLayout

# Floating-point slack when comparing placed geometry; not a spacing value.
TOLERANCE = 0.05


@dataclass(frozen=True)
class Rule:
    id: str
    section: str
    summary: str


@dataclass(frozen=True)
class Violation:
    rule: Rule
    message: str
    element_ids: tuple[str, ...]


RULES = {rule.id: rule for rule in (
    Rule("bar_width", "§6.6.6.7", "Bars are 3/8 inch to 1 inch wide."),
    Rule("bar_spacing", "§6.6.6.7 (gap value from §3.4.3.12)",
         "Bars in a set are joined without overlapping; neighbouring sets are separated "
         "by at least 1/8 inch."),
    Rule("label_clearance", "§3.4.3.12, §6.6.6.6",
         "Labels are at least 1/8 inch (3 mm) from every other component."),
    Rule("label_owner_distance", "§3.4.3.12, §5.10.5",
         "A label is at most 1/4 inch (6 mm) from the component it belongs to."),
    Rule("fits_page", "§3.4.2", "The graphic fits 40 cells by 25 lines."),
    Rule("grid_spacing", "§3.4.3.9, §6.6.1.1", "Grid lines are at least 3/8 inch apart."),
    Rule("line_hierarchy", "§6.6.6.5, §3.4.3.8",
         "Grid and lead lines are the least distinct; axis lines are stronger."),
    Rule("primary_line_length", "§3.4.3.2", "Primary (axis) lines are at least 1/2 inch long."),
    Rule("tick_length", "§6.6.6.5", "Tick marks cross the axis 1/4 inch on each side."),
    Rule("numeric_indicator", "§6.6.6.2", "Numeric values use the numeric indicator."),
    Rule("all_bars_labeled", "§6.6.1.3", "No labels are omitted for bars."),
    Rule("key_before_graph", "§6.6.6.4", "The key is placed before (above) the graph."),
    Rule("key_order", "§6.6.6.4", "The key lists bars in the order they appear, left to right."),
    Rule("distinct_textures", "§6.6.6.7, §6.6.6.4",
         "Each set of bars has a distinct texture that matches its key sample."),
    Rule("key_sample_size", "§3.4.3.14", "Key texture samples are 1/2 inch high and 1 inch wide."),
    Rule("y_heading_length", "§6.6.6.6", "Each line of the vertical heading has at most 20 cells."),
    Rule("heading_placement", "§6.6.6.6",
         "The horizontal heading is left-justified with the first horizontal value, below "
         "the values; the vertical heading is aligned with the first cell of the vertical "
         "values. Headings are not centred on the plot."),
    Rule("texture_count", "§3.2.2", "A graphic uses at most five different area textures."),
    Rule("key_symbols", "§5.8.1.1-3, §5.8.1.6",
         "Alphabetic keys are 2-3 letters with a dot 3 or 6 letter and not aa/ac/cc/ca; "
         "numeric keys are numbers without a period; keys are distinct."),
    Rule("x_label_overflow", "§6.6.6.6, §5.8",
         "Category labels fit under their bars in some overflow mode, or need review."),
    Rule("key_format", "§5.8.4.3, §5.8.4.4",
         "Key explanations start in cell 6, level with the top of their texture sample."),
)}


def _violation(rule_id: str, message: str, *ids: str) -> Violation:
    return Violation(RULES[rule_id], message, tuple(ids))


def check_layout(layout: TactileLayout) -> list[Violation]:
    found: list[Violation] = []
    found += _bar_width(layout)
    found += _bar_spacing(layout)
    found += _label_clearance(layout)
    found += _label_owner_distance(layout)
    found += _fits_page(layout)
    found += _grid_spacing(layout)
    found += _line_hierarchy(layout)
    found += _primary_line_length(layout)
    found += _tick_length(layout)
    found += _numeric_indicator(layout)
    found += _all_bars_labeled(layout)
    found += _key_rules(layout)
    found += _y_heading_length(layout)
    found += _heading_placement(layout)
    found += _texture_count(layout)
    found += _key_format(layout)
    found += _key_symbols(layout)
    found += [_violation("x_label_overflow", reason) for reason in layout.needs_review]
    return found


def _key_symbols(layout):
    if not layout.categories_keyed:
        return
    symbols = [l for l in layout.labels if l.role == "category"]
    texts = [l.text for l in symbols]
    for label in symbols:
        if layout.x_label_mode == "abbreviation":
            valid = is_valid_abbreviation_key(label.text)
        else:
            valid = is_valid_numeric_key(label.text) and has_numeric_indicator(label.cells)
        if not valid:
            yield _violation("key_symbols", f"{label.id} key {label.text!r} breaks the "
                             f"{layout.x_label_mode} key rules", label.id)
        if texts.count(label.text) > 1:
            yield _violation("key_symbols", f"key {label.text!r} is used twice", label.id)


def _heading_placement(layout):
    by_role = {}
    for label in layout.labels:
        by_role.setdefault(label.role, []).append(label)
    categories = sorted(by_role.get("category", []), key=lambda l: l.x)
    x_heading = sorted(by_role.get("x_heading", []), key=lambda l: l.y)
    if categories and x_heading:
        first = categories[0]
        if abs(x_heading[0].x - first.x) > TOLERANCE:
            yield _violation("heading_placement", f"{x_heading[0].id} starts at "
                             f"x={x_heading[0].x:.1f} mm, not with the first value at "
                             f"x={first.x:.1f} mm", x_heading[0].id, first.id)
        if x_heading[0].y <= first.y:
            yield _violation("heading_placement", f"{x_heading[0].id} is not below the values",
                             x_heading[0].id)
    values = by_role.get("y_value", [])
    if values:
        first_cell = min(l.x for l in values)
        for heading in by_role.get("y_heading", []):
            if abs(heading.x - first_cell) > TOLERANCE:
                yield _violation("heading_placement", f"{heading.id} is not aligned with the "
                                 "first cell of the vertical values", heading.id)


def _texture_count(layout):
    count = len(set(layout.series_textures))
    if count > MAX_AREA_TEXTURES:
        yield _violation("texture_count", f"{count} area textures (max {MAX_AREA_TEXTURES})")


def _key_format(layout):
    text_x = layout.usable.x1 + (KEY_TEXT_CELL - 1) * CELL_PITCH
    for label in layout.labels:
        if label.role != "key" or label.owner_id is None:
            continue
        sample = layout.by_id(label.owner_id)
        first_line = label.id == f"key-{getattr(sample, 'series_index', '')}"
        if sample is None or not first_line:
            continue
        if abs(label.x - text_x) > TOLERANCE or abs(label.y - sample.y) > TOLERANCE:
            yield _violation("key_format", f"{label.id} should start in cell 6, level with "
                             f"the top of {sample.id}", label.id, sample.id)


def _bar_width(layout):
    for bar in layout.bars:
        if not BAR_MIN_WIDTH - TOLERANCE <= bar.width <= BAR_MAX_WIDTH + TOLERANCE:
            yield _violation("bar_width", f"{bar.id} is {bar.width:.1f} mm wide "
                             f"(allowed {BAR_MIN_WIDTH:.1f}-{BAR_MAX_WIDTH:.1f} mm)", bar.id)


def _bar_spacing(layout):
    bars = sorted(layout.bars, key=lambda b: b.x)
    for left, right in zip(bars, bars[1:]):
        same_set = left.category_index == right.category_index
        minimum = BAR_GAP if same_set else GROUP_GAP
        gap = right.bounds.x1 - left.bounds.x2
        if gap < minimum - TOLERANCE:
            kind = "bars in a set" if same_set else "sets of bars"
            yield _violation("bar_spacing", f"{left.id} and {right.id} ({kind}) are "
                             f"{gap:.1f} mm apart (min {minimum:.1f} mm)", left.id, right.id)


def _label_clearance(layout):
    others = [*layout.lines, *layout.bars, *layout.key_samples]
    for label in layout.labels:
        for other in others:
            gap = label.bounds.gap(other.bounds)
            if gap < LABEL_MIN_GAP - TOLERANCE:
                yield _violation("label_clearance", f"{label.id} ({label.text!r}) is "
                                 f"{gap:.1f} mm from {other.id}", label.id, other.id)
    for a, b in combinations(layout.labels, 2):
        gap = a.bounds.gap(b.bounds)
        if gap < LABEL_MIN_GAP - TOLERANCE:
            yield _violation("label_clearance", f"{a.id} ({a.text!r}) is {gap:.1f} mm "
                             f"from {b.id} ({b.text!r})", a.id, b.id)


def _label_owner_distance(layout):
    for label in layout.labels:
        if label.owner_id is None:
            continue
        owner = layout.by_id(label.owner_id)
        if owner is None:
            yield _violation("label_owner_distance",
                             f"{label.id} belongs to missing {label.owner_id}", label.id)
            continue
        gap = label.bounds.gap(owner.bounds)
        if gap > LABEL_MAX_GAP + TOLERANCE:
            yield _violation("label_owner_distance", f"{label.id} is {gap:.1f} mm from "
                             f"{owner.id} (max {LABEL_MAX_GAP:.1f} mm without a lead line)",
                             label.id, owner.id)


def _fits_page(layout):
    for element in layout.elements():
        if not element.bounds.inside(layout.usable):
            yield _violation("fits_page", f"{element.id} extends outside the "
                             "40-cell by 25-line area", element.id)


def _grid_spacing(layout):
    ys = sorted({round(line.y1, 3) for line in layout.lines if line.kind == "grid"}
                | {round(line.y1, 3) for line in layout.lines
                   if line.kind == "axis" and line.y1 == line.y2})
    for a, b in zip(ys, ys[1:]):
        if b - a < GRID_MIN_SPACING - TOLERANCE:
            yield _violation("grid_spacing", f"grid lines at y={a:.1f} and y={b:.1f} mm "
                             f"are {b - a:.1f} mm apart")


def _line_hierarchy(layout):
    axes = [l for l in layout.lines if l.kind == "axis"]
    weak = [l for l in layout.lines if l.kind in ("grid", "lead")]
    if not axes:
        return
    weakest_axis = min(l.strength for l in axes)
    for line in weak:
        if line.strength >= weakest_axis:
            yield _violation("line_hierarchy", f"{line.id} ({line.kind}) is as strong as "
                             "an axis line", line.id)


def _primary_line_length(layout):
    for line in layout.lines:
        if line.kind == "axis" and line.length < PRIMARY_LINE_MIN_LENGTH - TOLERANCE:
            yield _violation("primary_line_length",
                             f"{line.id} is {line.length:.1f} mm long", line.id)


def _tick_length(layout):
    for line in layout.lines:
        if line.kind == "tick" and abs(line.length - 2 * TICK_HALF_LENGTH) > 0.5:
            yield _violation("tick_length", f"{line.id} is {line.length:.1f} mm long "
                             f"(expected {2 * TICK_HALF_LENGTH:.1f} mm)", line.id)


def _numeric_indicator(layout):
    for label in layout.labels:
        if label.numeric and not has_numeric_indicator(label.cells):
            yield _violation("numeric_indicator",
                             f"{label.id} ({label.text!r}) has no numeric indicator", label.id)


def _all_bars_labeled(layout):
    labeled = {l.category_index for l in layout.labels if l.role == "category"}
    keyed = {l.category_index for l in layout.labels if l.role == "key"
             and l.category_index is not None}
    for j in range(layout.category_count):
        if j not in labeled:
            yield _violation("all_bars_labeled", f"category {j} has no label")
        elif layout.categories_keyed and j not in keyed:
            yield _violation("all_bars_labeled", f"category {j} is keyed but missing from the key")


def _key_rules(layout):
    samples = sorted(layout.key_samples, key=lambda s: s.y)
    if len(layout.series_textures) > 1 and not samples:
        yield _violation("key_before_graph", "several sets of bars but no key")
    key_items = [*samples, *(l for l in layout.labels if l.role == "key")]
    for item in key_items:
        if item.bounds.y2 > layout.plot.y1:
            yield _violation("key_before_graph", f"{item.id} is not above the graph", item.id)

    order = [s.series_index for s in samples]
    bar_order = []
    for bar in sorted(layout.bars, key=lambda b: b.x):
        if bar.series_index not in bar_order:
            bar_order.append(bar.series_index)
    if samples and order != bar_order:
        yield _violation("key_order", f"key order {order} differs from bar order {bar_order}")

    if len(set(layout.series_textures)) != len(layout.series_textures):
        yield _violation("distinct_textures", "two sets of bars share a texture")
    for bar in layout.bars:
        if bar.texture != layout.series_textures[bar.series_index]:
            yield _violation("distinct_textures", f"{bar.id} does not use its series texture",
                             bar.id)
    for sample in samples:
        if sample.texture != layout.series_textures[sample.series_index]:
            yield _violation("distinct_textures", f"{sample.id} texture does not match its bars",
                             sample.id)
        if (abs(sample.width - KEY_SAMPLE_WIDTH) > 0.5
                or abs(sample.height - KEY_SAMPLE_HEIGHT) > 0.5):
            yield _violation("key_sample_size", f"{sample.id} is {sample.width:.1f} x "
                             f"{sample.height:.1f} mm", sample.id)


def _y_heading_length(layout):
    for label in layout.labels:
        if label.role == "y_heading" and len(label.cells) > Y_HEADING_MAX_CELLS:
            yield _violation("y_heading_length",
                             f"{label.id} has {len(label.cells)} cells", label.id)


def format_report(violations: list[Violation]) -> str:
    lines = [f"Checked {len(RULES)} BANA rules: "
             + ("all passed." if not violations else f"{len(violations)} problem(s) found.")]
    for v in violations:
        lines.append(f"  [{v.rule.id}] {v.rule.section}: {v.message}")
    return "\n".join(lines)
