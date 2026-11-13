"""Braille value labels for the bars of a vertical bar chart.

Text: a value is rounded to the decimals of the y-axis tick step, then to at most
``max(VALUE_LABEL_SIG_DIGITS, significant digits of the ticks)`` significant digits;
a nonzero value is never shown as 0 (it keeps one significant digit instead).

Placement (``plan_value_labels``): centred above the bar at the 1/8 inch label
clearance (§3.4.3.12). A label wider than its bar, or one that would touch a
neighbour, is tried with fewer decimals (the same for every label on the page, down
to the fewest that keep each value nonzero), then raised on a 3/4 inch lead line
(§3.4.3.12), then omitted. The planner checks real distances to every bar, line,
label and key sample, so the layout passes ``label_clearance`` by construction.
Negative values are formatted but not placed: the layout rejects them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from src.tactile.constants import (
    LABEL_HEIGHT, LABEL_MIN_GAP, LEAD_LINE_LENGTH,
    VALUE_LABEL_SIG_DIGITS,
)

_EPS = 0.01      # keeps a label just outside the minimum clearance despite float noise


def tick_decimals(step: float) -> int:
    """Decimals of the y-axis tick labels (the same rule as ``_format_value``)."""
    return max(0, -math.floor(math.log10(step))) if step < 1 else 0


def _sig_digits(text: str) -> int:
    digits = text.replace("-", "").replace(".", "")
    return len(digits.lstrip("0") if "." in text else digits.lstrip("0").rstrip("0")) or 1


def significant_cap(tick_texts: list[str]) -> int:
    return max([VALUE_LABEL_SIG_DIGITS, *(_sig_digits(t) for t in tick_texts)])


def value_text(value: float, decimals: int, cap: int) -> str:
    """``value`` rounded to ``decimals`` places and at most ``cap`` significant digits."""
    if value == 0:
        return "0"
    exact = Decimal(repr(float(value)))
    magnitude = exact.adjusted()                      # floor(log10(|value|))
    places = min(decimals, cap - 1 - magnitude)
    if round_to(exact, places) == 0:                  # never show a nonzero value as 0
        places = -magnitude
    rounded = round_to(exact, places)
    text = f"{rounded:.{max(0, places)}f}"
    return "0" if float(text) == 0 else text


def round_to(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class ValuePlan:
    labels: tuple          # BrailleLabel
    leads: tuple           # Line
    omitted: tuple         # bar ids with no room for a label
    staggered: tuple       # bar ids whose label is raised on a lead line
    decimals: int


def plan_value_labels(bars, skipped, forced_omit, fixed_labels, fixed_lines, samples, usable,
                      step: float, tick_texts: list[str], make_label, make_lead) -> ValuePlan:
    """Choose text and position for the label of every bar not in ``skipped``.

    ``fixed_*`` are the page's other labels, non-grid lines (grid lines are cut
    around labels afterwards) and key samples. ``make_label(id, text, x, y, owner)``
    and ``make_lead(id, x, y1, y2)`` build the layout elements.
    """
    todo = [b for b in bars if b.id not in skipped]
    cap = significant_cap(tick_texts)
    start = tick_decimals(step)
    smallest = [abs(b.value) for b in todo if b.value]
    floor = max(0, -math.floor(math.log10(min(smallest)))) if smallest else 0
    floor = min(floor, start)

    def place(decimals, stagger):
        labels, leads, omitted, raised = [], [], [], []
        for bar in todo:
            if bar.id in forced_omit:
                omitted.append(bar.id)
                continue
            text = value_text(bar.value, decimals, cap)
            made = None
            for lift in ((0.0, LEAD_LINE_LENGTH) if stagger else (0.0,)):
                made = _try(bar, text, lift, bars, labels, leads, fixed_labels, fixed_lines,
                            samples, usable, make_label, make_lead)
                if made:
                    break
            if made is None:
                omitted.append(bar.id)
                continue
            labels.append(made[0])
            if made[1] is not None:
                leads.append(made[1])
                raised.append(bar.id)
        return ValuePlan(tuple(labels), tuple(leads), tuple(omitted), tuple(raised), decimals)

    for decimals in range(start, floor - 1, -1):
        plan = place(decimals, False)
        if all(bar_id in forced_omit for bar_id in plan.omitted):
            return plan
    return place(floor, True)


def _try(bar, text, lift, bars, placed, leads, fixed_labels, fixed_lines, samples, usable,
         make_label, make_lead):
    """One candidate: directly above the bar (``lift`` 0) or raised on a lead line."""
    centre = bar.x + bar.width / 2
    top = bar.bounds.y1
    owner, lead = bar.id, None
    bottom = top - LABEL_MIN_GAP - _EPS - lift
    if lift:
        lead = make_lead(f"lead-{bar.series_index}-{bar.category_index}", centre, top, top - lift)
        owner = lead.id
        bottom -= lead.width / 2         # the clearance is measured from the line's edge
    label = make_label(f"value-{bar.series_index}-{bar.category_index}", text, 0.0,
                       bottom - LABEL_HEIGHT, owner)
    label = _centred(label, centre, make_label)
    if lift == 0.0 and label.width > bar.bounds.x2 - bar.bounds.x1:
        return None                                   # wider than its bar
    box = label.bounds
    if not box.inside(usable):
        return None
    if any(box.gap(other.bounds) < LABEL_MIN_GAP for other in (*fixed_labels, *placed)):
        return None
    if any(box.gap(other.bounds) < LABEL_MIN_GAP for other in (*bars, *samples)):
        return None
    obstacles = [*fixed_lines, *leads]
    if any(box.gap(line.bounds) < LABEL_MIN_GAP for line in obstacles):
        return None
    if lead is not None:
        others = (*fixed_labels, *placed)
        if any(lead.bounds.gap(other.bounds) < LABEL_MIN_GAP for other in others):
            return None
    return label, lead


def _centred(label, centre, make_label):
    return make_label(label.id, label.text, centre - label.width / 2, label.y, label.owner_id)

