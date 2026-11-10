"""Decide whether a reconstructed ``VerticalBarSpec`` can become a tactile chart.

``check_completeness`` sorts a spec into one of three states:

* ``ready``: every title, category and series name is present and nothing
  suspicious was flagged; ``spec_to_chart`` can build a ``VerticalBarChart``.
* ``needs_input``: only text is missing (titles, unreadable categories,
  unnamed series). The user can supply it with ``apply_overrides``.
* ``cannot_build``: the numbers or structure are not trustworthy. Overrides
  cannot fix this; the user needs a clearer image or a hand-written chart.

Nothing here invents text or numbers. Overrides supply text only.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field, replace

import numpy as np

from src.tactile.braille import to_braille
from src.contracts import (
    BarSeries,
    DetectionFrame,
    ChartInterpretationError,
    VerticalBarChart,
    VerticalBarSpec,
)

READY, NEEDS_INPUT, CANNOT_BUILD = "ready", "needs_input", "cannot_build"

# Flags that mean the numbers or structure cannot be trusted. Any one of them
# makes the spec ``cannot_build``.
BLOCKING_FLAGS = {
    # bars and category labels do not pair one to one
    "category_count_mismatch": "The number of bars does not match the number of category labels.",
    # two bars claimed the same category slot
    "category_ambiguous": "Some bars could not be matched to a single category label.",
    # legend entries and bar colours do not pair up
    "series_ambiguous": "The legend entries could not be matched to the bar colours.",
    # a surviving tick label like "04" has a leading zero, so a decimal point was probably lost
    "y_tick_leading_zero": ("A y-axis tick label has a leading zero (like '04'), so a decimal "
                            "point may have been lost and the axis scale could be wrong."),
    # two equally good scales fit the y ticks
    "y_calibration_ambiguous": "The y-axis scale could not be settled from the tick labels.",
    # the zero line is outside the plot, so heights do not map to values
    "baseline_outside_plot": "The zero line of the y axis is not inside the plot area.",
    # a bar runs past the last tick, so its value is cut off
    "value_clipped_at_axis": "At least one bar runs past the end of the y axis.",
}

# Blocking flags that describe ambiguity rather than a missing piece of the
# chart. ``--force`` may continue past these (the user is told loudly).
FORCEABLE_FLAGS = frozenset({
    "category_count_mismatch", "category_ambiguous", "series_ambiguous",
    "y_calibration_ambiguous",
})

# Flags worth telling the user about that do not stop a build. Each is matched
# by name, or by the prefix before the colon.
NOTE_FLAGS = {
    # values measured from pixels because no bar was detected
    "bars_rescued": "{n} bar value(s) were measured from pixels because YOLO did not detect the bar.",
    # a rescue found no bar pixels; {n} counts those values, drawn as zero
    "value_inferred_zero": ("No bar was visible for {n} value(s), so they are drawn as 0. "
                            "The data may include values below the axis that are not drawn."),
    # legend swatches with no detected bars became series measured from pixels
    "series_from_swatch": ("{n} series had no detected bars and were built from the legend; "
                           "their values were measured from pixels and are usually near zero."),
    # some tick labels were not read or were dropped, leaving gaps in the tick values
    "y_ticks_not_arithmetic": ("The y-axis tick values are not evenly spaced, probably because some "
                               "tick labels were missed. Bar values use the line through the ticks "
                               "that were read."),
    # y tick labels that disagreed with the others were ignored
    "y_ticks_dropped": "{n} y-axis tick label(s) were ignored because they disagreed with the rest.",
    # the title box reaches the image edge, so the title text may be cut off
    "title_touches_right_edge": "The chart title touches the right edge of the image and may be cut off.",
}

CUT_OFF_REASON = ("The {what} appears cut off at the image edge. Enter the full title, "
                  "or enter the same text to confirm it.")

NEXT_STEPS = ("Try a clearer or higher-resolution image, or write the chart by hand as "
              "a VerticalBarChart JSON file (see tests/fixtures/tactile/rainfall.json) "
              "and run `python -m src.tactile --chart FILE --output OUT.svg`.")

_CATEGORY_KEY = re.compile(r"category\[(\d+)\]")


@dataclass(frozen=True)
class MissingItem:
    field: str        # title, x_axis_title, y_axis_title, series_name[i] or category[j]
    reason: str
    cli_flag: str     # e.g. "--title" or "--category 2=TEXT"


@dataclass(frozen=True)
class Completeness:
    status: str
    notes: list[str] = field(default_factory=list)
    missing: list[MissingItem] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    forceable: bool = False   # every blocker is an ambiguity ``--force`` may skip


@dataclass(frozen=True)
class Overrides:
    """Text the user supplies. Bar values cannot be overridden."""

    title: str | None = None
    x_axis_title: str | None = None
    y_axis_title: str | None = None
    categories: dict[int, str] = field(default_factory=dict)
    series_names: dict[int, str] = field(default_factory=dict)


# A category this far (in years) from the median year is treated as a misread.
YEAR_SPREAD = 50
# A one-character category is suspicious when the other labels are this long.
SHORT_CATEGORY_MEDIAN = 4


def _suspicious_categories(spec: VerticalBarSpec) -> list[tuple[int, str]]:
    """Categories that read cleanly but look like OCR misreads, skipping any the user supplied."""
    supplied = {int(f.split(":", 1)[1]) for f in spec.flags if f.startswith("category_supplied:")}
    cats = spec.categories
    found: list[tuple[int, str]] = []
    if cats and statistics.median(len(c.strip()) for c in cats) >= SHORT_CATEGORY_MEDIAN:
        found += [(j, f"Category {j} ({c!r}) is a single character but the other categories "
                      "are longer, so it was probably misread.")
                  for j, c in enumerate(cats) if len(c.strip()) == 1]
    if cats and all(re.fullmatch(r"\d{4}", c) for c in cats):
        middle = statistics.median(int(c) for c in cats)
        found += [(j, f"Category {j} ({c!r}) is more than {YEAR_SPREAD} years from the "
                      f"median year ({middle:g}), so it was probably misread.")
                  for j, c in enumerate(cats) if abs(int(c) - middle) > YEAR_SPREAD]
    return [(j, reason) for j, reason in found if j not in supplied]


# A title whose ink runs to the right edge of the image (chart title) or whose
# box touches the top edge (rotated y-axis title) was cut off by the image.
EDGE_COLUMNS = 3      # image columns checked for ink at the right edge
EDGE_INK_PIXELS = 2   # dark pixels in those columns that count as ink
TOP_EDGE_PIXELS = 2   # a y-axis title box starting this close to the top is cut off
_DARK = 128           # greyscale value below which a pixel is ink


def cut_off_flags(frame, image) -> list[str]:
    """Flags for titles the image edge cuts off: ``title_cut_off``, ``y_axis_title_cut_off``.

    ``image`` is a BGR array as ``cv2.imread`` returns it. Uses the same title
    detections as ``reconstruct_vbar`` (best detection at its confidence floor).
    """
    from src.reconstruct import MIN_CONFIDENCE, _best, _group_roles

    roles = _group_roles(frame, MIN_CONFIDENCE)
    flags = []
    title = _best(roles.get("chart_title", []))
    if title is not None:
        gray = image[:, :, :3] @ np.array([0.114, 0.587, 0.299])
        rows = gray[int(title.box.y1):math.ceil(title.box.y2), -EDGE_COLUMNS:]
        if int((rows < _DARK).sum()) >= EDGE_INK_PIXELS:
            flags.append("title_cut_off")
    y_title = _best(roles.get("y_axis_title", []))
    if y_title is not None and y_title.box.y1 <= TOP_EDGE_PIXELS:
        flags.append("y_axis_title_cut_off")
    return flags


def _flag_name(flag: str) -> str:
    return flag.split(":", 1)[0]


def _unreadable_categories(spec: VerticalBarSpec) -> list[int]:
    marked = {int(m.group(1)) for key in spec.confidence if (m := _CATEGORY_KEY.fullmatch(key))}
    marked |= {j for j, text in enumerate(spec.categories) if not text.strip()}
    return sorted(j for j in marked if j < len(spec.categories))


def _unencodable(text: str) -> str | None:
    """Characters the layout's braille encoder rejects in ``text``, or None if it encodes."""
    try:
        to_braille(text)
        return None
    except ValueError as exc:
        bad = sorted({ch for ch in text if not ch.isspace() and _rejects(ch)})
        return " ".join(repr(ch) for ch in bad) if bad else str(exc)


def _rejects(char: str) -> bool:
    try:
        to_braille(char)
        return False
    except ValueError:
        return True


def check_completeness(spec: VerticalBarSpec) -> Completeness:
    blockers: list[str] = []
    forceable = True
    for flag in spec.flags:
        message = BLOCKING_FLAGS.get(_flag_name(flag))
        if message:
            blockers.append(f"{message} ({_flag_name(flag)}) {NEXT_STEPS}")
            forceable = forceable and _flag_name(flag) in FORCEABLE_FLAGS
    if not spec.categories:
        blockers.append(f"No category labels were read. {NEXT_STEPS}")
        forceable = False
    if any(value is None for series in spec.series for value in series.values):
        blockers.append(f"Some bar values could not be read. {NEXT_STEPS}")
        forceable = False

    notes = []
    for flag in spec.flags:
        name, _, count = flag.partition(":")
        if name == "value_inferred_zero":
            count = sum(1 for series in spec.series
                        for value, rescued in zip(series.values, series.value_rescued)
                        if rescued and value == 0.0)
        if name in NOTE_FLAGS:
            notes.append(NOTE_FLAGS[name].format(n=count))

    if blockers:
        return Completeness(CANNOT_BUILD, notes, [], blockers, forceable)

    missing: list[MissingItem] = []
    if spec.title is None or "title_missing" in spec.flags:
        missing.append(MissingItem("title", "No chart title was read.", "--title"))
    elif "title_cut_off" in spec.flags:
        missing.append(MissingItem("title", CUT_OFF_REASON.format(what="chart title"), "--title"))
    if spec.x_axis_title is None or "x_axis_title_missing" in spec.flags:
        missing.append(MissingItem("x_axis_title", "No x-axis title was read.", "--x-title"))
    if spec.y_axis_title is None or "y_axis_title_missing" in spec.flags:
        missing.append(MissingItem("y_axis_title", "No y-axis title was read.", "--y-title"))
    elif "y_axis_title_cut_off" in spec.flags:
        missing.append(MissingItem("y_axis_title", CUT_OFF_REASON.format(what="y-axis title"),
                                   "--y-title"))
    for j in _unreadable_categories(spec):
        shown = spec.categories[j] or "(empty)"
        missing.append(MissingItem(
            f"category[{j}]", f"Category {j} was unreadable (read as {shown!r}).",
            f"--category {j}=TEXT"))
    listed_now = {m.field for m in missing}
    for j, reason in _suspicious_categories(spec):
        if f"category[{j}]" not in listed_now:
            missing.append(MissingItem(f"category[{j}]", reason, f"--category {j}=TEXT"))
    if len(spec.series) > 1:
        for i, series in enumerate(spec.series):
            if not series.name:
                colour = f" (colour {series.source_color})" if series.source_color else ""
                missing.append(MissingItem(
                    f"series_name[{i}]", f"Series {i}{colour} has no legend name.",
                    f"--series {i}=NAME"))

    # Every string the layout encodes must pass the layout's own braille encoder.
    listed = {m.field for m in missing}
    strings = [("title", "The chart title", "--title", spec.title),
               ("x_axis_title", "The x-axis title", "--x-title", spec.x_axis_title),
               ("y_axis_title", "The y-axis title", "--y-title", spec.y_axis_title)]
    strings += [(f"category[{j}]", f"Category {j}", f"--category {j}=TEXT", text)
                for j, text in enumerate(spec.categories)]
    if len(spec.series) > 1:
        strings += [(f"series_name[{i}]", f"Series {i} name", f"--series {i}=NAME", s.name)
                    for i, s in enumerate(spec.series)]
    for name, label, cli_flag, text in strings:
        if name in listed or not text:
            continue
        bad = _unencodable(text)
        if bad:
            missing.append(MissingItem(
                name, f"{label} ({text!r}) has characters that cannot be written in "
                      f"braille: {bad}. It was probably misread.", cli_flag))
    return Completeness(NEEDS_INPUT if missing else READY, notes, missing, [], False)


def completeness_from_error(error: ChartInterpretationError,
                            frame: DetectionFrame | None = None,
                            min_confidence: float = 0.3) -> Completeness:
    """Turn a reconstruction failure into a plain-language ``cannot_build``."""
    def count(role: str) -> int:
        return sum(1 for d in frame.detections
                   if d.role == role and d.confidence >= min_confidence) if frame else 0

    text = str(error)
    if "no x tick labels" in text:
        message = "No x tick labels were found."
        titles = count("x_axis_title")
        if titles:
            message += f" {titles} label(s) may have been detected as axis titles."
    elif "fewer than 2 readable y tick labels" in text or "y-axis calibration" in text:
        message = "Fewer than 2 y-axis tick labels could be read, so bar heights cannot be turned into values."
    elif "no vertical bars" in text:
        message = "No vertical bars were found in the plot area."
        if count("horizontal_bar"):
            message += " The chart may have horizontal bars."
    else:
        message = f"The chart could not be interpreted ({text})."
    return Completeness(CANNOT_BUILD, [], [], [f"{message} {NEXT_STEPS}"], False)


def apply_overrides(spec: VerticalBarSpec,
                    overrides: Overrides) -> tuple[VerticalBarSpec, list[str]]:
    """Return a copy of ``spec`` with user-supplied text filled in.

    Matching flags and per-category confidences are removed. Flags that make a
    spec ``cannot_build`` are never touched.
    """
    for label, value in (("title", overrides.title), ("x_axis_title", overrides.x_axis_title),
                         ("y_axis_title", overrides.y_axis_title),
                         *((f"category {j}", t) for j, t in overrides.categories.items()),
                         *((f"series {i}", t) for i, t in overrides.series_names.items())):
        if value is not None and not value.strip():
            raise ValueError(f"override for {label} is empty")
    for j in overrides.categories:
        if not 0 <= j < len(spec.categories):
            raise ValueError(f"category index {j} is out of range (0 to {len(spec.categories) - 1})")
    for i in overrides.series_names:
        if not 0 <= i < len(spec.series):
            raise ValueError(f"series index {i} is out of range (0 to {len(spec.series) - 1})")

    flags = list(spec.flags)
    confidence = dict(spec.confidence)
    notes: list[str] = []

    def drop(*names: str) -> None:
        flags[:] = [f for f in flags if f not in names]

    title, x_title, y_title = spec.title, spec.x_axis_title, spec.y_axis_title
    if overrides.title is not None:
        title = overrides.title
        drop("title_missing", "title_touches_right_edge", "title_cut_off")
        confidence.pop("title", None)
        notes.append("chart title supplied by user")
    if overrides.x_axis_title is not None:
        x_title = overrides.x_axis_title
        drop("x_axis_title_missing")
        confidence.pop("x_axis_title", None)
        notes.append("x-axis title supplied by user")
    if overrides.y_axis_title is not None:
        y_title = overrides.y_axis_title
        drop("y_axis_title_missing", "y_axis_title_cut_off")
        confidence.pop("y_axis_title", None)
        notes.append("y-axis title supplied by user")

    categories = list(spec.categories)
    for j, text in sorted(overrides.categories.items()):
        categories[j] = text
        confidence.pop(f"category[{j}]", None)
        flags.append(f"category_supplied:{j}")
        notes.append(f"category {j} supplied by user: {text!r}")
    if overrides.categories and not any(_CATEGORY_KEY.fullmatch(k) for k in confidence):
        drop("category_unreadable")

    series = list(spec.series)
    for i, name in sorted(overrides.series_names.items()):
        series[i] = replace(series[i], name=name)
        notes.append(f"series {i} name supplied by user: {name!r}")

    return replace(spec, title=title, x_axis_title=x_title, y_axis_title=y_title,
                   categories=categories, series=series, flags=flags,
                   confidence=confidence), notes


# Negative values smaller than this share of the chart maximum are pixel-offset
# noise from measuring a bar at the zero line, not real data.
NEGATIVE_NOISE_FRACTION = 0.005


def spec_to_chart(spec: VerticalBarSpec) -> VerticalBarChart:
    """Build the layout's chart from a spec whose status is ``ready``."""
    return spec_to_chart_with_notes(spec)[0]


def spec_to_chart_with_notes(spec: VerticalBarSpec) -> tuple[VerticalBarChart, list[str]]:
    """Like ``spec_to_chart``, also returning notes about values it had to adjust.

    Negative values under 0.5% of the chart maximum become 0.0. Larger
    negatives are left alone, so the layout still rejects them.
    """
    result = check_completeness(spec)
    if result.status != READY:
        raise ValueError(
            f"spec is not ready ({result.status}); resolve it before building a chart: "
            + "; ".join(result.blockers or [f"{m.field}: {m.reason}" for m in result.missing]))
    biggest = max(abs(v) for series in spec.series for v in series.values)
    limit = NEGATIVE_NOISE_FRACTION * biggest
    zeroed = 0
    series_out = []
    for series in spec.series:
        values = []
        for value in series.values:
            if biggest and -limit < value < 0:
                value, zeroed = 0.0, zeroed + 1
            values.append(float(value))
        series_out.append(BarSeries(series.name or None, tuple(values)))
    notes = []
    if zeroed:
        notes.append(f"{zeroed} value(s) slightly below zero (under 0.5% of the chart maximum) "
                     "were treated as 0.")
    chart = VerticalBarChart(
        spec.source_key, spec.title, spec.x_axis_title, spec.y_axis_title,
        tuple(spec.categories), tuple(series_out))
    return chart, notes
