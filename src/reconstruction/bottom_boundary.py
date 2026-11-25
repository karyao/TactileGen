"""Conservative shared-bottom boundary for vertical charts without a plot area."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

from src.contracts import DetectionFrame


MIN_CONFIDENCE = 0.3
MIN_SHARE = 0.75
WINDOW_HEIGHT_SHARE = 0.025
MAX_OUTLIER_WIDTH_SHARE = 0.6


@dataclass(frozen=True)
class BottomBoundary:
    frame: DetectionFrame
    baseline_y: float
    confidence_share: float
    retained_bar_ids: tuple[str, ...]
    rejected_bar_ids: tuple[str, ...]
    rejected_tick_ids: tuple[str, ...]


def filter_vertical_bottom(frame: DetectionFrame) -> BottomBoundary | None:
    """Keep a dominant bottom group only when other bars look like title artifacts."""
    if any(d.role == "plot_area" and d.confidence >= MIN_CONFIDENCE
           for d in frame.detections):
        return None
    bars = [d for d in frame.detections if d.role == "vertical_bar"]
    qualifying = sorted((d for d in bars if d.confidence >= MIN_CONFIDENCE),
                        key=lambda d: (d.box.y2, d.id))
    if len(qualifying) < 2:
        return None
    window = WINDOW_HEIGHT_SHARE * frame.height
    groups = [tuple(d for d in qualifying if start.box.y2 <= d.box.y2 <= start.box.y2 + window)
              for start in qualifying]
    group = max(groups, key=lambda ds: (sum(d.confidence for d in ds), len(ds),
                                        -min(d.box.y2 for d in ds)))
    total = sum(d.confidence for d in qualifying)
    share = sum(d.confidence for d in group) / total
    if len(group) < 2 or share < MIN_SHARE:
        return None
    lower = min(d.box.y2 for d in group)
    upper = lower + window
    retained = [d for d in bars if lower <= d.box.y2 <= upper]
    rejected = [d for d in bars if d not in retained]
    median_width = median(d.box.x2 - d.box.x1 for d in group)
    highest_top = min(d.box.y1 for d in group)
    if any(d.box.x2 - d.box.x1 >= MAX_OUTLIER_WIDTH_SHARE * median_width
           or d.box.y2 >= highest_top for d in rejected):
        return None
    left = min(d.box.x1 for d in retained) - median_width
    right = max(d.box.x2 for d in retained) + median_width
    rejected_ticks = [d for d in frame.detections if d.role == "x_tick_label"
                      and not left <= (d.box.x1 + d.box.x2) / 2 <= right]
    if not rejected and not rejected_ticks:
        return None
    removed_ids = {d.id for d in (*rejected, *rejected_ticks)}
    return BottomBoundary(
        frame=replace(frame, detections=tuple(d for d in frame.detections
                                              if d.id not in removed_ids)),
        baseline_y=median(d.box.y2 for d in group),
        confidence_share=share,
        retained_bar_ids=tuple(d.id for d in retained),
        rejected_bar_ids=tuple(d.id for d in rejected),
        rejected_tick_ids=tuple(d.id for d in rejected_ticks),
    )
