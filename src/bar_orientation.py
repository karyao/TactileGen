"""Select bar orientation from raw detections without inference dependencies.

Resolved means an orientation was selected, not that a chart is reconstructable.
Scores include duplicate and outside-plot detections and depend on upstream YOLO
settings. These thresholds are heuristics, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum
from typing import Literal, cast

from src.contracts import DetectionFrame


Orientation = Literal["vertical", "horizontal"]
OrientationPreference = Literal["auto", "vertical", "horizontal"]
OrientationStatus = Literal["resolved", "uncertain"]
OrientationSource = Literal["automatic", "manual"]
OrientationReason = Literal[
    "no_qualifying_bars", "insufficient_evidence", "ambiguous_orientation",
    "automatic_selection", "requested_orientation_unavailable", "manual_selection",
]

# Currently matches reconstruct.MIN_CONFIDENCE. Review both constants whenever
# confidence policy changes; importing reconstruction here would add dependencies.
MIN_BAR_CONFIDENCE = 0.30
MIN_ORIENTATION_SCORE = 0.60
MIN_ORIENTATION_SHARE = 0.75


@dataclass(frozen=True)
class BarOrientationResult:
    """One decision, with raw evidence and an effective frame only if resolved.

    ``source`` describes the decision mode, including unsuccessful manual requests.
    Uncertain results have no orientation/frame and suppress no detections.
    """

    preference: OrientationPreference
    orientation: Orientation | None
    status: OrientationStatus
    source: OrientationSource
    reason: OrientationReason
    vertical_score: float
    horizontal_score: float
    suppressed_ids: tuple[str, ...]
    effective_frame: DetectionFrame | None


def resolve_bar_orientation(
    frame: DetectionFrame, preference: str = "auto",
) -> BarOrientationResult:
    """Select an orientation and remove opposite bars without mutating raw input.

    Preference must be exactly auto, vertical, or horizontal. Manual requests
    require a bar at the confidence floor but bypass the score/share thresholds.
    Geometry and non-bar roles never contribute to the scores. Retained detection
    objects, including low-confidence selected bars, are reused in original order.
    """
    if not isinstance(preference, str) or preference not in ("auto", "vertical", "horizontal"):
        raise ValueError("preference must be 'auto', 'vertical', or 'horizontal'")

    vertical_score = fsum(d.confidence for d in frame.detections
                          if d.role == "vertical_bar" and d.confidence >= MIN_BAR_CONFIDENCE)
    horizontal_score = fsum(d.confidence for d in frame.detections
                            if d.role == "horizontal_bar" and d.confidence >= MIN_BAR_CONFIDENCE)
    orientation: Orientation | None = None
    reason: OrientationReason
    source: OrientationSource = "automatic" if preference == "auto" else "manual"
    if preference == "auto":
        winning_score = max(vertical_score, horizontal_score)
        if winning_score == 0:
            reason = "no_qualifying_bars"
        elif winning_score < MIN_ORIENTATION_SCORE:
            reason = "insufficient_evidence"
        elif winning_score < MIN_ORIENTATION_SHARE * (vertical_score + horizontal_score):
            reason = "ambiguous_orientation"
        else:
            orientation = "vertical" if vertical_score > horizontal_score else "horizontal"
            reason = "automatic_selection"
    else:
        requested_score = vertical_score if preference == "vertical" else horizontal_score
        if requested_score == 0:
            reason = "requested_orientation_unavailable"
        else:
            orientation = cast(Orientation, preference)
            reason = "manual_selection"

    effective_frame = None
    suppressed_ids: tuple[str, ...] = ()
    if orientation is not None:
        opposite_role = "horizontal_bar" if orientation == "vertical" else "vertical_bar"
        suppressed_ids = tuple(d.id for d in frame.detections if d.role == opposite_role)
        effective_frame = replace(
            frame, detections=tuple(d for d in frame.detections if d.role != opposite_role),
        )

    return BarOrientationResult(
        preference=cast(OrientationPreference, preference),
        orientation=orientation,
        status="resolved" if orientation is not None else "uncertain",
        source=source,
        reason=reason,
        vertical_score=vertical_score,
        horizontal_score=horizontal_score,
        suppressed_ids=suppressed_ids,
        effective_frame=effective_frame,
    )
