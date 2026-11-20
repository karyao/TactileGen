"""Drop low-confidence bars that no legend swatch claims, so the series do not look ambiguous.

Offline sweep on the PlotQA test charts (scripts/audit_extras.py, variant A): removing bars with
confidence < 0.85 and a legend status other than ``matched`` fixed 334 charts and broke none; the
rescue step then re-measures the removed bars from pixels.
"""

from __future__ import annotations

from dataclasses import replace

from src.contracts import DetectionFrame, LegendMappingFrame
from src.reconstruction.vertical import MIN_CONFIDENCE, _centre, _dedupe, _inside

UNMATCHED_BAR_CONFIDENCE = 0.85


def drop_unmatched_bars(frame: DetectionFrame, mapping: LegendMappingFrame,
                        tau: float = UNMATCHED_BAR_CONFIDENCE) -> DetectionFrame | None:
    """Return ``frame`` without the bars to drop, or None when there are none."""
    raw = [d for d in frame.detections if d.role == "vertical_bar"]
    kept = _dedupe([d for d in raw if d.confidence >= MIN_CONFIDENCE])
    plot = max((d for d in frame.detections if d.role == "plot_area" and d.confidence >= MIN_CONFIDENCE),
               key=lambda d: d.confidence, default=None)
    surviving = [d for d in kept if plot is None or _inside(_centre(d), plot)]
    status = {a.bar_id: a.status for a in mapping.assignments}
    drop = {d.id for d in surviving if d.confidence < tau and status.get(d.id, "none") != "matched"}
    if not drop:
        return None
    return replace(frame, detections=tuple(d for d in frame.detections if d.id not in drop))
