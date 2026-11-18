"""Concrete pass-through filter results for tests of filter consumers."""
from src.contracts import LegendBarAssignment, LegendMappingFrame
from src.reconstruction.legend_matcher import LegendFilterResult


def passthrough_filter(image, frame, ocr, **kwargs):
    mapping = LegendMappingFrame(frame.source_key, tuple(
        LegendBarAssignment(d.id) for d in frame.detections
        if d.role == kwargs.get("bar_role", "vertical_bar")
    ))
    return LegendFilterResult(frame, mapping)
