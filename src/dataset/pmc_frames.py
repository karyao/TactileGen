"""Ground-truth DetectionFrame and OCRFrame for a PMC bar chart (plots_detection COCO).

Roles assume a vertical chart: ``xlabel`` are the category labels along the bottom and
``ylabel`` the value-axis labels at the left. (In the dataset ``ylabel`` always means the
value axis; for horizontal charts these roles would swap.) Boxes are clipped to the image
and dropped when they keep less than a pixel of width or height. ``other``, ``mark_label``,
``tick_grouping``, ``legend_title`` and the axis-area classes have no role here.
"""
from src.contracts import Box, Detection, DetectionFrame, OCRFrame, RecognizedText, TEXT_ROLES

ROLE_BY_COCO_CATEGORY = {
    1: "x_axis_title", 2: "y_axis_title", 3: "plot_area", 5: "x_tick_label", 6: "y_tick_label",
    7: "chart_title", 8: "x_tick", 9: "y_tick", 10: "legend_swatch", 11: "legend_label",
    13: "legend", 15: "value_label", 19: "vertical_bar",
}


def ground_truth_frames(source_key: str, image: dict, annotations: list[dict]):
    """-> (DetectionFrame, OCRFrame) with confidence 1.0 everywhere.

    ``image`` has ``width`` and ``height``; each annotation has ``id``, ``category_id``,
    ``bbox`` (x, y, width, height) and, for text classes, ``text``.
    """
    width, height = image["width"], image["height"]
    detections, regions = [], []
    for a in annotations:
        role = ROLE_BY_COCO_CATEGORY.get(a["category_id"])
        if role is None:
            continue
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        detection = Detection(f"gt{a['id']}", role, Box(x1, y1, x2, y2), 1.0)
        detections.append(detection)
        if role in TEXT_ROLES and "text" in a:
            regions.append(RecognizedText(detection.id, a["text"], 1.0))
    return (DetectionFrame(source_key, width, height, tuple(detections)),
            OCRFrame(source_key, tuple(regions)))
