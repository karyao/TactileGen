"""Ground-truth frames from plots_detection annotations."""

import unittest

from src.dataset.pmc_frames import ground_truth_frames

IMAGE = {"width": 100, "height": 80}


def ann(i, category, bbox, text=None):
    out = {"id": i, "category_id": category, "bbox": bbox}
    if text is not None:
        out["text"] = text
    return out


class GroundTruthFramesTest(unittest.TestCase):
    def test_roles_text_and_confidence(self):
        det, ocr = ground_truth_frames("val/a.jpg", IMAGE, [
            ann(1, 19, [10, 20, 10, 40]),
            ann(2, 5, [8, 62, 14, 10], "A"),
            ann(3, 6, [0, 20, 8, 8], "50"),
            ann(4, 9, [8, 22, 3, 3]),
            ann(5, 15, [10, 14, 10, 6], "42"),
            ann(6, 4, [0, 0, 5, 5], "other is skipped"),
        ])
        self.assertEqual({d.role for d in det.detections},
                         {"vertical_bar", "x_tick_label", "y_tick_label", "y_tick", "value_label"})
        self.assertTrue(all(d.confidence == 1.0 for d in det.detections))
        self.assertEqual({r.detection_id: r.text for r in ocr.regions}, {"gt2": "A", "gt3": "50", "gt5": "42"})
        self.assertEqual(det.source_key, ocr.source_key)

    def test_boxes_are_clipped_and_degenerate_boxes_dropped(self):
        det, _ = ground_truth_frames("val/a.jpg", IMAGE, [
            ann(1, 19, [90, 70, 30, 30]),     # runs off the corner
            ann(2, 19, [10, 10, 0.4, 20]),    # under a pixel wide
            ann(3, 19, [150, 10, 10, 10]),    # entirely outside
        ])
        self.assertEqual(len(det.detections), 1)
        box = det.detections[0].box
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (90, 70, 100, 80))


if __name__ == "__main__":
    unittest.main()
