"""Helpers of the OCRBackend scoring script (the backend itself is not part of the repository)."""

import unittest

from src.baseline.ocr_backend_eval import ROLE_BY_CLASS, clip
from src.baseline.ocr_eval import CLASSES
from src.contracts import ROLES, TEXT_ROLES


class BackendEvalHelpersTest(unittest.TestCase):
    def test_every_scored_class_maps_to_a_text_role(self):
        self.assertEqual(set(ROLE_BY_CLASS), set(CLASSES.values()))
        for role in ROLE_BY_CLASS.values():
            self.assertIn(role, ROLES)
            self.assertIn(role, TEXT_ROLES)

    def test_clip_keeps_boxes_inside_the_image_and_drops_degenerate_ones(self):
        box = clip((90, 70, 30, 30), 100, 80)
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (90, 70, 100, 80))
        self.assertIsNone(clip((10, 10, 0.4, 20), 100, 80))     # under a pixel wide
        self.assertIsNone(clip((150, 10, 10, 10), 100, 80))     # entirely outside


if __name__ == "__main__":
    unittest.main()
