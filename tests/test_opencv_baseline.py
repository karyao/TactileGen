"""Evaluator contract and a small image-level detector check."""

import tempfile
import unittest
from pathlib import Path

from src.baseline.opencv_bars import Box, detect_vertical_bars, load_yolo_boxes, match_boxes
from src.baseline.opencv_charts import predict_chart


class BaselineScoringTest(unittest.TestCase):
    def test_one_to_one_matching(self):
        actual = [Box(10, 10, 30, 80), Box(50, 10, 70, 80)]
        predicted = [Box(10, 10, 30, 80), Box(11, 11, 31, 81)]
        self.assertEqual(match_boxes(predicted, actual), 1)

    def test_yolo_decoder_filters_other_chart_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.txt"
            path.write_text("1 0.2 0.5 0.1 0.4\n4 0.5 0.1 0.5 0.1\n", encoding="utf-8")
            boxes = load_yolo_boxes(path, 1, (100, 100), 15)
            self.assertEqual(len(boxes), 1)
            self.assertAlmostEqual(boxes[0].x1, 15)
            self.assertAlmostEqual(boxes[0].y2, 70)

    def test_missing_label_is_error(self):
        with self.assertRaises(FileNotFoundError):
            load_yolo_boxes(Path("/missing/0.txt"), 1, (100, 100), 15)

    def test_detector_uses_image_pixels(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are not installed")
        image = np.full((400, 500, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 30), (50, 350), (0, 0, 0), 2)
        cv2.line(image, (50, 350), (450, 350), (0, 0, 0), 2)
        cv2.rectangle(image, (100, 150), (124, 349), (255, 0, 0), -1)
        cv2.rectangle(image, (200, 100), (224, 349), (0, 0, 0), -1)
        boxes = detect_vertical_bars(image)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(any(abs(box.x1 - 100) <= 2 for box in boxes))
        self.assertTrue(any(abs(box.x1 - 200) <= 2 for box in boxes))

    def test_four_type_router_uses_image_geometry(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are not installed")
        for kind in ("hbar_categorical", "vbar_categorical", "dot_line", "line"):
            with self.subTest(kind=kind):
                image = np.full((400, 500, 3), 255, dtype=np.uint8)
                cv2.line(image, (50, 30), (50, 350), (0, 0, 0), 2)
                cv2.line(image, (50, 350), (450, 350), (0, 0, 0), 2)
                if kind == "hbar_categorical":
                    cv2.rectangle(image, (51, 80), (250, 103), (255, 0, 0), -1)
                    cv2.rectangle(image, (51, 170), (350, 193), (0, 0, 0), -1)
                elif kind == "vbar_categorical":
                    cv2.rectangle(image, (100, 150), (124, 349), (255, 0, 0), -1)
                    cv2.rectangle(image, (200, 100), (224, 349), (0, 0, 0), -1)
                elif kind == "dot_line":
                    for x, y in ((100, 100), (180, 130), (260, 90), (340, 150)):
                        cv2.circle(image, (x, y), 7, (0, 100, 230), -1)
                else:
                    cv2.line(image, (90, 100), (180, 120), (255, 0, 0), 2)
                    cv2.line(image, (180, 120), (270, 90), (255, 0, 0), 2)
                    cv2.line(image, (270, 90), (360, 150), (255, 0, 0), 2)
                predicted_type, boxes = predict_chart(image)
                self.assertEqual(predicted_type, kind)
                self.assertGreaterEqual(len(boxes), 2)


if __name__ == "__main__":
    unittest.main()
