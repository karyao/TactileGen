"""In-memory legend-to-bar matcher behavior and boundary coverage."""

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.contract_io import load_detection_frame, load_ocr_frame
from src.contracts import Box, Detection, DetectionFrame, OCRFrame, RecognizedText
from src.legend_matcher import (
    _classify_legend_scores,
    _crop,
    _pair_legend_items,
    _score_legend_to_bars,
    match_legend_to_bars,
)


FIXTURES = Path(__file__).parent / "fixtures/contracts"


def detection(identifier, role, box):
    return Detection(identifier, role, Box(*box), 0.9)


def frame(*items, width=80, height=60):
    return DetectionFrame("input/chart.png", width, height, items)


class LegendMatcherTest(unittest.TestCase):
    def test_crop_uses_proportional_inset_and_rejects_consumed_interiors(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        large = detection("large", "vertical_bar", (5, 5, 25, 25))
        tiny = detection("tiny", "vertical_bar", (5, 5, 7, 7))
        self.assertEqual(_crop(image, large).shape, (14, 14, 3))
        self.assertIsNone(_crop(image, tiny))

    def test_checked_in_fixture_matches_bars_without_running_ocr(self):
        detections = load_detection_frame(FIXTURES / "detections.json")
        ocr = load_ocr_frame(FIXTURES / "ocr.json")
        image = cv2.imread(str(FIXTURES / "val/example.png"))

        mapping = match_legend_to_bars(image, detections, ocr)

        self.assertEqual(
            [(item.bar_id, item.series_name, item.status) for item in mapping.assignments],
            [
                ("bar_a_n", "N", "matched"),
                ("bar_a_s", "S", "matched"),
                ("bar_b_n", "N", "matched"),
                ("bar_b_s", "S", "matched"),
            ],
        )

    def test_normal_absence_and_ocr_failure_return_one_row_per_bar(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(detection("bar", "vertical_bar", (5, 5, 15, 50)))
        available = match_legend_to_bars(
            image, detections, OCRFrame("input/chart.png", ())
        )
        unavailable = match_legend_to_bars(image, detections, None)
        self.assertEqual(available.assignments[0].status, "no_legend")
        self.assertEqual(unavailable.assignments[0].status, "ocr_unavailable")

    def test_single_candidate_matches_and_strips_ocr_text(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[11:19, 11:19] = (10, 20, 30)
        image[21:49, 11:19] = (10, 20, 30)
        detections = frame(
            detection("swatch", "legend_swatch", (10, 10, 20, 20)),
            detection("label", "legend_label", (22, 10, 35, 20)),
            detection("bar", "vertical_bar", (10, 20, 20, 50)),
        )
        ocr = OCRFrame("input/chart.png", (RecognizedText("label", "  North  ", 1),))
        assignment = match_legend_to_bars(image, detections, ocr).assignments[0]
        self.assertEqual(assignment.status, "matched")
        self.assertEqual(assignment.series_name, "North")

    def test_tight_single_parent_keeps_outlying_child_eligible(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[11:19, 6:14] = (10, 20, 30)
        image[21:49, 6:14] = (10, 20, 30)
        detections = frame(
            detection("legend", "legend", (15, 5, 40, 25)),
            detection("swatch", "legend_swatch", (5, 10, 15, 20)),
            detection("label", "legend_label", (17, 10, 35, 20)),
            detection("bar", "vertical_bar", (5, 20, 15, 50)),
        )
        ocr = OCRFrame(
            "input/chart.png", (RecognizedText("label", "North", 1),)
        )

        assignment = match_legend_to_bars(image, detections, ocr).assignments[0]

        self.assertEqual(assignment.status, "matched")
        self.assertEqual(assignment.series_name, "North")

    def test_multiple_rows_match_and_duplicate_colors_abstain(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[6:14, 6:14] = (10, 20, 30)
        image[21:29, 6:14] = (40, 50, 60)
        image[6:44, 41:49] = (10, 20, 30)
        image[6:44, 56:64] = (40, 50, 60)
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)),
            detection("l1", "legend_label", (17, 5, 30, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)),
            detection("l2", "legend_label", (17, 20, 30, 30)),
            detection("b1", "vertical_bar", (40, 5, 50, 45)),
            detection("b2", "vertical_bar", (55, 5, 65, 45)),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("l1", "North", 1),
            RecognizedText("l2", "South", 1),
        ))

        assignments = match_legend_to_bars(image, detections, ocr).assignments

        self.assertEqual(
            [(item.series_name, item.status) for item in assignments],
            [("North", "matched"), ("South", "matched")],
        )
        image[21:29, 6:14] = (10, 20, 30)
        duplicate = match_legend_to_bars(image, detections, ocr).assignments[0]
        self.assertEqual(duplicate.status, "ambiguous")
        self.assertIsNone(duplicate.series_name)

    def test_partial_ocr_never_assigns_an_unreadable_series_another_name(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[6:14, 6:14] = (10, 20, 30)
        image[21:29, 6:14] = (40, 50, 60)
        image[6:44, 41:49] = (10, 20, 30)
        image[6:44, 56:64] = (40, 50, 60)
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)),
            detection("l1", "legend_label", (17, 5, 30, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)),
            detection("l2", "legend_label", (17, 20, 30, 30)),
            detection("b1", "vertical_bar", (40, 5, 50, 45)),
            detection("b2", "vertical_bar", (55, 5, 65, 45)),
        )
        partial_ocr = OCRFrame(
            "input/chart.png", (RecognizedText("l1", "North", 1),)
        )

        assignments = match_legend_to_bars(
            image, detections, partial_ocr
        ).assignments

        self.assertEqual(assignments[0].status, "matched")
        self.assertEqual(assignments[0].series_name, "North")
        self.assertEqual(assignments[1].status, "unreadable_label")
        self.assertIsNone(assignments[1].series_name)

        image[21:29, 6:14] = (10, 20, 30)
        ambiguous = match_legend_to_bars(
            image, detections, partial_ocr
        ).assignments[0]
        self.assertEqual(ambiguous.status, "ambiguous")

    def test_pairing_maximizes_cardinality_before_geometry_cost(self):
        detections = frame(
            detection("s1", "legend_swatch", (0, 0, 10, 10)),
            detection("s2", "legend_swatch", (0, 9, 10, 19)),
            detection("l1", "legend_label", (10, 8, 20, 12)),
            detection("l2", "legend_label", (11, 0, 20, 5)),
            width=40,
            height=30,
        )

        pairs, status = _pair_legend_items(detections)

        self.assertIsNone(status)
        self.assertEqual(
            [(swatch.id, label.id) for swatch, label in pairs],
            [("s1", "l2"), ("s2", "l1")],
        )

    def test_blank_or_absent_label_text_is_unreadable(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(
            detection("swatch", "legend_swatch", (10, 10, 20, 20)),
            detection("label", "legend_label", (22, 10, 35, 20)),
            detection("bar", "vertical_bar", (10, 20, 20, 50)),
        )
        for regions in ((), (RecognizedText("label", " \n ", 0),)):
            with self.subTest(regions=regions):
                mapping = match_legend_to_bars(
                    image, detections, OCRFrame("input/chart.png", regions)
                )
                self.assertEqual(mapping.assignments[0].status, "unreadable_label")

    def test_unusable_swatch_candidate_prevents_unsafe_partial_matching(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(
            detection("usable", "legend_swatch", (5, 5, 15, 15)),
            detection("usable-label", "legend_label", (17, 5, 30, 15)),
            detection("tiny", "legend_swatch", (5, 20, 7, 22)),
            detection("tiny-label", "legend_label", (9, 20, 25, 25)),
            detection("bar", "vertical_bar", (40, 5, 50, 50)),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("usable-label", "A", 1),
            RecognizedText("tiny-label", "B", 1),
        ))

        assignment = match_legend_to_bars(image, detections, ocr).assignments[0]

        self.assertEqual(assignment.status, "unusable_crop")

    def test_threshold_boundaries_and_candidate_diagnostics(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)),
            detection("l1", "legend_label", (17, 5, 25, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)),
            detection("l2", "legend_label", (17, 20, 25, 30)),
            detection("bar", "vertical_bar", (40, 5, 50, 50)),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("l1", "A", 1), RecognizedText("l2", "B", 1),
        ))
        with patch("src.legend_matcher._candidate_error", side_effect=[0.05, 0.07]):
            matched = match_legend_to_bars(
                image, detections, ocr, max_error=0.05, min_margin=0.02
            ).assignments[0]
        self.assertEqual(matched.status, "matched")
        with patch("src.legend_matcher._candidate_error", side_effect=[0.05, 0.069]):
            ambiguous = match_legend_to_bars(
                image, detections, ocr, max_error=0.05, min_margin=0.02
            ).assignments[0]
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.series_name)
        self.assertEqual(ambiguous.swatch_id, "s1")
        with patch("src.legend_matcher._candidate_error", side_effect=[0.051, 0.20]):
            weak = match_legend_to_bars(
                image, detections, ocr, max_error=0.05, min_margin=0.02
            ).assignments[0]
        self.assertEqual(weak.status, "weak_match")

    def test_reusable_scores_match_direct_classification_across_grid(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)),
            detection("l1", "legend_label", (17, 5, 25, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)),
            detection("l2", "legend_label", (17, 20, 25, 30)),
            detection("bar", "vertical_bar", (40, 5, 50, 50)),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("l1", "A", 1), RecognizedText("l2", "B", 1),
        ))
        scored = _score_legend_to_bars(image, detections, ocr)
        for max_error in (0.03, 0.05, 0.10):
            for min_margin in (0.01, 0.02, 0.03):
                with self.subTest(max_error=max_error, min_margin=min_margin):
                    direct = match_legend_to_bars(
                        image, detections, ocr,
                        max_error=max_error, min_margin=min_margin,
                    )
                    cached = _classify_legend_scores(
                        scored, max_error=max_error, min_margin=min_margin
                    )
                    self.assertEqual(cached, direct)

    def test_rejects_bad_images_sources_and_thresholds_without_mutation(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        original = image.copy()
        detections = frame()
        match_legend_to_bars(image, detections, OCRFrame("input/chart.png", ()))
        self.assertTrue(np.array_equal(image, original))
        with self.assertRaisesRegex(ValueError, "BGR"):
            match_legend_to_bars(image[:, :, 0], detections, None)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            match_legend_to_bars(image[:-1], detections, None)
        with self.assertRaisesRegex(ValueError, "different source"):
            match_legend_to_bars(
                image, detections, OCRFrame("other/chart.png", ())
            )
        for value in (True, -0.1, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    match_legend_to_bars(image, detections, None, max_error=value)


if __name__ == "__main__":
    unittest.main()
