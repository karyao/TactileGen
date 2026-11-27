"""Legend matching and baseline validation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.baseline.legend_matching import (
    TruthChart,
    _report,
    _score_end_to_end,
    _score_legend_to_bars as baseline_score_legend_to_bars,
    calibrate,
    evaluate,
    load_truth_charts,
)
from src.contract_io import load_detection_frame, load_ocr_frame
from src.contracts import (
    Box,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendEntry,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
)
from src.reconstruction.legend_matcher import (
    _classify_legend_scores,
    _crop,
    _pair_legend_items,
    _repair_collisions,
    _score_legend_to_bars,
    filter_bars_by_legend,
    match_legend_to_bars,
)


# Legend matcher

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
        with patch("src.reconstruction.legend_matcher._candidate_error", side_effect=[0.05, 0.07]):
            matched = match_legend_to_bars(
                image, detections, ocr, max_error=0.05, min_margin=0.02
            ).assignments[0]
        self.assertEqual(matched.status, "matched")
        with patch("src.reconstruction.legend_matcher._candidate_error", side_effect=[0.05, 0.069]):
            ambiguous = match_legend_to_bars(
                image, detections, ocr, max_error=0.05, min_margin=0.02
            ).assignments[0]
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.series_name)
        self.assertEqual(ambiguous.swatch_id, "s1")
        with patch("src.reconstruction.legend_matcher._candidate_error", side_effect=[0.051, 0.20]):
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


class LegendEntriesTest(unittest.TestCase):
    """The mapping keeps every legend pair, including those no bar matched."""

    def three_entries(self):
        image = np.full((90, 80, 3), 255, dtype=np.uint8)
        colours = ((200, 30, 30), (30, 200, 30), (30, 30, 200))
        items = []
        for row, colour in enumerate(colours):
            top = 10 + 20 * row
            image[top + 1:top + 9, 11:19] = colour
            items += [detection(f"sw{row}", "legend_swatch", (10, top, 20, top + 10)),
                      detection(f"lb{row}", "legend_label", (22, top, 35, top + 10))]
        image[11:79, 51:69] = colours[1]          # only the middle colour has a bar
        items.append(detection("bar", "vertical_bar", (50, 10, 70, 80)))
        ocr = OCRFrame("input/chart.png", tuple(
            RecognizedText(f"lb{row}", name, 0.95)
            for row, name in enumerate(("Alpha", "Beta", "Gamma"))))
        return image, frame(*items, height=90), ocr

    def test_entries_list_every_pair_in_legend_order_including_unmatched(self):
        image, detections, ocr = self.three_entries()
        mapping = match_legend_to_bars(image, detections, ocr)
        self.assertEqual(
            [(a.bar_id, a.series_name, a.status) for a in mapping.assignments],
            [("bar", "Beta", "matched")])
        self.assertEqual(mapping.entries, (
            LegendEntry("sw0", "lb0", "Alpha"),
            LegendEntry("sw1", "lb1", "Beta"),
            LegendEntry("sw2", "lb2", "Gamma"),
        ))

    def test_unreadable_label_text_gives_an_entry_without_a_name(self):
        image, detections, _ = self.three_entries()
        ocr = OCRFrame("input/chart.png", (RecognizedText("lb0", "Alpha", 0.95),
                                           RecognizedText("lb1", "Beta", 0.95),
                                           RecognizedText("lb2", "  ", 0.1)))
        entries = match_legend_to_bars(image, detections, ocr).entries
        self.assertEqual([entry.name for entry in entries], ["Alpha", "Beta", None])

    def test_no_entries_without_a_legend_or_without_ocr(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        detections = frame(detection("bar", "vertical_bar", (5, 5, 15, 50)))
        self.assertEqual(match_legend_to_bars(
            image, detections, OCRFrame("input/chart.png", ())).entries, ())
        self.assertEqual(match_legend_to_bars(image, detections, None).entries, ())


# Legend matching baseline

class LegendMatchingBaselineTest(unittest.TestCase):
    def _dataset(self, root: Path):
        images = root / "dataset/images/val"
        images.mkdir(parents=True)
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[11:19, 11:19] = (40, 80, 120)
        image[21:49, 11:19] = (40, 80, 120)
        cv2.imwrite(str(images / "1.png"), image)
        annotation = [{
            "image_index": 1,
            "type": "vbar_categorical",
            "models": [{
                "name": "North",
                "bboxes": [{"x": 10, "y": 20, "w": 10, "h": 30}],
            }],
            "general_figure_info": {"legend": {
                "bbox": {"x": 8, "y": 8, "w": 32, "h": 14},
                "items": [{
                    "model": "North",
                    "preview": {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}},
                    "label": {
                        "text": "North",
                        "bbox": {"x": 22, "y": 10, "w": 15, "h": 10},
                    },
                }],
            }},
        }]
        annotations = root / "annotations.json"
        annotations.write_text(json.dumps(annotation), encoding="utf-8")
        return annotations, root / "dataset"

    def test_oracle_evaluation_writes_metrics_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, data_root = self._dataset(root)
            charts = load_truth_charts(annotations, data_root, "val")
            self.assertEqual(len(charts), 1)
            output = root / "output"
            report = evaluate(
                annotations=annotations,
                data_root=data_root,
                split="val",
                output_dir=output,
                track="oracle",
            )
            self.assertEqual(report["oracle"]["precision"], 1.0)
            self.assertEqual(report["oracle"]["recall"], 1.0)
            self.assertTrue((output / "metrics.json").is_file())
            rows = (output / "assignments.csv").read_text(encoding="utf-8")
            self.assertIn("North", rows)
            self.assertIn("matched", rows)

    def test_calibration_uses_documented_tie_break(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations, data_root = self._dataset(root)
            original_imread = cv2.imread
            with patch(
                "src.baseline.legend_matching.cv2.imread",
                side_effect=original_imread,
            ) as imread, patch(
                "src.baseline.legend_matching._score_legend_to_bars",
                wraps=baseline_score_legend_to_bars,
            ) as score:
                report = calibrate(
                    annotations=annotations,
                    data_root=data_root,
                    split="val",
                    output_dir=root / "calibration",
                )
            self.assertEqual(report["chosen"]["max_error"], 0.03)
            self.assertEqual(report["chosen"]["min_margin"], 0.03)
            self.assertEqual(report["chosen"]["f1"], 1.0)
            self.assertEqual(imread.call_count, 1)
            self.assertEqual(score.call_count, 1)

    def test_end_to_end_counts_correct_wrong_missed_and_false_positive_bars(self):
        def bar(identifier, x1, x2):
            return Detection(
                identifier, "vertical_bar", Box(x1, 20, x2, 50), 1.0
            )

        truth_frame = DetectionFrame("val/1.png", 100, 60, (
            bar("t1", 5, 15), bar("t2", 25, 35), bar("t3", 45, 55),
        ))
        chart = TruthChart(
            Path("unused.png"),
            truth_frame,
            OCRFrame("val/1.png", ()),
            {"t1": "A", "t2": "B", "t3": "C"},
        )
        predicted = DetectionFrame("val/1.png", 100, 60, (
            bar("p1", 5, 15), bar("p2", 25, 35), bar("fp", 75, 85),
        ))
        ocr = OCRFrame("val/1.png", (
            RecognizedText("l1", "A", 0.9),
            RecognizedText("l2", "Wrong", 0.8),
            RecognizedText("l3", "Ghost", 0.7),
        ))
        mapping = LegendMappingFrame("val/1.png", (
            LegendBarAssignment("p1", "s1", "l1", "A", 0.99, "matched"),
            LegendBarAssignment("p2", "s2", "l2", "Wrong", 0.98, "matched"),
            LegendBarAssignment("fp", "s3", "l3", "Ghost", 0.97, "matched"),
        ))

        with patch(
            "src.baseline.legend_matching.match_legend_to_bars",
            return_value=mapping,
        ):
            counts, rows = _score_end_to_end(
                chart,
                np.zeros((60, 100, 3), dtype=np.uint8),
                predicted,
                ocr,
                0.5,
                0.03,
                0.01,
            )
        report = _report(counts)

        self.assertEqual(report["truth_bars"], 3)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["incorrect"], 2)
        self.assertEqual(report["emitted_assignments"], 3)
        self.assertAlmostEqual(report["precision"], 1 / 3)
        self.assertAlmostEqual(report["recall"], 1 / 3)
        self.assertAlmostEqual(report["coverage"], 2 / 3)
        self.assertEqual(report["all_bars_correct_chart_accuracy"], 0.0)
        self.assertEqual(report["bar_detection"]["true_positive"], 2)
        self.assertEqual(report["bar_detection"]["false_positive"], 1)
        self.assertEqual(report["bar_detection"]["false_negative"], 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["status"] for row in rows}, {"matched", "missed_detection"}
        )

    def test_programmatic_evaluate_rejects_invalid_settings_before_io(self):
        common = dict(
            annotations=Path("missing.json"), data_root=Path("missing"),
            split="val", output_dir=Path("unused"),
        )
        for name, value in (
            ("track", "typo"), ("match_iou", -0.1),
            ("max_error", 1.1), ("min_margin", float("nan")),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    evaluate(**common, **{name: value})


class ColourResolutionTest(unittest.TestCase):
    """Bars the template error leaves weak/ambiguous are resolved by median colour."""

    def _chart(self, second):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[6:14, 6:14] = (10, 20, 30)
        image[21:29, 6:14] = second
        image[6:44, 41:49] = (10, 20, 30)
        image[6:44, 56:64] = second
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)),
            detection("l1", "legend_label", (17, 5, 30, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)),
            detection("l2", "legend_label", (17, 20, 30, 30)),
            detection("b1", "vertical_bar", (40, 5, 50, 45)),
            detection("b2", "vertical_bar", (55, 5, 65, 45)),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("l1", "North", 1), RecognizedText("l2", "South", 1)))
        return image, detections, ocr

    def test_similar_hues_over_the_margin_are_named(self):
        # 30 apart in RGB: the squared template error cannot separate them, the colours can
        image, detections, ocr = self._chart((10, 50, 30))
        assignments = match_legend_to_bars(image, detections, ocr).assignments
        self.assertEqual([(a.series_name, a.status) for a in assignments],
                         [("North", "matched"), ("South", "matched")])

    def test_hues_inside_the_margin_stay_unnamed(self):
        image, detections, ocr = self._chart((10, 40, 30))  # 20 apart
        assignments = match_legend_to_bars(image, detections, ocr).assignments
        self.assertTrue(all(a.series_name is None for a in assignments))

    def test_one_bar_that_matches_no_swatch_leaves_the_whole_chart_unresolved(self):
        # a mix of named and unresolved bars would make reconstruct stop rescuing bars
        image, detections, ocr = self._chart((10, 50, 30))
        image[6:44, 70:78] = (200, 10, 200)
        detections = frame(*detections.detections, detection("b3", "vertical_bar", (69, 5, 79, 45)))
        assignments = match_legend_to_bars(image, detections, ocr).assignments
        self.assertTrue(all(a.series_name is None for a in assignments[:2]))


class CollisionRepairTest(unittest.TestCase):
    """Two bars of one series in one category slot cannot both be right: reassign every bar by colour."""

    NORTH, SOUTH = (10, 20, 30), (200, 60, 30)

    def _chart(self, south=SOUTH):
        image = np.full((60, 130, 3), 255, dtype=np.uint8)
        image[6:14, 6:14] = self.NORTH
        image[21:29, 6:14] = south
        bars = [("a1", 40, self.NORTH), ("b1", 51, south), ("a2", 80, self.NORTH), ("b2", 91, south)]
        for _, x, colour in bars:
            image[10:50, x:x + 10] = colour
        detections = frame(
            detection("s1", "legend_swatch", (5, 5, 15, 15)), detection("l1", "legend_label", (17, 5, 30, 15)),
            detection("s2", "legend_swatch", (5, 20, 15, 30)), detection("l2", "legend_label", (17, 20, 30, 30)),
            *(detection(name, "vertical_bar", (x, 10, x + 10, 50)) for name, x, _ in bars),
            width=130)
        ocr = OCRFrame("input/chart.png", (RecognizedText("l1", "North", 1), RecognizedText("l2", "South", 1)))
        return image, detections, ocr

    @staticmethod
    def _mapping(series_by_bar):
        entries = (LegendEntry("s1", "l1", "North"), LegendEntry("s2", "l2", "South"))
        return LegendMappingFrame("input/chart.png", tuple(
            LegendBarAssignment(bar, "s1", "l1", name, 0.9, "matched") for bar, name in series_by_bar.items()), entries)

    def _names(self, mapping):
        return {a.bar_id: a.series_name for a in mapping.assignments}

    def test_two_bars_of_one_series_in_a_slot_are_reassigned_by_colour(self):
        image, detections, _ = self._chart()
        collided = self._mapping({"a1": "North", "b1": "North", "a2": "North", "b2": "North"})
        repaired = _repair_collisions(image, detections, collided, "vertical_bar")
        self.assertEqual(self._names(repaired), {"a1": "North", "b1": "South", "a2": "North", "b2": "South"})
        self.assertTrue(all(a.status == "matched" for a in repaired.assignments))

    def test_a_mapping_without_a_collision_is_left_alone(self):
        image, detections, _ = self._chart()
        fine = self._mapping({"a1": "North", "b1": "South", "a2": "North", "b2": "South"})
        self.assertIs(_repair_collisions(image, detections, fine, "vertical_bar"), fine)

    def test_a_collision_between_colours_inside_the_margin_is_left_alone(self):
        image, detections, _ = self._chart(south=(10, 40, 30))   # 20 apart: colour cannot decide
        collided = self._mapping({"a1": "North", "b1": "North", "a2": "North", "b2": "North"})
        self.assertIs(_repair_collisions(image, detections, collided, "vertical_bar"), collided)

    def test_one_bar_that_matches_no_swatch_leaves_the_collision_unrepaired(self):
        image, detections, _ = self._chart()
        image[10:50, 91:101] = (200, 10, 200)                     # b2 matches neither swatch
        collided = self._mapping({"a1": "North", "b1": "North", "a2": "North", "b2": "North"})
        self.assertIs(_repair_collisions(image, detections, collided, "vertical_bar"), collided)

    def test_filter_bars_by_legend_applies_the_repair(self):
        image, detections, ocr = self._chart()
        collided = self._mapping({"a1": "North", "b1": "North", "a2": "North", "b2": "North"})
        with patch("src.reconstruction.legend_matcher._classify_legend_scores", return_value=collided):
            result = filter_bars_by_legend(image, detections, ocr)
        self.assertEqual(self._names(result.mapping), {"a1": "North", "b1": "South", "a2": "North", "b2": "South"})


if __name__ == "__main__":
    unittest.main()
