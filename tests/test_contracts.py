"""Stage-handoff examples and compatibility checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.baseline.opencv_bars import Box as BaselineBox
from src.contract_io import (
    load_detection_frame,
    load_ocr_frame,
    load_vertical_bar_chart,
    write_detection_frame,
    write_ocr_frame,
    write_vertical_bar_chart,
)
from src.contracts import (
    CLASS_NAMES,
    BarSeries,
    Box,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
    VerticalBarChart,
    join_legend_assignments,
    join_text_regions,
    validate_unit_interval,
)
from src.dataset.prepare_plotqa import CLASS_NAMES as ConverterClassNames


FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class ContractTest(unittest.TestCase):
    def test_unit_interval_validation_normalizes_and_rejects_invalid_values(self):
        self.assertEqual(validate_unit_interval(0, "threshold"), 0.0)
        self.assertEqual(validate_unit_interval(0.5, "threshold"), 0.5)
        self.assertEqual(validate_unit_interval(1, "threshold"), 1.0)
        for value in (True, "0.5", -0.1, 1.1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "threshold must be a number from 0 to 1"
                ):
                    validate_unit_interval(value, "threshold")

    def test_legend_mapping_validates_statuses_and_exact_bar_coverage(self):
        frame = DetectionFrame("input/chart.png", 20, 20, (
            Detection("bar", "vertical_bar", Box(0, 0, 5, 10), 0.9),
            Detection("swatch", "legend_swatch", Box(10, 0, 15, 5), 0.9),
            Detection("label", "legend_label", Box(15, 0, 20, 5), 0.9),
        ))
        mapping = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment(
                "bar", "swatch", "label", "North", 0.9, "matched"
            ),
        ))
        self.assertEqual(join_legend_assignments(frame, mapping)[0][0].id, "bar")
        with self.assertRaisesRegex(ValueError, "detection order"):
            join_legend_assignments(
                frame, LegendMappingFrame("input/chart.png", ())
            )
        with self.assertRaises(ValueError):
            LegendBarAssignment("bar", status="unknown")
        with self.assertRaises(ValueError):
            LegendBarAssignment("bar", match_score=True, status="weak_match")

    def test_legend_mapping_can_cover_the_bars_of_a_horizontal_chart(self):
        frame = DetectionFrame("input/chart.png", 20, 20, (
            Detection("h0", "horizontal_bar", Box(0, 0, 10, 4), 0.9),
            Detection("h1", "horizontal_bar", Box(0, 6, 8, 10), 0.9),
        ))
        mapping = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment("h0"), LegendBarAssignment("h1"),
        ))
        self.assertEqual([bar.id for bar, _ in join_legend_assignments(frame, mapping)],
                         ["h0", "h1"])
        with self.assertRaisesRegex(ValueError, "detection order"):
            join_legend_assignments(frame, LegendMappingFrame(
                "input/chart.png", (LegendBarAssignment("h1"), LegendBarAssignment("h0"))))

    def test_existing_class_order_and_box_are_shared(self):
        self.assertEqual(CLASS_NAMES, (
            "horizontal_bar", "vertical_bar", "dot", "line_segment",
            "chart_title", "x_axis_title", "y_axis_title", "x_tick_label",
            "y_tick_label", "x_axis_rule", "y_axis_rule", "plot_area",
            "legend", "legend_swatch", "legend_label",
        ))
        self.assertIs(ConverterClassNames, CLASS_NAMES)
        self.assertIs(BaselineBox, Box)

    def test_fixture_handoffs_and_round_trip(self):
        detections = load_detection_frame(FIXTURES / "detections.json")
        ocr = load_ocr_frame(FIXTURES / "ocr.json")
        chart = load_vertical_bar_chart(FIXTURES / "vertical_bar_chart.json")
        self.assertEqual(detections.source_key, ocr.source_key)
        self.assertEqual(ocr.source_key, chart.source_key)
        self.assertEqual(detections.source_key, "val/example.png")
        self.assertEqual(len(join_text_regions(detections, ocr)), len(ocr.regions))
        self.assertEqual(len(chart.series), 2)
        self.assertEqual(chart.series[0].values, (4, 6))

        from src.dataset.prepare_plotqa import png_size
        self.assertEqual(
            png_size(FIXTURES / detections.source_key),
            (detections.width, detections.height),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_detection_frame(root / "detections.json", detections)
            write_ocr_frame(root / "ocr.json", ocr)
            write_vertical_bar_chart(root / "chart.json", chart)
            self.assertEqual(load_detection_frame(root / "detections.json"), detections)
            self.assertEqual(load_ocr_frame(root / "ocr.json"), ocr)
            self.assertEqual(load_vertical_bar_chart(root / "chart.json"), chart)

    def test_rejects_unknown_role_box_and_confidence(self):
        with self.assertRaisesRegex(ValueError, "source_key"):
            DetectionFrame("22410", 10, 10, ())
        with self.assertRaisesRegex(ValueError, "Unknown detection role"):
            Detection("a", "not_a_role", Box(0, 0, 1, 1), 0.5)
        with self.assertRaisesRegex(ValueError, "Box outside image"):
            DetectionFrame("val/a.png", 10, 10, (
                Detection("a", "vertical_bar", Box(9, 0, 11, 2), 0.5),
            ))
        with self.assertRaisesRegex(ValueError, "confidence"):
            Detection("a", "vertical_bar", Box(0, 0, 1, 1), math.nan)

    def test_rejects_duplicate_and_broken_ocr_links(self):
        detection = Detection("a", "chart_title", Box(0, 0, 5, 5), 0.9)
        with self.assertRaisesRegex(ValueError, "Duplicate detection ID"):
            DetectionFrame("val/a.png", 10, 10, (detection, detection))
        frame = DetectionFrame("val/a.png", 10, 10, (detection,))
        with self.assertRaisesRegex(ValueError, "different source images"):
            join_text_regions(frame, OCRFrame("test/a.png", ()))
        with self.assertRaisesRegex(ValueError, "unknown detection"):
            join_text_regions(frame, OCRFrame("val/a.png", (
                RecognizedText("missing", "A", 0.9),
            )))
        non_text = DetectionFrame("val/a.png", 10, 10, (
            Detection("bar", "vertical_bar", Box(0, 0, 5, 5), 0.9),
        ))
        with self.assertRaisesRegex(ValueError, "non-text role"):
            join_text_regions(non_text, OCRFrame("val/a.png", (
                RecognizedText("bar", "A", 0.9),
            )))

    def test_rejects_incomplete_chart(self):
        with self.assertRaisesRegex(ValueError, "category count"):
            VerticalBarChart("val/a.png", "Title", "X", "Y", ("A", "B"), (
                BarSeries("North", (1,)),
            ))

    def test_rejects_unsupported_json_version(self):
        data = json.loads((FIXTURES / "detections.json").read_text(encoding="utf-8"))
        data["schema_version"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version 1"):
                load_detection_frame(path)


if __name__ == "__main__":
    unittest.main()


class BarLessSeriesContractTest(unittest.TestCase):
    def test_a_series_with_no_visible_bars_has_a_name_and_no_values(self):
        from src.contracts import BarSeries, VerticalBarChart
        series = BarSeries("Vietnam", (), has_bars=False)
        chart = VerticalBarChart("demo/a.png", "T", "X", "Y", ("a", "b"),
                                 (BarSeries("Kenya", (1.0, 2.0)), series))
        self.assertFalse(chart.series[1].has_bars)
        with self.assertRaises(ValueError):
            BarSeries(None, (), has_bars=False)
        with self.assertRaises(ValueError):
            BarSeries("Vietnam", (0.0, 0.0), has_bars=False)

    def test_a_chart_needs_at_least_one_series_with_bars(self):
        from src.contracts import BarSeries, VerticalBarChart
        with self.assertRaises(ValueError):
            VerticalBarChart("demo/a.png", "T", "X", "Y", ("a",),
                             (BarSeries("Vietnam", (), has_bars=False),))

    def test_series_spec_with_no_visible_bars_cannot_carry_values(self):
        from src.contracts import SeriesSpec, VerticalBarSpec, YAxisSpec
        def spec(series):
            return VerticalBarSpec("val/1.png", "T", "X", "Y", YAxisSpec(0, 1, [0, 1]), ["a", "b"],
                                   series, False, [], {})
        ok = spec([SeriesSpec("Vietnam", "#112233", [None, None], no_visible_bars=True)])
        self.assertTrue(ok.series[0].no_visible_bars)
        for bad in (SeriesSpec(None, "#112233", [None, None], no_visible_bars=True),
                    SeriesSpec("V", "#112233", [0.0, None], no_visible_bars=True),
                    SeriesSpec("V", "#112233", [None, None], [False, False], no_visible_bars=True)):
            with self.assertRaises(ValueError):
                spec([bad])

    def test_legend_entries_are_validated_and_do_not_share_detections(self):
        from src.contracts import LegendEntry, LegendMappingFrame
        with self.assertRaises(ValueError):
            LegendEntry("", "lb")
        frame_ = LegendMappingFrame("val/1.png", (), (LegendEntry("s0", "l0", "A"), LegendEntry("s1", "l1")))
        self.assertEqual(len(frame_.entries), 2)
        with self.assertRaises(ValueError):
            LegendMappingFrame("val/1.png", (), (LegendEntry("s0", "l0"), LegendEntry("s0", "l1")))
        with self.assertRaises(ValueError):
            LegendMappingFrame("val/1.png", (), ("not an entry",))
        self.assertEqual(LegendMappingFrame("val/1.png", ()).entries, ())

    def test_chart_json_round_trip_writes_has_bars_only_when_false(self):
        import json
        import tempfile
        from pathlib import Path
        from src.contract_io import load_vertical_bar_chart, write_vertical_bar_chart
        from src.contracts import BarSeries, VerticalBarChart
        chart = VerticalBarChart("demo/a.png", "T", "X", "Y", ("a", "b"),
                                 (BarSeries("Kenya", (1.0, 2.0)), BarSeries("Vietnam", (), has_bars=False)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            write_vertical_bar_chart(path, chart)
            data = json.loads(path.read_text())["series"]
            self.assertNotIn("has_bars", data[0])
            self.assertIs(data[1]["has_bars"], False)
            self.assertEqual(load_vertical_bar_chart(path), chart)
