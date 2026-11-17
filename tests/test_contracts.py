"""Shared contracts, palette, and presentation helpers."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.baseline.opencv_bars import Box as BaselineBox
from src.borealis import BOREALIS_COLORS, BOREALIS_CSS, hex_to_bgr
from src.contract_io import (
    load_detection_frame,
    load_ocr_frame,
    load_vertical_bar_chart,
    write_detection_frame,
    write_ocr_frame,
    write_vertical_bar_chart,
)
from src.contracts import (
    BarSeries,
    Box,
    CLASS_NAMES,
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
import src.presentation as presentation
from src.presentation import (
    LEGEND_TABLE_HEADERS,
    OCR_TABLE_HEADERS,
    detection_rows,
    legend_rows,
    ocr_rows,
    render_detection_overlay,
    summarize_detections,
)


# Contracts

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


# Borealis

def contrast(left: str, right: str) -> float:
    def luminance(value: str) -> float:
        channels = [int(value[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045
            else math.pow((channel + 0.055) / 1.055, 2.4)
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    brighter, darker = sorted((luminance(left), luminance(right)), reverse=True)
    return (brighter + 0.05) / (darker + 0.05)


class BorealisTest(unittest.TestCase):
    def test_exact_palette_is_immutable(self):
        self.assertEqual(dict(BOREALIS_COLORS), {
            "midnight": "#001322", "deep_sky": "#003866",
            "aurora": "#669ECC", "glow": "#CCDFFF",
            "text": "#F7FBFF", "muted": "#B8CCE0",
            "panel": "#06243B", "elevated": "#0A304B",
            "sun": "#F4C95D",
        })
        with self.assertRaises(TypeError):
            BOREALIS_COLORS["midnight"] = "#FFFFFF"

    def test_hex_to_bgr_validates_and_converts(self):
        self.assertEqual(hex_to_bgr("#CCDFFF"), (255, 223, 204))
        self.assertEqual(hex_to_bgr("#001322"), (34, 19, 0))
        for invalid in (None, "001322", "#123", "#GG0011", "#00132200"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "#RRGGBB"):
                    hex_to_bgr(invalid)

    def test_overlay_colors_resolve_from_shared_palette(self):
        self.assertEqual(presentation._SOFT_GLOW, hex_to_bgr(BOREALIS_COLORS["glow"]))
        self.assertEqual(presentation._AURORA_BLUE, hex_to_bgr(BOREALIS_COLORS["aurora"]))
        self.assertEqual(presentation._DEEP_SKY, hex_to_bgr(BOREALIS_COLORS["deep_sky"]))
        self.assertEqual(presentation._PANEL_NAVY, hex_to_bgr(BOREALIS_COLORS["panel"]))
        self.assertEqual(presentation._HALO, hex_to_bgr(BOREALIS_COLORS["text"]))

    def test_css_contains_tokens_layout_and_focus_contract(self):
        for name, value in BOREALIS_COLORS.items():
            self.assertIn(f"--borealis-{name.replace('_', '-')}: {value};", BOREALIS_CSS)
        self.assertIn("background: var(--borealis-midnight)", BOREALIS_CSS)
        self.assertIn("--borealis-content-width: 1200px", BOREALIS_CSS)
        self.assertIn("--borealis-space-1: 8px", BOREALIS_CSS)
        self.assertIn("--borealis-radius-sm: 4px", BOREALIS_CSS)
        self.assertIn("--borealis-radius-md: 4px", BOREALIS_CSS)
        self.assertIn("--borealis-shadow:", BOREALIS_CSS)
        self.assertIn("--borealis-focus-width: 3px", BOREALIS_CSS)
        self.assertIn(":focus-visible", BOREALIS_CSS)
        self.assertIn(
            "outline: var(--borealis-focus-width) solid var(--borealis-sun)",
            BOREALIS_CSS,
        )
        for class_name in (
            "app", "header", "actions", "scope", "control-rail", "upload-image",
            "summary", "workspace", "workspace-grid", "error", "empty",
        ):
            self.assertIn(f".borealis-{class_name}", BOREALIS_CSS)
        self.assertIn("@media (max-width: 900px)", BOREALIS_CSS)
        self.assertIn("@media (max-width: 600px)", BOREALIS_CSS)
        self.assertIn(".borealis-workspace-heading .borealis-workspace-heading", BOREALIS_CSS)
        self.assertIn(".borealis-workspace-heading,", BOREALIS_CSS)
        self.assertIn("margin-block: 0 16px !important", BOREALIS_CSS)
        self.assertIn("padding: 16px !important", BOREALIS_CSS)
        self.assertIn("padding: 16px 12px !important", BOREALIS_CSS)

    def test_text_and_control_pairs_meet_wcag_aa(self):
        pairs = (
            ("text", "midnight"), ("text", "panel"), ("muted", "panel"),
            ("glow", "midnight"), ("text", "deep_sky"),
            ("aurora", "midnight"), ("aurora", "panel"),
            ("sun", "midnight"), ("midnight", "sun"),
        )
        for foreground, background in pairs:
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(
                    contrast(BOREALIS_COLORS[foreground], BOREALIS_COLORS[background]), 4.5
                )


# Presentation

def sample_frame(*detections: Detection, width: int = 120, height: int = 100):
    return DetectionFrame("input/chart.png", width, height, detections)


class PresentationTest(unittest.TestCase):
    def test_module_import_does_not_require_frontend_or_model_packages(self):
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'gradio', 'ultralytics'}:
        raise ImportError(f'blocked dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import src.presentation
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_summary_groups_roles_in_contract_order(self):
        frame = sample_frame(
            Detection("title", "chart_title", Box(5, 5, 60, 15), 0.8),
            Detection("bar-1", "vertical_bar", Box(20, 30, 35, 90), 0.95),
            Detection("bar-2", "vertical_bar", Box(50, 40, 65, 90), 0.9),
        )

        summary = summarize_detections(frame)

        self.assertIn("**Detected 3 chart elements**", summary)
        self.assertIn("Confidence range: 80.0% to 95.0%.", summary)
        self.assertIn("- Vertical bar: 2", summary)
        self.assertIn("- Chart title: 1", summary)
        self.assertIn("- Chart text has not been read.", summary)
        self.assertTrue(summary.endswith("- Numeric bar values are not reconstructed."))

    def test_summary_treats_zero_detections_as_valid(self):
        summary = summarize_detections(sample_frame())
        self.assertIn("**Detected 0 chart elements**", summary)
        self.assertIn("- No roles were detected.", summary)
        self.assertNotIn("Confidence range", summary)

    def test_ocr_summary_and_rows_use_joined_detection_geometry(self):
        frame = sample_frame(
            Detection("title", "chart_title", Box(1, 2, 30, 12), 0.9),
            Detection("bar", "vertical_bar", Box(10, 20, 20, 80), 0.8),
            Detection("tick", "x_tick_label", Box(40, 85, 55, 95), 0.85),
        )
        ocr = OCRFrame("input/chart.png", (
            RecognizedText("title", "Sales", 0.9),
            RecognizedText("tick", "", 0.0),
        ))

        summary = summarize_detections(frame, ocr=ocr)
        rows = ocr_rows(frame, ocr)

        self.assertIn("- OCR: 1 of 2 text regions readable", summary)
        self.assertEqual(OCR_TABLE_HEADERS, (
            "Detection ID", "Role", "Recognized text", "OCR confidence",
            "X1", "Y1", "X2", "Y2",
        ))
        self.assertEqual(rows, [
            ["title", "chart title", "Sales", 0.9, 1, 2, 30, 12],
            ["tick", "x tick label", "", 0.0, 40, 85, 55, 95],
        ])

    def test_table_preserves_order_and_numeric_values(self):
        frame = sample_frame(
            Detection("det-2", "x_tick_label", Box(1.5, 2.5, 10.5, 12.5), 0.875),
            Detection("det-1", "dot", Box(20, 21, 24, 25), 0.5),
        )

        rows = detection_rows(frame)

        self.assertEqual([row["Detection ID"] for row in rows], ["det-2", "det-1"])
        self.assertEqual(rows[0], {
            "Detection ID": "det-2", "Role": "x tick label", "Confidence": 0.875,
            "X1": 1.5, "Y1": 2.5, "X2": 10.5, "Y2": 12.5,
        })
        self.assertEqual(list(rows[0]), [
            "Detection ID", "Role", "Confidence", "X1", "Y1", "X2", "Y2",
        ])

    def test_overlay_draws_every_group_without_mutating_input(self):
        frame = sample_frame(
            Detection("bar", "vertical_bar", Box(10, 25, 25, 90), 0.91),
            Detection("title", "chart_title", Box(35, 10, 75, 20), 0.82),
            Detection("axis", "x_axis_rule", Box(5, 90, 100, 94), 0.77),
            Detection("legend", "legend_swatch", Box(100, 25, 115, 40), 0.88),
        )
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        original = image.copy()

        overlay = render_detection_overlay(image, frame)

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(overlay.shape, image.shape)
        self.assertFalse(np.array_equal(overlay, image))
        for x, y in ((10, 30), (35, 15), (5, 90), (100, 30)):
            self.assertTrue(np.any(overlay[y, x]))

    def test_legend_mapping_summary_rows_and_overlay(self):
        frame = sample_frame(
            Detection("bar", "vertical_bar", Box(10, 25, 25, 90), 0.91),
            Detection("swatch", "legend_swatch", Box(80, 10, 90, 20), 0.9),
            Detection("label", "legend_label", Box(92, 10, 110, 20), 0.9),
        )
        mapping = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment(
                "bar", "swatch", "label", "North", 0.98, "matched"
            ),
        ))
        summary = summarize_detections(frame, legend_mapping=mapping)
        rows = legend_rows(frame, mapping)
        overlay = render_detection_overlay(
            np.zeros((100, 120, 3), dtype=np.uint8),
            frame,
            legend_mapping=mapping,
        )
        self.assertIn(
            "- Legend matching: 1 of 1 vertical bars matched; 0 unmatched",
            summary,
        )
        self.assertEqual(LEGEND_TABLE_HEADERS, (
            "Bar ID", "Series", "Color similarity", "Swatch ID", "Label ID", "Status",
        ))
        self.assertEqual(rows, [["bar", "North", 0.98, "swatch", "label", "matched"]])
        self.assertTrue(np.any(overlay))

    def test_structure_outline_is_dashed_and_solid_outline_is_continuous(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        solid = render_detection_overlay(image, sample_frame(
            Detection("bar", "vertical_bar", Box(10, 20, 100, 80), 0.9),
        ))
        dashed = render_detection_overlay(image, sample_frame(
            Detection("plot", "plot_area", Box(10, 20, 100, 80), 0.9),
        ))

        # Away from the label, a long solid edge has fewer black gaps than a dashed edge.
        solid_gaps = np.count_nonzero(np.all(solid[20, 20:95] == 0, axis=1))
        dashed_gaps = np.count_nonzero(np.all(dashed[20, 20:95] == 0, axis=1))
        self.assertLess(solid_gaps, dashed_gaps)

    def test_labels_and_boxes_remain_within_edge_boundaries(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        frame = sample_frame(
            Detection("edge", "legend_label", Box(45, 1, 60, 12), 0.99),
            width=60, height=40,
        )
        overlay = render_detection_overlay(image, frame)
        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.any(overlay[0:20, 40:60]))

    def test_overlay_rejects_invalid_or_mismatched_images(self):
        frame = sample_frame()
        cases = (
            (object(), TypeError),
            (np.zeros((100, 120), dtype=np.uint8), ValueError),
            (np.zeros((100, 120, 4), dtype=np.uint8), ValueError),
            (np.zeros((100, 120, 3), dtype=np.float32), ValueError),
            (np.zeros((99, 120, 3), dtype=np.uint8), ValueError),
        )
        for image, error in cases:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(error):
                    render_detection_overlay(image, frame)


if __name__ == "__main__":
    unittest.main()
