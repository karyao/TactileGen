from tests.gradio.callback_helpers import invoke
"""Gradio startup, analysis, review, callbacks, components, theme, and diagnostics."""

import contextlib
from dataclasses import asdict, fields, replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from PIL import Image
import cv2
import numpy as np
import pytest

from src.borealis import BOREALIS_CSS, borealis_theme
from src.contract_io import load_ocr_frame
from src.contracts import (
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
    SeriesSpec,
    VerticalBarSpec,
    YAxisSpec,
)
from src.diagnostics import AnalysisDiagnostics
from src.gradio_app import (
    AnalysisResult,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_WEIGHTS,
    LEGEND_WARNING,
    MAX_UPLOAD_BYTES,
    ManualGenerationBlocked,
    OCR_WARNING,
    RECONSTRUCTION_WARNING,
    ReviewModel,
    UploadValidationError,
    alert_message,
    analyze_image,
    analyzing_status,
    build_app,
    build_review_model,
    complete_status,
    completeness_summary,
    decode_upload,
    generate_review_submission,
    generation_report,
    main,
    parse_args,
    ready_status,
    review_overrides,
    review_text_rows,
    review_value_rows,
    update_review_draft,
)
from src.gradio_ui.components import AnalysisComponents, Controls, GenerationComponents
from src.rapidocr_ocr import RapidOCRBackend
from src.tactile.completeness import CANNOT_BUILD, Completeness, MissingItem, NEEDS_INPUT, READY

from tests.test_ocr import IMAGE, SHORT, collector, fake_rapidocr, output
from tests.test_reconstruct import base_chart


# Gradio fixtures

def make_spec(**changes):
    fields = dict(
        source_key="input/chart.png",
        title="Sales",
        x_axis_title="Year",
        y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]),
        categories=["2024"],
        series=[SeriesSpec(None, "#112233", [50.0])],
        bars_adjacent=False,
        flags=[],
        confidence={},
    )
    fields.update(changes)
    return VerticalBarSpec(**fields)


class GradioTestCase(unittest.TestCase):
    def _image(
        self,
        root: Path,
        name: str = "chart.png",
        *,
        size=(100, 80),
        format: str | None = None,
        rgb=(255, 255, 255),
    ) -> Path:
        path = root / name
        Image.new("RGB", size, rgb).save(path, format=format)
        return path


# Gradio app

class GradioCliTest(GradioTestCase):
    def test_argument_defaults_and_port_validation(self):
        args = parse_args([])
        self.assertEqual(args.weights, DEFAULT_WEIGHTS)
        self.assertEqual(args.host, DEFAULT_HOST)
        self.assertEqual(args.port, DEFAULT_PORT)
        self.assertIsNone(args.device)
        self.assertEqual(args.ocr, "rapidocr")
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(parse_args(["--imgsz", "641"]).imgsz, 641)
        self.assertEqual(parse_args(["--imgsz", "1280"]).imgsz, 1280)
        for value in ("0", "-1", "640.5", "no", "True", "1281", "100000"):
            with self.subTest(imgsz=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--imgsz", value])
        self.assertEqual(parse_args(["--ocr", "rapidocr"]).ocr, "rapidocr")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--port", "0"])
            with self.assertRaises(SystemExit):
                parse_args(["--port", "not-a-port"])

    def test_main_loads_once_and_uses_private_styled_launch(self):
        model = object()
        demo = Mock()
        theme = object()
        with patch("src.gradio_app._import_gradio"),\
             patch("src.gradio_app.load_model", return_value=model) as load,\
             patch("src.gradio_app.build_app", return_value=demo) as build,\
             patch("src.gradio_app.borealis_theme", return_value=theme):
            main(["--weights", "custom.pt", "--device", "cpu",
                  "--host", "0.0.0.0", "--port", "9000", "--imgsz", "960"])
        load.assert_called_once_with(Path("custom.pt"))
        build.assert_called_once_with(model, ocr_engine="rapidocr", device="cpu", imgsz=960)
        kwargs = demo.launch.call_args.kwargs
        self.assertEqual(kwargs["server_name"], "0.0.0.0")
        self.assertEqual(kwargs["server_port"], 9000)
        self.assertIs(kwargs["share"], False)
        self.assertIs(kwargs["show_error"], False)
        self.assertEqual(kwargs["max_file_size"], MAX_UPLOAD_BYTES)
        self.assertIs(kwargs["theme"], theme)
        self.assertEqual(kwargs["css"], BOREALIS_CSS)
        self.assertIs(kwargs["run_history"], False)
        self.assertIs(kwargs["enable_monitoring"], False)


# Gradio analysis

class GradioAnalysisTest(GradioTestCase):
    def test_decode_accepts_supported_content_and_ignores_filename_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("chart.png", "PNG"),
                ("chart.jpg", "JPEG"),
                ("chart.webp", "WEBP"),
                ("renamed.data", "PNG"),
            )
            for name, image_format in cases:
                with self.subTest(name=name):
                    path = self._image(root, name, format=image_format)
                    returned_path, image = decode_upload(path)
                    self.assertEqual(returned_path, path)
                    self.assertEqual(image.shape, (80, 100, 3))

    def test_analysis_forwards_custom_size_and_collector(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            diagnostics = AnalysisDiagnostics(settings={})
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame) as detect,\
                 patch("src.gradio_ui.analysis.run_ocr", return_value=OCRFrame(frame.source_key, ())):
                analyze_image(object(), image, .25, .7, imgsz=1280, diagnostics=diagnostics)
            self.assertEqual(detect.call_args.kwargs["imgsz"], 1280)
            self.assertIs(detect.call_args.kwargs["diagnostics"], diagnostics)
            self.assertEqual(diagnostics.record["settings"]["imgsz"], 1280)

    def test_valid_analysis_forwards_settings_and_swaps_bgr_to_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(
                Path(temporary), rgb=(30, 20, 10)  # OpenCV reads this as BGR 10,20,30.
            )
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            progress = Mock()
            model = object()
            backend = object()
            ocr = OCRFrame("input/chart.png", ())
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame) as run,\
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr) as recognize:
                result = analyze_image(
                    model,
                    image_path,
                    0.4,
                    0.6,
                    language="eng",
                    ocr_backend=backend,
                    device="cpu",
                    progress=progress,
                )
            run.assert_called_once_with(
                model, image_path, confidence=0.4, nms_iou=0.6, device="cpu", imgsz=640
            )
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.detection_count, 1)
            self.assertEqual(result.ocr_frame, ocr)
            recognize.assert_called_once()
            self.assertIs(recognize.call_args.args[2], backend)
            self.assertEqual(recognize.call_args.args[3], "eng")
            self.assertIn("**Detected 1 chart element**", result.summary)
            self.assertIn("- OCR: 0 of 0 text regions readable", result.summary)
            self.assertEqual(tuple(result.overlay_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(progress.call_count, 7)

    def test_reconstruction_reuses_results_and_applies_edge_completeness(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", (LegendBarAssignment("bar"),))
            reconstructed = make_spec()
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame),\
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),\
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping),\
                 patch("src.gradio_ui.analysis.reconstruct_vbar",
                       return_value=reconstructed) as reconstruct,\
                 patch("src.gradio_ui.analysis.cut_off_flags",
                       return_value=["title_cut_off"]) as edge_flags:
                result = analyze_image(
                    object(), image_path, 0.25, 0.70, ocr_backend=object()
                )

            args = reconstruct.call_args.args
            self.assertEqual(args[:2], (frame, ocr))
            self.assertEqual(reconstruct.call_args.kwargs, {"legend_mapping": mapping})
            edge_flags.assert_called_once_with(frame, args[2])
            self.assertEqual(result.spec.flags, ["title_cut_off"])
            self.assertEqual(result.completeness.status, NEEDS_INPUT)
            self.assertEqual(result.completeness.missing[0].field, "title")
            self.assertIsNone(result.reconstruction_warning)

    def test_horizontal_chart_uses_the_horizontal_matcher_and_reconstruction(self):
        for diagnostics_enabled in (False, True):
            with self.subTest(diagnostics=diagnostics_enabled), tempfile.TemporaryDirectory() as temporary:
                image_path = self._image(Path(temporary))
                frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "horizontal_bar", Box(60, 20, 75, 70), .9),
            ))
                ocr = OCRFrame("input/chart.png", ())
                mapping = LegendMappingFrame("input/chart.png", (LegendBarAssignment("bar"),))
                spec = replace(make_spec(), orientation="horizontal")
                diagnostics = AnalysisDiagnostics(settings={}) if diagnostics_enabled else None
                with (
                    patch("src.gradio_ui.analysis.run_detection", return_value=frame),
                    patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),
                    patch("src.gradio_ui.analysis.match_legend_to_bars") as vertical_match,
                    patch("src.gradio_ui.analysis.reconstruct_vbar") as vertical,
                    patch("src.gradio_ui.analysis.match_legend_to_hbars", autospec=True,
                          return_value=mapping) as horizontal_match,
                    patch("src.gradio_ui.analysis.reconstruct_hbar", autospec=True,
                          return_value=spec) as horizontal,
                    patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]),
                ):
                    result = analyze_image(
                        object(), image_path, 0.25, 0.70, ocr_backend=object(),
                        diagnostics=diagnostics,
                    )

                vertical_match.assert_not_called()
                vertical.assert_not_called()
                self.assertEqual(horizontal_match.call_args.args[1:], (frame, ocr))
                self.assertEqual(horizontal.call_args.kwargs, {"legend_mapping": mapping})
                self.assertEqual(result.spec.orientation, "horizontal")
                self.assertIsNone(result.legend_warning)
                self.assertIsNone(result.reconstruction_warning)
                if diagnostics is not None:
                    for stage in ("legend_matching", "reconstruction"):
                        self.assertEqual(diagnostics.record["stages"][stage]["outcome"], "succeeded")

    def test_interpretation_error_keeps_partial_results_and_explains_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", (LegendBarAssignment("bar"),))
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame),\
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),\
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping),\
                 patch("src.gradio_ui.analysis.reconstruct_vbar", side_effect=
                       ChartInterpretationError("no x tick labels detected")):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70, ocr_backend=object()
                )

            self.assertIsNone(result.spec)
            self.assertEqual(result.completeness.status, CANNOT_BUILD)
            self.assertIn("No x tick labels", result.completeness.blockers[0])
            self.assertIsNotNone(result.overlay_rgb)
            self.assertIsNone(result.reconstruction_warning)

    def test_unexpected_completeness_failure_does_not_publish_unchecked_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", (LegendBarAssignment("bar"),))
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame),\
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),\
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping),\
                 patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=make_spec()),\
                 patch("src.gradio_ui.analysis.check_completeness",
                       side_effect=RuntimeError("classification failed")),\
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70, ocr_backend=object()
                )

            self.assertIsNone(result.spec)
            self.assertIsNone(result.completeness)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIsNotNone(result.overlay_rgb)

    def test_ocr_failure_keeps_successful_detection_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("title", "chart_title", Box(5, 5, 50, 15), 0.9),
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame),\
                 patch("src.gradio_ui.analysis.run_ocr", side_effect=RuntimeError("missing")),\
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.detection_frame, frame)
            self.assertIsNone(result.ocr_frame)
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIn("**Detected 2 chart elements**", result.summary)
            self.assertIsNotNone(result.overlay_rgb)

    def test_matcher_failure_is_independent_and_can_coexist_with_ocr_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(5, 5, 20, 70), 0.9),
            ))
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame),\
                 patch("src.gradio_ui.analysis.run_ocr", side_effect=RuntimeError("ocr")),\
                 patch("src.gradio_ui.analysis.match_legend_to_bars",
                       side_effect=RuntimeError("matcher")),\
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.legend_warning, LEGEND_WARNING)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIsNone(result.legend_mapping)
            self.assertNotIn("Legend matching:", result.summary)

    def test_invalid_upload_never_calls_model_and_mismatched_frame_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "broken.png"
            broken.write_bytes(b"not an image")
            with patch("src.gradio_ui.analysis.run_detection") as run:
                with self.assertRaises(UploadValidationError):
                    analyze_image(object(), broken, 0.25, 0.70)
                run.assert_not_called()

            valid = self._image(root)
            mismatched = DetectionFrame("input/chart.png", 99, 80, ())
            with patch("src.gradio_ui.analysis.run_detection", return_value=mismatched):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    analyze_image(object(), valid, 0.25, 0.70)

    def test_upload_validation_rejects_missing_size_dimensions_and_bad_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "directory"
            directory.mkdir()
            for missing in (None, root / "missing.png", directory):
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(UploadValidationError, "Choose"):
                        decode_upload(missing)

            empty = root / "empty.png"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(UploadValidationError, "could not be read"):
                decode_upload(empty)

            text = root / "chart.png"
            text.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(UploadValidationError, "could not be read"):
                decode_upload(text)

            gif = self._image(root, "chart.gif", format="GIF")
            with self.assertRaisesRegex(UploadValidationError, "PNG, JPEG, or WebP"):
                decode_upload(gif)

            too_large = root / "large.png"
            with too_large.open("wb") as stream:
                stream.truncate(MAX_UPLOAD_BYTES + 1)
            with self.assertRaisesRegex(UploadValidationError, "20 MB"):
                decode_upload(too_large)

            normal = self._image(root, "dimensions.png")
            fake_source = Mock()
            fake_source.__enter__ = Mock(return_value=fake_source)
            fake_source.__exit__ = Mock(return_value=False)
            fake_source.format = "PNG"
            fake_source.size = (10_000, 4_001)
            with patch("PIL.Image.open", return_value=fake_source):
                with self.assertRaisesRegex(UploadValidationError, "40 megapixels"):
                    decode_upload(normal)


# Gradio review

class GradioReviewTest(GradioTestCase):
    def test_accessible_status_helpers_cover_ready_busy_success_zero_and_error(self):
        self.assertIn('role="status"', ready_status())
        self.assertIn('aria-live="polite"', analyzing_status())
        self.assertIn("Analysis complete", complete_status())
        self.assertIn("Analysis complete", complete_status(0))
        self.assertIn("Analysis complete", complete_status(3, 2))
        self.assertIn('role="alert"', alert_message("Try again"))

    def test_completeness_summary_covers_all_states_and_details(self):
        self.assertIn("Reconstruction: Ready", completeness_summary(Completeness(READY)))
        needs_input = completeness_summary(Completeness(
            NEEDS_INPUT,
            notes=["Check the inferred value."],
            missing=[MissingItem("title", "No title was read.", "--title")],
        ))
        self.assertIn("Reconstruction: Needs input", needs_input)
        self.assertIn("Check the inferred value", needs_input)
        self.assertIn("`title`: No title was read", needs_input)
        cannot_build = completeness_summary(Completeness(
            CANNOT_BUILD, blockers=["The axis scale is unavailable."]
        ))
        self.assertIn("Reconstruction: Cannot build", cannot_build)
        self.assertIn("The axis scale is unavailable", cannot_build)

    def test_review_model_has_stable_order_and_read_only_values(self):
        spec = make_spec(
            categories=["2023", "2024"],
            series=[
                SeriesSpec("North", "#112233", [10.0, 20.0]),
                SeriesSpec("South", "#445566", [30.0, 40.0]),
            ],
        )
        model = build_review_model(spec, Completeness(READY))

        self.assertEqual(
            [binding.key for binding in model.bindings],
            [
                "title", "x_axis_title", "y_axis_title",
                "category[0]", "category[1]",
                "series_name[0]", "series_name[1]",
            ],
        )
        self.assertEqual(review_text_rows(model)[3], ["Category 0", "2023"])
        self.assertEqual(
            review_value_rows(spec),
            [
                [0, "2023", 0, "North", 10.0],
                [1, "2024", 0, "North", 20.0],
                [0, "2023", 1, "South", 30.0],
                [1, "2024", 1, "South", 40.0],
            ],
        )

    def test_review_mapping_ignores_submitted_labels_and_preserves_spec(self):
        spec = make_spec(
            categories=["2023", "2024"],
            series=[SeriesSpec(None, "#112233", [25.0, 50.0])],
        )
        model = build_review_model(spec, Completeness(READY))
        rows = review_text_rows(model)
        rows[0] = ["category[1]", "Corrected sales"]
        rows[3] = ["title", "FY 2023"]
        rows[4] = ["anything", "FY 2024"]

        updated = update_review_draft(model, rows)
        overrides = review_overrides(updated)

        self.assertEqual(overrides.title, "Corrected sales")
        self.assertEqual(overrides.categories, {0: "FY 2023", 1: "FY 2024"})
        self.assertEqual(spec.title, "Sales")
        self.assertEqual(spec.categories, ["2023", "2024"])
        self.assertEqual(review_text_rows(updated)[0][0], "Chart title")
        self.assertEqual(review_text_rows(updated)[3][0], "Category 0")

    def test_review_confirmation_and_blank_validation(self):
        spec = make_spec(flags=["title_cut_off"])
        completeness = Completeness(
            NEEDS_INPUT,
            missing=[MissingItem("title", "Confirm title.", "--title")],
        )
        model = build_review_model(spec, completeness)
        self.assertTrue(model.bindings[0].confirmation_required)
        self.assertEqual(review_overrides(model).title, "Sales")

        rows = review_text_rows(model)
        rows[1][1] = "   "
        draft = update_review_draft(model, rows)
        self.assertEqual(draft.draft_values[1], "   ")
        with self.assertRaisesRegex(ValueError, "X-axis title cannot be blank"):
            review_overrides(draft)

        unnamed = build_review_model(make_spec(), Completeness(READY))
        self.assertFalse(unnamed.bindings[-1].required)
        self.assertEqual(review_overrides(unnamed).series_names, {})

    def test_review_draft_rejects_malformed_table_shapes_and_cells(self):
        model = build_review_model(make_spec(), Completeness(READY))
        rows = review_text_rows(model)
        invalid = (
            rows[:-1],
            [*rows, ["extra", "value"]],
            [["only one column"], *rows[1:]],
            [["Chart title", 42], *rows[1:]],
        )
        for submitted in invalid:
            with self.subTest(submitted=submitted):
                with self.assertRaises(ValueError):
                    update_review_draft(model, submitted)
        self.assertEqual(model.spec.title, "Sales")
        self.assertEqual(model.draft_values[0], "Sales")

    def test_manual_generation_applies_and_commits_review_text(self):
        spec = make_spec(flags=["title_cut_off"])
        assessment = Completeness(
            NEEDS_INPUT,
            missing=[MissingItem("title", "Confirm title.", "--title")],
        )
        model = build_review_model(spec, assessment)
        rows = review_text_rows(model)
        rows[0][1] = "Confirmed sales"
        rows[3][1] = "FY 2024"

        with tempfile.TemporaryDirectory() as root:
            result = generate_review_submission(model, rows, root)
            self.assertEqual(result.completeness.status, READY)
            self.assertEqual(result.review_model.spec.title, "Confirmed sales")
            self.assertEqual(result.review_model.spec.categories, ["FY 2024"])
            self.assertNotIn("title_cut_off", result.review_model.spec.flags)
            self.assertTrue(result.generation.zip_path.is_file())
            report = generation_report(result)

        self.assertIn("chart title supplied by user", report)
        self.assertIn("category 0 supplied by user", report)
        self.assertIn("Tactile graphic", report)
        self.assertIn("Checked", report)
        self.assertIn("annotated review copies", report)

    def test_manual_generation_blocks_invalid_text_and_structural_uncertainty(self):
        model = build_review_model(make_spec(), Completeness(READY))
        rows = review_text_rows(model)
        rows[0][1] = "Sales 😀"
        with tempfile.TemporaryDirectory() as root, self.assertRaises(
            ManualGenerationBlocked
        ) as raised:
            generate_review_submission(model, rows, root)
        self.assertEqual(raised.exception.completeness.status, NEEDS_INPUT)
        self.assertIn("cannot be written in braille",
                      raised.exception.completeness.missing[0].reason)

        blocked_spec = make_spec(flags=["category_ambiguous"])
        blocked = build_review_model(
            blocked_spec,
            Completeness(CANNOT_BUILD, blockers=["Ambiguous categories."]),
        )
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ManualGenerationBlocked) as raised:
                generate_review_submission(blocked, review_text_rows(blocked), root)
            self.assertEqual(raised.exception.completeness.status, CANNOT_BUILD)
            self.assertFalse(any(Path(root).iterdir()))

        blank = review_text_rows(model)
        blank[1][1] = "   "
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(
            ValueError, "X-axis title cannot be blank"
        ):
            generate_review_submission(model, blank, root)


# Gradio components

class ComponentOrderingTest(unittest.TestCase):
    def test_named_updates_follow_component_order_regardless_of_keyword_order(self):
        for group_type in (AnalysisComponents, GenerationComponents):
            with self.subTest(group=group_type.__name__):
                names = [field.name for field in fields(group_type)]
                components = {name: object() for name in names}
                group = group_type(**components)
                updates = {name: f"update:{name}" for name in reversed(names)}

                self.assertEqual(group.components(), list(components.values()))
                self.assertEqual(
                    group.updates(**updates),
                    tuple(f"update:{name}" for name in names),
                )

    def test_missing_or_unknown_outputs_are_rejected(self):
        values = {field.name: None for field in fields(AnalysisComponents)}
        group = AnalysisComponents(**values)
        missing = dict(values)
        del missing["diagnostics_info"]
        with self.assertRaisesRegex(ValueError, "diagnostics_info"):
            group.updates(**missing)
        with self.assertRaisesRegex(ValueError, "typo"):
            group.updates(**values, typo=None)

    def test_reset_controls_include_each_preset_in_component_order(self):
        values = {field.name: field.name for field in fields(Controls)}
        values["size_presets"] = ("small", "medium", "large")
        controls = Controls(**values)
        updates = {
            name: tuple(f"reset:{preset}" for preset in value)
            if name == "size_presets" else f"reset:{value}"
            for name, value in reversed(list(values.items()))
        }
        self.assertEqual(
            controls.updates(**updates),
            tuple(f"reset:{component}" for component in controls.components()),
        )
        with self.assertRaisesRegex(ValueError, "preset"):
            controls.updates(**{**updates, "size_presets": ()})

    def test_application_logic_does_not_import_optional_gradio_dependency(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.gradio_ui.analysis; "
                "import src.gradio_ui.review; assert 'gradio' not in sys.modules",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


# Gradio ui

class GradioUiTest(GradioTestCase):
    def test_size_control_uses_startup_default_forwards_selection_and_resets(self):
        demo = build_app(object(), ocr_backend=object(), imgsz=960)
        try:
            binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            control = next(c for c in binding.inputs if c.label == "YOLO inference size (pixels)")
            self.assertEqual(control.value, 960)
            self.assertEqual(control.maximum, 1280)
            presets = [fn for fn in demo.fns.values()
                       if fn.outputs == [control] and not fn.inputs]
            self.assertEqual([invoke(fn.fn) for fn in presets], [640, 960, 1280])
            self.assertIsNone(control.precision)  # Never round fractional input into a valid size.
            for selected in (None, 1280):
                with patch("src.gradio_ui.callbacks.analyze_image", side_effect=UploadValidationError("Choose a chart.")) as analyze:
                    args = (None, .25, .7, "eng")
                    invoke(binding.fn, *args, *(() if selected is None else (selected,)))
                self.assertEqual(analyze.call_args.kwargs["imgsz"], selected or 960)
            clear = next(fn for fn in demo.fns.values() if fn.api_name == "clear_all")
            updates = invoke(clear.fn)
            index = clear.outputs.index(control)
            self.assertEqual(updates[index]["value"], 960)
            self.assertTrue(updates[index]["interactive"])
        finally:
            demo.close(verbose=False)

    def test_invalid_size_is_reported_with_diagnostics_before_inference(self):
        from zipfile import ZipFile

        demo = build_app(object(), ocr_backend=object())
        try:
            binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            archive_index = next(i for i, c in enumerate(binding.outputs)
                                 if getattr(c, "label", None) == "Download analysis diagnostics (ZIP)")
            for size in (640.5, 1281, 100000):
                with self.subTest(size=size):
                    with patch("src.gradio_ui.analysis.run_detection") as detect:
                        updates = invoke(binding.fn, None, .25, .7, "eng", size)
                    detect.assert_not_called()
                    with ZipFile(updates[archive_index]["value"]) as archive:
                        record = json.loads(archive.read("diagnostics.json"))
                    self.assertEqual(record["settings"]["imgsz"], size)
                    self.assertEqual(record["stages"]["validation"]["outcome"], "failed")
                    self.assertIn("imgsz", record["exceptions"][0]["message"])
                    self.assertIsNone(record["metadata"]["detection"]["tensor_shape"])
        finally:
            demo.close(verbose=False)

    def test_creator_workspace_disclosures_preserve_result_bindings(self):
        demo = build_app(object(), device="cpu")
        config = demo.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        disclosures = {
            item["props"]["label"]: item
            for item in components.values() if item["type"] == "accordion"
        }

        def descendants(node):
            result = {node["id"]}
            for child in node.get("children", []):
                result.update(descendants(child))
            return result

        def layout_node(node, target):
            if node["id"] == target:
                return node
            for child in node.get("children", []):
                found = layout_node(child, target)
                if found is not None:
                    return found
            return None

        analysis_details = disclosures["Analysis details"]
        specification = disclosures["Chart specification"]
        for disclosure in (analysis_details, specification):
            self.assertFalse(disclosure["props"]["open"])
        detail_ids = descendants(layout_node(config["layout"], analysis_details["id"]))
        spec_ids = descendants(layout_node(config["layout"], specification["id"]))
        detail_labels = {components[key]["props"].get("label") for key in detail_ids}
        self.assertTrue({"Detection summary", "Recognized text", "Legend mappings"}
                        <= detail_labels)
        self.assertTrue(any(components[key]["props"].get("label") ==
                            "Reconstructed VerticalBarSpec" for key in spec_ids))
        secondary_ids = detail_ids | spec_ids
        for component in components.values():
            classes = component["props"].get("elem_classes", [])
            if any(name in classes for name in (
                "borealis-review", "borealis-error", "borealis-result-image",
                "borealis-tactile-gallery",
            )):
                self.assertNotIn(component["id"], secondary_ids)

        analyze = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = invoke(ready)
        self.assertEqual(len(ready_updates), 27)
        self.assertIn("Your chart workspace", ready_updates[1]["value"])
        frame = DetectionFrame("input/chart.png", 20, 20, ())
        result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 0,
            frame, None, None, spec=make_spec(), completeness=Completeness(READY),
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=result):
            updates = invoke(analyze.fn, "chart.png", 0.25, 0.70, "eng")
        by_class = {
            name: update
            for output, update in zip(analyze.outputs, updates)
            for name in output.elem_classes
        }
        self.assertTrue(by_class["borealis-result-image"]["visible"])
        self.assertTrue(by_class["borealis-review"]["visible"])
        self.assertTrue(by_class["borealis-generate"]["interactive"])
        self.assertTrue(by_class["borealis-spec-json"]["visible"])
        self.assertFalse(by_class["borealis-empty"]["visible"])

        blocked = AnalysisResult(
            "partial summary", result.overlay_rgb, 0, frame, None, None,
            spec=make_spec(),
            completeness=Completeness(CANNOT_BUILD, blockers=["Ambiguous structure."]),
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=blocked):
            blocked_updates = invoke(analyze.fn, "chart.png", 0.25, 0.70, "eng")
        self.assertFalse(blocked_updates[13]["interactive"])
        self.assertTrue(blocked_updates[9]["visible"])
        with patch("src.gradio_ui.callbacks.analyze_image", side_effect=UploadValidationError("Choose a chart.")):
            error_updates = invoke(analyze.fn, None, 0.25, 0.70, "eng")
        self.assertTrue(error_updates[23]["visible"])
        self.assertIn('role="alert"', error_updates[23]["value"])
        self.assertEqual(len(invoke(ready)), 27)

    def test_image_change_clears_diagnostics_with_all_analysis_results(self):
        demo = build_app(object(), device="cpu")
        reset = next(fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        analyze = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        config = demo.get_config_file()
        image = next(component for component in config["components"]
                     if component["props"].get("label") == "Bar chart image")
        event = next(dependency for dependency in config["dependencies"]
                     if any(tuple(target) == (image["id"], "change")
                            for target in dependency["targets"]))

        self.assertEqual(event["outputs"], [output._id for output in reset.outputs])
        self.assertEqual(reset.outputs, analyze.outputs)
        self.assertFalse(event["queue"])
        self.assertEqual(event["api_visibility"], "private")
        updates = invoke(reset.fn)
        self.assertEqual(len(updates), len(reset.outputs[:27]))
        for name in ("diagnostics_download", "diagnostics_info", "orientation_output"):
            index = [field.name for field in fields(AnalysisComponents)].index(name)
            self.assertFalse(updates[index]["visible"])
            self.assertEqual(updates[index]["value"], None if name == "diagnostics_download" else "")

    def test_app_factory_wires_controls_states_and_private_events(self):
        with patch("src.gradio_app.load_model") as load,\
             patch("gradio.Blocks.launch") as launch:
            demo = build_app(object(), device="cpu")
        load.assert_not_called()
        launch.assert_not_called()
        config = demo.get_config_file()
        components = config["components"]
        labels = {component["props"].get("label") for component in components}
        self.assertIn("Bar chart image", labels)
        self.assertIn("Minimum detection confidence", labels)
        self.assertIn("NMS IoU for duplicate suppression", labels)
        self.assertIn("Tesseract language", labels)
        self.assertIn("OCR engine", labels)
        self.assertIn("Annotated detections", labels)
        self.assertIn("Download analysis diagnostics (ZIP)", labels)
        self.assertIn("Recognized text", labels)
        self.assertIn("Legend mappings", labels)
        self.assertIn("Reconstruction completeness", labels)
        self.assertIn("Reconstructed text", labels)
        self.assertIn("Reconstructed values (read-only)", labels)
        self.assertIn("Reconstructed VerticalBarSpec", labels)
        self.assertIn("Tactile review pages", labels)
        self.assertIn("Tactile generation report", labels)

        by_class = {}
        for component in components:
            for class_name in component["props"].get("elem_classes", []):
                by_class.setdefault(class_name, component)
        for class_name in (
            "borealis-scope", "borealis-control-rail", "borealis-workspace",
            "borealis-status", "borealis-empty", "borealis-summary",
            "borealis-error",
        ):
            self.assertIn(class_name, by_class)
        self.assertTrue(by_class["borealis-empty"]["props"]["visible"])
        self.assertFalse(by_class["borealis-summary"]["props"]["visible"])
        self.assertFalse(by_class["borealis-error"]["props"]["visible"])
        annotated = next(c for c in components
                         if c["props"].get("label") == "Annotated detections")
        self.assertFalse(annotated["props"]["visible"])
        recognized = next(c for c in components
                          if c["props"].get("label") == "Recognized text")
        self.assertFalse(recognized["props"]["visible"])

        inference = [dependency for dependency in config["dependencies"]
                     if dependency.get("api_name") == "analyze"]
        self.assertEqual(len(inference), 1)
        self.assertEqual(inference[0]["api_visibility"], "private")
        self.assertEqual(inference[0]["show_progress"], "full")
        inference_fn = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        self.assertEqual(inference_fn.concurrency_limit, 1)
        self.assertEqual(inference_fn.concurrency_id, "tactilegen-inference")

        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = invoke(ready)
        self.assertEqual(len(ready_updates), 27)
        self.assertTrue(ready_updates[1]["visible"])
        self.assertIsNone(ready_updates[8])
        self.assertTrue(all(not update["visible"] for update in ready_updates[2:8]))
        self.assertFalse(ready_updates[9]["visible"])
        self.assertFalse(ready_updates[12]["visible"])
        self.assertFalse(ready_updates[13]["interactive"])
        self.assertTrue(all(not update["visible"] for update in ready_updates[14:24]))
        self.assertEqual(inference_fn.outputs[24].label, "Download analysis diagnostics (ZIP)")
        self.assertIsNone(ready_updates[24]["value"])
        self.assertFalse(ready_updates[24]["visible"])
        self.assertEqual(ready_updates[25]["value"], "")
        self.assertFalse(ready_updates[25]["visible"])

        analyze = inference_fn.fn
        display_frame = DetectionFrame("input/chart.png", 2, 2, (
            Detection("title", "chart_title", Box(0, 0, 2, 1), 0.9),
        ))
        display_ocr = OCRFrame("input/chart.png", (
            RecognizedText("title", "Sales", 0.8),
        ))
        success = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, display_ocr, None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=success):
            success_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertIn("Analysis complete", success_updates[0]["value"])
        self.assertEqual(success_updates[2]["value"], success.summary)
        self.assertTrue(success_updates[2]["visible"])
        self.assertTrue(success_updates[3]["visible"])
        self.assertTrue(success_updates[4]["visible"])
        self.assertEqual(success_updates[4]["value"][0][:4],
                         ["title", "chart title", "Sales", 0.8])
        self.assertTrue(success_updates[5]["visible"])
        self.assertEqual(load_ocr_frame(Path(success_updates[5]["value"])), display_ocr)
        self.assertFalse(success_updates[6]["visible"])
        self.assertFalse(success_updates[7]["visible"])
        self.assertIsNone(success_updates[8])
        self.assertFalse(success_updates[9]["visible"])
        self.assertFalse(success_updates[12]["visible"])
        self.assertFalse(success_updates[13]["interactive"])
        self.assertTrue(all(not success_updates[index]["visible"]
                            for index in range(14, 24)))
        self.assertIs(success_updates[3]["value"], success.overlay_rgb)
        self.assertTrue(success_updates[24]["visible"])
        self.assertTrue(Path(success_updates[24]["value"]).is_file())
        self.assertTrue(success_updates[25]["visible"])
        self.assertIn("Analysis run:", success_updates[25]["value"])

        mapped_frame = DetectionFrame("input/chart.png", 20, 20, (
            Detection("bar", "vertical_bar", Box(0, 5, 5, 20), 0.9),
            Detection("swatch", "legend_swatch", Box(8, 0, 12, 4), 0.9),
            Detection("label", "legend_label", Box(13, 0, 20, 4), 0.9),
        ))
        mapped = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment(
                "bar", "swatch", "label", "North", 0.98, "matched"
            ),
        ))
        reconstructed = make_spec()
        mapped_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None,
            reconstructed, Completeness(READY), None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=mapped_result):
            mapped_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertTrue(mapped_updates[6]["visible"])
        self.assertEqual(
            mapped_updates[6]["value"][0],
            ["bar", "North", 0.98, "swatch", "label", "matched"],
        )
        self.assertTrue(mapped_updates[7]["visible"])
        self.assertIn("Reconstruction: Ready", mapped_updates[7]["value"])
        self.assertIsInstance(mapped_updates[8], ReviewModel)
        self.assertTrue(mapped_updates[9]["visible"])
        self.assertEqual(mapped_updates[10]["value"][0], ["Chart title", "Sales"])
        self.assertEqual(mapped_updates[11]["value"],
                         [[0, "2024", 0, "(unnamed)", 50.0]])
        self.assertTrue(mapped_updates[13]["interactive"])
        self.assertTrue(mapped_updates[14]["visible"])
        self.assertEqual(mapped_updates[14]["value"], asdict(reconstructed))
        self.assertTrue(mapped_updates[15]["visible"])
        spec_path = Path(mapped_updates[15]["value"])
        self.assertEqual(json.loads(spec_path.read_text(encoding="utf-8")),
                         asdict(reconstructed))
        self.assertRegex(spec_path.name, r"chart_spec_[0-9a-f]{32}\.json")

        uncertain_result = replace(
            mapped_result, summary="Bar orientation is unclear.",
            ocr_frame=None, legend_mapping=None, spec=None, completeness=None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=uncertain_result):
            uncertain_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertIsNone(uncertain_updates[8])
        self.assertFalse(uncertain_updates[13]["interactive"])
        for index in (4, 5, 6, 7, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            self.assertFalse(uncertain_updates[index]["visible"])

        spec_blocked_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None, reconstructed,
            Completeness(CANNOT_BUILD, blockers=["Ambiguous structure."]), None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=spec_blocked_result):
            spec_blocked_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertIsInstance(spec_blocked_updates[8], ReviewModel)
        self.assertTrue(spec_blocked_updates[9]["visible"])
        self.assertFalse(spec_blocked_updates[13]["interactive"])
        self.assertEqual(spec_blocked_updates[11]["value"],
                         [[0, "2024", 0, "(unnamed)", 50.0]])

        blocked_ocr = OCRFrame("input/chart.png", (
            RecognizedText("label", "North", 0.9),
        ))
        blocked_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, blocked_ocr, None, mapped, None, None,
            Completeness(
                CANNOT_BUILD,
                blockers=["No x tick labels were found. Try a clearer image."],
            ),
            None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=blocked_result):
            blocked_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertIn("Analysis complete", blocked_updates[0]["value"])
        for index in (2, 3, 4, 5, 6, 7):
            self.assertTrue(blocked_updates[index]["visible"])
        self.assertIn("Reconstruction: Cannot build", blocked_updates[7]["value"])
        self.assertIn("No x tick labels", blocked_updates[7]["value"])
        self.assertIsNone(blocked_updates[8])
        self.assertFalse(blocked_updates[9]["visible"])
        self.assertFalse(blocked_updates[14]["visible"])
        self.assertFalse(blocked_updates[15]["visible"])
        self.assertTrue(all(not blocked_updates[index]["visible"]
                            for index in (20, 21, 22, 23)))

        warning = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, None, OCR_WARNING,
            reconstruction_warning=RECONSTRUCTION_WARNING,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=warning):
            warning_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertTrue(warning_updates[2]["visible"])
        self.assertTrue(warning_updates[3]["visible"])
        self.assertFalse(warning_updates[4]["visible"])
        self.assertTrue(warning_updates[20]["visible"])
        self.assertFalse(warning_updates[21]["visible"])
        self.assertTrue(warning_updates[22]["visible"])
        self.assertIn('role="alert"', warning_updates[22]["value"])
        self.assertIn("reconstruction was unavailable", warning_updates[22]["value"])
        self.assertFalse(warning_updates[23]["visible"])
        with patch(
            "src.gradio_ui.callbacks.analyze_image",
            side_effect=UploadValidationError("Upload a supported image."),
        ):
            failure_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertFalse(failure_updates[2]["visible"])
        self.assertFalse(failure_updates[3]["visible"])
        self.assertFalse(failure_updates[4]["visible"])
        self.assertTrue(failure_updates[23]["visible"])
        self.assertIn('role="alert"', failure_updates[23]["value"])

        private_path = "/private/tmp/gradio/secret-chart.png"
        with self.assertLogs("src.gradio_app", level="ERROR"), patch(
            "src.gradio_ui.callbacks.analyze_image",
            side_effect=RuntimeError(private_path),
        ):
            unexpected_updates = invoke(analyze, "chart.png", 0.25, 0.70, "eng")
        self.assertTrue(unexpected_updates[23]["visible"])
        self.assertNotIn(private_path, unexpected_updates[23]["value"])
        self.assertNotIn("RuntimeError", unexpected_updates[23]["value"])

        change_events = [dependency for dependency in config["dependencies"]
                         if dependency["targets"] and dependency["targets"][0][1] == "change"]
        self.assertEqual(len(change_events), 1)
        self.assertTrue(all(not event["queue"] for event in change_events))
        self.assertEqual(change_events[0]["api_visibility"], "private")
        self.assertEqual(change_events[0]["outputs"],
                         [output._id for output in inference_fn.outputs])
        review_events = [dependency for dependency in config["dependencies"]
                         if dependency.get("api_name") == "update_review"]
        self.assertEqual(len(review_events), 1)
        self.assertEqual(review_events[0]["api_visibility"], "private")
        self.assertFalse(review_events[0]["queue"])
        review_fn = next(fn.fn for fn in demo.fns.values()
                         if fn.api_name == "update_review")
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=mapped_result):
            review_model = invoke(analyze, "chart.png", .25, .7, "eng")[8]
        invalid_review = invoke(review_fn, review_model, review_text_rows(review_model)[:-1])
        self.assertIs(invalid_review[0], review_model)
        self.assertTrue(invalid_review[2]["visible"])
        changed_rows = review_text_rows(review_model)
        changed_rows[0] = ["category[0]", "Updated title"]
        valid_review = invoke(review_fn, review_model, changed_rows)
        self.assertEqual(valid_review[0].draft_values[0], "Updated title")
        self.assertEqual(valid_review[1]["value"][0][0], "Chart title")
        self.assertFalse(valid_review[2]["visible"])

        generation_events = [
            dependency for dependency in config["dependencies"]
            if dependency.get("api_name") == "generate_tactile"
        ]
        self.assertEqual(len(generation_events), 1)
        self.assertEqual(generation_events[0]["api_visibility"], "private")
        generate_fn = next(
            fn for fn in demo.fns.values() if fn.api_name == "generate_tactile"
        )
        self.assertEqual(generate_fn.concurrency_limit, 1)
        self.assertEqual(generate_fn.concurrency_id, "tactilegen-generation")
        review_model = valid_review[0]
        generated_rows = review_text_rows(review_model)
        generated_rows[0][1] = "Updated sales"
        generated = invoke(generate_fn.fn, review_model, generated_rows)
        self.assertEqual(generated[1].spec.title, "Updated sales")
        self.assertEqual(generated[2]["value"][0], ["Chart title", "Updated sales"])
        self.assertTrue(generated[5]["visible"])
        self.assertTrue(generated[7]["visible"])
        self.assertTrue(generated[8]["visible"])
        self.assertTrue(generated[9]["visible"])
        self.assertFalse(generated[10]["visible"])
        demo.close(verbose=False)


# Theme

class GradioThemeTest(unittest.TestCase):
    def test_theme_uses_borealis_component_values(self):
        import gradio as gr

        theme = borealis_theme()
        self.assertIsInstance(theme, gr.themes.Base)
        values = theme.to_dict()["theme"]
        expected = {
            "body_background_fill": "#001322",
            "body_text_color": "#F7FBFF",
            "body_text_color_subdued": "#B8CCE0",
            "block_background_fill": "#06243B",
            "input_background_fill": "#0A304B",
            "input_border_color_focus": "#F4C95D",
            "button_primary_background_fill": "#F4C95D",
            "button_primary_text_color": "#001322",
            "button_secondary_background_fill": "transparent",
            "button_secondary_border_color": "#669ECC",
            "table_even_background_fill": "#001322",
            "table_odd_background_fill": "#06243B",
            "error_background_fill": "#0A304B",
            "error_text_color": "#F7FBFF",
        }
        for name, value in expected.items():
            self.assertEqual(values[name], value)


    def test_import_is_gradio_independent_and_factory_error_is_actionable(self):
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == 'gradio':
        raise ImportError(f'blocked dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import src.borealis
import src.presentation
try:
    src.borealis.borealis_theme()
except RuntimeError as exc:
    assert 'requirements-app.txt' in str(exc)
else:
    raise AssertionError('theme factory should require Gradio')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[2],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


# Diagnostics

class GradioDiagnosticsTest(unittest.TestCase):
    def test_optional_collector_preserves_exceptions_and_invalid_settings(self):
        collector = AnalysisDiagnostics(settings={"confidence": float("nan")})
        with self.assertRaises(UploadValidationError),\
             patch("src.gradio_ui.analysis.run_detection") as detect:
            analyze_image(object(), None, .25, .7, diagnostics=collector)
        detect.assert_not_called()
        collector.finish(failed=True)
        self.assertEqual(collector.record["stages"]["validation"]["outcome"], "failed")
        with tempfile.TemporaryDirectory() as temporary:
            path = collector.persist(Path(temporary))
            self.assertEqual(json.loads(path.read_text())["settings"]["confidence"], "nan")


    def test_event_records_success_failures_and_repeated_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "chart.png"
            Image.new("RGB", (100, 80), "white").save(image)
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("title", "chart_title", Box(5, 5, 50, 15), .9),
                Detection("tick", "y_tick_label", Box(5, 20, 15, 30), .9),
                Detection("bar", "vertical_bar", Box(60, 20, 75, 70), .9),
            ))
            ocr = OCRFrame(frame.source_key, (
                RecognizedText("title", "Sales", .8), RecognizedText("tick", "", 0),
            ))
            # Use real record persistence while retaining a copy beyond app shutdown.
            records = []
            paths = []
            persist = AnalysisDiagnostics.persist
            def save(collector, artifact_root):
                path = persist(collector, artifact_root)
                paths.append(path)
                records.append(json.loads(path.read_text()))
                return path
            demo = build_app(object(), ocr_backend=object(), device="cpu")
            event_binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            event = event_binding.fn
            error_index = max(
                index for index, output in enumerate(event_binding.outputs)
                if type(output).__name__ == "HTML" and "borealis-error" in output.elem_classes
            )
            try:
                with patch.object(AnalysisDiagnostics, "persist", save),\
                     patch("src.gradio_ui.analysis.run_detection", return_value=frame) as detect,\
                     patch("src.gradio_ui.analysis.run_ocr", return_value=ocr) as recognize,\
                     patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=object()),\
                     patch("src.gradio_ui.analysis.render_detection_overlay", side_effect=lambda image, *a, **k: image),\
                     patch("src.gradio_ui.analysis.summarize_detections", return_value="summary"),\
                     patch("src.gradio_ui.callbacks.legend_rows", return_value=[]),\
                     patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=make_spec()) as reconstruct,\
                     patch("src.gradio_ui.analysis.cut_off_flags", return_value=[]),\
                     self.assertLogs("src.gradio_app", level="INFO") as logs:
                    success = invoke(event, str(image), .25, .7, "eng")
                    invoke(event, None, .25, .7, "eng")
                    detect.side_effect = RuntimeError("detector unavailable")
                    invoke(event, str(image), .25, .7, "eng")
                    detect.side_effect = None
                    recognize.side_effect = RuntimeError("OCR unavailable")
                    partial = invoke(event, str(image), .25, .7, "eng")
                    recognize.side_effect = None
                    reconstruct.side_effect = ChartInterpretationError("fewer than 2 readable y tick labels")
                    invoke(event, str(image), .25, .7, "eng")
                    reconstruct.side_effect = None
                    with patch("src.gradio_ui.callbacks.write_ocr_download", side_effect=OSError("export unavailable")):
                        invoke(event, str(image), .25, .7, "eng")
                    with patch("src.diagnostics.os.replace", side_effect=OSError("disk full")),\
                         patch.object(AnalysisDiagnostics, "persist", persist):
                        storage_failure = invoke(event, str(image), .25, .7, "eng")
                    reconstruct.return_value = make_spec(flags=["y_ticks_dropped:1"])
                    invoke(event, str(image), .25, .7, "eng")
                self.assertFalse(success[error_index]["visible"])
                self.assertFalse(partial[error_index]["visible"])
                self.assertFalse(storage_failure[error_index]["visible"])
                self.assertEqual(len(records), 7)
                self.assertEqual(len(set(r["run_id"] for r in records)), 7)
                self.assertEqual(records[0]["outcome"], "succeeded")
                self.assertTrue(all(s["outcome"] == "succeeded" for s in records[0]["stages"].values()))
                self.assertEqual(records[0]["counts"], {
                    "detections": 3, "text_regions": 2, "ocr_returned_regions": 2,
                    "ocr_readable": 1, "ocr_empty": 1,
                })
                for record, stage in ((records[1], "validation"), (records[2], "detection"),
                                      (records[5], "result_preparation")):
                    self.assertEqual(record["outcome"], "failed")
                    self.assertEqual(record["stages"][stage]["outcome"], "failed")
                    self.assertTrue(record["exceptions"])
                self.assertEqual(records[1]["stages"]["detection"]["outcome"], "skipped")
                self.assertEqual(records[2]["stages"]["ocr"]["outcome"], "skipped")
                self.assertEqual(records[3]["outcome"], "completed_with_warnings")
                self.assertIsNone(records[3]["counts"]["ocr_readable"])
                self.assertEqual(records[3]["stages"]["reconstruction"]["outcome"], "skipped")
                self.assertEqual(records[3]["stages"]["presentation"]["outcome"], "succeeded")
                self.assertEqual(records[4]["stages"]["reconstruction"]["outcome"], "blocked")
                self.assertEqual(records[4]["completeness_status"], "cannot_build")
                self.assertEqual(records[0]["exceptions"], [])
                self.assertEqual(records[6]["spec_flags"], ["y_ticks_dropped:1"])
                self.assertIn({"code": "y_ticks_dropped", "stage": "reconstruction"},
                              records[6]["warnings"])
                self.assertEqual(records[6]["outcome"], "completed_with_warnings")
                terminal = "\n".join(logs.output)
                self.assertEqual(terminal.count("analysis outcome="), 8)
                for record in records:
                    run_logs = [line for line in logs.output if f"[run {record['run_id']}]" in line]
                    self.assertEqual(sum("analysis outcome=" in line for line in run_logs), 1)
                    self.assertIn("analysis outcome=", run_logs[-1])
                    self.assertIn(f"warnings={len(record['warnings'])}", run_logs[-1])
                    self.assertIn("diagnostic ZIP created", run_logs[-2])
                for record, path in zip(records, paths):
                    self.assertIn(f"[run {record['run_id']}]", terminal)
                    self.assertIn(str(path), terminal)
                self.assertIn("detector unavailable", terminal)
                self.assertNotIn("Traceback (most recent call last)", terminal)
            finally:
                demo.close(verbose=False)


    def test_app_passes_collector_to_real_ocr_orchestration(self):
        module, _, _ = fake_rapidocr([output(" label ", .8)])
        with patch("src.rapidocr_ocr._load_rapidocr", return_value=module):
            backend = RapidOCRBackend()
        frame = DetectionFrame("input/chart.png", 60, 40, (
            SHORT, Detection("bar", "vertical_bar", Box(40, 20, 50, 35), .9),
        ))
        diagnostics = collector()
        with patch("src.gradio_ui.analysis.decode_upload", return_value=(Path("chart.png"), IMAGE)),\
             patch("src.gradio_ui.analysis.run_detection", return_value=frame):
            result = analyze_image(object(), "chart.png", .25, .7,
                                   ocr_backend=backend, diagnostics=diagnostics)
        trace = diagnostics.record["ocr_trace"]
        self.assertEqual(result.ocr_frame.regions[0].text, "label")
        self.assertEqual(trace["attempts"][0]["raw_output"]["text"], " label ")
        self.assertEqual(trace["results"][0]["application_result"]["text"], "label")


@pytest.mark.parametrize('failure', [None, 'detection', 'ocr', 'legend_matching', 'reconstruction', 'presentation'])
def test_zip_preserves_partial_stage_outputs(tmp_path, failure):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    diagnostic = AnalysisDiagnostics(settings={'confidence': .25})
    functions = {'detection': 'run_detection', 'ocr': 'run_ocr',
                 'legend_matching': 'match_legend_to_bars', 'reconstruction': 'reconstruct_vbar',
                 'presentation': 'render_detection_overlay'}
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch('src.gradio_ui.analysis.run_detection', return_value=frame))
        stack.enter_context(patch('src.gradio_ui.analysis.run_ocr', return_value=ocr))
        if failure:
            stack.enter_context(patch('src.gradio_ui.analysis.' + functions[failure], side_effect=RuntimeError('injected failure')))
        try:
            result = analyze_image(object(), image_path, .25, .7, diagnostics=diagnostic)
            assert result.diagnostics is diagnostic
        except RuntimeError:
            assert failure in ('detection', 'presentation')
        diagnostic.finish(failed=failure in ('detection', 'presentation'))
    path = diagnostic.export_zip(tmp_path)
    with ZipFile(path) as archive:
        report = json.loads(archive.read('diagnostics.json'))
        assert 'injected failure' in archive.read('report.txt').decode() if failure else not report['exceptions']
        assert 'input.png' not in archive.namelist()
        if failure != 'detection':
            assert 'detections.json' in archive.namelist()
            crops = [e for e in report['events'] if e['code'] == 'source_crop']
            assert len(crops) == len(ocr.regions)
            assert all(e['filename'] in archive.namelist() for e in crops)
            assert cv2.imdecode(np.frombuffer(archive.read(crops[0]['filename']), dtype=np.uint8), 1) is not None
        if failure not in ('detection', 'ocr'):
            assert 'ocr.json' in archive.namelist()
        if failure in ('reconstruction', 'presentation', None):
            assert 'legend.json' in archive.namelist()
        if failure == 'presentation':
            assert 'spec.json' in archive.namelist()


def test_gradio_download_reset_and_export_failure(tmp_path):
    chart = base_chart()
    image_path = tmp_path / 'chart.png'
    cv2.imwrite(str(image_path), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(k, text, conf)
                     for k, (text, conf) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    event = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(event.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_ui.analysis.run_detection', return_value=frame),\
             patch('src.gradio_ui.analysis.run_ocr', return_value=ocr):
            success = invoke(event.fn, str(image_path), .25, .7, 'eng')
            assert success[index]['visible']
            with ZipFile(success[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert 'annotated.png' in archive.namelist()
                assert record['outputs']['spec']['categories'] == ['C0', 'C1', 'C2']
            rejected = invoke(event.fn, None, .25, .7, 'eng')
            assert rejected[index]['visible']
            assert rejected[index]['value'] != success[index]['value']
            with ZipFile(rejected[index]['value']) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                assert record['outcome'] == 'failed'
                assert 'input.png' not in archive.namelist()
            with patch('src.diagnostics.os.replace', side_effect=OSError('disk full')):
                failed_export = invoke(event.fn, str(image_path), .25, .7, 'eng')
            assert not failed_export[index]['visible']
            assert 'Diagnostic export failed' in failed_export[index+1]['value']
            assert failed_export[8] is not None  # Review remains available.
        for name in ('ready_results', 'clear_all'):
            callback = next(fn for fn in demo.fns.values() if fn.api_name == name)
            updates = invoke(callback.fn)
            offset = 12 if name == "clear_all" else 0
            assert not updates[index+offset]['visible']
            assert updates[index+offset]['value'] is None
    finally:
        demo.close(verbose=False)


def test_report_failure_keeps_app_analysis_and_shows_specific_error(tmp_path):
    chart = base_chart()
    image = tmp_path / 'chart.png'
    assert cv2.imwrite(str(image), chart.image)
    frame = chart.frame()
    ocr = OCRFrame(frame.source_key, tuple(RecognizedText(key, text, confidence)
                   for key, (text, confidence) in chart.texts.items()))
    demo = build_app(object(), ocr_backend=object(), device='cpu')
    binding = next(fn for fn in demo.fns.values() if fn.api_name == 'analyze')
    index = next(i for i, component in enumerate(binding.outputs)
                 if getattr(component, 'label', '') == 'Download analysis diagnostics (ZIP)')
    try:
        with patch('src.gradio_ui.analysis.run_detection', return_value=frame),\
             patch('src.gradio_ui.analysis.run_ocr', return_value=ocr),\
             patch('src.diagnostics_report.DiagnosticReport.render', side_effect=RuntimeError('formatting failed')):
            updates = invoke(binding.fn, str(image), .25, .7, 'eng')
        assert updates[8] is not None  # Analysis review state survived packaging failure.
        assert not updates[index]['visible'] and updates[index]['value'] is None
        assert 'Diagnostic export failed' in updates[index + 1]['value']
        assert 'report generation' in updates[index + 1]['value']
        assert 'formatting failed' in updates[index + 1]['value']
    finally:
        demo.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
