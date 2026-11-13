"""Runnable Gradio vertical slice without loading a real checkpoint."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from src.borealis import BOREALIS_CSS
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
from src.gradio_app import (
    AnalysisResult,
    DEFAULT_CONFIDENCE,
    DEFAULT_HOST,
    DEFAULT_NMS_IOU,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_PORT,
    DEFAULT_WEIGHTS,
    MAX_UPLOAD_BYTES,
    LEGEND_WARNING,
    OCR_WARNING,
    RECONSTRUCTION_WARNING,
    ManualGenerationBlocked,
    ReviewModel,
    UploadValidationError,
    alert_message,
    analyze_image,
    analyzing_status,
    build_review_model,
    build_app,
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
from src.tactile.completeness import (
    CANNOT_BUILD,
    NEEDS_INPUT,
    READY,
    Completeness,
    MissingItem,
)


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


class GradioAppTest(unittest.TestCase):
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

    def test_argument_defaults_and_port_validation(self):
        args = parse_args([])
        self.assertEqual(args.weights, DEFAULT_WEIGHTS)
        self.assertEqual(args.host, DEFAULT_HOST)
        self.assertEqual(args.port, DEFAULT_PORT)
        self.assertIsNone(args.device)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--port", "0"])
            with self.assertRaises(SystemExit):
                parse_args(["--port", "not-a-port"])

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

    def test_valid_analysis_forwards_settings_and_swaps_bgr_to_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(
                Path(temporary), rgb=(30, 20, 10)  # OpenCV reads this as BGR 10,20,30.
            )
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            progress = Mock()
            model = object()
            backend = object()
            ocr = OCRFrame("input/chart.png", ())
            with patch("src.gradio_app.run_detection", return_value=frame) as run, \
                 patch("src.gradio_app.run_ocr", return_value=ocr) as recognize:
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
                model, image_path, confidence=0.4, nms_iou=0.6, device="cpu"
            )
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.detection_count, 0)
            self.assertEqual(result.ocr_frame, ocr)
            recognize.assert_called_once()
            self.assertIs(recognize.call_args.args[2], backend)
            self.assertEqual(recognize.call_args.args[3], "eng")
            self.assertIn("Detected 0 chart elements.", result.summary)
            self.assertIn("OCR attempted 0 text regions", result.summary)
            self.assertEqual(tuple(result.overlay_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(tuple(result.source_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(progress.call_count, 7)

    def test_reconstruction_reuses_results_and_applies_edge_completeness(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            reconstructed = make_spec()
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", return_value=ocr), \
                 patch("src.gradio_app.match_legend_to_bars", return_value=mapping), \
                 patch("src.gradio_app.reconstruct_vbar",
                       return_value=reconstructed) as reconstruct, \
                 patch("src.gradio_app.cut_off_flags",
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

    def test_interpretation_error_keeps_partial_results_and_explains_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", return_value=ocr), \
                 patch("src.gradio_app.match_legend_to_bars", return_value=mapping), \
                 patch("src.gradio_app.reconstruct_vbar", side_effect=
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
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", return_value=ocr), \
                 patch("src.gradio_app.match_legend_to_bars", return_value=mapping), \
                 patch("src.gradio_app.reconstruct_vbar", return_value=make_spec()), \
                 patch("src.gradio_app.check_completeness",
                       side_effect=RuntimeError("classification failed")), \
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
            ))
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", side_effect=RuntimeError("missing")), \
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.detection_frame, frame)
            self.assertIsNone(result.ocr_frame)
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIn("Detected 1 chart element.", result.summary)
            self.assertIsNotNone(result.overlay_rgb)

    def test_matcher_failure_is_independent_and_can_coexist_with_ocr_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(5, 5, 20, 70), 0.9),
            ))
            with patch("src.gradio_app.run_detection", return_value=frame), \
                 patch("src.gradio_app.run_ocr", side_effect=RuntimeError("ocr")), \
                 patch("src.gradio_app.match_legend_to_bars",
                       side_effect=RuntimeError("matcher")), \
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.legend_warning, LEGEND_WARNING)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIsNone(result.legend_mapping)
            self.assertNotIn("remained unmatched", result.summary)

    def test_invalid_upload_never_calls_model_and_mismatched_frame_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "broken.png"
            broken.write_bytes(b"not an image")
            with patch("src.gradio_app.run_detection") as run:
                with self.assertRaises(UploadValidationError):
                    analyze_image(object(), broken, 0.25, 0.70)
                run.assert_not_called()

            valid = self._image(root)
            mismatched = DetectionFrame("input/chart.png", 99, 80, ())
            with patch("src.gradio_app.run_detection", return_value=mismatched):
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

    def test_accessible_status_helpers_cover_ready_busy_success_zero_and_error(self):
        self.assertIn('role="status"', ready_status())
        self.assertIn('aria-live="polite"', analyzing_status())
        self.assertIn("No detections found", complete_status(0))
        self.assertIn("1 detection found", complete_status(1))
        self.assertIn("3 detections found", complete_status(3))
        self.assertIn("OCR attempted 2 text regions", complete_status(3, 2))
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

    def test_app_factory_wires_controls_states_and_private_events(self):
        with patch("src.gradio_app.load_model") as load, \
             patch("gradio.Blocks.launch") as launch:
            demo = build_app(object(), device="cpu")
        load.assert_not_called()
        launch.assert_not_called()
        config = demo.get_config_file()
        components = config["components"]
        labels = {component["props"].get("label") for component in components}
        self.assertIn("Vertical bar chart image", labels)
        self.assertIn("Minimum detection confidence", labels)
        self.assertIn("NMS IoU for duplicate suppression", labels)
        self.assertIn("OCR language", labels)
        self.assertIn("Annotated detections", labels)
        self.assertIn("Image view", labels)
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

        begin = next(fn.fn for fn in demo.fns.values() if fn.api_name == "begin_analysis")
        begin_updates = begin()
        self.assertEqual(len(begin_updates), 33)
        self.assertTrue(all(update["interactive"] is False for update in begin_updates[:6]))
        finish = next(fn.fn for fn in demo.fns.values() if fn.api_name == "finish_analysis")
        self.assertTrue(all(update["interactive"] is True for update in finish()))
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = ready()
        self.assertEqual(len(ready_updates), 27)
        self.assertTrue(ready_updates[1]["visible"])
        self.assertIsNone(ready_updates[8])
        self.assertTrue(all(not update["visible"] for update in ready_updates[2:8]))
        self.assertFalse(ready_updates[9]["visible"])
        self.assertFalse(ready_updates[12]["visible"])
        self.assertFalse(ready_updates[13]["interactive"])
        self.assertTrue(all(not update["visible"] for update in ready_updates[14:24]))
        self.assertIsNone(ready_updates[24])
        self.assertIsNone(ready_updates[25])
        self.assertFalse(ready_updates[26]["visible"])

        analyze = inference_fn.fn
        display_frame = DetectionFrame("input/chart.png", 2, 2, (
            Detection("title", "chart_title", Box(0, 0, 2, 1), 0.9),
        ))
        display_ocr = OCRFrame("input/chart.png", (
            RecognizedText("title", "Sales", 0.8),
        ))
        original = np.full((2, 2, 3), 7, dtype=np.uint8)
        success = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, display_ocr, None, source_rgb=original,
        )
        with patch("src.gradio_app.analyze_image", return_value=success):
            success_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertIn("1 detection found", success_updates[0]["value"])
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
        self.assertIs(success_updates[24], original)
        self.assertIs(success_updates[25], success.overlay_rgb)
        self.assertTrue(success_updates[26]["visible"])
        self.assertTrue(success_updates[26]["interactive"])

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
        with patch("src.gradio_app.analyze_image", return_value=mapped_result):
            mapped_updates = analyze("chart.png", 0.25, 0.70, "eng")
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

        spec_blocked_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None, reconstructed,
            Completeness(CANNOT_BUILD, blockers=["Ambiguous structure."]), None,
        )
        with patch("src.gradio_app.analyze_image", return_value=spec_blocked_result):
            spec_blocked_updates = analyze("chart.png", 0.25, 0.70, "eng")
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
        with patch("src.gradio_app.analyze_image", return_value=blocked_result):
            blocked_updates = analyze("chart.png", 0.25, 0.70, "eng")
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
        with patch("src.gradio_app.analyze_image", return_value=warning):
            warning_updates = analyze("chart.png", 0.25, 0.70, "eng")
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
            "src.gradio_app.analyze_image",
            side_effect=UploadValidationError("Upload a supported image."),
        ):
            failure_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertFalse(failure_updates[2]["visible"])
        self.assertFalse(failure_updates[3]["visible"])
        self.assertFalse(failure_updates[4]["visible"])
        self.assertTrue(failure_updates[23]["visible"])
        self.assertIn('role="alert"', failure_updates[23]["value"])

        private_path = "/private/tmp/gradio/secret-chart.png"
        with self.assertLogs("src.gradio_app", level="ERROR"), patch(
            "src.gradio_app.analyze_image",
            side_effect=RuntimeError(private_path),
        ):
            unexpected_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(unexpected_updates[23]["visible"])
        self.assertNotIn(private_path, unexpected_updates[23]["value"])
        self.assertNotIn("RuntimeError", unexpected_updates[23]["value"])

        change_events = [dependency for dependency in config["dependencies"]
                         if dependency["targets"] and dependency["targets"][0][1] == "change"]
        self.assertEqual(len(change_events), 2)
        self.assertTrue(all(not event["queue"] for event in change_events))
        toggle_events = [dependency for dependency in change_events
                         if dependency.get("api_name") == "toggle_annotations"]
        self.assertEqual(len(toggle_events), 1)
        self.assertEqual(toggle_events[0]["api_visibility"], "private")
        toggle_fn = next(fn.fn for fn in demo.fns.values()
                         if fn.api_name == "toggle_annotations")
        annotated_pixels = np.full((2, 2, 3), 9, dtype=np.uint8)
        source_pixels = np.full((2, 2, 3), 3, dtype=np.uint8)
        np.testing.assert_array_equal(
            toggle_fn("Original", source_pixels, annotated_pixels)["value"], source_pixels
        )
        np.testing.assert_array_equal(
            toggle_fn("Annotated", source_pixels, annotated_pixels)["value"],
            annotated_pixels,
        )
        review_events = [dependency for dependency in config["dependencies"]
                         if dependency.get("api_name") == "update_review"]
        self.assertEqual(len(review_events), 1)
        self.assertEqual(review_events[0]["api_visibility"], "private")
        self.assertFalse(review_events[0]["queue"])
        review_fn = next(fn.fn for fn in demo.fns.values()
                         if fn.api_name == "update_review")
        review_model = build_review_model(make_spec(), Completeness(READY))
        invalid_review = review_fn(review_model, review_text_rows(review_model)[:-1])
        self.assertIs(invalid_review[0], review_model)
        self.assertTrue(invalid_review[2]["visible"])
        changed_rows = review_text_rows(review_model)
        changed_rows[0] = ["category[0]", "Updated title"]
        valid_review = review_fn(review_model, changed_rows)
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
        generated_rows = review_text_rows(review_model)
        generated_rows[0][1] = "Updated sales"
        generated = generate_fn.fn(review_model, generated_rows)
        self.assertEqual(generated[1].spec.title, "Updated sales")
        self.assertEqual(generated[2]["value"][0], ["Chart title", "Updated sales"])
        self.assertTrue(generated[5]["visible"])
        self.assertTrue(generated[7]["visible"])
        self.assertTrue(generated[8]["visible"])
        self.assertTrue(generated[9]["visible"])
        self.assertFalse(generated[10]["visible"])
        demo.close(verbose=False)

    def test_main_loads_once_and_uses_private_styled_launch(self):
        model = object()
        backend = object()
        demo = Mock()
        theme = object()
        with patch("src.gradio_app._import_gradio"), \
             patch("src.gradio_app.load_model", return_value=model) as load, \
             patch("src.gradio_app.TesseractBackend", return_value=backend) as load_ocr, \
             patch("src.gradio_app.build_app", return_value=demo) as build, \
             patch("src.gradio_app.borealis_theme", return_value=theme):
            main(["--weights", "custom.pt", "--device", "cpu",
                  "--host", "0.0.0.0", "--port", "9000"])
        load.assert_called_once_with(Path("custom.pt"))
        load_ocr.assert_called_once_with()
        build.assert_called_once_with(model, ocr_backend=backend, device="cpu")
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


if __name__ == "__main__":
    unittest.main()
