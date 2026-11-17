"""Upload validation and chart analysis without loading a real checkpoint."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from src.contracts import (
    Box,
    ChartInterpretationError,
    Detection,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
)
from src.diagnostics import AnalysisDiagnostics
from src.gradio_app import (
    AnalysisResult,
    MAX_UPLOAD_BYTES,
    LEGEND_WARNING,
    OCR_WARNING,
    RECONSTRUCTION_WARNING,
    UploadValidationError,
    analyze_image,
    decode_upload,
)
from src.tactile.completeness import CANNOT_BUILD, NEEDS_INPUT

from tests.gradio_helpers import GradioTestCase, make_spec


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
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            diagnostics = AnalysisDiagnostics(settings={})
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame) as detect, \
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
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            progress = Mock()
            model = object()
            backend = object()
            ocr = OCRFrame("input/chart.png", ())
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame) as run, \
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
            self.assertEqual(result.detection_count, 0)
            self.assertEqual(result.ocr_frame, ocr)
            recognize.assert_called_once()
            self.assertIs(recognize.call_args.args[2], backend)
            self.assertEqual(recognize.call_args.args[3], "eng")
            self.assertIn("**Detected 0 chart elements**", result.summary)
            self.assertIn("- OCR: 0 of 0 text regions readable", result.summary)
            self.assertEqual(tuple(result.overlay_rgb[0, 0]), (30, 20, 10))
            self.assertEqual(progress.call_count, 7)

    def test_reconstruction_reuses_results_and_applies_edge_completeness(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            reconstructed = make_spec()
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr), \
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping), \
                 patch("src.gradio_ui.analysis.reconstruct_vbar",
                       return_value=reconstructed) as reconstruct, \
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
                frame = DetectionFrame("input/chart.png", 100, 80, ())
                ocr = OCRFrame("input/chart.png", ())
                mapping = LegendMappingFrame("input/chart.png", ())
                spec = replace(make_spec(), orientation="horizontal")
                diagnostics = AnalysisDiagnostics(settings={}) if diagnostics_enabled else None
                with (
                    patch("src.gradio_ui.analysis.run_detection", return_value=frame),
                    patch("src.gradio_ui.analysis.run_ocr", return_value=ocr),
                    patch("src.gradio_ui.analysis.bar_orientation", return_value="horizontal"),
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
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr), \
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping), \
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
            frame = DetectionFrame("input/chart.png", 100, 80, ())
            ocr = OCRFrame("input/chart.png", ())
            mapping = LegendMappingFrame("input/chart.png", ())
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
                 patch("src.gradio_ui.analysis.run_ocr", return_value=ocr), \
                 patch("src.gradio_ui.analysis.match_legend_to_bars", return_value=mapping), \
                 patch("src.gradio_ui.analysis.reconstruct_vbar", return_value=make_spec()), \
                 patch("src.gradio_ui.analysis.check_completeness",
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
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
                 patch("src.gradio_ui.analysis.run_ocr", side_effect=RuntimeError("missing")), \
                 self.assertLogs("src.gradio_app", level="ERROR"):
                result = analyze_image(
                    object(), image_path, 0.25, 0.70,
                    ocr_backend=object(), language="eng",
                )
            self.assertEqual(result.detection_frame, frame)
            self.assertIsNone(result.ocr_frame)
            self.assertEqual(result.ocr_warning, OCR_WARNING)
            self.assertEqual(result.reconstruction_warning, RECONSTRUCTION_WARNING)
            self.assertIn("**Detected 1 chart element**", result.summary)
            self.assertIsNotNone(result.overlay_rgb)

    def test_matcher_failure_is_independent_and_can_coexist_with_ocr_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = self._image(Path(temporary))
            frame = DetectionFrame("input/chart.png", 100, 80, (
                Detection("bar", "vertical_bar", Box(5, 5, 20, 70), 0.9),
            ))
            with patch("src.gradio_ui.analysis.run_detection", return_value=frame), \
                 patch("src.gradio_ui.analysis.run_ocr", side_effect=RuntimeError("ocr")), \
                 patch("src.gradio_ui.analysis.match_legend_to_bars",
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


if __name__ == "__main__":
    unittest.main()
