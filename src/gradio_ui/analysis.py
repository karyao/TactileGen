"""Validate an image and run the chart analysis stages, without depending on Gradio."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.contracts import (
    ChartInterpretationError,
    TEXT_ROLES,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
    VerticalBarSpec,
)
from src.detection import DEFAULT_IMGSZ, run_detection, validate_imgsz
from src.diagnostics import AnalysisDiagnostics
from src.gradio_ui.settings import (
    MAX_UPLOAD_BYTES,
    MAX_IMAGE_PIXELS,
    SUPPORTED_FORMATS,
    DEFAULT_OCR_LANGUAGE,
    OCR_WARNING,
    LEGEND_WARNING,
    RECONSTRUCTION_WARNING,
    LOGGER,
)
from src.legend_matcher import match_legend_to_bars
from src.ocr import run_ocr
from src.presentation import render_detection_overlay, summarize_detections
from src.reconstruct import reconstruct_vbar
from src.reconstruct_hbar import (
    bar_orientation,
    match_legend_to_hbars,
    reconstruct_hbar,
)
from src.tactile.completeness import (
    Completeness,
    check_completeness,
    completeness_from_error,
    cut_off_flags,
)
from src.tesseract_ocr import TesseractBackend


class UploadValidationError(ValueError):
    """An upload problem that is safe to explain in the browser."""


@dataclass(frozen=True)
class AnalysisResult:
    """Successful stages and warnings from one analysis, ready for presentation."""

    summary: str
    overlay_rgb: Any
    detection_count: int
    detection_frame: DetectionFrame
    ocr_frame: OCRFrame | None
    ocr_warning: str | None
    legend_mapping: LegendMappingFrame | None = None
    legend_warning: str | None = None
    spec: VerticalBarSpec | None = None
    completeness: Completeness | None = None
    reconstruction_warning: str | None = None
    diagnostics: AnalysisDiagnostics | None = None


def _report_progress(progress, fraction: float, description: str) -> None:
    if progress is not None:
        progress(fraction, desc=description)


def decode_upload(image_path: str | Path | None):
    """Validate uploaded content and return its server path and BGR pixels."""
    import cv2
    from PIL import Image, UnidentifiedImageError

    if image_path is None:
        raise UploadValidationError("Choose a chart image before analyzing it.")
    path = Path(image_path)
    if not path.is_file():
        raise UploadValidationError("Choose a chart image before analyzing it.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The image must be 20 MB or smaller.")
    try:
        with Image.open(path) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise UploadValidationError("Upload a PNG, JPEG, or WebP image.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    "The decoded image must contain no more than 40 megapixels."
                )
            source.verify()
    except UploadValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise UploadValidationError(
            "The decoded image must contain no more than 40 megapixels."
        ) from exc
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise UploadValidationError(
            "The uploaded image could not be read. Try exporting it again."
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise UploadValidationError(
            "The uploaded image could not be read. Try exporting it again."
        )
    return path, image


def analyze_image(
    model,
    image_path: str | Path,
    confidence: float,
    nms_iou: float,
    *,
    imgsz: int = DEFAULT_IMGSZ,
    language: str = DEFAULT_OCR_LANGUAGE,
    ocr_backend=None,
    ocr_backend_factory: Callable[[], Any] | None = None,
    device: str | None = None,
    progress: Callable[..., None] | None = None,
    diagnostics: AnalysisDiagnostics | None = None,
) -> AnalysisResult:
    """Validate, detect, recognize text, summarize, and annotate one chart."""
    if ocr_backend is not None and ocr_backend_factory is not None:
        raise ValueError("Supply either ocr_backend or ocr_backend_factory, not both")
    import cv2

    def stage(name, **kwargs):
        if diagnostics is None:
            return nullcontext()
        return diagnostics.stage(name, **kwargs)

    diagnostic_options = {"diagnostics": diagnostics} if diagnostics is not None else {}
    if diagnostics is not None:
        diagnostics.detection_request(imgsz)
    with stage("validation"):
        try:
            validate_imgsz(imgsz)
        except ValueError as exc:
            raise UploadValidationError(str(exc)) from exc
        _report_progress(progress, 0.05, "Validating image")
        path, image = decode_upload(image_path)
        if diagnostics is not None:
            diagnostics.record["image"] = {
                "name": path.name,
                "width": image.shape[1],
                "height": image.shape[0],
            }
    with stage("detection"):
        _report_progress(progress, 0.20, "Detecting chart elements")
        frame = run_detection(
            model, path, confidence=confidence, nms_iou=nms_iou, device=device,
            imgsz=imgsz, **diagnostic_options,
        )
        if image.shape[:2] != (frame.height, frame.width):
            raise ValueError("image dimensions do not match the detection frame")
        if diagnostics is not None:
            diagnostics.snapshot("detections", frame)
            diagnostics.source_images(image, frame)
            diagnostics.record["counts"].update(
                detections=len(frame.detections),
                text_regions=sum(d.role in TEXT_ROLES for d in frame.detections),
            )
    _report_progress(progress, 0.60, "Reading detected text")
    ocr_frame = None
    ocr_warning = None
    try:
        with stage("ocr"):
            backend = (
                ocr_backend_factory() if ocr_backend_factory is not None
                else ocr_backend if ocr_backend is not None else TesseractBackend()
            )
            if diagnostics is not None:
                diagnostics.capture_metadata(model, backend, language)
            ocr_frame = run_ocr(
                image, frame, backend, language,
                **diagnostic_options,
            )
            if diagnostics is not None:
                diagnostics.snapshot("ocr", ocr_frame)
                readable = sum(bool(r.text.strip()) for r in ocr_frame.regions)
                diagnostics.record["counts"].update(
                    ocr_returned_regions=len(ocr_frame.regions), ocr_readable=readable,
                    ocr_empty=len(ocr_frame.regions) - readable,
                )
                diagnostics.log(
                    "ocr text_regions=%s returned=%s readable=%s empty=%s",
                    diagnostics.record["counts"]["text_regions"], len(ocr_frame.regions),
                    readable, len(ocr_frame.regions) - readable,
                )
    except Exception:
        if diagnostics is not None:
            diagnostics.warning("ocr_unavailable", "ocr")
        else:
            LOGGER.exception("Chart text recognition failed")
        ocr_warning = OCR_WARNING
    _report_progress(progress, 0.80, "Matching legend series")
    horizontal = bar_orientation(frame) == "horizontal"
    # The horizontal matcher/reconstructor do not accept a diagnostics collector.
    reconstruction_options = {} if horizontal else diagnostic_options
    legend_mapping = None
    legend_warning = None
    try:
        with stage("legend_matching"):
            match_legend = match_legend_to_hbars if horizontal else match_legend_to_bars
            legend_mapping = match_legend(
                image, frame, ocr_frame,
                **reconstruction_options,
            )
            if diagnostics is not None:
                diagnostics.snapshot("legend", legend_mapping)
    except Exception:
        if diagnostics is not None:
            diagnostics.warning("legend_matching_unavailable", "legend_matching")
        else:
            LOGGER.exception("Legend-to-bar matching failed")
        legend_warning = LEGEND_WARNING
    _report_progress(progress, 0.90, "Reconstructing chart values")
    spec = None
    completeness = None
    reconstruction_warning = None
    if ocr_frame is None or legend_mapping is None:
        reconstruction_warning = RECONSTRUCTION_WARNING
        if diagnostics is not None:
            dependencies = (("ocr", ocr_frame), ("legend_matching", legend_mapping))
            missing = [name for name, value in dependencies if value is None]
            diagnostics.skip(
                "reconstruction",
                "unavailable dependencies: " + ", ".join(missing),
            )
            diagnostics.warning("reconstruction_unavailable", "reconstruction")
    else:
        try:
            with stage("reconstruction", blocked_errors=(ChartInterpretationError,)):
                reconstruct = reconstruct_hbar if horizontal else reconstruct_vbar
                reconstructed = reconstruct(
                    frame, ocr_frame, image, legend_mapping=legend_mapping,
                    **reconstruction_options,
                )
                edge_flags = cut_off_flags(frame, image, **diagnostic_options)
                candidate = replace(
                    reconstructed,
                    flags=[*reconstructed.flags, *edge_flags],
                )
                if diagnostics is not None:
                    diagnostics.snapshot("spec", candidate)
                    diagnostics.record["spec_flags"] = list(candidate.flags)
                    for flag in candidate.flags:
                        diagnostics.warning(
                            flag.split(":", 1)[0],
                            "reconstruction",
                            flag=flag,
                        )
                assessment = check_completeness(candidate, **diagnostic_options)
                spec = candidate
                completeness = assessment
        except ChartInterpretationError as exc:
            if diagnostics is not None:
                diagnostics.warning("chart_interpretation_blocked", "reconstruction")
            completeness = completeness_from_error(exc, frame)
        except Exception:
            if diagnostics is not None:
                diagnostics.warning("reconstruction_unavailable", "reconstruction")
            else:
                LOGGER.exception("Chart reconstruction failed")
            reconstruction_warning = RECONSTRUCTION_WARNING
    if diagnostics is not None and completeness is not None:
        diagnostics.record["completeness_status"] = completeness.status
        diagnostics.snapshot("completeness", completeness)
    with stage("presentation"):
        _report_progress(progress, 0.95, "Rendering annotations")
        overlay = render_detection_overlay(image, frame, legend_mapping=legend_mapping)
        if diagnostics is not None:
            diagnostics.image("annotated.png", overlay)
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        summary = summarize_detections(
            frame,
            ocr=ocr_frame,
            legend_mapping=legend_mapping,
        )
        _report_progress(progress, 1.0, "Analysis result ready")
    return AnalysisResult(
        summary=summary,
        overlay_rgb=rgb_overlay,
        detection_count=len(frame.detections),
        detection_frame=frame,
        ocr_frame=ocr_frame,
        ocr_warning=ocr_warning,
        legend_mapping=legend_mapping,
        legend_warning=legend_warning,
        spec=spec,
        completeness=completeness,
        reconstruction_warning=reconstruction_warning,
        diagnostics=diagnostics,
    )
