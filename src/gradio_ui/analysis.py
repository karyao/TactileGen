"""Validate an image and run the chart analysis stages, without depending on Gradio."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.bar_orientation import (
    MIN_BAR_CONFIDENCE,
    MIN_ORIENTATION_SCORE,
    MIN_ORIENTATION_SHARE,
    BarOrientationResult,
    resolve_bar_orientation,
)
from src.contracts import (
    ChartInterpretationError,
    TEXT_ROLES,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
    VerticalBarSpec,
)
from src.detection import DEFAULT_IMGSZ, run_detection, validate_imgsz
from src.diagnostics.tracking import AnalysisDiagnostics
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
from src.reconstruction.legend_matcher import filter_bars_by_legend, match_legend_to_bars
from src.reconstruction.bottom_boundary import filter_vertical_bottom
from src.bar_filter import drop_unmatched_bars
from src.ocr import run_ocr
from src.presentation import render_detection_overlay, summarize_detections
from src.reconstruction.vertical import reconstruct_vbar
from src.reconstruction.horizontal import (
    reconstruct_hbar,
)
from src.tactile.completeness import (
    Completeness,
    check_completeness,
    completeness_from_error,
    cut_off_flags,
)
from src.ocr.tesseract import TesseractBackend


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
    orientation_result: BarOrientationResult | None = None
    ocr_detection_frame: DetectionFrame | None = None

    def __post_init__(self) -> None:
        if (self.ocr_frame is None) != (self.ocr_detection_frame is None):
            raise ValueError("OCR results and their detection frame must be supplied together")


def orientation_summary(result: BarOrientationResult) -> str:
    """Describe the decision and any required rerun in the browser."""
    if result.status == "resolved":
        return (
            f"**Bar orientation: {result.orientation}** ({result.source}). "
            f"Suppressed {len(result.suppressed_ids)} opposite-orientation detections."
        )
    explanations = {
        "no_qualifying_bars": "No usable bars were detected.",
        "insufficient_evidence": "The detected bars provide too little evidence.",
        "ambiguous_orientation": "The detected bars provide conflicting orientation evidence.",
        "requested_orientation_unavailable": (
            f"No usable {result.preference} bars were detected. "
            "Check the requested orientation or image."
        ),
    }
    return (
        "Bar orientation is unclear. Choose Vertical or Horizontal and analyze again. "
        + explanations[result.reason]
        + " OCR, legend matching, and reconstruction were skipped."
    )


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
    bar_orientation: str = "auto",
    keep_legend_matches: bool = False,
    imgsz: int = DEFAULT_IMGSZ,
    language: str = DEFAULT_OCR_LANGUAGE,
    ocr_backend=None,
    ocr_backend_factory: Callable[[], Any] | None = None,
    device: str | None = None,
    progress: Callable[..., None] | None = None,
    diagnostics: AnalysisDiagnostics | None = None,
) -> AnalysisResult:
    """Detect, resolve bar orientation, then read and reconstruct one chart.

    Uncertain orientation returns non-bar annotations without starting OCR.
    ``detection_frame`` contains effective detections, or the non-bar display
    frame when uncertain; diagnostics retain the raw detector output.
    """
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
        raw_frame = run_detection(
            model, path, confidence=confidence, nms_iou=nms_iou, device=device,
            imgsz=imgsz, **diagnostic_options,
        )
        if image.shape[:2] != (raw_frame.height, raw_frame.width):
            raise ValueError("image dimensions do not match the detection frame")
        if diagnostics is not None:
            diagnostics.snapshot("detections", raw_frame)
            diagnostics.source_images(image, raw_frame)
            diagnostics.record["counts"].update(
                detections=len(raw_frame.detections),
                text_regions=sum(d.role in TEXT_ROLES for d in raw_frame.detections),
            )
    with stage("orientation_resolution"):
        orientation_result = resolve_bar_orientation(raw_frame, bar_orientation)
        if diagnostics is not None:
            resolved = orientation_result.status == "resolved"
            effective = orientation_result.effective_frame
            diagnostics.event(
                "orientation_resolution",
                "bar_orientation_resolved" if resolved else "bar_orientation_uncertain",
                preference=orientation_result.preference,
                status=orientation_result.status,
                source=orientation_result.source,
                reason=orientation_result.reason,
                orientation=orientation_result.orientation,
                vertical_score=orientation_result.vertical_score,
                horizontal_score=orientation_result.horizontal_score,
                min_bar_confidence=MIN_BAR_CONFIDENCE,
                min_orientation_score=MIN_ORIENTATION_SCORE,
                min_orientation_share=MIN_ORIENTATION_SHARE,
                suppressed_ids=list(orientation_result.suppressed_ids),
                raw_detections=len(raw_frame.detections),
                effective_detections=len(effective.detections) if resolved else None,
            )
            if resolved:
                assert effective is not None
                diagnostics.snapshot("effective_detections", effective)
                diagnostics.record["counts"]["effective_detections"] = (
                    len(effective.detections))
            else:
                diagnostics.warning("bar_orientation_uncertain",
                                    "orientation_resolution")
    if orientation_result.status == "resolved":
        frame = orientation_result.effective_frame
        assert frame is not None
        _report_progress(progress, 0.60, "Reading detected text")
        ocr_frame = None
        ocr_detection_frame = None
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
                ocr_detection_frame = frame
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
        horizontal = orientation_result.orientation == "horizontal"
        bottom_boundary = None if horizontal else filter_vertical_bottom(frame)
        if bottom_boundary is not None:
            frame = bottom_boundary.frame
            if diagnostics is not None:
                diagnostics.event(
                    "reconstruction", "vertical_bottom_boundary",
                    baseline_y=bottom_boundary.baseline_y,
                    confidence_share=bottom_boundary.confidence_share,
                    retained_bar_ids=list(bottom_boundary.retained_bar_ids),
                    rejected_bar_ids=list(bottom_boundary.rejected_bar_ids),
                    rejected_tick_ids=list(bottom_boundary.rejected_tick_ids),
                )
                diagnostics.snapshot("bottom_filtered_detections", frame)
        # The horizontal reconstructor does not accept a diagnostics collector.
        reconstruction_options = {} if horizontal else {
            **diagnostic_options,
            **({"provisional_baseline": bottom_boundary.baseline_y}
               if bottom_boundary is not None else {}),
        }
        legend_mapping = None
        legend_warning = None
        mismatched_bar_ids = ()
        removed_bar_ids = []
        selection_error = None
        try:
            with stage("legend_matching"):
                if not horizontal:
                    frame = drop_unmatched_bars(
                        frame, match_legend_to_bars(image, frame, ocr_frame, resolve_by_colour=False)
                    ) or frame
                colour_filter = filter_bars_by_legend(
                    image, frame, ocr_frame,
                    bar_role="horizontal_bar" if horizontal else "vertical_bar",
                    **diagnostic_options)
                legend_mapping = colour_filter.mapping
                mismatched_bar_ids = colour_filter.mismatched_bar_ids
                if mismatched_bar_ids:
                    legend_warning = (
                        f"{len(mismatched_bar_ids)} candidate bar detection(s) could not be "
                        "matched to the legend. Reconstruction may exclude some candidates; "
                        "review the series assignments."
                    )
                if diagnostics is not None:
                    diagnostics.snapshot("legend", legend_mapping)
                    diagnostics.record["counts"]["colour_mismatched_bars"] = len(mismatched_bar_ids)
                    diagnostics.record["counts"]["colour_kept_bars"] = sum(
                        item.outcome == "kept" for item in colour_filter.decisions)
                if keep_legend_matches:
                    # A colour verdict is independent of whether OCR could name
                    # the legend. Unknown/skipped colours are not verified matches.
                    kept_ids = {item.bar_id for item in colour_filter.decisions
                                if item.outcome == "kept"}
                    role = "horizontal_bar" if horizontal else "vertical_bar"
                    removed_bar_ids = [d.id for d in frame.detections
                                       if d.role == role and d.id not in kept_ids]
                    frame = replace(frame, detections=tuple(
                        d for d in frame.detections if d.role != role or d.id in kept_ids))
                    legend_mapping = replace(legend_mapping, assignments=tuple(
                        a for a in legend_mapping.assignments if a.bar_id in kept_ids))
                    mismatched_bar_ids = ()
                    legend_warning = None
                    if not kept_ids:
                        selection_error = (
                            "No bars could be verified against a detected legend colour. "
                            "Turn off Keep only legend-matched bars to review more detected bars, "
                            "or use an image with a clearer legend."
                        )
                    if diagnostics is not None:
                        diagnostics.event("legend_matching", "legend_bar_selection",
                            kept_bar_ids=sorted(kept_ids), removed_bar_ids=removed_bar_ids)
                        diagnostics.snapshot("selected_detections", frame)
                        diagnostics.record["counts"]["legend_removed_bars"] = len(removed_bar_ids)
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
        if selection_error:
            completeness = Completeness("cannot_build", blockers=[selection_error])
            if diagnostics is not None:
                diagnostics.skip("reconstruction", "no verified legend matches")
        elif ocr_frame is None or legend_mapping is None:
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
                    from src.reconstruction.axis_titles import resolve_axis_titles
                    frame = resolve_axis_titles(frame, ocr_frame,
                                                orientation="horizontal" if horizontal else "vertical",
                                                diagnostics=diagnostics)
                    reconstruct = reconstruct_hbar if horizontal else reconstruct_vbar
                    reconstructed = reconstruct(
                        frame, ocr_frame, image, legend_mapping=legend_mapping,
                        mismatched_bar_ids=mismatched_bar_ids, **reconstruction_options,
                        **({"rescue_bars": False} if keep_legend_matches else {}),
                    )
                    edge_flags = cut_off_flags(frame, image, **diagnostic_options)
                    candidate = replace(
                        reconstructed,
                        flags=[*reconstructed.flags, *edge_flags,
                               *([f"legend_bars_removed:{len(removed_bar_ids)}"]
                                 if removed_bar_ids else [])],
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
    else:
        # This frame is for presentation only; uncertainty has no effective frame.
        frame = replace(raw_frame, detections=tuple(
            d for d in raw_frame.detections
            if d.role not in ("vertical_bar", "horizontal_bar")
        ))
        ocr_frame = legend_mapping = spec = completeness = None
        ocr_detection_frame = None
        ocr_warning = legend_warning = reconstruction_warning = None
        if diagnostics is not None:
            reason = "bar_orientation_uncertain: " + orientation_result.reason
            for name in ("ocr", "legend_matching", "reconstruction"):
                diagnostics.skip(name, reason)
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
        summary = orientation_summary(orientation_result) + "\n\n" + summary
        if keep_legend_matches and orientation_result.status == "resolved":
            summary += (f"\n\n**Legend bar selection:** removed {len(removed_bar_ids)} "
                        "bar detection(s) without a verified legend colour match.")
        _report_progress(progress, 1.0, "Analysis result ready")
    return AnalysisResult(
        summary=summary,
        overlay_rgb=rgb_overlay,
        detection_count=len(frame.detections),
        detection_frame=frame,
        ocr_frame=ocr_frame,
        ocr_detection_frame=ocr_detection_frame,
        ocr_warning=ocr_warning,
        legend_mapping=legend_mapping,
        legend_warning=legend_warning,
        spec=spec,
        completeness=completeness,
        reconstruction_warning=reconstruction_warning,
        diagnostics=diagnostics,
        orientation_result=orientation_result,
    )
