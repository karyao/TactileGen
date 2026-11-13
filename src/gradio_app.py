"""Local Gradio application for inspecting one chart detection at a time."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import re
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.borealis import BOREALIS_CSS, borealis_theme
from src.contract_io import write_ocr_frame
from src.contracts import (
    ChartInterpretationError,
    DetectionFrame,
    LegendMappingFrame,
    OCRFrame,
    VerticalBarSpec,
)
from src.detection import DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU, load_model, run_detection
from src.legend_matcher import match_legend_to_bars
from src.ocr import run_ocr
from src.presentation import (
    LEGEND_TABLE_HEADERS,
    OCR_TABLE_HEADERS,
    legend_rows,
    ocr_rows,
    render_detection_overlay,
    summarize_detections,
)
from src.reconstruct import reconstruct_vbar
from src.tactile.completeness import (
    READY,
    Completeness,
    Overrides,
    apply_overrides,
    check_completeness,
    completeness_from_error,
    cut_off_flags,
)
from src.tactile.generation import (
    GenerationError,
    GenerationResult,
    cleanup_generation,
    generate_tactile_artifacts,
)
from src.tactile.rules import format_report
from src.tesseract_ocr import TesseractBackend


DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
EXAMPLE_IMAGE = Path(__file__).parents[1] / "tests/fixtures/contracts/val/example.png"
DEFAULT_OCR_LANGUAGE = "eng"
OCR_WARNING = (
    "Text recognition was unavailable. Detection results are still shown. "
    "Check the terminal for details."
)
LEGEND_WARNING = (
    "Legend matching was unavailable. Detection and OCR results are still shown. "
    "Check the terminal for details."
)
RECONSTRUCTION_WARNING = (
    "Chart reconstruction was unavailable. Detection, OCR, and legend results "
    "that completed successfully are still shown. Check the terminal for details."
)

LOGGER = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """An upload problem that is safe to explain in the browser."""


@dataclass(frozen=True)
class AnalysisResult:
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


@dataclass(frozen=True)
class ReviewFieldBinding:
    """One immutable mapping between a review row and a specification field."""

    key: str
    label: str
    initial_text: str
    required: bool
    confirmation_required: bool = False


@dataclass(frozen=True)
class ReviewModel:
    """Authoritative specification and the text draft shown for its review."""

    spec: VerticalBarSpec
    bindings: tuple[ReviewFieldBinding, ...]
    draft_values: tuple[str, ...]


@dataclass(frozen=True)
class ManualGenerationResult:
    """A corrected review draft committed by successful artifact generation."""

    review_model: ReviewModel
    completeness: Completeness
    generation: GenerationResult
    override_notes: tuple[str, ...]


class ManualGenerationBlocked(ValueError):
    """A valid review draft whose corrected specification is not ready."""

    def __init__(self, review_model: ReviewModel, completeness: Completeness) -> None:
        super().__init__("The corrected specification is not ready for generation.")
        self.review_model = review_model
        self.completeness = completeness


def build_review_model(spec: VerticalBarSpec, completeness: Completeness) -> ReviewModel:
    """Build stable review bindings without deriving keys from display text."""
    confirmation_fields = {item.field for item in completeness.missing}
    bindings = [
        ReviewFieldBinding(
            "title", "Chart title", spec.title or "", True,
            "title" in confirmation_fields,
        ),
        ReviewFieldBinding(
            "x_axis_title", "X-axis title", spec.x_axis_title or "", True,
            "x_axis_title" in confirmation_fields,
        ),
        ReviewFieldBinding(
            "y_axis_title", "Y-axis title", spec.y_axis_title or "", True,
            "y_axis_title" in confirmation_fields,
        ),
    ]
    bindings.extend(
        ReviewFieldBinding(
            f"category[{index}]", f"Category {index}", text, True,
            f"category[{index}]" in confirmation_fields,
        )
        for index, text in enumerate(spec.categories)
    )
    multiple_series = len(spec.series) > 1
    bindings.extend(
        ReviewFieldBinding(
            f"series_name[{index}]", f"Series {index} name", series.name or "",
            multiple_series or series.name is not None,
            f"series_name[{index}]" in confirmation_fields,
        )
        for index, series in enumerate(spec.series)
    )
    frozen = tuple(bindings)
    return ReviewModel(spec, frozen, tuple(item.initial_text for item in frozen))


def review_text_rows(model: ReviewModel) -> list[list[str]]:
    """Rows for the editable table; labels always come from server bindings."""
    return [[binding.label, value]
            for binding, value in zip(model.bindings, model.draft_values, strict=True)]


def review_value_rows(spec: VerticalBarSpec) -> list[list[Any]]:
    """Flatten immutable reconstructed values for a read-only review table."""
    rows = []
    for series_index, series in enumerate(spec.series):
        for category_index, (category, value) in enumerate(
            zip(spec.categories, series.values, strict=True)
        ):
            rows.append([
                category_index,
                category,
                series_index,
                series.name or "(unnamed)",
                value,
            ])
    return rows


def update_review_draft(model: ReviewModel, rows: Any) -> ReviewModel:
    """Accept only text cells and map them by immutable server-side row order."""
    if not isinstance(rows, (list, tuple)) or len(rows) != len(model.bindings):
        raise ValueError(f"review must contain exactly {len(model.bindings)} rows")
    values = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"review row {index} must contain exactly 2 columns")
        if not isinstance(row[1], str):
            raise ValueError(f"review text in row {index} must be a string")
        # row[0] is deliberately ignored: it is display text, never a field key.
        values.append(row[1])
    return replace(model, draft_values=tuple(values))


def review_overrides(model: ReviewModel) -> Overrides:
    """Convert a valid draft to indexed overrides for the later generate action."""
    titles: dict[str, str] = {}
    categories: dict[int, str] = {}
    series_names: dict[int, str] = {}
    for binding, draft in zip(model.bindings, model.draft_values, strict=True):
        value = draft.strip()
        if binding.required and not value:
            raise ValueError(f"{binding.label} cannot be blank")
        initial = binding.initial_text.strip()
        if value == initial and not binding.confirmation_required:
            continue
        if not value:
            continue
        if binding.key in {"title", "x_axis_title", "y_axis_title"}:
            titles[binding.key] = value
            continue
        match = re.fullmatch(r"category\[(\d+)]", binding.key)
        if match:
            categories[int(match.group(1))] = value
            continue
        match = re.fullmatch(r"series_name\[(\d+)]", binding.key)
        if match:
            series_names[int(match.group(1))] = value
            continue
        raise ValueError(f"unknown review field: {binding.key}")
    return Overrides(
        title=titles.get("title"),
        x_axis_title=titles.get("x_axis_title"),
        y_axis_title=titles.get("y_axis_title"),
        categories=categories,
        series_names=series_names,
    )


def generate_review_submission(
    model: ReviewModel,
    rows: Any,
    artifact_root: Path | str,
) -> ManualGenerationResult:
    """Validate the latest review table, apply text, and generate ready artifacts."""
    draft = update_review_draft(model, rows)
    corrected, notes = apply_overrides(model.spec, review_overrides(draft))
    completeness = check_completeness(corrected)
    if completeness.status != READY:
        raise ManualGenerationBlocked(draft, completeness)
    committed_review = build_review_model(corrected, completeness)
    generation = generate_tactile_artifacts(corrected, artifact_root)
    return ManualGenerationResult(
        review_model=committed_review,
        completeness=completeness,
        generation=generation,
        override_notes=tuple(notes),
    )


def generation_report(result: ManualGenerationResult) -> str:
    """Format generated-page metadata as an in-page review report."""
    lines = [
        "### Tactile generation complete",
        "",
        "The gallery contains annotated review copies. The ZIP contains only "
        "clean SVG pages for embossing.",
        "",
        "**Applied corrections**",
    ]
    if result.override_notes:
        lines.extend(f"- {note}" for note in result.override_notes)
    else:
        lines.append("- No text corrections were needed.")
    if result.generation.notes:
        lines.extend(("", "**Generation notes**"))
        lines.extend(f"- {note}" for note in result.generation.notes)

    transcriber_review = False
    for page in result.generation.pages:
        lines.extend(("", f"#### {page.caption}"))
        if page.notes:
            lines.extend(f"- Note: {note}" for note in page.notes)
        if page.needs_review:
            transcriber_review = True
            lines.extend(f"- Needs review: {note}" for note in page.needs_review)
        if page.violations:
            transcriber_review = True
        lines.extend(("", "```text", format_report(list(page.violations)), "```"))
    if transcriber_review:
        lines.extend((
            "",
            "**Transcriber review is required before production.**",
        ))
    else:
        lines.extend((
            "",
            "Automated checks found no rule violations; this does not replace "
            "professional transcriber review.",
        ))
    return "\n".join(lines)


def ready_status() -> str:
    return '<div role="status" aria-live="polite">Ready for a chart image.</div>'


def analyzing_status() -> str:
    return '<div role="status" aria-live="polite">Analyzing chart.</div>'


def complete_status(detection_count: int = 0, ocr_count: int | None = None) -> str:
    return (
        '<div role="status" aria-live="polite">Analysis result ready.</div>'
    )


def alert_message(message: str) -> str:
    return f'<div role="alert">{message}</div>'


def completeness_summary(result: Completeness) -> str:
    """Render a compact, readable reconstruction assessment."""
    labels = {
        "ready": "Ready",
        "needs_input": "Needs input",
        "cannot_build": "Cannot build",
    }
    lines = [f"### Reconstruction: {labels.get(result.status, result.status)}"]
    if result.notes:
        lines.extend(("", "**Notes**", *(f"- {note}" for note in result.notes)))
    if result.missing:
        lines.extend((
            "",
            "**Missing or unconfirmed text**",
            *(f"- `{item.field}`: {item.reason}" for item in result.missing),
        ))
    if result.blockers:
        lines.extend(("", "**Blockers**", *(f"- {item}" for item in result.blockers)))
    return "\n".join(lines)


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install the application dependencies: "
            "python -m pip install -r requirements-app.txt"
        ) from exc
    return gr


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or mps")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def _report_progress(progress, fraction: float, description: str) -> None:
    if progress is not None:
        progress(fraction, desc=description)


def decode_upload(image_path: str | Path):
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
    language: str = DEFAULT_OCR_LANGUAGE,
    ocr_backend=None,
    device: str | None = None,
    progress: Callable[..., None] | None = None,
) -> AnalysisResult:
    """Validate, detect, recognize text, summarize, and annotate one chart."""
    import cv2

    _report_progress(progress, 0.05, "Validating image")
    path, image = decode_upload(image_path)
    _report_progress(progress, 0.20, "Detecting chart elements")
    frame = run_detection(
        model, path, confidence=confidence, nms_iou=nms_iou, device=device
    )
    if image.shape[:2] != (frame.height, frame.width):
        raise ValueError("image dimensions do not match the detection frame")
    _report_progress(progress, 0.60, "Reading detected text")
    ocr_frame = None
    ocr_warning = None
    try:
        backend = ocr_backend if ocr_backend is not None else TesseractBackend()
        ocr_frame = run_ocr(image, frame, backend, language)
    except Exception:
        LOGGER.exception("Chart text recognition failed")
        ocr_warning = OCR_WARNING
    _report_progress(progress, 0.80, "Matching legend series")
    legend_mapping = None
    legend_warning = None
    try:
        legend_mapping = match_legend_to_bars(image, frame, ocr_frame)
    except Exception:
        LOGGER.exception("Legend-to-bar matching failed")
        legend_warning = LEGEND_WARNING
    _report_progress(progress, 0.90, "Reconstructing chart values")
    spec = None
    completeness = None
    reconstruction_warning = None
    if ocr_frame is None or legend_mapping is None:
        reconstruction_warning = RECONSTRUCTION_WARNING
    else:
        try:
            reconstructed = reconstruct_vbar(
                frame, ocr_frame, image, legend_mapping=legend_mapping
            )
            candidate = replace(
                reconstructed,
                flags=[*reconstructed.flags, *cut_off_flags(frame, image)],
            )
            assessment = check_completeness(candidate)
            spec = candidate
            completeness = assessment
        except ChartInterpretationError as exc:
            completeness = completeness_from_error(exc, frame)
        except Exception:
            LOGGER.exception("Chart reconstruction failed")
            reconstruction_warning = RECONSTRUCTION_WARNING
    _report_progress(progress, 0.95, "Rendering annotations")
    overlay = render_detection_overlay(image, frame, legend_mapping=legend_mapping)
    rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    _report_progress(progress, 1.0, "Analysis result ready")
    return AnalysisResult(
        summary=summarize_detections(
            frame, ocr=ocr_frame, legend_mapping=legend_mapping
        ),
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
    )


def _download_path(root: Path, image_path: str | Path, ocr: OCRFrame) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(image_path).stem).strip("._-")
    path = root / f"{stem or 'chart'}_ocr_{uuid.uuid4().hex}.json"
    write_ocr_frame(path, ocr)
    return path


def _spec_download_path(
    root: Path, image_path: str | Path, spec: VerticalBarSpec
) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(image_path).stem).strip("._-")
    path = root / f"{stem or 'chart'}_spec_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")
    return path


def build_app(model, *, ocr_backend=None, device: str | None = None):
    """Build the interface around an already-loaded model without launching it."""
    gr = _import_gradio()
    backend = ocr_backend if ocr_backend is not None else TesseractBackend()
    artifacts = tempfile.TemporaryDirectory(prefix="tactilegen-gradio-ocr-")
    atexit.register(artifacts.cleanup)
    artifact_root = Path(artifacts.name)

    def result_updates(
        status: str,
        *,
        empty_message: str = "",
        summary: str | None = None,
        overlay=None,
        ocr_table: list[list[Any]] | None = None,
        ocr_download_path: Path | None = None,
        legend_table: list[list[Any]] | None = None,
        completeness_text: str | None = None,
        review_model: ReviewModel | None = None,
        generate_enabled: bool = False,
        spec_json: dict[str, Any] | None = None,
        spec_download_path: Path | None = None,
        generation_gallery: list[tuple[str, str]] | None = None,
        generation_report_text: str | None = None,
        generation_zip_path: Path | None = None,
        ocr_warning: str | None = None,
        legend_warning: str | None = None,
        reconstruction_warning: str | None = None,
        error_message: str | None = None,
    ):
        return (
            gr.update(value=status),
            gr.update(value=empty_message, visible=bool(empty_message)),
            gr.update(value=summary or "", visible=summary is not None),
            gr.update(value=overlay, visible=overlay is not None),
            gr.update(value=ocr_table or [], visible=ocr_table is not None),
            gr.update(
                value=ocr_download_path,
                visible=ocr_download_path is not None,
            ),
            gr.update(value=legend_table or [], visible=legend_table is not None),
            gr.update(
                value=completeness_text or "",
                visible=completeness_text is not None,
            ),
            review_model,
            gr.update(visible=review_model is not None),
            gr.update(
                value=review_text_rows(review_model) if review_model else [],
                row_count=len(review_model.bindings) if review_model else 0,
            ),
            gr.update(
                value=review_value_rows(review_model.spec) if review_model else [],
            ),
            gr.update(value="", visible=False),
            gr.update(interactive=generate_enabled),
            gr.update(value=spec_json or {}, visible=spec_json is not None),
            gr.update(
                value=spec_download_path,
                visible=spec_download_path is not None,
            ),
            gr.update(
                value=generation_gallery or [],
                visible=generation_gallery is not None,
            ),
            gr.update(
                value=generation_report_text or "",
                visible=generation_report_text is not None,
            ),
            gr.update(
                value=generation_zip_path,
                visible=generation_zip_path is not None,
            ),
            gr.update(value="", visible=False),
            gr.update(
                value=alert_message(ocr_warning) if ocr_warning else "",
                visible=ocr_warning is not None,
            ),
            gr.update(
                value=alert_message(legend_warning) if legend_warning else "",
                visible=legend_warning is not None,
            ),
            gr.update(
                value=(alert_message(reconstruction_warning)
                       if reconstruction_warning else ""),
                visible=reconstruction_warning is not None,
            ),
            gr.update(
                value=alert_message(error_message) if error_message else "",
                visible=error_message is not None,
            ),
        )

    def ready_results():
        return result_updates(
            ready_status(),
            empty_message="Upload a chart and choose Analyze chart.",
        )

    def begin_analysis():
        return (
            *(gr.update(interactive=False) for _ in range(6)),
            *result_updates(analyzing_status()),
        )

    def analyze_event(image_path, confidence, nms_iou, language, progress=gr.Progress()):
        try:
            result = analyze_image(
                model,
                image_path,
                confidence,
                nms_iou,
                language=language,
                ocr_backend=backend,
                device=device,
                progress=progress,
            )
            ocr_count = len(result.ocr_frame.regions) if result.ocr_frame is not None else None
            rows = (
                ocr_rows(result.detection_frame, result.ocr_frame)
                if result.ocr_frame is not None else []
            )
            download = (
                _download_path(artifact_root, image_path, result.ocr_frame)
                if result.ocr_frame is not None else None
            )
            mapping_rows = (
                legend_rows(result.detection_frame, result.legend_mapping)
                if result.legend_mapping is not None else []
            )
            show_mapping = result.legend_mapping is not None and bool(mapping_rows)
            spec_data = asdict(result.spec) if result.spec is not None else None
            review = (
                build_review_model(result.spec, result.completeness)
                if result.spec is not None and result.completeness is not None else None
            )
            spec_download = (
                _spec_download_path(artifact_root, image_path, result.spec)
                if result.spec is not None else None
            )
            return result_updates(
                complete_status(result.detection_count, ocr_count),
                summary=result.summary,
                overlay=result.overlay_rgb,
                ocr_table=rows if result.ocr_frame is not None else None,
                ocr_download_path=download,
                legend_table=mapping_rows if show_mapping else None,
                completeness_text=(
                    completeness_summary(result.completeness)
                    if result.completeness is not None else None
                ),
                review_model=review,
                generate_enabled=(
                    result.completeness is not None
                    and result.completeness.status != "cannot_build"
                ),
                spec_json=spec_data,
                spec_download_path=spec_download,
                ocr_warning=result.ocr_warning,
                legend_warning=result.legend_warning,
                reconstruction_warning=result.reconstruction_warning,
            )
        except UploadValidationError as exc:
            LOGGER.info("Rejected upload: %s", exc)
            message = str(exc)
        except Exception:
            LOGGER.exception("Chart analysis failed")
            message = "Analysis failed. Check the terminal and try another image."
        return result_updates(ready_status(), error_message=message)

    def update_review_event(model, rows):
        if model is None:
            return None, gr.update(value=[]), gr.update(value="", visible=False)
        try:
            updated = update_review_draft(model, rows)
        except ValueError as exc:
            return (
                model,
                gr.update(value=review_text_rows(model)),
                gr.update(value=alert_message(str(exc)), visible=True),
            )
        return (
            updated,
            gr.update(value=review_text_rows(updated)),
            gr.update(value="", visible=False),
        )

    def generate_event(model, rows):
        unchanged = gr.update()
        if model is None:
            return (
                unchanged, unchanged, unchanged, unchanged, unchanged,
                unchanged, unchanged, unchanged, unchanged, unchanged,
                gr.update(
                    value=alert_message("Analyze a chart before generating."),
                    visible=True,
                ),
            )
        submitted = None
        try:
            submitted = generate_review_submission(model, rows, artifact_root)
            corrected = submitted.review_model.spec
            spec_path = _spec_download_path(
                artifact_root, corrected.source_key, corrected
            )
            gallery = [
                (str(page.review_svg), page.caption)
                for page in submitted.generation.pages
            ]
            return (
                gr.update(value=completeness_summary(submitted.completeness)),
                submitted.review_model,
                gr.update(value=review_text_rows(submitted.review_model)),
                gr.update(value=review_value_rows(corrected)),
                gr.update(value="", visible=False),
                gr.update(value=asdict(corrected), visible=True),
                gr.update(value=spec_path, visible=True),
                gr.update(value=gallery, visible=True),
                gr.update(value=generation_report(submitted), visible=True),
                gr.update(value=submitted.generation.zip_path, visible=True),
                gr.update(value="", visible=False),
            )
        except ManualGenerationBlocked as exc:
            return (
                gr.update(value=completeness_summary(exc.completeness), visible=True),
                exc.review_model,
                gr.update(value=review_text_rows(exc.review_model)),
                unchanged,
                unchanged,
                unchanged,
                unchanged,
                unchanged,
                unchanged,
                unchanged,
                gr.update(value=alert_message(str(exc)), visible=True),
            )
        except ValueError as exc:
            if submitted is not None:
                try:
                    cleanup_generation(submitted.generation, artifact_root)
                except Exception:
                    LOGGER.exception("Could not clean up unpublished tactile artifacts")
            return (
                unchanged, model, unchanged, unchanged,
                gr.update(value=alert_message(str(exc)), visible=True),
                unchanged, unchanged, unchanged, unchanged, unchanged, unchanged,
            )
        except GenerationError as exc:
            LOGGER.exception("Tactile artifact generation failed at %s", exc.stage)
            return (
                unchanged, model, unchanged, unchanged, unchanged,
                unchanged, unchanged, unchanged, unchanged, unchanged,
                gr.update(value=alert_message(str(exc)), visible=True),
            )
        except Exception:
            if submitted is not None:
                try:
                    cleanup_generation(submitted.generation, artifact_root)
                except Exception:
                    LOGGER.exception("Could not clean up unpublished tactile artifacts")
            LOGGER.exception("Manual tactile generation failed")
            return (
                unchanged, model, unchanged, unchanged, unchanged,
                unchanged, unchanged, unchanged, unchanged, unchanged,
                gr.update(
                    value=alert_message(
                        "Tactile generation failed. Check the terminal and try again."
                    ),
                    visible=True,
                ),
            )

    def finish_analysis():
        return tuple(gr.update(interactive=True) for _ in range(6))

    def clear_all():
        return (
            gr.update(value=None, interactive=True),
            gr.update(value=DEFAULT_CONFIDENCE, interactive=True),
            gr.update(value=DEFAULT_NMS_IOU, interactive=True),
            gr.update(value=DEFAULT_OCR_LANGUAGE, interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            *ready_results(),
        )

    with gr.Blocks(
        title="TactileGen — Statistical Chart Element Detector",
        analytics_enabled=False,
        fill_width=False,
    ) as demo:
        with gr.Column(elem_classes="borealis-app"):
            with gr.Column(elem_classes="borealis-header"):
                gr.Markdown(
                    "**Statistical chart element detector**\n\n"
                    "# TactileGen\n\n"
                    "Inspect the regions detected by YOLO in a vertical categorical "
                    "bar chart, read detected text, and associate legend series "
                    "with bars. Reconstruct chart labels, series, and numeric values "
                    "when the image contains enough information."
                )
                gr.Markdown(
                    "**Scope:** Vertical categorical bar charts only. The detector "
                    "returns chart elements, bounding boxes, OCR text, provisional "
                    "legend-to-bar matches, and a completeness-checked reconstruction. "
                    "Incomplete results remain available for inspection.",
                    elem_classes="borealis-scope",
                )

            with gr.Row(elem_classes="borealis-workspace-grid"):
                with gr.Column(elem_classes="borealis-control-rail"):
                    gr.Markdown("## Input", elem_classes="borealis-section-title")
                    image_input = gr.Image(
                        type="filepath",
                        sources=["upload"],
                        image_mode=None,
                        format="png",
                        label="Vertical bar chart image",
                        placeholder="Upload a chart image",
                        buttons=["fullscreen"],
                        alt_text="Preview of the uploaded chart",
                        elem_classes="borealis-upload-image",
                    )
                    gr.Markdown(
                        "PNG, JPEG, or WebP · up to 20 MB and 40 megapixels.",
                        elem_classes="borealis-helper",
                    )
                    gr.Examples(
                        examples=[[str(EXAMPLE_IMAGE)]],
                        inputs=[image_input],
                        label="Example chart",
                        run_on_click=False,
                        cache_examples=False,
                    )
                    confidence = gr.Slider(
                        0.05,
                        1.0,
                        value=DEFAULT_CONFIDENCE,
                        step=0.05,
                        label="Minimum detection confidence",
                        info="Hide detections below this confidence score.",
                        elem_classes="borealis-setting",
                    )
                    with gr.Accordion(
                        "Advanced settings",
                        open=False,
                        elem_classes="borealis-advanced",
                    ):
                        nms_iou = gr.Slider(
                            0.10,
                            0.95,
                            value=DEFAULT_NMS_IOU,
                            step=0.05,
                            label="NMS IoU for duplicate suppression",
                            info=(
                                "Controls duplicate suppression during inference. "
                                "This is not evaluation or box-matching IoU."
                            ),
                            elem_classes="borealis-setting",
                        )
                        language = gr.Textbox(
                            value=DEFAULT_OCR_LANGUAGE,
                            label="OCR language",
                            info="Tesseract language code, such as eng or eng+fra.",
                            elem_classes="borealis-setting",
                        )
                    with gr.Row(elem_classes="borealis-actions"):
                        analyze_button = gr.Button("Analyze chart", variant="primary")
                        clear_button = gr.Button("Clear", variant="secondary")

                with gr.Column(elem_classes="borealis-workspace"):
                    gr.Markdown(
                        "## Analysis result",
                        elem_classes=[
                            "borealis-section-title",
                            "borealis-workspace-heading",
                        ],
                    )
                    status_output = gr.HTML(
                        ready_status(), elem_classes="borealis-status"
                    )
                    empty_output = gr.Markdown(
                        "Upload a chart and choose Analyze chart.",
                        visible=True,
                        elem_classes="borealis-empty",
                    )
                    summary_output = gr.Markdown(
                        "",
                        label="Detection summary",
                        visible=False,
                        elem_classes="borealis-summary",
                    )
                    annotated_output = gr.Image(
                        type="numpy",
                        format="png",
                        label="Annotated detections",
                        interactive=False,
                        visible=False,
                        buttons=["download", "fullscreen"],
                        alt_text="Uploaded chart with labeled detection boxes",
                        elem_classes="borealis-result-image",
                    )
                    ocr_output = gr.Dataframe(
                        value=[],
                        headers=list(OCR_TABLE_HEADERS),
                        datatype=["str", "str", "str", "number",
                                  "number", "number", "number", "number"],
                        type="array",
                        label="Recognized text",
                        interactive=False,
                        visible=False,
                        wrap=True,
                        elem_classes="borealis-ocr-table",
                    )
                    ocr_download = gr.DownloadButton(
                        "Download OCR JSON",
                        value=None,
                        visible=False,
                        variant="secondary",
                        elem_classes="borealis-ocr-download",
                    )
                    legend_output = gr.Dataframe(
                        value=[],
                        headers=list(LEGEND_TABLE_HEADERS),
                        datatype=["str", "str", "number", "str", "str", "str"],
                        type="array",
                        label="Legend mappings",
                        interactive=False,
                        visible=False,
                        wrap=True,
                        elem_classes="borealis-legend-table",
                    )
                    completeness_output = gr.Markdown(
                        "",
                        label="Reconstruction completeness",
                        visible=False,
                        elem_classes="borealis-summary",
                    )
                    review_state = gr.State(value=None)
                    with gr.Column(
                        visible=False,
                        elem_classes="borealis-review",
                    ) as review_panel:
                        gr.Markdown(
                            "## Review reconstructed chart\n\n"
                            "Correct reconstructed text below. Numeric values are "
                            "read-only. These draft corrections are not applied to "
                            "the specification yet.",
                            elem_classes="borealis-summary",
                        )
                        review_fields = gr.Dataframe(
                            value=[],
                            headers=["Field", "Text"],
                            datatype=["str", "str"],
                            type="array",
                            label="Reconstructed text",
                            interactive=True,
                            visible=True,
                            wrap=True,
                            column_count=2,
                            static_columns=[0],
                            show_search="none",
                            elem_classes="borealis-review-fields",
                        )
                        review_values = gr.Dataframe(
                            value=[],
                            headers=[
                                "Category #", "Category", "Series #", "Series",
                                "Reconstructed value",
                            ],
                            datatype=["number", "str", "number", "str", "number"],
                            type="array",
                            label="Reconstructed values (read-only)",
                            interactive=False,
                            visible=True,
                            wrap=True,
                            elem_classes="borealis-review-values",
                        )
                        review_error_output = gr.HTML(
                            "", visible=False, elem_classes="borealis-error"
                        )
                        generate_button = gr.Button(
                            "Generate tactile graphic",
                            variant="primary",
                            interactive=False,
                            elem_classes="borealis-generate",
                        )
                    spec_output = gr.JSON(
                        value={},
                        label="Reconstructed VerticalBarSpec",
                        visible=False,
                        elem_classes="borealis-spec-json",
                    )
                    spec_download = gr.DownloadButton(
                        "Download VerticalBarSpec JSON",
                        value=None,
                        visible=False,
                        variant="secondary",
                        elem_classes="borealis-spec-download",
                    )
                    generation_gallery = gr.Gallery(
                        value=[],
                        label="Tactile review pages",
                        visible=False,
                        columns=2,
                        object_fit="contain",
                        buttons=["fullscreen"],
                        elem_classes="borealis-tactile-gallery",
                    )
                    generation_report_output = gr.Markdown(
                        "",
                        label="Tactile generation report",
                        visible=False,
                        elem_classes="borealis-summary",
                    )
                    generation_download = gr.DownloadButton(
                        "Download clean tactile SVGs (ZIP)",
                        value=None,
                        visible=False,
                        variant="primary",
                        elem_classes="borealis-tactile-download",
                    )
                    generation_error_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )
                    ocr_warning_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )
                    legend_warning_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )
                    reconstruction_warning_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )
                    error_output = gr.HTML(
                        "", visible=False, elem_classes="borealis-error"
                    )

        image_input.change(
            ready_results,
            outputs=[
                status_output,
                empty_output,
                summary_output,
                annotated_output,
                ocr_output,
                ocr_download,
                legend_output,
                completeness_output,
                review_state,
                review_panel,
                review_fields,
                review_values,
                review_error_output,
                generate_button,
                spec_output,
                spec_download,
                generation_gallery,
                generation_report_output,
                generation_download,
                generation_error_output,
                ocr_warning_output,
                legend_warning_output,
                reconstruction_warning_output,
                error_output,
            ],
            queue=False,
            api_visibility="private",
        )

        controls = [
            image_input, confidence, nms_iou, language, analyze_button, clear_button
        ]
        result_components = [
            status_output,
            empty_output,
            summary_output,
            annotated_output,
            ocr_output,
            ocr_download,
            legend_output,
            completeness_output,
            review_state,
            review_panel,
            review_fields,
            review_values,
            review_error_output,
            generate_button,
            spec_output,
            spec_download,
            generation_gallery,
            generation_report_output,
            generation_download,
            generation_error_output,
            ocr_warning_output,
            legend_warning_output,
            reconstruction_warning_output,
            error_output,
        ]
        start = analyze_button.click(
            begin_analysis,
            outputs=[*controls, *result_components],
            queue=False,
            trigger_mode="once",
            api_visibility="private",
        )
        analysis = start.then(
            analyze_event,
            inputs=[image_input, confidence, nms_iou, language],
            outputs=result_components,
            show_progress="full",
            show_progress_on=annotated_output,
            concurrency_limit=1,
            concurrency_id="tactilegen-inference",
            api_name="analyze",
            api_description="Detect and annotate chart elements in one uploaded image.",
            api_visibility="private",
        )
        analysis.then(
            finish_analysis,
            outputs=controls,
            queue=False,
            api_visibility="private",
        )
        review_fields.input(
            update_review_event,
            inputs=[review_state, review_fields],
            outputs=[review_state, review_fields, review_error_output],
            queue=False,
            api_name="update_review",
            api_visibility="private",
        )
        generate_button.click(
            generate_event,
            inputs=[review_state, review_fields],
            outputs=[
                completeness_output,
                review_state,
                review_fields,
                review_values,
                review_error_output,
                spec_output,
                spec_download,
                generation_gallery,
                generation_report_output,
                generation_download,
                generation_error_output,
            ],
            show_progress="full",
            show_progress_on=generation_gallery,
            concurrency_limit=1,
            concurrency_id="tactilegen-generation",
            api_name="generate_tactile",
            api_description="Generate reviewed tactile SVG artifacts.",
            api_visibility="private",
        )
        clear_button.click(
            clear_all,
            outputs=[*controls, *result_components],
            queue=False,
            api_visibility="private",
        )

    return demo.queue(default_concurrency_limit=1, max_size=4)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    try:
        _import_gradio()
        model = load_model(args.weights)
        ocr_backend = TesseractBackend()
        demo = build_app(model, ocr_backend=ocr_backend, device=args.device)
        theme = borealis_theme()
    except Exception as exc:
        parser.error(f"Could not start TactileGen: {exc}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=False,
        max_file_size=MAX_UPLOAD_BYTES,
        theme=theme,
        css=BOREALIS_CSS,
        run_history=False,
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
