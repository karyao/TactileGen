"""Handle browser actions and translate their results into named component updates."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.detection import DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU
from src.diagnostics import AnalysisDiagnostics
from src.gradio_ui.analysis import UploadValidationError, analyze_image
from src.gradio_ui.artifacts import write_ocr_download, write_spec_download
from src.gradio_ui.components import Controls, AnalysisComponents, GenerationComponents
from src.gradio_ui.formatting import (
    generation_report,
    ready_status,
    analyzing_status,
    complete_status,
    alert_message,
    completeness_summary,
)
from src.gradio_ui.review import (
    ReviewModel,
    ManualGenerationBlocked,
    build_review_model,
    review_text_rows,
    review_value_rows,
    update_review_draft,
    generate_review_submission,
)
from src.gradio_ui.settings import DEFAULT_OCR_LANGUAGE, LOGGER
from src.presentation import legend_rows, ocr_rows
from src.tactile.generation import GenerationError, cleanup_generation


@dataclass(frozen=True)
class AppCallbacks:
    """The event handlers registered by the layout builder."""

    ready_results: Callable[..., Any]
    begin_analysis: Callable[..., Any]
    analyze_event: Callable[..., Any]
    update_review_event: Callable[..., Any]
    generate_event: Callable[..., Any]
    finish_analysis: Callable[..., Any]
    clear_all: Callable[..., Any]


def create_callbacks(
    *,
    gr,
    model,
    backend,
    device: str | None,
    initial_imgsz: int,
    artifact_root: Path,
    controls: Controls,
    results: AnalysisComponents,
    generation: GenerationComponents,
) -> AppCallbacks:
    """Bind app resources once; keep each user action in a small named handler."""

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
    ) -> dict[str, Any]:
        """Build a complete result reset, then fill in the available analysis."""
        return dict(
            status_output=gr.update(value=status),
            empty_output=gr.update(value=empty_message, visible=bool(empty_message)),
            summary_output=gr.update(value=summary or '', visible=summary is not None),
            annotated_output=gr.update(value=overlay, visible=overlay is not None),
            ocr_output=gr.update(value=ocr_table or [], visible=ocr_table is not None),
            ocr_download=gr.update(
                value=ocr_download_path,
                visible=ocr_download_path is not None,
            ),
            legend_output=gr.update(
                value=legend_table or [],
                visible=legend_table is not None,
            ),
            completeness_output=gr.update(
                value=completeness_text or '',
                visible=completeness_text is not None,
            ),
            review_state=review_model,
            review_panel=gr.update(visible=review_model is not None),
            review_fields=gr.update(
                value=review_text_rows(review_model) if review_model else [],
                row_count=len(review_model.bindings) if review_model else 0,
            ),
            review_values=gr.update(
                value=review_value_rows(review_model.spec) if review_model else [],
            ),
            review_error_output=gr.update(value='', visible=False),
            generate_button=gr.update(interactive=generate_enabled),
            spec_output=gr.update(value=spec_json or {}, visible=spec_json is not None),
            spec_download=gr.update(
                value=spec_download_path,
                visible=spec_download_path is not None,
            ),
            generation_gallery=gr.update(
                value=generation_gallery or [],
                visible=generation_gallery is not None,
            ),
            generation_report_output=gr.update(
                value=generation_report_text or '',
                visible=generation_report_text is not None,
            ),
            generation_download=gr.update(
                value=generation_zip_path,
                visible=generation_zip_path is not None,
            ),
            generation_error_output=gr.update(value='', visible=False),
            ocr_warning_output=gr.update(
                value=alert_message(ocr_warning) if ocr_warning else '',
                visible=ocr_warning is not None,
            ),
            legend_warning_output=gr.update(
                value=alert_message(legend_warning) if legend_warning else '',
                visible=legend_warning is not None,
            ),
            reconstruction_warning_output=gr.update(
                value=alert_message(reconstruction_warning) if reconstruction_warning else '',
                visible=reconstruction_warning is not None,
            ),
            error_output=gr.update(
                value=alert_message(error_message) if error_message else '',
                visible=error_message is not None,
            ),
            diagnostics_download=gr.update(value=None, visible=False),
            diagnostics_info=gr.update(value="", visible=False),
        )

    def ready_results():
        return results.updates(**result_updates(
            ready_status(),
            empty_message=(
                "### Your chart workspace\n\n"
                "Upload a vertical or horizontal bar chart, then select Analyze chart."
            ),
        ))

    def begin_analysis():
        return (
            *(gr.update(interactive=False) for _ in controls.components()),
            *results.updates(**result_updates(analyzing_status())),
        )

    def analyze_event(image_path, confidence, nms_iou, language, imgsz=initial_imgsz,
                      progress=gr.Progress()):
        diagnostics = AnalysisDiagnostics(settings={
            "confidence": confidence, "nms_iou": nms_iou,
            "device": device, "language": language, "imgsz": imgsz,
        })
        failed = False
        updates = None
        try:
            diagnostics.capture_metadata(model, backend, language)
            diagnostics.detection_request(imgsz)
            result = analyze_image(
                model,
                image_path,
                confidence,
                nms_iou,
                imgsz=imgsz,
                language=language,
                ocr_backend=backend,
                device=device,
                progress=progress,
                diagnostics=diagnostics,
            )
            with diagnostics.stage("result_preparation"):
                ocr_count = len(result.ocr_frame.regions) if result.ocr_frame is not None else None
                rows = (
                    ocr_rows(result.detection_frame, result.ocr_frame)
                    if result.ocr_frame is not None else []
                )
                download = (
                    write_ocr_download(artifact_root, image_path, result.ocr_frame)
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
                    write_spec_download(artifact_root, image_path, result.spec)
                    if result.spec is not None else None
                )
                updates = result_updates(
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
            failed = True
            diagnostics.fail(exc)
            diagnostics.warning("upload_rejected", "validation")
            message = str(exc)
        except Exception as exc:
            failed = True
            diagnostics.fail(exc)
            diagnostics.warning("analysis_failed", "analysis")
            message = "Analysis failed. Check the terminal and try another image."
        finally:
            diagnostics.finish(failed=failed)
            record_path = diagnostics.persist(artifact_root)
            diagnostic_path = diagnostics.export_zip(artifact_root)
            diagnostics.log_summary(
                record_path=record_path,
                archive_path=diagnostic_path,
            )
        if updates is None:
            updates = result_updates(ready_status(), error_message=message)
        diagnostic_text = f"Analysis run: `{diagnostics.run_id}`"
        if diagnostic_path is None:
            diagnostic_text += " — Diagnostic export failed. Check the terminal for details."
            if diagnostics.export_error:
                detail = html.escape(diagnostics.export_error).replace("\n", " ")
                diagnostic_text += " " + re.sub(
                    r"([\\`*_{}\[\]()#+.!|>])",
                    r"\\\1",
                    detail,
                )
        else:
            diagnostic_text += (
                " — Extract the ZIP and open **report.md** "
                "to follow findings to their evidence."
            )
        updates["diagnostics_download"] = gr.update(
            value=diagnostic_path, visible=diagnostic_path is not None,
        )
        updates["diagnostics_info"] = gr.update(value=diagnostic_text, visible=True)
        return results.updates(**updates)

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
            return generation.updates(
                completeness_output=unchanged,
                review_state=unchanged,
                review_fields=unchanged,
                review_values=unchanged,
                review_error_output=unchanged,
                spec_output=unchanged,
                spec_download=unchanged,
                generation_gallery=unchanged,
                generation_report_output=unchanged,
                generation_download=unchanged,
                generation_error_output=gr.update(
                    value=alert_message('Analyze a chart before generating.'),
                    visible=True,
                ),
            )
        submitted = None
        try:
            submitted = generate_review_submission(model, rows, artifact_root)
            corrected = submitted.review_model.spec
            spec_path = write_spec_download(
                artifact_root, corrected.source_key, corrected
            )
            gallery = [
                (str(page.review_svg), page.caption)
                for page in submitted.generation.pages
            ]
            return generation.updates(
                completeness_output=gr.update(value=completeness_summary(submitted.completeness)),
                review_state=submitted.review_model,
                review_fields=gr.update(value=review_text_rows(submitted.review_model)),
                review_values=gr.update(value=review_value_rows(corrected)),
                review_error_output=gr.update(value='', visible=False),
                spec_output=gr.update(value=asdict(corrected), visible=True),
                spec_download=gr.update(value=spec_path, visible=True),
                generation_gallery=gr.update(value=gallery, visible=True),
                generation_report_output=gr.update(
                    value=generation_report(submitted),
                    visible=True,
                ),
                generation_download=gr.update(
                    value=submitted.generation.zip_path,
                    visible=True,
                ),
                generation_error_output=gr.update(value='', visible=False),
            )
        except ManualGenerationBlocked as exc:
            return generation.updates(
                completeness_output=gr.update(
                    value=completeness_summary(exc.completeness),
                    visible=True,
                ),
                review_state=exc.review_model,
                review_fields=gr.update(value=review_text_rows(exc.review_model)),
                review_values=unchanged,
                review_error_output=unchanged,
                spec_output=unchanged,
                spec_download=unchanged,
                generation_gallery=unchanged,
                generation_report_output=unchanged,
                generation_download=unchanged,
                generation_error_output=gr.update(
                    value=alert_message(str(exc)),
                    visible=True,
                ),
            )
        except ValueError as exc:
            if submitted is not None:
                try:
                    cleanup_generation(submitted.generation, artifact_root)
                except Exception:
                    LOGGER.exception("Could not clean up unpublished tactile artifacts")
            return generation.updates(
                completeness_output=unchanged,
                review_state=model,
                review_fields=unchanged,
                review_values=unchanged,
                review_error_output=gr.update(
                    value=alert_message(str(exc)),
                    visible=True,
                ),
                spec_output=unchanged,
                spec_download=unchanged,
                generation_gallery=unchanged,
                generation_report_output=unchanged,
                generation_download=unchanged,
                generation_error_output=unchanged,
            )
        except GenerationError as exc:
            LOGGER.exception("Tactile artifact generation failed at %s", exc.stage)
            return generation.updates(
                completeness_output=unchanged,
                review_state=model,
                review_fields=unchanged,
                review_values=unchanged,
                review_error_output=unchanged,
                spec_output=unchanged,
                spec_download=unchanged,
                generation_gallery=unchanged,
                generation_report_output=unchanged,
                generation_download=unchanged,
                generation_error_output=gr.update(
                    value=alert_message(str(exc)),
                    visible=True,
                ),
            )
        except Exception:
            if submitted is not None:
                try:
                    cleanup_generation(submitted.generation, artifact_root)
                except Exception:
                    LOGGER.exception("Could not clean up unpublished tactile artifacts")
            LOGGER.exception("Manual tactile generation failed")
            return generation.updates(
                completeness_output=unchanged,
                review_state=model,
                review_fields=unchanged,
                review_values=unchanged,
                review_error_output=unchanged,
                spec_output=unchanged,
                spec_download=unchanged,
                generation_gallery=unchanged,
                generation_report_output=unchanged,
                generation_download=unchanged,
                generation_error_output=gr.update(
                    value=alert_message(
                        "Tactile generation failed. Check the terminal and try again."
                    ),
                    visible=True,
                ),
            )

    def finish_analysis():
        return tuple(gr.update(interactive=True) for _ in controls.components())

    def clear_all():
        control_updates = controls.updates(
            image_input=gr.update(value=None, interactive=True),
            confidence=gr.update(value=DEFAULT_CONFIDENCE, interactive=True),
            nms_iou=gr.update(value=DEFAULT_NMS_IOU, interactive=True),
            language=gr.update(value=DEFAULT_OCR_LANGUAGE, interactive=True),
            inference_size=gr.update(value=initial_imgsz, interactive=True),
            analyze_button=gr.update(interactive=True),
            clear_button=gr.update(interactive=True),
            size_presets=tuple(
                gr.update(interactive=True) for _ in controls.size_presets
            ),
        )
        return (*control_updates, *ready_results())

    return AppCallbacks(
        ready_results=ready_results,
        begin_analysis=begin_analysis,
        analyze_event=analyze_event,
        update_review_event=update_review_event,
        generate_event=generate_event,
        finish_analysis=finish_analysis,
        clear_all=clear_all,
    )
