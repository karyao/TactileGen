"""Handle browser actions and translate their results into named component updates."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import html
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from src.contracts import effective_omit_legend
from src.detection import DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU
from src.diagnostics.tracking import AnalysisDiagnostics
from src.gradio_ui.run_tracking import GuardedUpdates
from src.gradio_ui.analysis import UploadValidationError, analyze_image, orientation_summary
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
    orientation_changed: Callable[..., Any]
    analyze_event: Callable[..., Any]
    admit_analysis: Callable[..., Any]
    admit_generation: Callable[..., Any]
    update_review_event: Callable[..., Any]
    update_legend_event: Callable[..., Any]
    generate_event: Callable[..., Any]
    clear_all: Callable[..., Any]
    unload: Callable[..., Any]


def create_callbacks(
    *,
    gr,
    tracker,
    model,
    backends,
    initial_ocr_engine: str,
    device: str | None,
    initial_imgsz: int,
    artifact_root: Path,
    controls: Controls,
    results: AnalysisComponents,
    generation: GenerationComponents,
) -> AppCallbacks:
    """Bind app resources once; keep each user action in a small named handler."""

    admissions = {}
    workers = set()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tactilegen")

    def retire(key):
        for token, entry in list(admissions.items()):
            if entry[0] == key:
                admissions.pop(token)
                entry[-1]()

    def session(request):
        if request is None or not request.session_hash:
            raise ValueError("A browser session is required")
        return request.session_hash

    def result_updates(
        status: str,
        *,
        empty_message: str = "",
        summary: str | None = None,
        orientation_text: str | None = None,
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
            include_legend=gr.update(
                value=not effective_omit_legend(len(review_model.spec.series), review_model.omit_legend)
                    if review_model else True,
                interactive=len(review_model.spec.series) == 1 if review_model else True,
            ),
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
            orientation_output=gr.update(
                value=orientation_text or "", visible=orientation_text is not None,
            ),
        )

    def ready_results():
        return results.updates(**result_updates(
            ready_status(),
            empty_message=(
                "### Your chart workspace\n\n"
                "Upload a vertical or horizontal bar chart, then select Analyze chart."
            ),
        ))

    def orientation_changed():
        return results.updates(**result_updates(
            ready_status(),
            empty_message="Chart settings changed. Select Analyze chart to update the results.",
        ))

    def analyze_work(image_path, confidence, nms_iou, language, imgsz=initial_imgsz,
                      ocr_engine=initial_ocr_engine, bar_orientation="auto",
                      keep_legend_matches=False,
                      progress=None, diagnostics=None, owned_files=None):
        failed = False
        updates = None
        try:
            diagnostics.capture_metadata(model, None, language)
            diagnostics.detection_request(imgsz)
            result = analyze_image(
                model,
                image_path,
                confidence,
                nms_iou,
                imgsz=imgsz,
                bar_orientation=bar_orientation,
                keep_legend_matches=keep_legend_matches,
                language=language,
                ocr_backend_factory=lambda: backends.resolve(ocr_engine),
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
                if download is not None:
                    owned_files.append(download)
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
                if spec_download is not None:
                    owned_files.append(spec_download)
                updates = result_updates(
                    complete_status(result.detection_count, ocr_count),
                    summary=result.summary,
                    orientation_text=(
                        orientation_summary(result.orientation_result)
                        if result.orientation_result is not None else None
                    ),
                    overlay=result.overlay_rgb,
                    ocr_table=rows if result.ocr_frame is not None else None,
                    ocr_download_path=download,
                    legend_table=mapping_rows if show_mapping else None,
                    completeness_text=(
                        completeness_summary(result.completeness, result.spec)
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

    def update_review_event(model, rows, omit_legend=None):
        if model is None:
            return None, gr.update(value=[]), gr.update(value="", visible=False)
        try:
            updated = update_review_draft(model, rows, omit_legend=omit_legend)
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

    def generate_work(model, rows, owned_files=None, owned_generations=None):
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
            owned_generations.append(submitted.generation)
            corrected = submitted.review_model.spec
            spec_path = write_spec_download(
                artifact_root, corrected.source_key, corrected
            )
            owned_files.append(spec_path)
            gallery = [
                (str(page.review_svg), page.caption)
                for page in submitted.generation.pages
            ]
            return generation.updates(
                completeness_output=gr.update(value=completeness_summary(submitted.completeness, corrected)),
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
                    value=completeness_summary(exc.completeness, exc.review_model.spec),
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

    def clear_all():
        control_updates = controls.updates(
            image_input=gr.update(value=None, interactive=True),
            confidence=gr.update(value=DEFAULT_CONFIDENCE, interactive=True),
            nms_iou=gr.update(value=DEFAULT_NMS_IOU, interactive=True),
            language=gr.update(value=DEFAULT_OCR_LANGUAGE, interactive=True),
            inference_size=gr.update(value=initial_imgsz, interactive=True),
            ocr_engine=gr.update(value=initial_ocr_engine, interactive=True),
            bar_orientation=gr.update(value="auto", interactive=True),
            keep_legend_matches=gr.update(value=False, interactive=True),
            analyze_button=gr.update(interactive=True),
            clear_button=gr.update(interactive=True),
            size_presets=tuple(
                gr.update(interactive=True) for _ in controls.size_presets
            ),
        )
        return (*control_updates, *raw_ready())

    # Async wrappers own publication; workers only prepare candidate updates.
    raw_ready, raw_orientation, raw_clear = ready_results, orientation_changed, clear_all

    async def reset(request, values):
        key = session(request)
        async with tracker.publication(key):
            retire(key)
            tracker.invalidate(key)
            return GuardedUpdates((*values, *(gr.update(interactive=True) for _ in controls.components())),
                                  key, tracker.latest(key))

    async def ready_results(request: gr.Request):
        return await reset(request, raw_ready())

    async def orientation_changed(request: gr.Request):
        return await reset(request, raw_orientation())

    async def clear_all(request: gr.Request):
        key = session(request)
        async with tracker.publication(key):
            retire(key)
            tracker.invalidate(key)
            return GuardedUpdates(raw_clear(), key, tracker.latest(key))

    async def unload(request: gr.Request):
        key = session(request)
        async with tracker.publication(key):
            retire(key)
            tracker.remove(key)

    def discard_files(paths, generations=()):
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not remove unpublished download")
        for generated in generations:
            try:
                cleanup_generation(generated, artifact_root)
            except Exception:
                LOGGER.exception("Could not remove unpublished generation")

    def unused_files(paths, updates, group):
        values = dict(zip((field.name for field in fields(group)), updates))
        published = {values.get(name, {}).get("value")
                     for name in ("ocr_download", "spec_download")}
        return [path for path in paths if path not in published]

    async def admit_analysis(image_path, confidence, nms_iou, language, imgsz=initial_imgsz,
                             ocr_engine=initial_ocr_engine, bar_orientation="auto",
                             keep_legend_matches=False,
                             request: gr.Request = None):
        key = session(request)
        diagnostics = AnalysisDiagnostics(settings={
            "confidence": confidence, "nms_iou": nms_iou, "device": device,
            "language": language, "imgsz": imgsz, "ocr_engine": ocr_engine,
            "bar_orientation": bar_orientation,
            "keep_legend_matches": keep_legend_matches,
        })
        async with tracker.publication(key):
            retire(key)
            ticket = tracker.start(key, diagnostics.run_id)
            args = (image_path, confidence, nms_iou, language, imgsz, ocr_engine,
                    bar_orientation, keep_legend_matches)
            admissions[diagnostics.run_id] = (key, ticket, args, diagnostics, tracker.retain(key))
            return GuardedUpdates((*results.updates(**result_updates(analyzing_status())),
                *(gr.update(interactive=control is controls.clear_button) for control in controls.components()),
                gr.update(value=diagnostics.run_id)), key, ticket)

    def consume(key, token):
        if not isinstance(token, str):
            return None
        entry = admissions.get(token)
        if entry is None or entry[0] != key:
            LOGGER.info("Discarded missing or consumed operation admission")
            return None
        admissions.pop(token)
        if not tracker.current(key, entry[1]):
            LOGGER.info("Discarded outdated admission for run %s", entry[1].run_id)
            entry[-1]()
            return None
        return entry

    async def run_worker(key, ticket, function, args, owned_files, owned_generations=()):
        release = tracker.retain(key)
        abandoned = Event()
        future = executor.submit(function, *args)
        workers.add(future)
        def finished(done):
            workers.discard(done)
            if abandoned.is_set():
                discard_files(owned_files, owned_generations)
            release()
        future.add_done_callback(finished)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            abandoned.set()
            if future.done():
                discard_files(owned_files, owned_generations)
            async with tracker.publication(key):
                if tracker.current(key, ticket):
                    tracker.invalidate(key)
            raise

    async def analyze_event(image_path, confidence, nms_iou, language, imgsz=initial_imgsz,
                            ocr_engine=initial_ocr_engine, bar_orientation="auto", admission=None,
                            keep_legend_matches=False,
                            request: gr.Request = None, progress=gr.Progress()):
        key = session(request)
        entry = consume(key, admission)
        if entry is None:
            yield tuple(gr.skip() for _ in [*results.components(), *controls.components()])
            return
        _, ticket, args, diagnostics, release = entry
        release()
        # The first yield resets old results and disables settings in this lifecycle.
        yield GuardedUpdates((*results.updates(**result_updates(analyzing_status())),
               *(gr.update(interactive=control is controls.clear_button) for control in controls.components())), key, ticket)
        def report(*args, **kwargs):
            if tracker.current(key, ticket):
                progress(*args, **kwargs)
        owned_files = []
        if not tracker.current(key, ticket):
            yield tuple(gr.skip() for _ in [*results.components(), *controls.components()])
            return
        updates = await run_worker(key, ticket, analyze_work,
            (*args, report, diagnostics, owned_files), owned_files)
        if not tracker.current(key, ticket):
            discard_files(owned_files)
            yield tuple(gr.skip() for _ in [*results.components(), *controls.components()])
            return
        discard_files(unused_files(owned_files, updates, results))
        updates = list(updates)
        index = results.components().index(results.review_state)
        if updates[index] is not None:
            updates[index] = replace(updates[index], run_id=ticket.run_id)
        yield GuardedUpdates((*updates, *(gr.update(interactive=True) for _ in controls.components())),
                             key, ticket, lambda: discard_files(owned_files))

    raw_review = update_review_event

    def review_update(model, rows, request, omit_legend=None):
        key = session(request)
        if model is None:
            return tuple(gr.skip() for _ in range(3))
        updated = raw_review(model, rows, omit_legend)
        if (updated[0].draft_values != model.draft_values
                or updated[0].omit_legend != model.omit_legend):
            ticket = tracker.advance(key, model.run_id, model.review_revision, edit=True)
            if ticket is None:
                return tuple(gr.skip() for _ in range(3))
            updated = (replace(updated[0], review_revision=ticket.review_revision), *updated[1:])
        else:
            # Validate even rejected edits before returning a model or an error.
            if not tracker.matches_review(key, model.run_id, model.review_revision):
                return tuple(gr.skip() for _ in range(3))
        return updated

    async def update_review_event(model, rows, stamp=None, request: gr.Request = None,
                                  omit_legend=None):
        key = session(request)
        async with tracker.publication(key):
            if model is None or stamp != [model.run_id, model.review_revision]:
                return tuple(gr.skip() for _ in range(3))
            updates = review_update(model, rows, request, omit_legend)
            return GuardedUpdates(updates, key, tracker.latest(key))

    async def update_legend_event(model, rows, stamp, include_legend, request: gr.Request):
        return await update_review_event(model, rows, stamp, request, not include_legend)

    async def admit_generation(model, rows, stamp, include_legend=None, request: gr.Request = None):
        key = session(request)
        async with tracker.publication(key):
            if (model is None or stamp != [model.run_id, model.review_revision]
                    or not tracker.matches_review(key, model.run_id, model.review_revision)):
                return GuardedUpdates((gr.skip(), gr.skip(), gr.skip(), gr.update(value=None)),
                                      key, tracker.latest(key))
            try:
                draft = update_review_draft(model, rows,
                    omit_legend=None if include_legend is None else not include_legend)
            except ValueError as exc:
                return GuardedUpdates((gr.skip(), gr.skip(),
                    gr.update(value=alert_message(str(exc)), visible=True), gr.update(value=None)),
                    key, tracker.latest(key))
            changed = (draft.draft_values != model.draft_values
                       or draft.omit_legend != model.omit_legend)
            ticket = tracker.advance(key, model.run_id, model.review_revision, edit=changed)
            draft = replace(draft, review_revision=ticket.review_revision)
            token = f"{ticket.run_id}:{ticket.review_revision}:{ticket.generation_token}"
            retire(key)
            admissions[token] = (key, ticket, draft, [list(row) for row in rows], tracker.retain(key))
            return GuardedUpdates((draft, gr.update(value=review_text_rows(draft)),
                gr.update(value="", visible=False), gr.update(value=token)), key, ticket)

    async def generate_event(model, rows, admission=None, request: gr.Request = None):
        key = session(request)
        entry = consume(key, admission)
        if entry is None:
            return tuple(gr.skip() for _ in generation.components())
        _, ticket, model, rows, release = entry
        release()
        owned_files, owned_generations = [], []
        updates = await run_worker(key, ticket, generate_work,
            (model, rows, owned_files, owned_generations), owned_files, owned_generations)
        if not tracker.current(key, ticket):
            discard_files(owned_files, owned_generations)
            return tuple(gr.skip() for _ in generation.components())
        discard_files(unused_files(owned_files, updates, generation))
        updates = list(updates)
        index = generation.components().index(generation.review_state)
        if isinstance(updates[index], ReviewModel):
            updates[index] = replace(updates[index], run_id=ticket.run_id,
                                     review_revision=ticket.review_revision)
        return GuardedUpdates(updates, key, ticket,
                              lambda: discard_files(owned_files, owned_generations))

    for callback in (ready_results, orientation_changed, clear_all, unload,
                     analyze_event, update_review_event, generate_event,
                     admit_analysis, admit_generation, update_legend_event):
        callback.__annotations__["request"] = gr.Request

    analyze_event.admit = admit_analysis
    generate_event.admit = admit_generation

    return AppCallbacks(
        ready_results=ready_results,
        orientation_changed=orientation_changed,
        analyze_event=analyze_event,
        admit_analysis=admit_analysis,
        admit_generation=admit_generation,
        update_review_event=update_review_event,
        update_legend_event=update_legend_event,
        generate_event=generate_event,
        clear_all=clear_all,
        unload=unload,
    )
