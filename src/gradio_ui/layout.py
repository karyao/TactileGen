"""Build the page, then connect its named components to browser event handlers."""

from __future__ import annotations

import atexit
import tempfile
from pathlib import Path

from src.detection import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_NMS_IOU,
    MAX_IMGSZ,
    validate_imgsz,
)
from src.diagnostics.tracking import configure_analysis_logging
from src.gradio_ui.run_tracking import RunTracker, guarded_blocks
from src.gradio_ui.callbacks import create_callbacks
from src.gradio_ui.components import Controls, AnalysisComponents, GenerationComponents
from src.gradio_ui.dependencies import import_gradio
from src.gradio_ui.formatting import ready_status
from src.gradio_ui.settings import EXAMPLE_IMAGE, DEFAULT_OCR_LANGUAGE, DEFAULT_OCR_ENGINE
from src.gradio_ui.ocr import OCRBackendResolver
from src.presentation import LEGEND_TABLE_HEADERS, OCR_TABLE_HEADERS


def build_app(
    model,
    *,
    ocr_backend=None,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    device: str | None = None,
    imgsz: int = DEFAULT_IMGSZ,
):
    """Build without launching; an injected backend serves ``ocr_engine`` only."""
    initial_imgsz = validate_imgsz(imgsz)
    gr = import_gradio()
    configure_analysis_logging()
    backends = OCRBackendResolver(ocr_engine, ocr_backend)
    artifacts = tempfile.TemporaryDirectory(prefix="tactilegen-gradio-ocr-")
    atexit.register(artifacts.cleanup)
    artifact_root = Path(artifacts.name)

    tracker = RunTracker()
    with guarded_blocks(gr, tracker)(
        title="TactileGen — Prepare charts for tactile reading",
        analytics_enabled=False,
        fill_width=False,
    ) as demo:
        with gr.Column(elem_classes="borealis-app"):
            with gr.Column(elem_classes="borealis-header"):
                gr.Markdown(
                    "# TactileGen\n\n"
                    "**Prepare charts for tactile reading.**\n\n"
                    "Analyze a vertical or horizontal bar chart, review its labels and values, "
                    "and generate tactile pages."
                )

            with gr.Row(elem_classes="borealis-workspace-grid"):
                controls = _build_controls(gr, initial_imgsz, ocr_engine)
                results = _build_results(gr)

        analysis_admission = gr.Textbox(visible=False)
        generation_admission = gr.Textbox(visible=False)
        review_stamp = gr.JSON(visible=False)
        demo.review_stamp = review_stamp
        demo.analysis_controls = controls.components()
        demo.publication_errors = [results.error_output, results.generation_error_output,
                                   results.review_error_output]
        generation = GenerationComponents(
            completeness_output=results.completeness_output,
            review_state=results.review_state,
            review_fields=results.review_fields,
            review_values=results.review_values,
            review_error_output=results.review_error_output,
            spec_output=results.spec_output,
            spec_download=results.spec_download,
            generation_gallery=results.generation_gallery,
            generation_report_output=results.generation_report_output,
            generation_download=results.generation_download,
            generation_error_output=results.generation_error_output,
        )
        callbacks = create_callbacks(
            tracker=tracker, gr=gr, model=model, backends=backends, device=device,
            initial_ocr_engine=ocr_engine,
            initial_imgsz=initial_imgsz, artifact_root=artifact_root,
            controls=controls, results=results, generation=generation,
        )
        controls.image_input.change(
            callbacks.ready_results,
            outputs=[*results.components(), *controls.components(), review_stamp],
            queue=False,
            api_visibility="private",
        )
        controls.bar_orientation.input(
            callbacks.orientation_changed,
            outputs=[*results.components(), *controls.components(), review_stamp],
            queue=False,
            api_visibility="private",
        )
        controls.keep_legend_matches.input(
            callbacks.orientation_changed,
            outputs=[*results.components(), *controls.components(), review_stamp],
            queue=False, api_visibility="private",
        )
        demo.unload(callbacks.unload)
        analysis_start = controls.analyze_button.click(
            callbacks.admit_analysis,
            inputs=[
                controls.image_input,
                controls.confidence,
                controls.nms_iou,
                controls.language,
                controls.inference_size,
                controls.ocr_engine,
                controls.bar_orientation,
                controls.keep_legend_matches,
            ],
            outputs=[*results.components(), *controls.components(), analysis_admission, review_stamp],
            queue=False, trigger_mode="once", api_name="admit_analysis", api_visibility="private",
        )
        analysis_start.then(
            callbacks.analyze_event,
            inputs=[
                controls.image_input,
                controls.confidence,
                controls.nms_iou,
                controls.language,
                controls.inference_size,
                controls.ocr_engine,
                controls.bar_orientation,
                analysis_admission,
                controls.keep_legend_matches,
            ],
            outputs=[*results.components(), *controls.components(), review_stamp],
            show_progress="full", show_progress_on=results.annotated_output,
            concurrency_limit=1, concurrency_id="tactilegen-inference",
            api_name="analyze", api_visibility="private",
        )
        results.review_fields.input(
            callbacks.update_review_event,
            inputs=[results.review_state, results.review_fields, review_stamp],
            outputs=[
                results.review_state,
                results.review_fields,
                results.review_error_output,
                review_stamp,
            ],
            queue=False,
            api_name="update_review",
            api_visibility="private",
        )
        generation_start = results.generate_button.click(
            callbacks.admit_generation,
            inputs=[results.review_state, results.review_fields, review_stamp, results.include_legend],
            outputs=[results.review_state, results.review_fields, results.review_error_output,
                     generation_admission, review_stamp],
            queue=False, api_name="admit_generation", api_visibility="private",
        )
        results.include_legend.input(
            callbacks.update_legend_event,
            inputs=[results.review_state, results.review_fields, review_stamp, results.include_legend],
            outputs=[results.review_state, results.review_fields, results.review_error_output,
                     review_stamp],
            queue=False, api_name="update_legend", api_visibility="private",
        )
        generation_start.then(
            callbacks.generate_event,
            inputs=[results.review_state, results.review_fields, generation_admission],
            outputs=[*generation.components(), review_stamp],
            show_progress="full", show_progress_on=results.generation_gallery,
            concurrency_limit=1, concurrency_id="tactilegen-generation",
            api_name="generate_tactile", api_visibility="private",
        )
        controls.clear_button.click(
            callbacks.clear_all,
            outputs=[*controls.components(), *results.components(), review_stamp],
            queue=False,
            api_visibility="private",
        )

    return demo.queue(default_concurrency_limit=1, max_size=4)


def _build_controls(gr, initial_imgsz: int, initial_ocr_engine: str) -> Controls:
    """Create the upload/settings panel and its action buttons."""
    with gr.Column(elem_classes="borealis-control-rail"):
        gr.Markdown("## Chart input", elem_classes="borealis-section-title")
        gr.Markdown(
            "Supports vertical and horizontal categorical bar charts.",
            elem_classes="borealis-scope",
        )
        image_input = gr.Image(
            type="filepath",
            sources=["upload"],
            image_mode=None,
            format="png",
            label="Bar chart image",
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
        ocr_engine = gr.Dropdown(
            choices=[("RapidOCR", "rapidocr"), ("Tesseract", "tesseract")],
            value=initial_ocr_engine,
            label="OCR engine",
            info="Choose the text recognition engine for the next analysis.",
            elem_classes="borealis-setting",
        )
        bar_orientation = gr.Dropdown(
            choices=[("Auto", "auto"), ("Vertical", "vertical"), ("Horizontal", "horizontal")],
            value="auto",
            label="Bar orientation",
            info="Choose a direction if Auto is unclear, then analyze again.",
            elem_classes="borealis-setting",
        )
        with gr.Accordion(
            "Advanced settings",
            open=False,
            elem_classes="borealis-advanced",
        ):
            inference_size = gr.Number(
                value=initial_imgsz,
                precision=None,
                minimum=1,
                maximum=MAX_IMGSZ,
                step=1,
                label="YOLO inference size (pixels)",
                info=(
                    f"Maximum {MAX_IMGSZ} pixels. Larger sizes may improve "
                    "small-element detection and increase runtime. "
                    "YOLO may adjust this size to its model stride."
                ),
                elem_classes="borealis-setting",
            )
            with gr.Row():
                size_presets = [
                    gr.Button(label, size="sm", variant="secondary")
                    for label in ("640 (default)", "960", "1280 (maximum)")
                ]
            for button, size in zip(size_presets, (640, 960, MAX_IMGSZ)):
                button.click(
                    lambda value=size: value,
                    inputs=[], outputs=inference_size,
                    queue=False, api_visibility="private",
                )
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
                label="Tesseract language",
                info=("Tesseract code, such as eng or eng+fra. "
                      "RapidOCR uses its multilingual model and ignores this field."),
                elem_classes="borealis-setting",
            )
        with gr.Column(elem_classes="borealis-legend-option"):
            keep_legend_matches = gr.Checkbox(
                label="Keep only legend-matched bars",
                value=False,
                container=False,
                elem_classes="borealis-legend-checkbox",
            )
            gr.Markdown(
                "Remove bars whose colours cannot be matched to the detected legend. "
                "Leave off to keep all detected bars. Select Analyze chart after changing this.",
                elem_classes=["borealis-helper", "borealis-legend-help"],
            )
        with gr.Row(elem_classes="borealis-actions"):
            analyze_button = gr.Button("Analyze chart", variant="primary")
            clear_button = gr.Button("Clear", variant="secondary")


    return Controls(
        image_input=image_input,
        confidence=confidence,
        nms_iou=nms_iou,
        language=language,
        inference_size=inference_size,
        ocr_engine=ocr_engine,
        bar_orientation=bar_orientation,
        analyze_button=analyze_button,
        clear_button=clear_button,
        size_presets=tuple(size_presets),
        keep_legend_matches=keep_legend_matches,
    )


def _build_results(gr) -> AnalysisComponents:
    """Create the workspace, review panel, and downloadable result components."""
    with gr.Column(elem_classes="borealis-workspace"):
        gr.Markdown(
            "## Chart workspace",
            elem_classes=[
                "borealis-section-title",
                "borealis-workspace-heading",
            ],
        )
        status_output = gr.HTML(
            ready_status(), elem_classes="borealis-status"
        )
        empty_output = gr.Markdown(
            "### Your chart workspace\n\n"
            "Upload a vertical or horizontal bar chart, then select Analyze chart.",
            visible=True,
            elem_classes="borealis-empty",
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
        orientation_output = gr.Markdown(
            "", label="Bar orientation result", visible=False,
            elem_classes="borealis-summary",
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
                "## Review your chart\n\n"
                "Check the text and values, then generate your tactile graphic.",
                elem_classes=["borealis-summary", "borealis-review-heading"],
            )
            with gr.Column(elem_classes="borealis-legend-option"):
                include_legend = gr.Checkbox(
                    label="Include legend",
                    value=True,
                    container=False,
                    elem_classes="borealis-legend-checkbox",
                )
                gr.Markdown(
                    "No legend in the original? Turn this off to leave series names blank. "
                    "All bars and values stay in your chart.",
                    elem_classes=["borealis-helper", "borealis-legend-help"],
                )
            review_fields = gr.Dataframe(
                value=[],
                headers=["Field", "Text"],
                datatype=["str", "str"],
                type="array",
                label="Chart text · editable",
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
                label="Bar values · read-only",
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
        with gr.Accordion(
            "Analysis details", open=False, elem_classes="borealis-details",
        ):
            gr.Markdown(
                "Inspect detected regions, recognized text, and legend mappings. "
                "For diagnostic explanations, extract the diagnostics ZIP "
                "and open report.md.",
                elem_classes="borealis-helper",
            )
            diagnostics_info = gr.Markdown("", visible=False)
            diagnostics_download = gr.DownloadButton(
                "Download analysis diagnostics (ZIP)", value=None,
                visible=False, variant="secondary",
            )
            summary_output = gr.Markdown(
                "",
                label="Detection summary",
                visible=False,
                elem_classes="borealis-summary",
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
        with gr.Accordion(
            "Chart specification", open=False, elem_classes="borealis-details",
        ):
            gr.Markdown(
                "Download reconstructed labels, series, and values as JSON.",
                elem_classes="borealis-helper",
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


    return AnalysisComponents(
        status_output=status_output,
        empty_output=empty_output,
        summary_output=summary_output,
        annotated_output=annotated_output,
        ocr_output=ocr_output,
        ocr_download=ocr_download,
        legend_output=legend_output,
        completeness_output=completeness_output,
        review_state=review_state,
        review_panel=review_panel,
        review_fields=review_fields,
        review_values=review_values,
        review_error_output=review_error_output,
        generate_button=generate_button,
        spec_output=spec_output,
        spec_download=spec_download,
        generation_gallery=generation_gallery,
        generation_report_output=generation_report_output,
        generation_download=generation_download,
        generation_error_output=generation_error_output,
        ocr_warning_output=ocr_warning_output,
        legend_warning_output=legend_warning_output,
        reconstruction_warning_output=reconstruction_warning_output,
        error_output=error_output,
        diagnostics_download=diagnostics_download,
        diagnostics_info=diagnostics_info,
        orientation_output=orientation_output,
        include_legend=include_legend,
    )

