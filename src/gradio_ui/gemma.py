"""Independent Gemma chart review and lazy, serialized inference for Gradio."""
import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4

from src.baseline.gemma import GemmaChartExtractor, GemmaExtractionError
from src.contract_io import parse_bar_chart, write_bar_chart
from src.contracts import HorizontalBarChart
from src.gradio_ui.analysis import decode_upload
from src.gradio_ui.formatting import generation_report
from src.gradio_ui.run_tracking import GuardedUpdates
from src.tactile.generation import generate_chart_tactile_artifacts, GenerationError


def chart_data(chart):
    return json.loads(json.dumps({"schema_version": 1, "chart_type":
        "hbar_categorical" if isinstance(chart, HorizontalBarChart) else "vbar_categorical", **asdict(chart)}))


@dataclass(frozen=True)
class ChartReview:
    chart: object

    def text_rows(self):
        c = self.chart
        return [[label, value] for label, value in
                [("Chart title", c.title), ("X-axis title", c.x_axis_title),
                 ("Y-axis title", c.y_axis_title)]
                + [(f"Category {i}", v) for i, v in enumerate(c.categories)]
                + [(f"Series {i}", s.name or "") for i, s in enumerate(c.series)]]

    def value_rows(self):
        return [[f"Series {si}, category {ci}", value]
                for si, s in enumerate(self.chart.series) if s.has_bars
                for ci, value in enumerate(s.values)]

    def apply(self, texts, values, include_legend):
        expected_text, expected_values = self.text_rows(), self.value_rows()
        for rows, expected in ((texts, expected_text), (values, expected_values)):
            if not isinstance(rows, (list, tuple)) or len(rows) != len(expected):
                raise ValueError("Review row count changed. Extract the chart again.")
            if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in rows):
                raise ValueError("Each review row must have two cells.")
        if any(not isinstance(row[1], str) for row in texts):
            raise ValueError("Labels must be text.")
        numbers = []
        for row in values:
            v = row[1]
            try:
                finite = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("Values must be finite numbers; blanks are not allowed.")
            numbers.append(v)
        if type(include_legend) is not bool:
            raise ValueError("Legend selection must be a boolean.")
        data = chart_data(self.chart)
        labels = iter(row[1].strip() for row in texts)
        for field in ("title", "x_axis_title", "y_axis_title"):
            data[field] = next(labels)
        data["categories"] = [next(labels) for _ in data["categories"]]
        for series in data["series"]:
            series["name"] = next(labels) or None
        numbers = iter(numbers)
        for series in data["series"]:
            series["values"] = [next(numbers) for _ in series["values"]]
        data["omit_legend"] = not include_legend if len(data["series"]) == 1 else False
        return parse_bar_chart(data)


class GemmaBackend:
    def __init__(self, config=None, extractor=None):
        self.extractor = extractor if extractor is not None else GemmaChartExtractor(**(config or {}))
        self.lock = Lock()

    def extract(self, path):
        with self.lock:
            return self.extractor.extract_result(path)


def build_gemma_tab(gr, demo, tracker, artifact_root, backend):
    """Wire independent review tickets into the app's guarded publication."""
    gr.Markdown("## Read a chart with Gemma\nReview the extracted labels and values before generating tactile pages. "
                "The first extraction may download the checkpoint; accept its Hugging Face terms and authenticate first.")
    extractor = backend.extractor
    gr.Markdown(f"Model: `{getattr(extractor, 'model_id', 'injected')}` · "
                f"device: `{getattr(extractor, 'device', 'auto')}` · dtype: `{getattr(extractor, 'dtype', 'auto')}` · "
                f"revision: `{getattr(extractor, 'revision', None) or 'default'}` · token budget: `{getattr(extractor, 'max_new_tokens', 4096)}`")
    image = gr.Image(type="filepath", sources=["upload"], label="Gemma chart image")
    with gr.Row():
        extract = gr.Button("Extract chart", variant="primary")
        clear = gr.Button("Clear Gemma")
    status = gr.Markdown("Not loaded. Upload a vertical or horizontal bar chart.")
    texts = gr.Dataframe(headers=["Field", "Text"], datatype=["str", "str"], type="array",
                         interactive=True, static_columns=[0], column_count=2, label="Review labels (edit the Text column)")
    values = gr.Dataframe(headers=["Position", "Value"], datatype=["str", "number"], type="array",
                          interactive=True, static_columns=[0], column_count=2, label="Review values (edit the Value column)")
    legend = gr.Checkbox(value=True, label="Include legend (single series)")
    generate = gr.Button("Generate Gemma tactile pages")
    gallery = gr.Gallery(label="Tactile review copies", type="filepath")
    report = gr.Markdown()
    zip_file = gr.File(label="Clean tactile SVG ZIP")
    reviewed = gr.File(label="Reviewed chart JSON")
    with gr.Accordion("Gemma diagnostics", open=False):
        raw = gr.Textbox(label="Raw model response", interactive=False)
        metadata = gr.JSON(label="Extraction metadata")
        original = gr.File(label="Original extraction files", file_count="multiple")
    token = gr.Textbox(visible=False)
    stamp = gr.JSON(visible=False)
    demo.review_stamps = [stamp]
    demo.publication_errors = [*getattr(demo, "publication_errors", ()), status]
    outputs = [texts, values, legend, status, raw, metadata, original, reviewed, gallery, report, zip_file, token, stamp]
    sessions, admissions = {}, {}

    def key(request):
        if request is None or not request.session_hash:
            raise ValueError("A browser session is required")
        return "gemma:" + request.session_hash

    def cleanup(entry):
        if entry:
            cleanup(entry.get("generated"))
            shutil.rmtree(entry["root"], ignore_errors=True)

    def retire(k):
        cleanup(sessions.pop(k, None))
        for admission, entry in list(admissions.items()):
            if entry[0] == k:
                admissions.pop(admission)

    def empty(message):
        return [[], [], True, message, "", None, None, None, [], "", None, ""]

    async def reset(request: gr.Request):
        k = key(request)
        async with tracker.publication(k):
            tracker.invalidate(k)
            retire(k)
            return GuardedUpdates(empty("Ready. Upload or extract a chart."), k, tracker.latest(k))

    async def clear_all(request: gr.Request):
        result = await reset(request)
        return GuardedUpdates((None, *result), result.session, result.ticket)

    async def admit_extract(path, request: gr.Request):
        k = key(request)
        async with tracker.publication(k):
            retire(k)
            ticket = tracker.start(k, uuid4().hex)
            admission = uuid4().hex
            admissions[admission] = (k, ticket, path)
            updates = empty("Loading/extracting with Gemma. First use may take several minutes.")
            updates[-1] = admission
            return GuardedUpdates(updates, k, ticket)

    def extract_work(path):
        decode_upload(path)
        result = backend.extract(path)
        root = Path(tempfile.mkdtemp(prefix="gemma-", dir=artifact_root))
        try:
            write_bar_chart(root / "chart.json", result.chart)
            (root / "raw_response.txt").write_text(result.raw_response, encoding="utf-8")
            (root / "metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
            return {"root": root, "result": result, "review": ChartReview(result.chart)}
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    async def worker(function, *args, discard):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            def finished(done):
                if not done.cancelled() and done.exception() is None:
                    discard(done.result())
            task.add_done_callback(finished)
            raise

    async def extract_event(admission, request: gr.Request):
        k = key(request)
        entry = admissions.pop(admission, None)
        if not entry or entry[0] != k or not tracker.current(k, entry[1]):
            return tuple(gr.skip() for _ in outputs)
        ticket = entry[1]
        try:
            candidate = await worker(extract_work, entry[2], discard=cleanup)
        except Exception as exc:
            message = str(exc) if isinstance(exc, GemmaExtractionError) else "Gemma extraction failed. Check the image, installation, and model access."
            failed = empty(message)
            if isinstance(exc, GemmaExtractionError):
                failed[4], failed[5] = exc.raw_response or "", exc.metadata
            return GuardedUpdates(failed, k, ticket)
        async with tracker.publication(k):
            if not tracker.current(k, ticket):
                cleanup(candidate)
                return tuple(gr.skip() for _ in outputs)
            sessions[k] = candidate
            result, review = candidate["result"], candidate["review"]
            placeholders = result.metadata.get("placeholder_fields", [])
            message = "Ready for review. Verify labels and numeric values."
            if placeholders:
                message += " Placeholder titles: " + ", ".join(placeholders)
            files = [str(candidate["root"] / name) for name in ("chart.json", "raw_response.txt", "metadata.json")]
            return GuardedUpdates([review.text_rows(), review.value_rows(), not result.chart.omit_legend,
                message, result.raw_response, result.metadata, files, None, [], "", None, ""], k, ticket,
                cleanup=lambda: cleanup(candidate))

    async def edit(text_rows, value_rows, include, submitted_stamp, request: gr.Request):
        k = key(request)
        async with tracker.publication(k):
            current = tracker.latest(k)
            if not current or submitted_stamp != [current.run_id, current.review_revision] or k not in sessions:
                return tuple(gr.skip() for _ in outputs)
            ticket = tracker.advance(k, current.run_id, current.review_revision, edit=True)
            entry = sessions[k]
            cleanup(entry.pop("generated", None))
            try:
                entry["draft"] = entry["review"].apply(text_rows, value_rows, include)
                message = "Review updated. Generate tactile pages when ready."
            except ValueError:
                entry["draft"] = None
                message = "Invalid review. Keep row counts fixed, labels valid, and values finite numbers."
            return GuardedUpdates([gr.skip(), gr.skip(), gr.skip(), message, gr.skip(), gr.skip(),
                gr.skip(), None, [], "", None, ""], k, ticket)

    async def admit_generate(text_rows, value_rows, include, submitted_stamp, request: gr.Request):
        k = key(request)
        async with tracker.publication(k):
            current = tracker.latest(k)
            if not current or submitted_stamp != [current.run_id, current.review_revision] or k not in sessions:
                return gr.skip()
            try:
                chart = sessions[k]["review"].apply(text_rows, value_rows, include)
            except ValueError:
                return "invalid"
            ticket = tracker.advance(k, current.run_id, current.review_revision)
            admission = uuid4().hex
            admissions[admission] = (k, ticket, chart)
            return admission

    def generation_work(chart):
        root = Path(tempfile.mkdtemp(prefix="gemma-", dir=artifact_root))
        try:
            generated = generate_chart_tactile_artifacts(chart, root)
            write_bar_chart(root / "reviewed.json", chart)
            return {"root": root, "generation": generated}
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    async def generate_event(admission, request: gr.Request):
        k = key(request)
        entry = admissions.pop(admission, None)
        if admission == "invalid" and k in sessions:
            return GuardedUpdates([gr.skip(), gr.skip(), gr.skip(),
                "Invalid review. Keep row counts fixed, labels valid, and values finite numbers.",
                gr.skip(), gr.skip(), gr.skip(), None, [], "", None, ""], k, tracker.latest(k))
        if not entry or entry[0] != k or not tracker.current(k, entry[1]):
            return tuple(gr.skip() for _ in outputs)
        ticket = entry[1]
        try:
            candidate = await worker(generation_work, entry[2], discard=cleanup)
        except Exception as exc:
            message = str(exc) if isinstance(exc, GenerationError) else "Tactile generation failed. Review labels, values, and renderer limits."
            return GuardedUpdates([gr.skip(), gr.skip(), gr.skip(), message, gr.skip(), gr.skip(), gr.skip(),
                None, [], "", None, ""], k, ticket)
        async with tracker.publication(k):
            if not tracker.current(k, ticket):
                cleanup(candidate)
                return tuple(gr.skip() for _ in outputs)
            session = sessions[k]
            cleanup(session.pop("generated", None))
            session["generated"] = candidate
            generated = candidate["generation"]
            return GuardedUpdates([gr.skip(), gr.skip(), gr.skip(), "Tactile pages generated. Review layout findings.",
                gr.skip(), gr.skip(), gr.skip(), str(candidate["root"] / "reviewed.json"),
                [(str(p.review_svg), p.caption) for p in generated.pages],
                generation_report(SimpleNamespace(generation=generated, override_notes=())), str(generated.zip_path), ""],
                k, ticket, cleanup=lambda: cleanup(candidate))

    async def unload_gemma(request: gr.Request):
        k = key(request)
        async with tracker.publication(k):
            retire(k)
            tracker.remove(k)

    image.change(reset, outputs=outputs, queue=False, api_name="reset_gemma", api_visibility="private")
    clear.click(clear_all, outputs=[image, *outputs], queue=False, api_name="clear_gemma", api_visibility="private")
    extract.click(admit_extract, inputs=[image], outputs=outputs, queue=False, api_name="admit_gemma", api_visibility="private").then(
        extract_event, inputs=[token], outputs=outputs, concurrency_limit=1,
        concurrency_id="tactilegen-inference", api_name="extract_gemma", api_visibility="private")
    for component in (texts, values, legend):
        component.input(edit, inputs=[texts, values, legend, stamp], outputs=outputs, queue=False, api_name="edit_gemma_" + str(component._id), api_visibility="private")
    generate.click(admit_generate, inputs=[texts, values, legend, stamp], outputs=[token], queue=False, api_name="admit_gemma_generation", api_visibility="private").then(
        generate_event, inputs=[token], outputs=outputs, concurrency_limit=1,
        concurrency_id="tactilegen-generation", api_name="generate_gemma", api_visibility="private")
    demo.unload(unload_gemma)
