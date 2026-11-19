"""Gemma review and real Gradio publication with injected inference."""
import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import gradio as gr
from gradio.state_holder import SessionState
import pytest

from src.baseline.vlm_response import parse_chart_response
from src.gradio_app import build_app
from src.gradio_ui.gemma import ChartReview


def result(filename="vertical_response.json"):
    raw = (Path("tests/fixtures/vlm") / filename).read_text()
    return SimpleNamespace(chart=parse_chart_response(raw, image="chart.png").chart,
                           raw_response=raw, metadata={"placeholder_fields": []})


@pytest.mark.parametrize("filename", ["vertical_response.json", "horizontal_response.json"])
def test_review_edits_and_validation(filename):
    source = result(filename)
    review = ChartReview(source.chart)
    texts, values = review.text_rows(), review.value_rows()
    texts[0] = ["forged index", "Corrected title"]
    values[0] = ["forged index", 42]
    corrected = review.apply(texts, values, True)
    assert corrected.title == "Corrected title"
    assert corrected.series[0].values[0] == 42
    assert source.chart.title != corrected.title
    for bad in (True, None, "42", float("nan"), float("inf")):
        values[0][1] = bad
        with pytest.raises(ValueError):
            review.apply(texts, values, True)
    with pytest.raises(ValueError):
        review.apply(texts[:-1], review.value_rows(), True)


class Browser:
    def __init__(self, app, key="gemma-test"):
        self.app, self.key = app, key
        self.state = SessionState(app)
        self.request = gr.Request(session_hash=key)

    def fn(self, name):
        return next(f for f in self.app.fns.values() if f.api_name == name)

    async def call(self, name, inputs=()):
        return await self.app.process_api(self.fn(name), list(inputs), state=self.state,
            request=self.request, session_hash=self.key)

    async def extract(self):
        self.fn("admit_gemma").inputs[0].preprocess = lambda _: "chart.png"
        admitted = await self.call("admit_gemma", [None])
        return await self.call("extract_gemma", [admitted["data"][-2]])


@pytest.mark.parametrize("filename", ["vertical_response.json", "horizontal_response.json"])
def test_extract_review_generate_download(filename):
    async def scenario():
        source = result(filename)
        extractor = SimpleNamespace(extract_result=lambda _: source)
        app = build_app(None, workflow="gemma", gemma_extractor=extractor)
        browser = Browser(app)
        try:
            with patch("src.gradio_ui.gemma.decode_upload"):
                extracted = await browser.extract()
            assert "Ready for review" in extracted["data"][3]
            texts, values = ChartReview(source.chart).text_rows(), ChartReview(source.chart).value_rows()
            values[0][1] = 42
            admitted = await browser.call("admit_gemma_generation", [
                {"data": texts, "headers": ["Field", "Text"]}, {"data": values, "headers": ["Position", "Value"]}, True, extracted["data"][-1]["value"].root])
            generated = await browser.call("generate_gemma", [admitted["data"][0]])
            assert "generated" in generated["data"][3]
            assert generated["data"][8]
            import json
            reviewed = json.loads(Path(generated["data"][7]["path"]).read_text())
            assert reviewed["series"][0]["values"][0] == 42
            zip_path = generated["data"][10]["path"]
            with ZipFile(zip_path) as archive:
                assert archive.namelist()
                assert all("review" not in n for n in archive.namelist())
        finally:
            app.close(verbose=False)
    asyncio.run(scenario())


def test_late_extraction_does_not_restore_cleared_result():
    async def scenario():
        entered, release = Event(), Event()
        def blocked(_):
            entered.set()
            assert release.wait(5)
            return result()
        app = build_app(None, workflow="gemma", gemma_extractor=SimpleNamespace(extract_result=blocked))
        browser = Browser(app)
        try:
            with patch("src.gradio_ui.gemma.decode_upload"):
                task = asyncio.create_task(browser.extract())
                assert await asyncio.to_thread(entered.wait, 5)
                await browser.call("reset_gemma")
                release.set()
                response = await task
                assert all(item.get("__type__") == "update" for item in response["data"])
        finally:
            release.set()
            app.close(verbose=False)
    asyncio.run(scenario())


def test_gemma_only_startup_skips_detector():
    from unittest.mock import Mock
    from src.gradio_app import main
    demo = Mock()
    with patch("src.gradio_app.load_model") as detector, patch("src.gradio_app.build_app", return_value=demo) as build:
        main(["--workflow", "gemma", "--gemma-device", "cpu"])
    detector.assert_not_called()
    assert build.call_args.args == (None,)
    assert build.call_args.kwargs["gemma_config"]["device"] == "cpu"


def test_late_generation_is_discarded_after_edit():
    async def scenario():
        entered, release = Event(), Event()
        from src.gradio_ui.gemma import generate_chart_tactile_artifacts as real_generate
        produced = []
        def blocked(*args):
            entered.set()
            assert release.wait(5)
            generated = real_generate(*args)
            produced.append(generated.output_dir.parent)
            return generated
        app = build_app(None, workflow="gemma", gemma_extractor=SimpleNamespace(extract_result=lambda _: result()))
        b = Browser(app)
        try:
            with patch("src.gradio_ui.gemma.decode_upload"):
                extracted = await b.extract()
            review = ChartReview(result().chart)
            inputs = [{"data": review.text_rows(), "headers": ["Field", "Text"]},
                      {"data": review.value_rows(), "headers": ["Position", "Value"]},
                      True, extracted["data"][-1]["value"].root]
            admitted = await b.call("admit_gemma_generation", inputs)
            with patch("src.gradio_ui.gemma.generate_chart_tactile_artifacts", side_effect=blocked):
                task = asyncio.create_task(b.call("generate_gemma", [admitted["data"][0]]))
                assert await asyncio.to_thread(entered.wait, 5)
                edit_name = next(f.api_name for f in app.fns.values() if f.api_name.startswith("edit_gemma_"))
                edited = await b.call(edit_name, inputs)
                assert "updated" in edited["data"][3]
                release.set()
                stale = await task
                assert all(item.get("__type__") == "update" for item in stale["data"])
                assert all(not path.exists() for path in produced)
        finally:
            release.set()
            app.close(verbose=False)
    asyncio.run(scenario())


def test_tab_namespaces_remain_independent():
    from tests.gradio.session_helpers import AppSession
    async def scenario():
        app = build_app(object(), gemma_extractor=SimpleNamespace(extract_result=lambda _: result()))
        detector = AppSession(app, "shared")
        gemma = Browser(app, "shared")
        try:
            await detector.analyze_stub()
            before = detector.model
            with patch("src.gradio_ui.gemma.decode_upload"):
                extracted = await gemma.extract()
            assert detector.model == before
            await detector.call("clear_all")
            assert extracted["data"][-1]["value"].root[0]
            review = ChartReview(result().chart)
            admitted = await gemma.call("admit_gemma_generation", [
                {"data": review.text_rows(), "headers": ["Field", "Text"]},
                {"data": review.value_rows(), "headers": ["Position", "Value"]},
                True, extracted["data"][-1]["value"].root])
            assert isinstance(admitted["data"][0], str)
        finally:
            app.close(verbose=False)
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["dependencies", "loading", "inference", "parsing"])
def test_extraction_failure_is_visible_and_preserves_diagnostics(stage):
    from src.baseline.gemma import GemmaExtractionError
    def fail(_):
        raise GemmaExtractionError(stage, "Safe failure guidance", {}, "invalid JSON" if stage == "parsing" else None)
    async def scenario():
        app = build_app(None, workflow="gemma", gemma_extractor=SimpleNamespace(extract_result=fail))
        try:
            with patch("src.gradio_ui.gemma.decode_upload"):
                response = await Browser(app).extract()
            assert response["data"][3] == "Safe failure guidance"
            if stage == "parsing":
                assert response["data"][4] == "invalid JSON"
        finally:
            app.close(verbose=False)
    asyncio.run(scenario())


def test_backend_serializes_concurrent_sessions():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from src.gradio_ui.gemma import GemmaBackend
    active = 0
    gate = Lock()
    def extract(_):
        nonlocal active
        with gate:
            active += 1
            assert active == 1
        try:
            return result()
        finally:
            with gate:
                active -= 1
    backend = GemmaBackend(extractor=SimpleNamespace(extract_result=extract))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(backend.extract, ["first", "second"] * 5))
    assert len(results) == 10
