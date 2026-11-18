"""Deterministic late-result tests through Gradio's real session processing."""
import asyncio
from dataclasses import replace
from threading import Event
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np
import pytest

from src.contracts import DetectionFrame
from src.gradio_app import AnalysisResult, build_app
from src.gradio_ui.run_tracking import RunTracker
from src.gradio_ui.review import review_text_rows
from src.tactile.completeness import Completeness, READY
from tests.gradio.test_gradio_app import make_spec


def result(title="Original"):
    return AnalysisResult("summary", np.zeros((80, 100, 3), dtype=np.uint8), 0,
                          DetectionFrame("input/chart.png", 100, 80, ()), None, None,
                          spec=make_spec(title=title), completeness=Completeness(READY))


class Browser:
    def __init__(self, app, key):
        self.app, self.key = app, key
        self.state = SessionState(app)
        self.request = gr.Request(session_hash=key)
        component = self.fn("analyze").inputs[0]
        component.preprocess = lambda value: "chart.png"

    def fn(self, name):
        return next(f for f in self.app.fns.values() if f.api_name == name)

    async def call(self, name, inputs=(), iterator=None):
        inputs = list(inputs)
        if name in ("generate_tactile", "update_review"):
            inputs[1] = {"data": inputs[1], "headers": ["Field", "Text"]}
        return await self.app.process_api(self.fn(name), inputs, state=self.state,
                                         request=self.request, session_hash=self.key, iterator=iterator)

    async def analyze(self):
        inputs = [None, .25, .7, "eng", 640, "rapidocr", "auto"]
        first = await self.call("analyze", inputs)
        return await self.call("analyze", inputs, first["iterator"])

    @property
    def model(self):
        return self.state[self.fn("generate_tactile").inputs[0]._id]


async def wait_started(event):
    assert await asyncio.to_thread(event.wait, 5), "worker did not start"


@pytest.mark.parametrize("invalidation", ["orientation_changed", "clear_all", "ready_results", "edit", "new_analysis", "unload"])
@pytest.mark.parametrize("failure", [False, True])
def test_late_generation_cannot_restore_state(invalidation, failure):
    async def scenario():
        app = build_app(object())
        browser = Browser(app, "one")
        entered, release = Event(), Event()
        from src.gradio_ui.callbacks import generate_review_submission as real_generate
        produced = []

        def blocked(*args):
            entered.set()
            assert release.wait(5)
            if failure:
                raise RuntimeError("late failure")
            submission = real_generate(*args)
            produced.append(submission.generation.output_dir)
            return submission

        try:
            with patch("src.gradio_ui.callbacks.analyze_image", return_value=result()):
                initial = await browser.analyze()
                # Initial and completion yields disable and enable controls without changing their values.
                assert initial["data"][-1]["interactive"] is True
                with patch("src.gradio_ui.callbacks.generate_review_submission", side_effect=blocked):
                    task = asyncio.create_task(browser.call("generate_tactile", [None, review_text_rows(browser.model)]))
                    await wait_started(entered)
                    if invalidation == "edit":
                        rows = review_text_rows(browser.model)
                        rows[0][1] = "New draft"
                        await browser.call("update_review", [None, rows])
                        expected = browser.model
                    elif invalidation == "new_analysis":
                        with patch("src.gradio_ui.callbacks.analyze_image", return_value=result("New analysis")):
                            await browser.analyze()
                        expected = browser.model
                    elif invalidation == "unload":
                        unload = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == "unload")
                        await unload.fn(browser.request)
                        expected = browser.model
                    else:
                        await browser.call(invalidation)
                        expected = None
                    release.set()
                    completed = await task
                    assert browser.model == expected
                    assert all(item is None or (isinstance(item, dict) and item.get("__type__") == "update" and "value" not in item) for item in completed["data"]), completed["data"]
                    assert all(not path.exists() for path in produced)
        finally:
            release.set()
            app.close(verbose=False)
    asyncio.run(scenario())


def test_stale_analysis_and_independent_sessions():
    async def scenario():
        app = build_app(object())
        one, two = Browser(app, "one"), Browser(app, "two")
        entered, release = Event(), Event()
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return result("Old")
        try:
            with patch("src.gradio_ui.callbacks.analyze_image", side_effect=blocked):
                task = asyncio.create_task(one.analyze())
                await wait_started(entered)
                await one.call("clear_all")
                with patch("src.gradio_ui.callbacks.analyze_image", return_value=result("Other session")):
                    await two.analyze()
                release.set()
                completed = await task
            assert one.model is None
            assert two.model.spec.title == "Other session"
            assert all(item is None or (isinstance(item, dict) and item.get("__type__") == "update" and "value" not in item) for item in completed["data"]), completed["data"]
        finally:
            release.set()
            app.close(verbose=False)
    asyncio.run(scenario())


def test_tracker_revisions_and_removal():
    tracker = RunTracker()
    old = tracker.start("one", "run-a")
    generation = tracker.advance("one", "run-a", 0)
    assert not tracker.current("one", old)
    assert tracker.matches_review("one", "run-a", 0)
    edited = tracker.advance("one", "run-a", 0, edit=True)
    assert not tracker.current("one", generation)
    assert edited.review_revision == 1
    assert tracker.advance("one", "run-a", 0) is None
    tracker.remove("one")
    assert not tracker.current("one", edited)


def test_missing_session_fails_closed():
    async def scenario():
        app = build_app(object())
        try:
            callback = next(f for f in app.fns.values() if f.api_name == "orientation_changed")
            with pytest.raises(ValueError, match="session"):
                await callback.fn(gr.Request())
        finally:
            app.close(verbose=False)
    asyncio.run(scenario())


def test_reset_waits_for_postprocessing_and_remains_final():
    async def scenario():
        import gradio.processing_utils as processing
        app = build_app(object())
        browser = Browser(app, "one")
        entered, release = asyncio.Event(), asyncio.Event()
        original = processing.async_move_files_to_cache
        once = False
        async def blocked(*args, **kwargs):
            nonlocal once
            if kwargs.get("postprocess") and not once:
                once = True
                entered.set()
                await release.wait()
            return await original(*args, **kwargs)
        try:
            with patch("src.gradio_ui.callbacks.analyze_image", return_value=result()):
                await browser.analyze()
            with patch.object(processing, "async_move_files_to_cache", side_effect=blocked):
                generated = asyncio.create_task(browser.call("generate_tactile", [None, review_text_rows(browser.model)]))
                await asyncio.wait_for(entered.wait(), 5)
                reset = asyncio.create_task(browser.call("orientation_changed"))
                await asyncio.sleep(0)  # Let reset contend for the publication lock.
                assert not reset.done()
                release.set()
                await generated
                await reset
                assert browser.model is None
        finally:
            release.set()
            app.close(verbose=False)
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["ocr_export", "generation_report"])
def test_preparation_failure_cleans_unpublished_json(stage):
    async def scenario():
        from src.contracts import OCRFrame
        import src.gradio_ui.callbacks as callbacks
        app = build_app(object())
        browser = Browser(app, "one")
        created = []
        name = "write_ocr_download" if stage == "ocr_export" else "write_spec_download"
        original = getattr(callbacks, name)
        def capture(*args, **kwargs):
            path = original(*args, **kwargs)
            created.append(path)
            return path
        try:
            candidate = result()
            if stage == "ocr_export":
                candidate = replace(candidate, ocr_frame=OCRFrame("input/chart.png", ()), legend_mapping=object())
            with patch.object(callbacks, "analyze_image", return_value=candidate):
                if stage == "ocr_export":
                    with patch.object(callbacks, name, side_effect=capture), patch.object(callbacks, "legend_rows", side_effect=RuntimeError("after export")):
                        await browser.analyze()
                else:
                    await browser.analyze()
                    with patch.object(callbacks, name, side_effect=capture), patch.object(callbacks, "generation_report", side_effect=RuntimeError("after export")):
                        await browser.call("generate_tactile", [None, review_text_rows(browser.model)])
            assert created
            assert all(not path.exists() for path in created)
        finally:
            app.close(verbose=False)
    asyncio.run(scenario())
