"""Deterministic late-result tests through Gradio's real session processing."""
import asyncio
from dataclasses import replace
from threading import Event
from unittest.mock import patch

import gradio as gr
import pytest

from src.gradio_ui.run_tracking import RunTracker
from src.gradio_ui.review import review_text_rows
from tests.gradio.session_helpers import AppSession, make_analysis_result, wait_started


@pytest.mark.parametrize("invalidation", ["orientation_changed", "clear_all", "ready_results", "edit", "new_analysis", "unload"])
@pytest.mark.parametrize("failure", [False, True])
def test_late_generation_cannot_restore_state(app, invalidation, failure):
    async def scenario():
        browser = AppSession(app, "one")
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
            initial = await browser.analyze_stub()
            assert initial["data"][-2]["interactive"] is True
            with patch("src.gradio_ui.callbacks.generate_review_submission", side_effect=blocked):
                task = asyncio.create_task(browser.call("generate_tactile", [None, review_text_rows(browser.model)]))
                await wait_started(entered)
                if invalidation == "edit":
                    rows = review_text_rows(browser.model)
                    rows[0][1] = "New draft"
                    await browser.call("update_review", [None, rows])
                    expected = browser.model
                elif invalidation == "new_analysis":
                    await browser.analyze_stub("New analysis")
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

                for update in completed["data"]:
                    if update is None:
                        continue

                    assert isinstance(update, dict), update
                    assert update.get("__type__") == "update", update
                    assert "value" not in update, update

                for path in produced:
                    assert not path.exists(), f"Stale output still exists: {path}"
        finally:
            release.set()
    asyncio.run(scenario())


def test_stale_analysis_and_independent_sessions(app):
    async def scenario():
        one, two = AppSession(app, "one"), AppSession(app, "two")
        entered, release = Event(), Event()
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return make_analysis_result("Old")
        try:
            with patch("src.gradio_ui.callbacks.analyze_image", side_effect=blocked):
                task = asyncio.create_task(one.analyze())
                await wait_started(entered)
                await one.call("clear_all")
                await two.analyze_stub("Other session")
                release.set()
                completed = await task
            assert one.model is None
            assert two.model.spec.title == "Other session"
            assert all(item is None or (isinstance(item, dict) and item.get("__type__") == "update" and "value" not in item) for item in completed["data"]), completed["data"]
        finally:
            release.set()
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


def test_missing_session_fails_closed(app):
    async def scenario():
        callback = next(f for f in app.fns.values() if f.api_name == "orientation_changed")
        with pytest.raises(ValueError, match="session"):
            await callback.fn(gr.Request())
    asyncio.run(scenario())


def test_reset_waits_for_postprocessing_and_remains_final(app):
    async def scenario():
        import gradio.processing_utils as processing
        browser = AppSession(app, "one")
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
            await browser.analyze_stub()
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
    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["ocr_export", "generation_report"])
def test_preparation_failure_cleans_unpublished_json(app, stage):
    async def scenario():
        from src.contracts import OCRFrame
        import src.gradio_ui.callbacks as callbacks
        browser = AppSession(app, "one")
        created = []
        name = "write_ocr_download" if stage == "ocr_export" else "write_spec_download"
        original = getattr(callbacks, name)
        def capture(*args, **kwargs):
            path = original(*args, **kwargs)
            created.append(path)
            return path
        candidate = make_analysis_result()
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
    asyncio.run(scenario())
