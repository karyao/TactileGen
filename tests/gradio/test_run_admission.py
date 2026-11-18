"""Admission snapshots, cancelled workers, and transactional publication."""
import asyncio
from threading import Event
from unittest.mock import patch

from gradio.data_classes import PredictBodyInternal
from starlette.requests import Request
import pytest

from src.gradio_ui.review import review_text_rows
from src.gradio_ui.run_tracking import RunTracker
from tests.gradio.session_helpers import AppSession, make_analysis_result, wait_started


@pytest.mark.parametrize("reset", ["clear_all", "orientation_changed", "ready_results"])
def test_actual_queue_rejects_invalidated_analysis_admission(app, reset):
    async def scenario():
        running_session = AppSession(app, "running")
        queued_session = AppSession(app, "queued")
        queue = app._queue
        args = [None, .25, .7, "eng", 640, "rapidocr", "auto"]
        request = Request({"type": "http", "method": "POST", "path": "/gradio_api/queue/join",
                           "headers": [], "query_string": b"", "server": ("test", 80)})
        async def enqueue(browser):
            admitted = await browser.call("admit_analysis", args)
            token = admitted["data"][-2]["value"]
            body = PredictBodyInternal(session_hash=browser.key, fn_index=browser.find_callback("analyze")._id,
                                       data=[*args, token])
            success, _, _ = await queue.push(body, request, None)
            assert success
        await enqueue(running_session)
        first = queue.get_events()
        group = queue.event_queue_per_concurrency_id["tactilegen-inference"]
        group.current_concurrency = 1
        await enqueue(queued_session)
        assert queue.get_events() is None
        await queued_session.call(reset)
        group.current_concurrency = 0
        events, _, _ = queue.get_events()
        with patch("src.gradio_ui.callbacks.analyze_image") as analyze:
            response = await queued_session.call("analyze", events[0].data.data)
            analyze.assert_not_called()
        assert queued_session.model is None
        assert all(value is None or "value" not in value for value in response["data"])
    asyncio.run(scenario())


def test_generation_rejects_old_client_revision_and_freezes_admitted_rows(app):
    async def scenario():
        browser = AppSession(app, "one")
        await browser.analyze_stub()
        old = browser.model
        old_rows = review_text_rows(old)
        rows = review_text_rows(old)
        rows[0][1] = "New draft"
        await browser.call("update_review", [None, rows])
        refused = await browser.call("admit_generation", [None, old_rows, [old.run_id, old.review_revision]])
        assert refused["data"][3]["value"] is None
        assert browser.model.draft_values[0] == "New draft"
        current = browser.model
        admitted = await browser.call("admit_generation", [None, rows, [current.run_id, current.review_revision]])
        token = admitted["data"][3]["value"]
        # The queued stage ignores live/stale request-table rows in favor of admission snapshot.
        await browser.call("generate_tactile", [None, old_rows, token])
        assert browser.model.spec.title == "New draft"
        with patch("src.gradio_ui.callbacks.generate_review_submission") as generate:
            await browser.call("generate_tactile", [None, rows, token])
            generate.assert_not_called()  # one-use token
    asyncio.run(scenario())


def test_cancelled_generation_cleans_files_after_worker_finishes(app):
    async def scenario():
        import src.gradio_ui.callbacks as callbacks
        browser = AppSession(app, "one")
        entered, release, cleaned = Event(), Event(), Event()
        original = callbacks.generate_review_submission
        real_cleanup = callbacks.cleanup_generation
        paths = []
        def blocked(*args):
            submitted = original(*args)
            paths.append(submitted.generation.output_dir)
            entered.set()
            assert release.wait(5)
            return submitted
        def cleanup(*args):
            try:
                return real_cleanup(*args)
            finally:
                cleaned.set()
        try:
            await browser.analyze_stub()
            with patch.object(callbacks, "generate_review_submission", side_effect=blocked), patch.object(callbacks, "cleanup_generation", side_effect=cleanup):
                task = asyncio.create_task(browser.call("generate_tactile", [None, review_text_rows(browser.model)]))
                await wait_started(entered)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
                assert await asyncio.to_thread(cleaned.wait, 5)
                assert all(not path.exists() for path in paths)
        finally:
            release.set()
    asyncio.run(scenario())


def test_expired_lock_is_reclaimed_only_after_all_users_exit():
    async def scenario():
        tracker = RunTracker()
        for index in range(2):
            key = str(index)
            async with tracker.publication(key):
                tracker.start(key, "run")
                release = tracker.retain(key)
                entry = tracker._publication_locks[key]
                tracker.remove(key)
                assert tracker._publication_locks[key] is entry
            assert key in tracker._publication_locks
            release()
            assert key not in tracker._publication_locks
        assert not tracker._publication_locks
    asyncio.run(scenario())


def test_failed_postprocessing_rolls_back_review_state(app):
    async def scenario():
        import gradio.processing_utils as processing
        browser = AppSession(app, "one")
        original = processing.async_move_files_to_cache
        calls = 0
        async def fail(*args, **kwargs):
            nonlocal calls
            if kwargs.get("postprocess"):
                calls += 1
                # Fail after review_state would already have been assigned by Gradio.
                if calls == 6:
                    raise RuntimeError("cache failure")
            return await original(*args, **kwargs)
        await browser.analyze_stub()
        model = browser.model
        with patch.object(processing, "async_move_files_to_cache", side_effect=fail):
            response = await browser.call("generate_tactile", [None, review_text_rows(model)])
        assert browser.model is None
        assert "Results could not be displayed" in str(response["data"])
        await browser.call("clear_all")
        assert browser.model is None
    asyncio.run(scenario())


def test_queued_generation_is_invalidated_by_a_newer_review(app):
    async def scenario():
        browser = AppSession(app, "one")
        await browser.analyze_stub()
        model = browser.model
        rows = review_text_rows(model)
        admitted = await browser.call("admit_generation", [None, rows, [model.run_id, model.review_revision]])
        token = admitted["data"][3]["value"]
        changed = review_text_rows(model)
        changed[0][1] = "Latest"
        await browser.call("update_review", [None, changed])
        with patch("src.gradio_ui.callbacks.generate_review_submission") as generate:
            await browser.call("generate_tactile", [None, rows, token])
            generate.assert_not_called()
        assert browser.model.draft_values[0] == "Latest"
        # A late edit event carrying an earlier client revision cannot reverse it.
        await browser.call("update_review", [None, rows, [model.run_id, model.review_revision]])
        assert browser.model.draft_values[0] == "Latest"
    asyncio.run(scenario())


def test_cancelled_analysis_cleans_export_after_thread_completes(app):
    async def scenario():
        from pathlib import Path
        import src.gradio_ui.callbacks as callbacks
        browser = AppSession(app, "one")
        entered, release, cleaned = Event(), Event(), Event()
        created = []
        write, unlink = callbacks.write_spec_download, Path.unlink
        def blocked(*args):
            path = write(*args)
            created.append(path)
            entered.set()
            assert release.wait(5)
            return path
        def observe(path, *args, **kwargs):
            value = unlink(path, *args, **kwargs)
            if path in created:
                cleaned.set()
            return value
        try:
            with patch.object(callbacks, "analyze_image", return_value=make_analysis_result()), patch.object(callbacks, "write_spec_download", side_effect=blocked), patch.object(Path, "unlink", observe):
                task = asyncio.create_task(browser.analyze())
                await wait_started(entered)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await browser.call("clear_all")
                release.set()
                assert await asyncio.to_thread(cleaned.wait, 5)
                assert all(not path.exists() for path in created)
                assert browser.model is None
        finally:
            release.set()
    asyncio.run(scenario())


def test_missing_or_foreign_admission_never_runs_analysis(app):
    async def scenario():
        a, b = AppSession(app, "a"), AppSession(app, "b")
        args = [None, .25, .7, "eng", 640, "rapidocr", "auto"]
        admitted = await a.call("admit_analysis", args)
        token = admitted["data"][-2]["value"]
        with patch("src.gradio_ui.callbacks.analyze_image") as analyze:
            await b.call("analyze", [*args, token])
            await b.call("analyze", [*args, None])
            analyze.assert_not_called()
        # The foreign request did not consume the owner's admission.
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=make_analysis_result()):
            first = await a.call("analyze", [*args, token])
            await a.call("analyze", [*args, token], first["iterator"])
        assert a.model is not None
    asyncio.run(scenario())


@pytest.mark.parametrize("rejected", [False, True])
def test_unchanged_or_rejected_edit_preserves_queued_generation(app, rejected):
    async def scenario():
        browser = AppSession(app, "one")
        await browser.analyze_stub()
        model = browser.model
        rows = review_text_rows(model)
        admitted = await browser.call("admit_generation", [None, rows, [model.run_id, model.review_revision]])
        token = admitted["data"][3]["value"]
        await browser.call("update_review", [None, rows[:-1] if rejected else rows])
        assert browser.model.review_revision == model.review_revision
        await browser.call("generate_tactile", [None, rows, token])
        assert browser.model.spec.title == "Original"
    asyncio.run(scenario())


def test_expired_publication_lock_survives_waiters():
    async def scenario():
        tracker = RunTracker()
        queued = asyncio.Event()
        first = tracker.publication("one")
        await first.__aenter__()
        entry = tracker._publication_locks["one"]
        async def waiter():
            queued.set()
            async with tracker.publication("one"):
                assert tracker._publication_locks["one"] is entry
        task = asyncio.create_task(waiter())
        await queued.wait()
        tracker.remove("one")
        assert entry.users == 2
        await first.__aexit__(None, None, None)
        await task
        assert "one" not in tracker._publication_locks
    asyncio.run(scenario())
