"""Invoke async browser callbacks with an explicit test session."""
import asyncio
import inspect
import gradio as gr


def invoke(fn, *args, **kwargs):
    if "request" in inspect.signature(fn).parameters:
        kwargs.setdefault("request", gr.Request(session_hash="test-session"))
    async def call():
        if "stamp" in inspect.signature(fn).parameters and "stamp" not in kwargs:
            model = args[0]
            kwargs["stamp"] = [model.run_id, model.review_revision] if model else None
        if getattr(fn, "admit", None) is not None and "admission" not in kwargs:
            prepare_args = args
            if fn.__name__ == "generate_event":
                model = args[0]
                prepare_args = (*args, [model.run_id, model.review_revision] if model else None)
            admitted = await fn.admit(*prepare_args, request=kwargs["request"])
            kwargs["admission"] = admitted[-1]["value"]
        result = fn(*args, **kwargs)
        if inspect.isasyncgen(result):
            final = None
            async for value in result:
                final = value
            return final[:27]
        if inspect.isawaitable(result):
            value = await result
            return value[:27] if fn.__name__ in ("ready_results", "orientation_changed") else value
        return result
    return asyncio.run(call())
