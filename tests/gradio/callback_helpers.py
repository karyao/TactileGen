"""Invoke async browser callbacks with an explicit test session."""
import asyncio
import inspect
import gradio as gr


def invoke(fn, *args, **kwargs):
    if "request" in inspect.signature(fn).parameters:
        kwargs.setdefault("request", gr.Request(session_hash="test-session"))
    result = fn(*args, **kwargs)
    if inspect.isasyncgen(result):
        async def finish():
            final = None
            async for value in result:
                final = value
            # Analysis appends control updates after its result components.
            return final[:27]
        return asyncio.run(finish())
    if inspect.isawaitable(result):
        value = asyncio.run(result)
        return value[:27] if fn.__name__ in ("ready_results", "orientation_changed") else value
    return result
