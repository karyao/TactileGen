"""Load the optional Gradio dependency only when the interface is requested."""

from __future__ import annotations


def import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install the application dependencies: "
            "python -m pip install -r requirements-app.txt"
        ) from exc
    return gr
