"""Load the optional Gradio dependency only when the interface is requested."""

from __future__ import annotations

import inspect


def import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install the application dependencies: "
            "python -m pip install -r requirements-app.txt"
        ) from exc
    return gr


def image_component(gr, *, alt_text: str, **kwargs):
    """Keep alt text on Gradio versions that support it, without breaking older 6.x.

    Component labels remain available on versions without image alt text.
    Inspect the signature rather than catching TypeError, so unrelated component
    construction errors still propagate.
    """
    if "alt_text" in inspect.signature(gr.Image).parameters:
        kwargs["alt_text"] = alt_text
    return gr.Image(**kwargs)


def launch_app(demo, **kwargs):
    """Disable optional Gradio services when the installed version exposes them.

    Older versions have no run-history option. Keep the required launch settings
    intact so unsupported settings or unrelated launch errors remain visible.
    """
    parameters = inspect.signature(demo.launch).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    for name in ("run_history", "enable_monitoring"):
        if name in parameters or accepts_kwargs:
            kwargs[name] = False
    return demo.launch(**kwargs)
