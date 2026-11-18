"""Small chart builders and a Gradio API session for callback tests.

AppSession uses real Gradio state and callbacks; it does not launch a browser.
"""
import asyncio
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import numpy as np

from src.contracts import DetectionFrame, SeriesSpec, VerticalBarSpec, YAxisSpec
from src.gradio_app import AnalysisResult
from src.tactile.completeness import Completeness, READY


def make_spec(**changes):
    fields = dict(
        source_key="input/chart.png",
        title="Sales",
        x_axis_title="Year",
        y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]),
        categories=["2024"],
        series=[SeriesSpec(None, "#112233", [50.0])],
        bars_adjacent=False,
        flags=[],
        confidence={},
    )
    fields.update(changes)
    return VerticalBarSpec(**fields)


def make_analysis_result(title="Original"):
    return AnalysisResult("summary", np.zeros((80, 100, 3), dtype=np.uint8), 0,
                          DetectionFrame("input/chart.png", 100, 80, ()), None, None,
                          spec=make_spec(title=title), completeness=Completeness(READY))


class AppSession:
    def __init__(self, app, key):
        self.app, self.key = app, key
        self.state = SessionState(app)
        self.request = gr.Request(session_hash=key)
        component = self.find_callback("analyze").inputs[0]
        component.preprocess = lambda value: "chart.png"

    def find_callback(self, name):
        return next(f for f in self.app.fns.values() if f.api_name == name)

    async def call(self, name, inputs=(), iterator=None):
        inputs = list(inputs)
        if name == "update_review" and len(inputs) == 2:
            model = self.model
            inputs.append([model.run_id, model.review_revision] if model else None)
        if name == "generate_tactile" and len(inputs) == 2:
            model = self.model
            admitted = await self.call("admit_generation", [*inputs, [model.run_id, model.review_revision] if model else None])
            token = admitted["data"][3].get("value")
            inputs.append(token)
        if name in ("generate_tactile", "update_review", "admit_generation") and isinstance(inputs[1], list):
            inputs[1] = {"data": inputs[1], "headers": ["Field", "Text"]}
        return await self.app.process_api(self.find_callback(name), inputs, state=self.state,
                                         request=self.request, session_hash=self.key, iterator=iterator)

    async def analyze(self):
        inputs = [None, .25, .7, "eng", 640, "rapidocr", "auto"]
        admitted = await self.call("admit_analysis", inputs)
        inputs.append(admitted["data"][-2]["value"])
        first = await self.call("analyze", inputs)
        return await self.call("analyze", inputs, first["iterator"])

    async def analyze_stub(self, title="Original"):
        """Run the real callbacks with a prepared result instead of model inference."""
        with patch("src.gradio_ui.callbacks.analyze_image",
                   return_value=make_analysis_result(title)):
            return await self.analyze()

    @property
    def model(self):
        return self.state[self.find_callback("generate_tactile").inputs[0]._id]


async def wait_started(event):
    assert await asyncio.to_thread(event.wait, 5), "worker did not start"

