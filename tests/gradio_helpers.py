"""Small chart/image fixtures shared by the Gradio test suites."""

import unittest
from pathlib import Path

from PIL import Image
from src.contracts import SeriesSpec, VerticalBarSpec, YAxisSpec


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



class GradioTestCase(unittest.TestCase):
    def _image(
        self,
        root: Path,
        name: str = "chart.png",
        *,
        size=(100, 80),
        format: str | None = None,
        rgb=(255, 255, 255),
    ) -> Path:
        path = root / name
        Image.new("RGB", size, rgb).save(path, format=format)
        return path

