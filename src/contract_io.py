"""JSON interchange for small chart-contract fixtures and optional inspection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.contracts import (
    BarSeries,
    Box,
    Detection,
    DetectionFrame,
    HorizontalBarChart,
    OCRFrame,
    RecognizedText,
    VerticalBarChart,
)


SCHEMA_VERSION = 1


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != SCHEMA_VERSION):
        raise ValueError(f"Expected chart contract schema_version {SCHEMA_VERSION}: {path}")
    return data


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_detection_frame(path: Path) -> DetectionFrame:
    data = _read(path)
    try:
        detections = tuple(
            Detection(item["id"], item["role"], Box(**item["box"]), item["confidence"])
            for item in _array(data["detections"], "detections")
        )
        return DetectionFrame(data["source_key"], data["width"], data["height"], detections)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid detection frame: {path}") from exc


def write_detection_frame(path: Path, frame: DetectionFrame) -> None:
    _write(path, {"schema_version": SCHEMA_VERSION, **asdict(frame)})


def load_ocr_frame(path: Path) -> OCRFrame:
    data = _read(path)
    try:
        regions = tuple(
            RecognizedText(item["detection_id"], item["text"], item["confidence"])
            for item in _array(data["regions"], "regions")
        )
        return OCRFrame(data["source_key"], regions)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid OCR frame: {path}") from exc


def write_ocr_frame(path: Path, frame: OCRFrame) -> None:
    _write(path, {"schema_version": SCHEMA_VERSION, **asdict(frame)})


_CHART_TYPES = {"vbar_categorical": VerticalBarChart, "hbar_categorical": HorizontalBarChart}


def load_vertical_bar_chart(path: Path) -> VerticalBarChart:
    chart = load_bar_chart(path)
    if not isinstance(chart, VerticalBarChart):
        raise ValueError(f"Expected vbar_categorical chart: {path}")
    return chart


def load_bar_chart(path: Path) -> VerticalBarChart | HorizontalBarChart:
    """A ``vbar_categorical`` or ``hbar_categorical`` chart file.

    For ``hbar_categorical`` the categories run top to bottom and the titles
    keep their printed axes (``x_axis_title`` is the value heading).
    """
    data = _read(path)
    chart_type = _CHART_TYPES.get(data.get("chart_type"))
    if chart_type is None:
        raise ValueError(f"Expected vbar_categorical or hbar_categorical chart: {path}")
    try:
        series = tuple(
            BarSeries(item["name"], tuple(_array(item["values"], "values")),
                      item.get("has_bars", True))
            for item in _array(data["series"], "series")
        )
        return chart_type(
            data["source_key"], data["title"], data["x_axis_title"],
            data["y_axis_title"], tuple(_array(data["categories"], "categories")), series,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid bar chart: {path}") from exc


def write_vertical_bar_chart(path: Path, chart: VerticalBarChart) -> None:
    write_bar_chart(path, chart)


def write_bar_chart(path: Path, chart: VerticalBarChart | HorizontalBarChart) -> None:
    chart_type = next(name for name, kind in _CHART_TYPES.items() if type(chart) is kind)
    data = asdict(chart)
    for series in data["series"]:  # written only for a series with no visible bars
        if series["has_bars"]:
            del series["has_bars"]
    _write(path, {
        "schema_version": SCHEMA_VERSION,
        "chart_type": chart_type,
        **data,
    })
