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
    """Load a schema-version-1 categorical bar chart file."""
    return parse_bar_chart(_read(path))


def _fields(data: Any, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(sorted(missing))}")
    unknown = data.keys() - required - optional
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(map(str, unknown)))}")


def parse_bar_chart(data: dict[str, Any]) -> VerticalBarChart | HorizontalBarChart:
    """Validate an already-decoded schema-version-1 chart without file I/O.

    For ``hbar_categorical`` the categories run top to bottom and the titles
    keep their printed axes (``x_axis_title`` is the value heading).
    Raises ``ValueError`` for invalid contracts; does not modify ``data``.
    """
    _fields(data, {
        "schema_version", "chart_type", "source_key", "title", "x_axis_title",
        "y_axis_title", "categories", "series",
    }, {"omit_legend"}, "chart")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Expected chart contract schema_version {SCHEMA_VERSION}")
    kind = data["chart_type"]
    if not isinstance(kind, str) or kind not in _CHART_TYPES:
        raise ValueError("Expected vbar_categorical or hbar_categorical chart")
    series = []
    for index, item in enumerate(_array(data["series"], "series")):
        name = f"series[{index}]"
        _fields(item, {"name", "values"}, {"has_bars", "rescued"}, name)
        has_bars = item.get("has_bars", True)
        if not isinstance(has_bars, bool):
            raise ValueError(f"{name}.has_bars must be a boolean")
        series.append(BarSeries(
            item["name"], tuple(_array(item["values"], f"{name}.values")),
            has_bars, tuple(_array(item.get("rescued", []), f"{name}.rescued")),
        ))
    return _CHART_TYPES[kind](
        data["source_key"], data["title"], data["x_axis_title"],
        data["y_axis_title"], tuple(_array(data["categories"], "categories")), tuple(series),
        omit_legend=data.get("omit_legend", False),
    )


def write_vertical_bar_chart(path: Path, chart: VerticalBarChart) -> None:
    write_bar_chart(path, chart)


def write_bar_chart(path: Path, chart: VerticalBarChart | HorizontalBarChart) -> None:
    chart_type = next(name for name, kind in _CHART_TYPES.items() if type(chart) is kind)
    data = asdict(chart)
    if not chart.omit_legend:
        del data["omit_legend"]
    for series in data["series"]:  # written only for a series with no visible bars
        if series["has_bars"]:
            del series["has_bars"]
        if not any(series["rescued"]):    # written only when some value was rescued
            del series["rescued"]
    _write(path, {
        "schema_version": SCHEMA_VERSION,
        "chart_type": chart_type,
        **data,
    })
