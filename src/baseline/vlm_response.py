"""Dependency-free adaptation of VLM response text to the chart contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from src.contract_io import SCHEMA_VERSION, parse_bar_chart
from src.contracts import HorizontalBarChart, VerticalBarChart


_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)
_TITLE_DEFAULTS = {
    "title": "Untitled chart",
    "x_axis_title": "X axis",
    "y_axis_title": "Y axis",
}


@dataclass(frozen=True)
class ParsedChartResponse:
    chart: VerticalBarChart | HorizontalBarChart
    placeholder_fields: tuple[str, ...]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"Duplicate JSON key: {key}")
        data[key] = value
    return data


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def parse_chart_response(
    response: str, *, image: str | Path, source_key: str | None = None,
) -> ParsedChartResponse:
    """Parse one JSON object, optionally inside a single ``json`` code fence.

    The adapter supplies the internal schema version; the model returns chart
    content only. The caller owns the source identity; ``image`` supplies only its filename
    for the default key and is not opened. Missing/null/blank titles receive
    placeholders tracked separately from the chart. Invalid responses raise
    ``ValueError`` without repairing or salvaging model output.
    """
    if not isinstance(response, str):
        raise ValueError("VLM response must be text containing a JSON object")
    text = response.strip()
    fence = _JSON_FENCE.fullmatch(text)
    if fence is not None:
        text = fence.group(1)
    try:
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid VLM response JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("VLM response must contain a JSON object")
    if set(data) == {"error"}:
        raise ValueError(f"VLM extraction failed: {data['error']}")

    data["schema_version"] = SCHEMA_VERSION
    data["source_key"] = source_key if source_key is not None else f"vlm/{Path(image).name}"
    placeholders = []
    for field, default in _TITLE_DEFAULTS.items():
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            data[field] = default
            placeholders.append(field)
    return ParsedChartResponse(parse_bar_chart(data), tuple(placeholders))
