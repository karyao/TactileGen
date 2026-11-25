"""End-to-end check of five PlotQA charts that only the short-bar remeasure reads correctly.

Their very short bars carry outline colour, which makes a spurious extra series unless the boxes are left
out of colour clustering and re-measured from pixels. Switching the remeasure off, or keeping every
all-short group (the first gate prototype), blocks all five with ``series_ambiguous``.

Needs the PlotQA test PNGs and annotations, the detector weights and both OCR engines, so it skips when
any of them is missing. The data folder defaults to ``<repo>/data``; set ``TACTILEGEN_DATA`` to use another.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TACTILEGEN_DATA", REPO / "data"))
PNG_DIR = DATA / "png_test" / "png"
ANNOTATIONS = DATA / "annotations_test.json"
WEIGHTS = REPO / "models" / "yolo_bars_hv" / "best.pt"
VALUE_TOLERANCE = 0.01  # of the ground-truth y-axis maximum, as in the full-run audit

HELPED_BY_REMEASURE = ("22448", "24105", "29012", "30059", "30724")


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


@pytest.fixture(scope="module")
def annotations():
    if not (PNG_DIR.is_dir() and ANNOTATIONS.is_file()):
        pytest.skip(f"PlotQA test data not found under {DATA} (set TACTILEGEN_DATA)")
    return {str(r["image_index"]): r for r in json.loads(ANNOTATIONS.read_text())
            if r["type"] == "vbar_categorical"}


@pytest.fixture(scope="module")
def analyse():
    if not WEIGHTS.is_file():
        pytest.skip(f"detector weights not found: {WEIGHTS}")
    from src.tactile import pipeline
    try:
        pipeline._ocr_backend("hybrid")
    except Exception as error:  # a missing engine or Tesseract binary
        pytest.skip(f"hybrid OCR unavailable: {error}")

    def run(chart_id):
        spec, error, _ = pipeline.analyse(PNG_DIR / f"{chart_id}.png", WEIGHTS, "hybrid")
        assert error is None, f"chart {chart_id} was blocked: {error}"
        return spec
    return run


def _axis_max(record, models) -> float:
    ticks = sorted({float(t) for t in record["general_figure_info"]["y_axis"]["major_ticks"]["values"]})
    values = [v for m in models for v in m["y"] if v is not None]
    return max(ticks) if ticks and max(ticks) > 0 else (max((abs(v) for v in values), default=1.0) or 1.0)


@pytest.mark.parametrize("chart_id", HELPED_BY_REMEASURE)
def test_chart_helped_by_short_bar_remeasure_is_read_correctly(chart_id, annotations, analyse):
    record = annotations[chart_id]
    models = record["models"]
    spec = analyse(chart_id)

    assert [_norm(c) for c in spec.categories] == [_norm(c) for c in models[0]["x"]]
    assert len(spec.series) == len(models), "a spurious or missing series"
    assert not any(f.startswith("series_ambiguous") for f in spec.flags)
    by_name = {_norm(m["name"]): m for m in models}
    assert {_norm(s.name) for s in spec.series} == set(by_name)
    limit = VALUE_TOLERANCE * _axis_max(record, models)
    for series in spec.series:
        truth = by_name[_norm(series.name)]["y"]
        assert len(series.values) == len(truth)
        for got, want in zip(series.values, truth):
            if want is None:
                continue
            assert got is not None, f"{series.name} has a value with no reading"
            assert abs(got - want) <= limit, f"{series.name}: {got} vs {want}"
    assert any(f.startswith("short_bars_remeasured") for f in spec.flags)
