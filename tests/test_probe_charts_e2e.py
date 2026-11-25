"""End-to-end check of the bars_mostly_missed pixel-probe rule on real PlotQA charts.

Released: charts whose small bars are slivers the pixel probe measures; the block was wrong (the spec is right).
Still blocked: charts with negative bars the chart does not draw (the axis starts at 0); the probe reads 0 for
them, the spec is wrong, so the block must stay.

Needs the PlotQA test PNGs and annotations, the detector weights and both OCR engines, so it skips when any of
them is missing. The data folder defaults to ``<repo>/data``; set ``TACTILEGEN_DATA`` to use another.
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

RELEASED = ("22643", "22780", "25234")
STILL_BLOCKED = ("22451", "22453", "22457", "22683")


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


def assert_matches_ground_truth(record, spec):
    """The full-run audit's definition of correct: categories, series names and every value within 1% of the axis max."""
    models = record["models"]
    ticks = sorted({float(t) for t in record["general_figure_info"]["y_axis"]["major_ticks"]["values"]})
    limit = VALUE_TOLERANCE * max(ticks)
    by_name = {_norm(m["name"]): m for m in models}
    assert [_norm(c) for c in spec.categories] == [_norm(c) for c in models[0]["x"]]
    assert {_norm(s.name) for s in spec.series} == set(by_name)
    for series in spec.series:
        for got, want in zip(series.values, by_name[_norm(series.name)]["y"]):
            if want is not None:
                assert got is not None and abs(got - want) <= limit, f"{series.name}: {got} vs {want}"


@pytest.mark.parametrize("chart_id", RELEASED)
def test_chart_with_measured_slivers_is_no_longer_blocked_and_is_correct(chart_id, annotations, analyse):
    from src.tactile.completeness import check_completeness

    spec = analyse(chart_id)
    assert not any(f.startswith("bars_mostly_missed") for f in spec.flags)
    assert not check_completeness(spec).blockers
    assert_matches_ground_truth(annotations[chart_id], spec)


@pytest.mark.parametrize("chart_id", STILL_BLOCKED)
def test_chart_with_undrawn_negative_bars_stays_blocked(chart_id, annotations, analyse):
    from src.tactile.completeness import check_completeness

    models = annotations[chart_id]["models"]
    assert any(v < 0 for m in models for v in m["y"] if v is not None), "not an undrawn-negative chart"
    spec = analyse(chart_id)
    assert any(f.startswith("bars_mostly_missed") for f in spec.flags)
    assessment = check_completeness(spec)
    assert assessment.status == "cannot_build" and assessment.forceable
