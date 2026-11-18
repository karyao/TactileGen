"""User-controlled legend omission preserves data and structural blockers."""
import asyncio
from dataclasses import replace
from unittest.mock import patch

import pytest

from src.contract_io import load_bar_chart, write_bar_chart
from src.gradio_ui.review import (
    ManualGenerationBlocked, build_review_model, generate_review_submission,
    review_overrides, review_text_rows, update_review_draft,
)
from src.reconstruction.horizontal import reconstruct_hbar, transpose_frame, transpose_image
from src.tactile.completeness import check_completeness, spec_to_chart
from src.tactile.layout import layout_vertical_bar_chart_pages
from src.tactile.layout_hbar import layout_horizontal_bar_chart_pages
from src.tactile.rules import check_layout
from tests.gradio.session_helpers import AppSession, make_analysis_result
from tests.test_reconstruct import GREEN, RED, _grouped


def no_legend_spec(horizontal=False, failure=None):
    chart = _grouped((RED, GREEN), legend=False)
    bars = [d for d in chart.detections if d.role == "vertical_bar"]
    if failure == "incomplete":
        chart.detections.remove(bars[2])
    elif failure == "order":
        for bar, colour in zip(bars[2:4], (GREEN, RED)):
            box = bar.box
            chart.paint((box.x1, box.y1, box.x2, box.y2), colour)
    spec = (reconstruct_hbar(transpose_frame(chart.frame()), chart.texts,
                             transpose_image(chart.image)) if horizontal else chart.run())
    return replace(spec, title="Sales", x_axis_title="Group", y_axis_title="Revenue")


@pytest.mark.parametrize("horizontal", [False, True])
def test_omit_legend_generates_all_bars_without_key(horizontal, tmp_path):
    spec = no_legend_spec(horizontal)
    assessment = check_completeness(spec)
    assert assessment.status == "needs_input"
    model = build_review_model(spec, assessment)
    rows = review_text_rows(model)
    with pytest.raises(ValueError, match="name cannot be blank"):
        review_overrides(model)
    draft = update_review_draft(model, rows, omit_legend=True)
    result = generate_review_submission(draft, rows, tmp_path)
    corrected = result.review_model.spec
    assert result.completeness.status == "ready"
    assert corrected.omit_legend
    assert corrected.series == spec.series
    assert result.generation.zip_path.is_file()
    chart = spec_to_chart(corrected)
    layout_fn = layout_horizontal_bar_chart_pages if horizontal else layout_vertical_bar_chart_pages
    layouts = layout_fn(chart)
    assert sum(len(page.bars) for page in layouts) == 6
    assert all(not page.key_samples for page in layouts)
    assert all(not label.id.startswith("key-") for page in layouts for label in page.labels)
    assert all(v.rule_id != "key_before_graph" for page in layouts for v in check_layout(page))
    for page in result.generation.pages:
        assert "key-sample" not in page.clean_svg.read_text()
        assert "key-sample" not in page.review_svg.read_text()
    path = tmp_path / "chart.json"
    write_bar_chart(path, chart)
    assert load_bar_chart(path) == chart
    restored = update_review_draft(result.review_model, rows, omit_legend=False)
    with pytest.raises(ValueError, match="name cannot be blank"):
        generate_review_submission(restored, rows, tmp_path)


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("failure", ["incomplete", "order", "collision", "unaccounted"])
def test_omit_legend_cannot_bypass_structural_blockers(horizontal, failure, tmp_path):
    spec = no_legend_spec(horizontal, failure)
    if failure in {"collision", "unaccounted"}:
        flag = "category_ambiguous" if failure == "collision" else "bar_colour_unaccounted:1"
        spec = replace(spec, flags=[*spec.flags, flag])
    model = build_review_model(spec, check_completeness(spec))
    rows = review_text_rows(model)
    draft = update_review_draft(model, rows, omit_legend=True)
    with pytest.raises(ManualGenerationBlocked):
        generate_review_submission(draft, rows, tmp_path)
    assert not list(tmp_path.iterdir())


def test_checkbox_updates_review_revision_generates_and_resets(app):
    async def scenario():
        session = AppSession(app, "omit-legend")
        spec = no_legend_spec()
        result = replace(make_analysis_result(), spec=spec, completeness=check_completeness(spec))
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=result):
            await session.analyze()
        model = session.model
        rows = review_text_rows(model)
        await session.call("update_legend", [None, rows,
            [model.run_id, model.review_revision], False])
        assert session.model.omit_legend
        assert session.model.review_revision == model.review_revision + 1
        await session.call("generate_tactile", [None, rows])
        assert session.model.spec.omit_legend
        assert session.model.spec.series == spec.series
        # Changing the image resets the checkbox and its server-side draft.
        response = await session.call("ready_results")
        assert session.model is None
        assert response["data"][27]["value"] is True
    asyncio.run(scenario())
