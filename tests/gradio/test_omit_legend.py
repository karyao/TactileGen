"""Only single-series charts may omit the tactile series key."""
import asyncio
from dataclasses import replace
from unittest.mock import patch

import pytest

from src.contract_io import load_bar_chart, write_bar_chart
from src.contracts import SeriesSpec
from src.gradio_ui.review import (
    ManualGenerationBlocked, build_review_model, generate_review_submission,
    review_overrides, review_text_rows, update_review_draft,
)
from src.reconstruction.horizontal import reconstruct_hbar, transpose_frame, transpose_image
from src.tactile.completeness import check_completeness, spec_to_chart
from src.tactile.layout import layout_vertical_bar_chart_pages
from src.tactile.layout_hbar import layout_horizontal_bar_chart_pages
from src.tactile.rules import check_layout
from tests.gradio.session_helpers import AppSession, make_analysis_result, make_spec
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


def named_rows(model):
    rows = review_text_rows(model)
    for binding, row in zip(model.bindings, rows):
        if binding.key.startswith("series_name["):
            row[1] = "Alpha" if binding.key == "series_name[0]" else "Beta"
    return rows


@pytest.mark.parametrize("horizontal", [False, True])
def test_multiple_series_require_names_and_key_despite_omission(horizontal, tmp_path):
    spec = replace(no_legend_spec(horizontal), omit_legend=True)
    assessment = check_completeness(spec)
    assert assessment.status == "needs_input"
    assert {"series_name[0]", "series_name[1]"} <= {m.field for m in assessment.missing}
    model = build_review_model(spec, assessment)
    assert not model.omit_legend
    rows = review_text_rows(model)
    draft = update_review_draft(model, rows, omit_legend=True)
    assert not draft.omit_legend
    # Even a stale or manually constructed review model cannot skip names.
    with pytest.raises(ValueError, match="name cannot be blank"):
        review_overrides(replace(draft, omit_legend=True))
    with pytest.raises(ValueError, match="name cannot be blank"):
        generate_review_submission(draft, rows, tmp_path)
    assert not list(tmp_path.iterdir())
    rows = named_rows(model)
    result = generate_review_submission(draft, rows, tmp_path)
    corrected = result.review_model.spec
    assert result.completeness.status == "ready"
    assert not corrected.omit_legend
    assert [s.values for s in corrected.series] == [s.values for s in spec.series]
    assert result.generation.zip_path.is_file()
    chart = spec_to_chart(replace(corrected, omit_legend=True))
    assert not chart.omit_legend
    # Old serialized charts remain readable, but direct layout cannot omit their key.
    path = tmp_path / "chart.json"
    write_bar_chart(path, replace(chart, omit_legend=True))
    chart = load_bar_chart(path)
    assert chart.omit_legend
    layout_fn = layout_horizontal_bar_chart_pages if horizontal else layout_vertical_bar_chart_pages
    layouts = layout_fn(chart)
    assert sum(len(page.bars) for page in layouts) == 6
    assert sum(len(page.key_samples) for page in layouts) >= 2
    assert all(v.rule.id != "key_before_graph" for page in layouts for v in check_layout(page))
    for page in result.generation.pages:
        assert "key-sample" in page.clean_svg.read_text()
        assert "key-sample" in page.review_svg.read_text()
    page = next(page for page in layouts if page.key_samples)
    assert [s.series_index for s in sorted(page.key_samples, key=lambda s: s.y)] == [0, 1]
    for bar in page.bars:
        sample = next(s for s in page.key_samples if s.series_index == bar.series_index)
        assert sample.texture == bar.texture
    broken = replace(page, key_samples=(), series_key_elsewhere=False, omit_legend=True)
    assert any(v.rule.id == "key_before_graph" for v in check_layout(broken))
    assert not any(v.rule.id == "key_before_graph" for v in check_layout(
        replace(broken, series_key_elsewhere=True)))


@pytest.mark.parametrize("horizontal", [False, True])
def test_single_series_can_omit_key(horizontal, tmp_path):
    spec = make_spec(series=[SeriesSpec("Sales", "#112233", [50.0])],
                     orientation="horizontal" if horizontal else "vertical")
    model = build_review_model(spec, check_completeness(spec))
    rows = review_text_rows(model)
    rows[-1][1] = ""
    draft = update_review_draft(model, rows, omit_legend=True)
    assert draft.omit_legend
    result = generate_review_submission(draft, rows, tmp_path)
    chart = spec_to_chart(result.review_model.spec)
    assert chart.omit_legend
    layout_fn = layout_horizontal_bar_chart_pages if horizontal else layout_vertical_bar_chart_pages
    assert all(not page.key_samples for page in layout_fn(chart))
    for page in result.generation.pages:
        assert "key-sample" not in page.clean_svg.read_text()
        assert "key-sample" not in page.review_svg.read_text()
    path = tmp_path / "chart.json"
    write_bar_chart(path, chart)
    assert load_bar_chart(path) == chart


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("failure", ["incomplete", "order", "collision", "unaccounted"])
def test_omit_legend_cannot_bypass_structural_blockers(horizontal, failure, tmp_path):
    spec = no_legend_spec(horizontal, failure)
    if failure in {"collision", "unaccounted"}:
        flag = "category_ambiguous" if failure == "collision" else "bar_colour_unaccounted:1"
        spec = replace(spec, flags=[*spec.flags, flag])
    model = build_review_model(spec, check_completeness(spec))
    rows = named_rows(model)
    draft = update_review_draft(model, rows, omit_legend=True)
    with pytest.raises(ManualGenerationBlocked):
        generate_review_submission(draft, rows, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("multiple", [False, True])
def test_checkbox_updates_review_revision_generates_and_resets(app, multiple):
    async def scenario():
        session = AppSession(app, "omit-legend")
        spec = no_legend_spec() if multiple else make_spec()
        result = replace(make_analysis_result(), spec=spec, completeness=check_completeness(spec))
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=result):
            response = await session.analyze()
        checkbox = next(i for i, component in enumerate(session.find_callback("analyze").outputs)
                        if getattr(component, "label", None) == "Include legend")
        assert response["data"][checkbox]["value"] is True
        assert response["data"][checkbox]["interactive"] is (not multiple)
        model = session.model
        rows = named_rows(model) if multiple else review_text_rows(model)
        await session.call("update_legend", [None, rows,
            [model.run_id, model.review_revision], False])
        assert session.model.omit_legend is (not multiple)
        assert session.model.review_revision == model.review_revision + 1
        admitted = await session.call("admit_generation", [None, rows,
            [session.model.run_id, session.model.review_revision], False])
        token = admitted["data"][3]["value"]
        await session.call("generate_tactile", [None, rows, token])
        assert session.model.spec.omit_legend is (not multiple)
        assert [s.values for s in session.model.spec.series] == [s.values for s in spec.series]
        # Changing the image resets the checkbox and its server-side draft.
        response = await session.call("ready_results")
        assert session.model is None
        checkbox = next(i for i, component in enumerate(session.find_callback("ready_results").outputs)
                        if getattr(component, "label", None) == "Include legend")
        assert response["data"][checkbox]["value"] is True
        assert response["data"][checkbox]["interactive"] is True
    asyncio.run(scenario())


@pytest.mark.parametrize("horizontal", [False, True])
def test_omission_keeps_key_and_note_for_series_without_bars(horizontal):
    spec = make_spec(
        omit_legend=True, orientation="horizontal" if horizontal else "vertical",
        series=[SeriesSpec("Alpha", "#112233", [50.0]),
                SeriesSpec("Beta", "#c81e1e", [None], no_visible_bars=True)],
    )
    chart = replace(spec_to_chart(spec), omit_legend=True)
    layout_fn = layout_horizontal_bar_chart_pages if horizontal else layout_vertical_bar_chart_pages
    pages = layout_fn(chart)
    assert sum(len(page.bars) for page in pages) == len(spec.categories)
    assert {s.series_index for page in pages for s in page.key_samples} == {0, 1}
    note = " ".join(label.text for page in pages for label in page.labels if label.role == "note")
    assert "No bars are visible for: Beta." in note
    assert all(not check_layout(page) for page in pages)


def test_omission_does_not_skip_series_braille_validation():
    spec = no_legend_spec()
    spec = replace(spec, omit_legend=True, series=[
        replace(spec.series[0], name="Alpha"), replace(spec.series[1], name="😀")])
    assessment = check_completeness(spec)
    assert assessment.status == "needs_input"
    assert any(m.field == "series_name[1]" and "braille" in m.reason
               for m in assessment.missing)
