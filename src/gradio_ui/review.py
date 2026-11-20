"""Keep review drafts separate from reconstructed values and generate approved artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.contracts import VerticalBarSpec, effective_omit_legend
from src.tactile.completeness import (
    READY,
    Completeness,
    Overrides,
    apply_overrides,
    check_completeness,
)
from src.tactile.generation import GenerationResult, generate_tactile_artifacts


@dataclass(frozen=True)
class ReviewFieldBinding:
    """One immutable mapping between a review row and a specification field."""

    key: str
    label: str
    initial_text: str
    required: bool
    confirmation_required: bool = False


@dataclass(frozen=True)
class ReviewModel:
    """Authoritative specification and the text draft shown for its review."""

    spec: VerticalBarSpec
    bindings: tuple[ReviewFieldBinding, ...]
    draft_values: tuple[str, ...]
    run_id: str | None = None
    review_revision: int = 0
    omit_legend: bool = False


@dataclass(frozen=True)
class ManualGenerationResult:
    """A corrected review draft committed by successful artifact generation."""

    review_model: ReviewModel
    completeness: Completeness
    generation: GenerationResult
    override_notes: tuple[str, ...]


class ManualGenerationBlocked(ValueError):
    """A valid review draft whose corrected specification is not ready."""

    def __init__(self, review_model: ReviewModel, completeness: Completeness) -> None:
        super().__init__("The corrected specification is not ready for generation.")
        self.review_model = review_model
        self.completeness = completeness


def build_review_model(spec: VerticalBarSpec, completeness: Completeness) -> ReviewModel:
    """Build stable review bindings without deriving keys from display text."""
    confirmation_fields = {item.field for item in completeness.missing}
    bindings = [
        ReviewFieldBinding(
            "title", "Chart title", spec.title or "", True,
            "title" in confirmation_fields,
        ),
        ReviewFieldBinding(
            "x_axis_title", "X-axis title", spec.x_axis_title or "", True,
            "x_axis_title" in confirmation_fields,
        ),
        ReviewFieldBinding(
            "y_axis_title", "Y-axis title", spec.y_axis_title or "", True,
            "y_axis_title" in confirmation_fields,
        ),
    ]
    bindings.extend(
        ReviewFieldBinding(
            f"category[{index}]", f"Category {index}", text, True,
            f"category[{index}]" in confirmation_fields,
        )
        for index, text in enumerate(spec.categories)
    )
    multiple_series = len(spec.series) > 1
    bindings.extend(
        ReviewFieldBinding(
            f"series_name[{index}]", f"Series {index} name", series.name or "",
            multiple_series or series.name is not None,
            f"series_name[{index}]" in confirmation_fields,
        )
        for index, series in enumerate(spec.series)
    )
    frozen = tuple(bindings)
    return ReviewModel(spec, frozen, tuple(item.initial_text for item in frozen),
                       omit_legend=effective_omit_legend(len(spec.series), spec.omit_legend))


def review_text_rows(model: ReviewModel) -> list[list[str]]:
    """Rows for the editable table; labels always come from server bindings."""
    return [[binding.label, value]
            for binding, value in zip(model.bindings, model.draft_values, strict=True)]


def review_value_rows(spec: VerticalBarSpec) -> list[list[Any]]:
    """Flatten immutable reconstructed values for a read-only review table."""
    rows = []
    for series_index, series in enumerate(spec.series):
        for category_index, (category, value) in enumerate(
            zip(spec.categories, series.values, strict=True)
        ):
            rows.append([
                category_index,
                category,
                series_index,
                series.name or "(unnamed)",
                value,
            ])
    return rows


def update_review_draft(model: ReviewModel, rows: Any, *,
                        omit_legend: bool | None = None) -> ReviewModel:
    """Accept only text cells and map them by immutable server-side row order."""
    if not isinstance(rows, (list, tuple)) or len(rows) != len(model.bindings):
        raise ValueError(f"review must contain exactly {len(model.bindings)} rows")
    values = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"review row {index} must contain exactly 2 columns")
        if not isinstance(row[1], str):
            raise ValueError(f"review text in row {index} must be a string")
        # row[0] is deliberately ignored: it is display text, never a field key.
        values.append(row[1])
    if omit_legend is not None and not isinstance(omit_legend, bool):
        raise ValueError("Omit legend must be a boolean")
    return replace(model, draft_values=tuple(values),
                   omit_legend=effective_omit_legend(
                       len(model.spec.series),
                       model.omit_legend if omit_legend is None else omit_legend))


def review_overrides(model: ReviewModel) -> Overrides:
    """Convert a valid draft to indexed overrides for the later generate action."""
    titles: dict[str, str] = {}
    categories: dict[int, str] = {}
    series_names: dict[int, str] = {}
    for binding, draft in zip(model.bindings, model.draft_values, strict=True):
        if (effective_omit_legend(len(model.spec.series), model.omit_legend)
                and binding.key.startswith("series_name[")):
            continue
        value = draft.strip()
        if binding.required and not value:
            raise ValueError(f"{binding.label} cannot be blank")
        initial = binding.initial_text.strip()
        if value == initial and not binding.confirmation_required:
            continue
        if not value:
            continue
        if binding.key in {"title", "x_axis_title", "y_axis_title"}:
            titles[binding.key] = value
            continue
        match = re.fullmatch(r"category\[(\d+)]", binding.key)
        if match:
            categories[int(match.group(1))] = value
            continue
        match = re.fullmatch(r"series_name\[(\d+)]", binding.key)
        if match:
            series_names[int(match.group(1))] = value
            continue
        raise ValueError(f"unknown review field: {binding.key}")
    return Overrides(
        title=titles.get("title"),
        x_axis_title=titles.get("x_axis_title"),
        y_axis_title=titles.get("y_axis_title"),
        categories=categories,
        series_names=series_names,
    )


def generate_review_submission(
    model: ReviewModel,
    rows: Any,
    artifact_root: Path | str,
) -> ManualGenerationResult:
    """Validate the latest review table, apply text, and generate ready artifacts."""
    draft = update_review_draft(model, rows)
    corrected, notes = apply_overrides(model.spec, review_overrides(draft))
    corrected = replace(corrected, omit_legend=draft.omit_legend)
    completeness = check_completeness(corrected)
    if completeness.status != READY:
        raise ManualGenerationBlocked(draft, completeness)
    committed_review = build_review_model(corrected, completeness)
    generation = generate_tactile_artifacts(corrected, artifact_root)
    return ManualGenerationResult(
        review_model=committed_review,
        completeness=completeness,
        generation=generation,
        override_notes=tuple(notes),
    )
