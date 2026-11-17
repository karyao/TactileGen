"""Named groups connect layout components to callback updates."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ComponentGroup:
    """Keep Gradio's positional boundary in one place; use names everywhere else."""

    def components(self) -> list[Any]:
        return [getattr(self, field.name) for field in fields(self)]

    def updates(self, **values: Any) -> tuple[Any, ...]:
        names = [field.name for field in fields(self)]
        if set(values) != set(names):
            missing = set(names) - set(values)
            unexpected = set(values) - set(names)
            raise ValueError(
                f"Invalid component updates: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return tuple(values[name] for name in names)


@dataclass(frozen=True)
class AnalysisComponents(ComponentGroup):
    """Outputs replaced together when an image is analyzed or cleared."""

    status_output: Any
    empty_output: Any
    summary_output: Any
    annotated_output: Any
    ocr_output: Any
    ocr_download: Any
    legend_output: Any
    completeness_output: Any
    review_state: Any
    review_panel: Any
    review_fields: Any
    review_values: Any
    review_error_output: Any
    generate_button: Any
    spec_output: Any
    spec_download: Any
    generation_gallery: Any
    generation_report_output: Any
    generation_download: Any
    generation_error_output: Any
    ocr_warning_output: Any
    legend_warning_output: Any
    reconstruction_warning_output: Any
    error_output: Any
    diagnostics_download: Any
    diagnostics_info: Any


@dataclass(frozen=True)
class GenerationComponents(ComponentGroup):
    """Outputs affected by the Generate tactile graphic button."""

    completeness_output: Any
    review_state: Any
    review_fields: Any
    review_values: Any
    review_error_output: Any
    spec_output: Any
    spec_download: Any
    generation_gallery: Any
    generation_report_output: Any
    generation_download: Any
    generation_error_output: Any


@dataclass(frozen=True)
class Controls:
    """Inputs and buttons disabled while analysis is running."""

    image_input: Any
    confidence: Any
    nms_iou: Any
    language: Any
    inference_size: Any
    ocr_engine: Any
    analyze_button: Any
    clear_button: Any
    size_presets: tuple[Any, ...]

    def components(self) -> list[Any]:
        return [
            self.image_input,
            self.confidence,
            self.nms_iou,
            self.language,
            self.inference_size,
            self.ocr_engine,
            self.analyze_button,
            self.clear_button,
            *self.size_presets,
        ]

    def updates(self, **values: Any) -> tuple[Any, ...]:
        """Use the same named control order for component binding and reset values."""
        updates = Controls(**values)
        if len(updates.size_presets) != len(self.size_presets):
            raise ValueError("Each preset button must have an update")
        return tuple(updates.components())
