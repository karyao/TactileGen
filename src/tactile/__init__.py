"""Tactile graphics: layout, BANA rule checks and SVG output for bar charts."""

from src.tactile.generation import (
    GeneratedPage,
    GenerationError,
    GenerationResult,
    cleanup_generation,
    generate_tactile_artifacts,
)
from src.tactile.layout import (
    TactileLayout,
    layout_vertical_bar_chart,
    layout_vertical_bar_chart_pages,
)
from src.tactile.rules import RULES, Violation, check_layout, format_report
from src.tactile.svg import render_svg

__all__ = [
    "GeneratedPage", "GenerationError", "GenerationResult", "RULES",
    "TactileLayout", "Violation", "check_layout", "cleanup_generation", "format_report",
    "generate_tactile_artifacts",
    "layout_vertical_bar_chart", "layout_vertical_bar_chart_pages", "render_svg",
]
