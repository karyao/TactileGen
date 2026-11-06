"""Tactile graphics: layout, BANA rule checks and SVG output for bar charts."""

from src.tactile.layout import TactileLayout, layout_vertical_bar_chart
from src.tactile.rules import RULES, Violation, check_layout, format_report
from src.tactile.svg import render_svg

__all__ = [
    "RULES", "TactileLayout", "Violation", "check_layout", "format_report",
    "layout_vertical_bar_chart", "render_svg",
]
