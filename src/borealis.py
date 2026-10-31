"""Shared Borealis design tokens for the local application and overlays."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType


BOREALIS_COLORS: Mapping[str, str] = MappingProxyType({
    "midnight": "#001322",
    "deep_sky": "#003866",
    "aurora": "#669ECC",
    "glow": "#CCDFFF",
    "text": "#F7FBFF",
    "muted": "#B8CCE0",
    "panel": "#06243B",
    "elevated": "#0A304B",
})

# Fixed ramps let Gradio derive component states without introducing colors
# unrelated to the approved anchors. They are implementation details rather
# than additional public brand tokens.
_AURORA_RAMP = (
    "#F4F8FC", "#E6F0F8", "#CCDFFF", "#A8C7E3", "#82ACD2", "#669ECC",
    "#4F84B0", "#3B688C", "#294C68", "#173248", "#0A1E2F",
)
_GLOW_RAMP = (
    "#F7FBFF", "#EDF5FF", "#E0EBFF", "#CCDFFF", "#B8D1F5", "#9EC0E8",
    "#7FA6CC", "#6087AB", "#416687", "#294963", "#142C42",
)
_NAVY_RAMP = (
    "#F7FBFF", "#E6EFF7", "#B8CCE0", "#8EAAC3", "#6688A5", "#466B88",
    "#30536F", "#1D3D56", "#0A304B", "#06243B", "#001322",
)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """Convert a CSS ``#RRGGBB`` color to an OpenCV BGR tuple."""
    if not isinstance(value, str) or re.fullmatch(r"#[0-9A-Fa-f]{6}", value) is None:
        raise ValueError("color must use #RRGGBB format")
    red, green, blue = (int(value[index:index + 2], 16) for index in (1, 3, 5))
    return blue, green, red


def borealis_theme():
    """Build the Gradio theme while keeping Gradio an optional app dependency."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install the application dependencies: "
            "python -m pip install -r requirements-app.txt"
        ) from exc

    primary = gr.themes.Color(*_AURORA_RAMP, name="borealis_aurora")
    secondary = gr.themes.Color(*_GLOW_RAMP, name="borealis_glow")
    neutral = gr.themes.Color(*_NAVY_RAMP, name="borealis_navy")
    spacing = gr.themes.Size("2px", "4px", "8px", "16px", "24px", "32px", "48px")
    radius = gr.themes.Size("2px", "4px", "8px", "10px", "12px", "16px", "20px")
    colors = BOREALIS_COLORS
    return gr.themes.Base(
        primary_hue=primary,
        secondary_hue=secondary,
        neutral_hue=neutral,
        spacing_size=spacing,
        radius_size=radius,
        font=("Inter", "Arial", "system-ui", "sans-serif"),
    ).set(
        body_background_fill=colors["midnight"],
        body_background_fill_dark=colors["midnight"],
        body_text_color=colors["text"],
        body_text_color_dark=colors["text"],
        body_text_color_subdued=colors["muted"],
        body_text_color_subdued_dark=colors["muted"],
        background_fill_primary=colors["midnight"],
        background_fill_primary_dark=colors["midnight"],
        background_fill_secondary=colors["panel"],
        background_fill_secondary_dark=colors["panel"],
        block_background_fill=colors["panel"],
        block_background_fill_dark=colors["panel"],
        block_border_color=colors["deep_sky"],
        block_border_color_dark=colors["deep_sky"],
        block_radius="12px",
        block_shadow="0 6px 18px rgba(0, 19, 34, 0.18)",
        block_shadow_dark="0 6px 18px rgba(0, 19, 34, 0.18)",
        panel_background_fill=colors["panel"],
        panel_background_fill_dark=colors["panel"],
        panel_border_color=colors["deep_sky"],
        panel_border_color_dark=colors["deep_sky"],
        input_background_fill=colors["elevated"],
        input_background_fill_dark=colors["elevated"],
        input_background_fill_focus=colors["elevated"],
        input_background_fill_focus_dark=colors["elevated"],
        input_border_color=colors["aurora"],
        input_border_color_dark=colors["aurora"],
        input_border_color_focus=colors["glow"],
        input_border_color_focus_dark=colors["glow"],
        input_placeholder_color=colors["muted"],
        input_placeholder_color_dark=colors["muted"],
        input_radius="10px",
        accordion_text_color=colors["text"],
        accordion_text_color_dark=colors["text"],
        slider_color=colors["aurora"],
        slider_color_dark=colors["aurora"],
        table_text_color=colors["text"],
        table_text_color_dark=colors["text"],
        table_border_color=colors["deep_sky"],
        table_border_color_dark=colors["deep_sky"],
        table_even_background_fill=colors["midnight"],
        table_even_background_fill_dark=colors["midnight"],
        table_odd_background_fill=colors["panel"],
        table_odd_background_fill_dark=colors["panel"],
        table_row_focus=colors["elevated"],
        table_row_focus_dark=colors["elevated"],
        button_primary_background_fill=colors["glow"],
        button_primary_background_fill_dark=colors["glow"],
        button_primary_background_fill_hover=colors["text"],
        button_primary_background_fill_hover_dark=colors["text"],
        button_primary_border_color=colors["glow"],
        button_primary_border_color_dark=colors["glow"],
        button_primary_text_color=colors["midnight"],
        button_primary_text_color_dark=colors["midnight"],
        button_primary_text_color_hover=colors["midnight"],
        button_primary_text_color_hover_dark=colors["midnight"],
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover=colors["elevated"],
        button_secondary_background_fill_hover_dark=colors["elevated"],
        button_secondary_border_color=colors["aurora"],
        button_secondary_border_color_dark=colors["aurora"],
        button_secondary_text_color=colors["text"],
        button_secondary_text_color_dark=colors["text"],
        button_secondary_text_color_hover=colors["text"],
        button_secondary_text_color_hover_dark=colors["text"],
        error_background_fill=colors["elevated"],
        error_background_fill_dark=colors["elevated"],
        error_border_color=colors["aurora"],
        error_border_color_dark=colors["aurora"],
        error_text_color=colors["text"],
        error_text_color_dark=colors["text"],
        error_icon_color=colors["glow"],
        error_icon_color_dark=colors["glow"],
        loader_color=colors["glow"],
        loader_color_dark=colors["glow"],
    )


BOREALIS_CSS = """
:root {
  --borealis-midnight: #001322;
  --borealis-deep-sky: #003866;
  --borealis-aurora: #669ECC;
  --borealis-glow: #CCDFFF;
  --borealis-text: #F7FBFF;
  --borealis-muted: #B8CCE0;
  --borealis-panel: #06243B;
  --borealis-elevated: #0A304B;
  --borealis-space-1: 8px;
  --borealis-space-2: 16px;
  --borealis-space-3: 24px;
  --borealis-space-4: 32px;
  --borealis-radius-sm: 10px;
  --borealis-radius-md: 12px;
  --borealis-content-width: 1200px;
  --borealis-font: Inter, Arial, system-ui, sans-serif;
  --borealis-shadow: 0 6px 18px rgba(0, 19, 34, 0.18);
  --borealis-focus-width: 3px;
  --borealis-focus-offset: 3px;
}

body,
.gradio-container {
  color: var(--borealis-text);
  font-family: var(--borealis-font);
  background: linear-gradient(135deg, #001322 0%, #002744 55%, #003866 100%);
}

.gradio-container {
  max-width: var(--borealis-content-width) !important;
  margin-inline: auto !important;
}

.borealis-app,
.borealis-header,
.borealis-nav,
.borealis-status,
.borealis-panel,
.borealis-scope,
.borealis-upload,
.borealis-summary,
.borealis-results,
.borealis-table,
.borealis-downloads,
.borealis-error,
.borealis-empty {
  box-sizing: border-box;
}

.borealis-app {
  width: 100%;
  max-width: var(--borealis-content-width);
  margin-inline: auto;
  padding: var(--borealis-space-4) var(--borealis-space-3);
}

.borealis-header {
  padding-block: var(--borealis-space-3);
  color: var(--borealis-text);
  gap: var(--borealis-space-1) !important;
}

.borealis-header .borealis-header {
  padding-block: 0;
}

.borealis-header h1,
.borealis-header p {
  margin-block: 0 !important;
}

.borealis-nav,
.borealis-downloads {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--borealis-space-1);
}

.borealis-panel,
.borealis-upload,
.borealis-summary,
.borealis-results,
.borealis-table,
.borealis-downloads,
.borealis-error {
  background: var(--borealis-panel);
  border: 1px solid var(--borealis-deep-sky);
  border-radius: var(--borealis-radius-md);
  padding: var(--borealis-space-3);
  box-shadow: var(--borealis-shadow);
}

.borealis-upload {
  border-color: var(--borealis-aurora);
  border-style: dashed;
}

.borealis-summary {
  background: var(--borealis-elevated);
}

.borealis-table {
  overflow-x: auto;
}

.borealis-scope,
.borealis-error {
  background: var(--borealis-panel);
  border-left: 4px solid var(--borealis-aurora);
  border-radius: var(--borealis-radius-sm);
  padding: var(--borealis-space-2);
}

.borealis-scope .borealis-scope,
.borealis-error .borealis-error {
  background: transparent;
  border: 0;
  padding: 0;
}

.borealis-status {
  color: var(--borealis-muted);
}

.borealis-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--borealis-aurora), var(--borealis-glow));
}

.borealis-empty {
  color: var(--borealis-muted);
  background: transparent;
  border: 0;
  padding: 0;
  box-shadow: none;
}

.borealis-app :focus-visible {
  outline: var(--borealis-focus-width) solid var(--borealis-glow);
  outline-offset: var(--borealis-focus-offset);
}

.borealis-two-column {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--borealis-space-3);
}

@media (max-width: 768px) {
  .borealis-two-column {
    grid-template-columns: minmax(0, 1fr);
  }

  .borealis-app {
    padding: var(--borealis-space-2);
  }
}
""".strip()
