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
    "sun": "#F4C95D",
})

# Fixed ramps let Gradio derive component states without introducing colors
# unrelated to the approved anchors. They are implementation details rather
# than additional public brand tokens.
_GLOW_RAMP = (
    "#F7FBFF", "#EDF5FF", "#E0EBFF", "#CCDFFF", "#B8D1F5", "#9EC0E8",
    "#7FA6CC", "#6087AB", "#416687", "#294963", "#142C42",
)
_NAVY_RAMP = (
    "#F7FBFF", "#E6EFF7", "#B8CCE0", "#8EAAC3", "#6688A5", "#466B88",
    "#30536F", "#1D3D56", "#0A304B", "#06243B", "#001322",
)
_SUN_RAMP = (
    "#FFFCF2", "#FFF8DF", "#FFF1BF", "#FCE59A", "#F8D975", "#F4C95D",
    "#D8AA3E", "#B58928", "#8E6818", "#66480D", "#3C2A07",
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

    primary = gr.themes.Color(*_SUN_RAMP, name="borealis_sun")
    secondary = gr.themes.Color(*_GLOW_RAMP, name="borealis_glow")
    neutral = gr.themes.Color(*_NAVY_RAMP, name="borealis_navy")
    spacing = gr.themes.Size("2px", "4px", "8px", "16px", "24px", "32px", "48px")
    radius = gr.themes.Size("0px", "2px", "4px", "4px", "4px", "4px", "4px")
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
        block_radius="4px",
        block_shadow="none",
        block_shadow_dark="none",
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
        input_border_color_focus=colors["sun"],
        input_border_color_focus_dark=colors["sun"],
        input_placeholder_color=colors["muted"],
        input_placeholder_color_dark=colors["muted"],
        input_radius="4px",
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
        button_primary_background_fill=colors["sun"],
        button_primary_background_fill_dark=colors["sun"],
        button_primary_background_fill_hover="#FFDC7A",
        button_primary_background_fill_hover_dark="#FFDC7A",
        button_primary_border_color=colors["sun"],
        button_primary_border_color_dark=colors["sun"],
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


_BOREALIS_CSS_TEMPLATE = """
:root {
  --borealis-midnight: #001322;
  --borealis-deep-sky: #003866;
  --borealis-aurora: #669ECC;
  --borealis-glow: #CCDFFF;
  --borealis-text: #F7FBFF;
  --borealis-muted: #B8CCE0;
  --borealis-panel: #06243B;
  --borealis-elevated: #0A304B;
  --borealis-sun: #F4C95D;
  --borealis-space-1: 8px;
  --borealis-space-2: 16px;
  --borealis-space-3: 24px;
  --borealis-space-4: 32px;
  --borealis-radius-sm: 4px;
  --borealis-radius-md: 4px;
  --borealis-content-width: 1200px;
  --borealis-font: Inter, Arial, system-ui, sans-serif;
  --borealis-shadow: none;
  --borealis-focus-width: 3px;
  --borealis-focus-offset: 3px;
}

body,
.gradio-container {
  color: var(--borealis-text);
  font-family: var(--borealis-font);
  background: var(--borealis-midnight);
}

.gradio-container {
  max-width: var(--borealis-content-width) !important;
  margin-inline: auto !important;
}

.borealis-app,
.borealis-header,
.borealis-actions,
.borealis-status,
.borealis-scope,
.borealis-control-rail,
.borealis-upload-image,
.borealis-image-view,
.borealis-summary,
.borealis-workspace,
.borealis-review,
.borealis-error,
.borealis-empty {
  box-sizing: border-box;
}

.borealis-app {
  width: 100%;
  max-width: var(--borealis-content-width);
  margin-inline: auto;
  padding: 40px var(--borealis-space-3) 64px;
  gap: var(--borealis-space-4) !important;
}

.borealis-header {
  padding: 0 0 32px;
  border-bottom: 1px solid var(--borealis-deep-sky);
  color: var(--borealis-text);
  gap: 8px !important;
}

.borealis-header .borealis-header {
  padding-block: 0;
}

.borealis-header h1,
.borealis-header p {
  margin-block: 0 !important;
}

.borealis-header h1 {
  margin-block: 0 16px !important;
  font-size: 40px;
  line-height: 1.15;
  letter-spacing: -0.035em;
}

.borealis-header strong {
  color: var(--borealis-text);
  font-size: 1rem;
  letter-spacing: 0;
  text-transform: none;
}

.borealis-actions {
  display: flex;
  align-items: center;
  gap: var(--borealis-space-1);
}

.borealis-workspace {
  background: var(--borealis-panel);
  border: 1px solid var(--borealis-deep-sky);
  border-radius: var(--borealis-radius-md);
  padding: var(--borealis-space-3);
  box-shadow: var(--borealis-shadow);
}

.borealis-review {
  padding-top: var(--borealis-space-2);
  border-top: 1px solid var(--borealis-deep-sky);
  gap: var(--borealis-space-2) !important;
}

.borealis-control-rail {
  min-width: 0;
  padding-block: 2px;
  gap: var(--borealis-space-2) !important;
}

.borealis-section-title h2 {
  margin: 0 !important;
  font-size: 18px;
  line-height: 1.4;
  letter-spacing: -0.01em;
}

.borealis-upload-image {
  border: 1px dashed var(--borealis-aurora) !important;
  border-radius: var(--borealis-radius-md) !important;
  min-height: 230px;
  overflow: hidden;
}

.borealis-helper,
.borealis-scope {
  color: var(--borealis-muted);
  font-size: 0.875rem;
}

.borealis-scope {
  padding: 0;
  max-width: 850px;
}

.borealis-scope .borealis-scope {
  border: 0;
  padding: 0;
}

.borealis-scope p,
.borealis-helper p,
.borealis-summary p {
  margin-block: 0 !important;
}

.borealis-workspace {
  min-width: 0;
  min-height: 560px;
  align-content: start;
  gap: var(--borealis-space-2) !important;
}

.borealis-workspace-heading,
.borealis-workspace-heading .borealis-workspace-heading {
  padding: 0;
  border: 0;
}

.borealis-summary {
  color: var(--borealis-text);
  padding: 0;
  border: 0;
  background: transparent;
  box-shadow: none;
  line-height: 1.5;
  font-size: 14px;
}

.borealis-setting,
.borealis-advanced {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.borealis-setting {
  min-height: 0 !important;
  padding: 16px !important;
}

.borealis-setting .head {
  gap: 8px;
  margin-bottom: 8px;
}

.borealis-setting .head > label,
.borealis-setting > label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.borealis-setting input[type="text"],
.borealis-setting textarea {
  box-sizing: border-box;
  min-height: 44px;
  padding: 12px !important;
}

.borealis-advanced {
  padding: 0 !important;
  border-block: 1px solid var(--borealis-deep-sky) !important;
  border-radius: 0 !important;
}

.borealis-advanced > .label-wrap {
  box-sizing: border-box;
  min-height: 44px;
  padding: 16px !important;
  margin-bottom: 0 !important;
  align-items: center;
  gap: 8px;
}

/* Controls supply the inset; the accordion body must not add another one. */
.borealis-advanced > [data-testid="accordion-content"],
.borealis-advanced > [data-testid="accordion-content"] > .form,
.borealis-advanced > [data-testid="accordion-content"] > .column,
.borealis-advanced > [data-testid="accordion-content"] > .column > .form {
  padding: 0 !important;
  gap: 0 !important;
}

.borealis-result-image {
  border-color: var(--borealis-deep-sky) !important;
  box-shadow: none !important;
}

.borealis-image-view {
  width: fit-content !important;
  min-width: 240px;
  margin-top: calc(-1 * var(--borealis-space-1));
  padding: 4px !important;
  background: var(--borealis-elevated) !important;
  border: 1px solid var(--borealis-deep-sky) !important;
  border-radius: var(--borealis-radius-sm) !important;
}

.borealis-image-view .wrap {
  gap: 4px !important;
}

.borealis-image-view label {
  margin: 0 !important;
  padding: 7px 12px !important;
  border-radius: 4px !important;
  font-weight: 600;
}

.borealis-error {
  border-left: 3px solid var(--borealis-aurora);
  padding: 12px 14px;
  background: var(--borealis-elevated);
  border-radius: 0 var(--borealis-radius-sm) var(--borealis-radius-sm) 0;
}

.borealis-status {
  color: var(--borealis-muted);
  font-size: 0.875rem;
  font-weight: 600;
}

.borealis-empty {
  color: var(--borealis-muted);
  background: transparent;
  border: 0;
  padding: 0;
  box-shadow: none;
}

.borealis-app :focus-visible {
  outline: var(--borealis-focus-width) solid var(--borealis-sun);
  outline-offset: var(--borealis-focus-offset);
}

.borealis-workspace-grid {
  display: grid !important;
  grid-template-columns: minmax(320px, 360px) minmax(0, 1fr);
  align-items: start;
  gap: var(--borealis-space-4) !important;
}

/* Section rules and spacing establish hierarchy without nested cards. */
.borealis-app {
  font-size: 16px;
  line-height: 1.5;
}

.borealis-header p {
  max-width: 680px;
  color: var(--borealis-muted);
}

.borealis-header p + p {
  margin-top: 8px !important;
}

.borealis-workspace-heading {
  padding-bottom: 16px !important;
  border-bottom: 1px solid var(--borealis-deep-sky) !important;
}

.borealis-workspace-heading .borealis-workspace-heading {
  padding: 0 !important;
  border: 0 !important;
}

.borealis-empty {
  padding: 48px 0;
  max-width: 420px;
}

.borealis-empty h3 {
  color: var(--borealis-text);
  font-size: 18px;
  margin: 0 0 8px;
}

.borealis-app button {
  min-height: 44px;
  border-radius: 4px !important;
  box-shadow: none !important;
}

.borealis-details {
  border: 0 !important;
  border-top: 1px solid var(--borealis-deep-sky) !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.borealis-details > .label-wrap {
  min-height: 44px;
  align-items: center;
  gap: 8px;
}

/* Centre the disclosure glyph's line box alongside the title in either state. */
.borealis-details > .label-wrap > .icon {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 16px;
  width: 16px;
  height: 16px;
  line-height: 1;
}

.borealis-details .borealis-summary {
  padding-bottom: 16px;
}

.borealis-workspace > *,
.borealis-review > * {
  min-width: 0;
  max-width: 100%;
}

.borealis-app .prose a {
  color: var(--borealis-glow);
}

@media (max-width: 900px) {
  .borealis-workspace-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .borealis-workspace {
    min-height: 420px;
  }
}

@media (max-width: 600px) {
  .borealis-app {
    padding: var(--borealis-space-2);
  }

  .borealis-header { padding-top: 8px; padding-bottom: 24px; }
  .borealis-header h1 { font-size: 32px; }

  .borealis-workspace {
    min-height: 360px;
    padding: var(--borealis-space-2);
  }

  .borealis-setting,
  .borealis-advanced > .label-wrap {
    padding: 16px 12px !important;
  }

  .borealis-actions {
    flex-direction: column;
  }

  .borealis-actions > * {
    width: 100%;
  }
}
""".strip()


BOREALIS_CSS = _BOREALIS_CSS_TEMPLATE
