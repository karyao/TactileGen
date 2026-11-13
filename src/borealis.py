"""Shared Borealis design tokens for the local application and overlays."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import quote


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
        input_border_color_focus=colors["sun"],
        input_border_color_focus_dark=colors["sun"],
        input_placeholder_color=colors["muted"],
        input_placeholder_color_dark=colors["muted"],
        input_radius="10px",
        accordion_text_color=colors["text"],
        accordion_text_color_dark=colors["text"],
        slider_color=colors["sun"],
        slider_color_dark=colors["sun"],
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
        loader_color=colors["sun"],
        loader_color_dark=colors["sun"],
    )


def _background_art_svg() -> str:
    """Build the decorative Braille-inspired chart background artwork."""
    marks: list[str] = []

    def add_dot(x: float, y: float, radius: float) -> None:
        marks.append(f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}"/>')

    def add_bar_graph(x: float, bottom: float, heights: tuple[int, ...]) -> None:
        step, pitch, pair_gap, radius = 11.0, 26.0, 8.0, 3.6
        for index, cells in enumerate(heights):
            left = x + index * pitch
            for row in range(cells):
                y = bottom - (row + 0.5) * step
                add_dot(left, y, radius)
                add_dot(left + pair_gap, y, radius)
        width = (len(heights) - 1) * pitch + pair_gap + 12
        marks.append(
            f'<rect x="{x - 6:g}" y="{bottom + 5:g}" '
            f'width="{width:g}" height="1.6"/>'
        )

    def add_line_graph(points: list[tuple[float, float]]) -> None:
        spacing = 16.0
        for index, (x, y) in enumerate(points):
            add_dot(x - 4, y, 4.4)
            add_dot(x + 4, y, 4.4)
            if index + 1 == len(points):
                continue
            next_x, next_y = points[index + 1]
            length = math.hypot(next_x - x, next_y - y)
            steps = int(length // spacing)
            for step in range(1, steps):
                ratio = step * spacing / length
                add_dot(
                    x + (next_x - x) * ratio,
                    y + (next_y - y) * ratio,
                    3.6,
                )

    add_bar_graph(80, 210, (4, 7, 5, 8, 6))
    add_line_graph([
        (1040, 230), (1105, 170), (1170, 200),
        (1235, 150), (1300, 185), (1360, 130),
    ])
    add_bar_graph(70, 650, (5, 8, 4, 6))

    stops = (
        '<stop offset="0" stop-color="#000000" stop-opacity="1"/>'
        '<stop offset="1" stop-color="#000000" stop-opacity="0"/>'
    )
    gradients = "".join(
        f'<linearGradient id="{identifier}" x1="{x1:g}" y1="{y1:g}" '
        f'x2="{x2:g}" y2="{y2:g}" gradientUnits="userSpaceOnUse">'
        f"{stops}"
        "</linearGradient>"
        for identifier, x1, y1, x2, y2 in (
            ("fade-left", 0, 0, 120, 0),
            ("fade-right", 1440, 0, 1320, 0),
            ("fade-top", 0, 0, 0, 110),
            ("fade-bottom", 0, 900, 0, 780),
        )
    )
    fades = "".join(
        f'<rect width="1440" height="900" fill="url(#{identifier})"/>'
        for identifier in ("fade-left", "fade-right", "fade-top", "fade-bottom")
    )
    art = "".join(marks)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" '
        'viewBox="0 0 1440 900">'
        "<defs>"
        f"{gradients}"
        '<mask id="edge-fade" maskUnits="userSpaceOnUse" '
        'x="0" y="0" width="1440" height="900">'
        '<rect width="1440" height="900" fill="#ffffff"/>'
        f"{fades}"
        "</mask>"
        "</defs>"
        f'<g mask="url(#edge-fade)" fill="{BOREALIS_COLORS["glow"]}" '
        'opacity="0.07">'
        f"{art}"
        "</g>"
        "</svg>"
    )


def _background_art_css() -> str:
    """Return the CSS rules that place the decorative background artwork."""
    uri = "data:image/svg+xml," + quote(_background_art_svg())
    return (
        "body,\n"
        ".gradio-container {\n"
        f'  background-image: url("{uri}");\n'
        "  background-repeat: no-repeat;\n"
        "  background-size: cover;\n"
        "  background-position: 15% 20%;\n"
        "  background-attachment: fixed;\n"
        "}\n"
        "\n"
        "@media (max-width: 600px) {\n"
        "  body,\n"
        "  .gradio-container {\n"
        "    background-size: 300vw auto;\n"
        "    background-position: left top;\n"
        "  }\n"
        "}\n"
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
  --borealis-radius-sm: 10px;
  --borealis-radius-md: 12px;
  --borealis-content-width: 1200px;
  --borealis-font: Inter, Arial, system-ui, sans-serif;
  --borealis-shadow: 0 10px 30px rgba(0, 12, 22, 0.22);
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
  padding: var(--borealis-space-2) var(--borealis-space-3) 40px;
  gap: var(--borealis-space-2) !important;
}

.borealis-header {
  padding: 0;
  color: var(--borealis-text);
  gap: 6px !important;
}

.borealis-header .borealis-header {
  padding-block: 0;
}

.borealis-header h1,
.borealis-header p {
  margin-block: 0 !important;
}

.borealis-header h1 {
  margin-block: 12px !important;
  font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1;
  letter-spacing: -0.04em;
}

.borealis-header strong {
  color: var(--borealis-sun);
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
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
  font-size: 1rem;
  letter-spacing: 0.02em;
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
  border-left: 2px solid var(--borealis-sun);
  padding-left: 12px;
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
  font-size: 0.8125rem;
}

.borealis-setting,
.borealis-advanced {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.borealis-setting {
  min-height: 0 !important;
  padding: 20px 16px !important;
}

.borealis-setting > div,
.borealis-setting .wrap {
  padding-block: 0 !important;
}

.borealis-advanced {
  border-block: 1px solid var(--borealis-deep-sky) !important;
  border-radius: 0 !important;
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
  border-radius: 7px !important;
  font-weight: 600;
}

.borealis-error {
  border-left: 3px solid var(--borealis-sun);
  padding: 12px 14px;
  background: var(--borealis-elevated);
  border-radius: 0 var(--borealis-radius-sm) var(--borealis-radius-sm) 0;
}

.borealis-status {
  color: var(--borealis-sun);
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

  .borealis-header {
    padding-top: var(--borealis-space-2);
  }

  .borealis-workspace {
    min-height: 360px;
    padding: var(--borealis-space-2);
  }

  .borealis-setting {
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


BOREALIS_CSS = _BOREALIS_CSS_TEMPLATE + "\n\n" + _background_art_css()
