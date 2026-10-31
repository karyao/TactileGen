"""Centralized Borealis design tokens and Gradio theme."""

import math
import subprocess
import sys
import unittest
from pathlib import Path

import src.presentation as presentation
from src.borealis import BOREALIS_COLORS, BOREALIS_CSS, borealis_theme, hex_to_bgr


def contrast(left: str, right: str) -> float:
    def luminance(value: str) -> float:
        channels = [int(value[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045
            else math.pow((channel + 0.055) / 1.055, 2.4)
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    brighter, darker = sorted((luminance(left), luminance(right)), reverse=True)
    return (brighter + 0.05) / (darker + 0.05)


class BorealisTest(unittest.TestCase):
    def test_exact_palette_is_immutable(self):
        self.assertEqual(dict(BOREALIS_COLORS), {
            "midnight": "#001322", "deep_sky": "#003866",
            "aurora": "#669ECC", "glow": "#CCDFFF",
            "text": "#F7FBFF", "muted": "#B8CCE0",
            "panel": "#06243B", "elevated": "#0A304B",
            "sun": "#F4C95D",
        })
        with self.assertRaises(TypeError):
            BOREALIS_COLORS["midnight"] = "#FFFFFF"

    def test_hex_to_bgr_validates_and_converts(self):
        self.assertEqual(hex_to_bgr("#CCDFFF"), (255, 223, 204))
        self.assertEqual(hex_to_bgr("#001322"), (34, 19, 0))
        for invalid in (None, "001322", "#123", "#GG0011", "#00132200"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "#RRGGBB"):
                    hex_to_bgr(invalid)

    def test_overlay_colors_resolve_from_shared_palette(self):
        self.assertEqual(presentation._SOFT_GLOW, hex_to_bgr(BOREALIS_COLORS["glow"]))
        self.assertEqual(presentation._AURORA_BLUE, hex_to_bgr(BOREALIS_COLORS["aurora"]))
        self.assertEqual(presentation._DEEP_SKY, hex_to_bgr(BOREALIS_COLORS["deep_sky"]))
        self.assertEqual(presentation._PANEL_NAVY, hex_to_bgr(BOREALIS_COLORS["panel"]))
        self.assertEqual(presentation._HALO, hex_to_bgr(BOREALIS_COLORS["text"]))

    def test_css_contains_tokens_layout_and_focus_contract(self):
        for name, value in BOREALIS_COLORS.items():
            self.assertIn(f"--borealis-{name.replace('_', '-')}: {value};", BOREALIS_CSS)
        self.assertIn("background: var(--borealis-midnight)", BOREALIS_CSS)
        self.assertIn("--borealis-content-width: 1200px", BOREALIS_CSS)
        self.assertIn("--borealis-space-1: 8px", BOREALIS_CSS)
        self.assertIn("--borealis-radius-sm: 10px", BOREALIS_CSS)
        self.assertIn("--borealis-radius-md: 12px", BOREALIS_CSS)
        self.assertIn("--borealis-shadow:", BOREALIS_CSS)
        self.assertIn("--borealis-focus-width: 3px", BOREALIS_CSS)
        self.assertIn(":focus-visible", BOREALIS_CSS)
        self.assertIn(
            "outline: var(--borealis-focus-width) solid var(--borealis-sun)",
            BOREALIS_CSS,
        )
        for class_name in (
            "app", "header", "actions", "scope", "control-rail", "upload-image",
            "summary", "workspace", "workspace-grid", "error", "empty",
        ):
            self.assertIn(f".borealis-{class_name}", BOREALIS_CSS)
        self.assertIn("@media (max-width: 900px)", BOREALIS_CSS)
        self.assertIn("@media (max-width: 600px)", BOREALIS_CSS)
        self.assertIn(".borealis-workspace-heading .borealis-workspace-heading", BOREALIS_CSS)
        self.assertIn(".borealis-workspace-heading,", BOREALIS_CSS)
        self.assertIn("margin-block: 12px !important", BOREALIS_CSS)
        self.assertIn("padding: 20px 16px !important", BOREALIS_CSS)
        self.assertIn("padding: 16px 12px !important", BOREALIS_CSS)

    def test_text_and_control_pairs_meet_wcag_aa(self):
        pairs = (
            ("text", "midnight"), ("text", "panel"), ("muted", "panel"),
            ("glow", "midnight"), ("text", "deep_sky"),
            ("aurora", "midnight"), ("aurora", "panel"),
            ("sun", "midnight"), ("midnight", "sun"),
        )
        for foreground, background in pairs:
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(
                    contrast(BOREALIS_COLORS[foreground], BOREALIS_COLORS[background]), 4.5
                )

    def test_theme_uses_borealis_component_values(self):
        import gradio as gr

        theme = borealis_theme()
        self.assertIsInstance(theme, gr.themes.Base)
        values = theme.to_dict()["theme"]
        expected = {
            "body_background_fill": "#001322",
            "body_text_color": "#F7FBFF",
            "body_text_color_subdued": "#B8CCE0",
            "block_background_fill": "#06243B",
            "input_background_fill": "#0A304B",
            "input_border_color_focus": "#F4C95D",
            "button_primary_background_fill": "#F4C95D",
            "button_primary_text_color": "#001322",
            "button_secondary_background_fill": "transparent",
            "button_secondary_border_color": "#669ECC",
            "table_even_background_fill": "#001322",
            "table_odd_background_fill": "#06243B",
            "error_background_fill": "#0A304B",
            "error_text_color": "#F7FBFF",
        }
        for name, value in expected.items():
            self.assertEqual(values[name], value)

    def test_import_is_gradio_independent_and_factory_error_is_actionable(self):
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == 'gradio':
        raise ImportError(f'blocked dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import src.borealis
import src.presentation
try:
    src.borealis.borealis_theme()
except RuntimeError as exc:
    assert 'requirements-app.txt' in str(exc)
else:
    raise AssertionError('theme factory should require Gradio')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
