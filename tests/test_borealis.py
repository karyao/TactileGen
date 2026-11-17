"""Centralized Borealis design tokens and Gradio theme."""

import math
import unittest

import src.presentation as presentation
from src.borealis import BOREALIS_COLORS, BOREALIS_CSS, hex_to_bgr


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
        self.assertIn("--borealis-radius-sm: 4px", BOREALIS_CSS)
        self.assertIn("--borealis-radius-md: 4px", BOREALIS_CSS)
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
        self.assertIn("margin-block: 0 16px !important", BOREALIS_CSS)
        self.assertIn("padding: 16px !important", BOREALIS_CSS)
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


if __name__ == "__main__":
    unittest.main()
