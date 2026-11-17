"""Borealis Gradio theme values and optional dependency behavior."""

from pathlib import Path
import subprocess
import sys
import unittest

from src.borealis import borealis_theme


class GradioThemeTest(unittest.TestCase):
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
            [sys.executable, "-c", script], cwd=Path(__file__).parents[2],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

