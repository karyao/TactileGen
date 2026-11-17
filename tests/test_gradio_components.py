"""Named updates must match the output order registered with Gradio."""

import subprocess
import sys
import unittest
from dataclasses import fields

from src.gradio_ui.components import AnalysisComponents, Controls, GenerationComponents


class ComponentOrderingTest(unittest.TestCase):
    def test_named_updates_follow_component_order_regardless_of_keyword_order(self):
        for group_type in (AnalysisComponents, GenerationComponents):
            with self.subTest(group=group_type.__name__):
                names = [field.name for field in fields(group_type)]
                components = {name: object() for name in names}
                group = group_type(**components)
                updates = {name: f"update:{name}" for name in reversed(names)}

                self.assertEqual(group.components(), list(components.values()))
                self.assertEqual(
                    group.updates(**updates),
                    tuple(f"update:{name}" for name in names),
                )

    def test_missing_or_unknown_outputs_are_rejected(self):
        values = {field.name: None for field in fields(AnalysisComponents)}
        group = AnalysisComponents(**values)
        missing = dict(values)
        del missing["diagnostics_info"]
        with self.assertRaisesRegex(ValueError, "diagnostics_info"):
            group.updates(**missing)
        with self.assertRaisesRegex(ValueError, "typo"):
            group.updates(**values, typo=None)

    def test_reset_controls_include_each_preset_in_component_order(self):
        values = {field.name: field.name for field in fields(Controls)}
        values["size_presets"] = ("small", "medium", "large")
        controls = Controls(**values)
        updates = {
            name: tuple(f"reset:{preset}" for preset in value)
            if name == "size_presets" else f"reset:{value}"
            for name, value in reversed(list(values.items()))
        }
        self.assertEqual(
            controls.updates(**updates),
            tuple(f"reset:{component}" for component in controls.components()),
        )
        with self.assertRaisesRegex(ValueError, "preset"):
            controls.updates(**{**updates, "size_presets": ()})

    def test_application_logic_does_not_import_optional_gradio_dependency(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.gradio_ui.analysis; "
                "import src.gradio_ui.review; assert 'gradio' not in sys.modules",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
