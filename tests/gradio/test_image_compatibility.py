"""Image accessibility metadata must not prevent startup on older Gradio 6.x."""
from types import SimpleNamespace
from unittest.mock import patch
import inspect

import gradio as gr
import pytest

from src.gradio_ui.dependencies import image_component
from src.gradio_ui.layout import build_app


def test_image_preserves_alt_text_when_supported():
    class Image:
        def __init__(self, *, label, alt_text=None):
            self.label, self.alt_text = label, alt_text
    component = image_component(SimpleNamespace(Image=Image), label='Chart', alt_text='Chart preview')
    assert component.label == 'Chart'
    assert component.alt_text == 'Chart preview'


def test_image_omits_only_unsupported_alt_text():
    class Image:
        def __init__(self, *, label):
            self.label = label
    component = image_component(SimpleNamespace(Image=Image), label='Chart', alt_text='Chart preview')
    assert component.label == 'Chart'
    with pytest.raises(TypeError):
        image_component(SimpleNamespace(Image=Image), label='Chart', alt_text='Chart preview', bogus=True)


def test_app_builds_with_image_signature_without_alt_text():
    original = gr.Image.__init__
    signature = inspect.signature(original)
    def legacy_init(self, *args, **kwargs):
        assert 'alt_text' not in kwargs
        return original(self, *args, **kwargs)
    legacy_init.__signature__ = signature.replace(parameters=[
        p for name, p in signature.parameters.items() if name != 'alt_text'])
    with patch.object(gr.Image, '__init__', legacy_init):
        demo = build_app(object(), ocr_backend=object())
    images = [block for block in demo.blocks.values() if isinstance(block, gr.Image)]
    assert {image.label for image in images} >= {'Bar chart image', 'Annotated detections'}
    demo.close()
