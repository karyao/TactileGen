"""Shared app lifecycle for Gradio tests."""
import pytest

from src.gradio_app import build_app


@pytest.fixture
def app():
    demo = build_app(object())
    try:
        yield demo
    finally:
        demo.close(verbose=False)
