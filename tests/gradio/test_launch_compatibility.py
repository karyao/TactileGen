"""Launch options must work with both older and newer Gradio 6.x APIs."""
from unittest.mock import patch

import pytest

from src.gradio_app import main
from src.gradio_ui.dependencies import launch_app


@pytest.mark.parametrize('supports_monitoring', [False, True])
def test_main_launches_without_run_history(supports_monitoring):
    calls = []
    class LegacyBlocks:
        def launch(self, *, server_name, server_port, share, show_error,
                   max_file_size, theme, css):
            calls.append(locals())
    class MonitoringBlocks(LegacyBlocks):
        def launch(self, *, enable_monitoring=None, **kwargs):
            assert enable_monitoring is False
            return super().launch(**kwargs)
    # Give the wrapper the explicit signature of the supported launch API.
    import inspect
    MonitoringBlocks.launch.__signature__ = inspect.signature(LegacyBlocks.launch).replace(
        parameters=[*inspect.signature(LegacyBlocks.launch).parameters.values(),
                    inspect.Parameter('enable_monitoring', inspect.Parameter.KEYWORD_ONLY, default=None)])
    demo = MonitoringBlocks() if supports_monitoring else LegacyBlocks()
    with patch('src.gradio_app._import_gradio'), \
         patch('src.gradio_app.load_model', return_value=object()), \
         patch('src.gradio_app.build_app', return_value=demo), \
         patch('src.gradio_app.borealis_theme', return_value='theme'):
        main(['--host', '127.0.0.1', '--port', '9000'])
    assert len(calls) == 1
    assert calls[0]['share'] is False
    assert calls[0]['server_port'] == 9000
    assert calls[0]['theme'] == 'theme'
    assert calls[0]['css']


def test_launch_disables_supported_services_and_returns_result():
    class ModernBlocks:
        def launch(self, *, run_history=None, enable_monitoring=None):
            assert run_history is False
            assert enable_monitoring is False
            return 'launched'
    assert launch_app(ModernBlocks()) == 'launched'


def test_launch_does_not_swallow_unrelated_errors():
    class Blocks:
        def launch(self):
            raise TypeError('unrelated launch error')
    with pytest.raises(TypeError, match='unrelated launch error'):
        launch_app(Blocks())
