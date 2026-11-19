"""Offline Gemma inference tests, with the processor/model boundary mocked."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.baseline.gemma import (
    DEFAULT_MODEL_ID, EXTRACTION_PROMPT, PROMPT_VERSION,
    GemmaChartExtractor, GemmaExtractionError, _resolve_runtime, _save_artifacts,
)
from src.contracts import HorizontalBarChart, VerticalBarChart


FIXTURES = Path(__file__).parent / "fixtures" / "vlm"


class Tensor:
    """Small tensor double that records dtype/device placement and slicing."""

    def __init__(self, values, floating=False, shape=None):
        self.values = values
        self.floating = floating
        self.shape = shape or (len(values),)
        self.to = Mock(return_value=self)

    def is_floating_point(self):
        return self.floating

    def __getitem__(self, key):
        return Tensor(self.values[key], self.floating)

    def __len__(self):
        return len(self.values)


def runtime(cuda=False, mps=False, bf16=False):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=cuda),
                             is_bf16_supported=Mock(return_value=bf16), synchronize=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps))),
        mps=SimpleNamespace(synchronize=Mock()),
        float32="float32", bfloat16="bfloat16", float16="float16", in_inference=False,
    )

    @contextmanager
    def inference_mode():
        torch.in_inference = True
        try:
            yield
        finally:
            torch.in_inference = False

    torch.inference_mode = inference_mode
    return torch


class GemmaConfigTest(unittest.TestCase):
    def test_invalid_configuration(self):
        for kwargs in (
            {"model_id": ""}, {"model_id": None}, {"revision": " "}, {"revision": 1},
            {"device": "gpu"}, {"device": []}, {"dtype": "int4"}, {"dtype": []},
            {"max_new_tokens": True}, {"max_new_tokens": 0}, {"max_new_tokens": -1},
            {"max_new_tokens": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GemmaChartExtractor(**kwargs)

    def test_import_and_construction_without_optional_packages(self):
        script = """
import sys
from src.baseline.gemma import GemmaChartExtractor
extractor = GemmaChartExtractor()
assert not any(name in sys.modules for name in ('torch', 'transformers', 'PIL', 'src.detection', 'src.ocr', 'src.reconstruction', 'src.tactile'))
"""
        subprocess.run([sys.executable, "-S", "-c", script], check=True, capture_output=True,
                       text=True, cwd=Path(__file__).resolve().parents[1])

    def test_device_priority_and_dtype(self):
        for cuda, mps, bf16, expected in (
            (True, True, True, ("cuda", "bfloat16")),
            (True, True, False, ("cuda", "float32")),
            (False, True, False, ("mps", "float32")),
            (False, False, False, ("cpu", "float32")),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(_resolve_runtime(runtime(cuda, mps, bf16), "auto", "auto"), expected)
        for device, dtype in (("cuda", "float16"), ("cuda", "float32"),
                              ("cuda", "bfloat16"), ("mps", "float16"), ("cpu", "float32")):
            self.assertEqual(_resolve_runtime(runtime(True, True, True), device, dtype), (device, dtype))

    def test_unavailable_devices_and_unsupported_dtypes(self):
        for device, dtype in (("cuda", "auto"), ("mps", "auto"),
                              ("cpu", "float16"), ("cpu", "bfloat16")):
            with self.subTest(device=device, dtype=dtype), self.assertRaises(ValueError):
                _resolve_runtime(runtime(), device, dtype)
        for device in ("cuda", "mps"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                _resolve_runtime(runtime(True, True, False), device, "bfloat16")


class GemmaInferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "chart.png"
        with Image.new("RGBA", (8, 6), (100, 50, 0, 128)) as image:
            image.save(self.image)
        self.torch = runtime()
        self.raw = (FIXTURES / "vertical_response.json").read_text()
        self.ids = Tensor([1, 2, 3], shape=(1, 3))
        self.mask = Tensor([1, 1, 1], shape=(1, 3))
        self.pixels = Tensor([0.1, 0.2], floating=True)
        self.processor = Mock()
        self.processor.apply_chat_template.side_effect = self.prepare
        self.processor.decode.side_effect = lambda *args, **kwargs: self.raw
        self.model = Mock(config=SimpleNamespace(_commit_hash="resolved-commit"))
        self.model.generate.side_effect = self.generate
        self.processor_class = Mock()
        self.processor_class.from_pretrained.return_value = self.processor
        self.model_class = Mock()
        self.model_class.from_pretrained.return_value = self.model
        self.dependencies = patch("src.baseline.gemma._load_dependencies", return_value=(
            self.torch, Image, self.processor_class, self.model_class)).start()
        self.addCleanup(patch.stopall)
        self.extractor = GemmaChartExtractor(revision="test-revision", max_new_tokens=2)

    def prepare(self, messages, **kwargs):
        self.rgb = messages[0]["content"][0]["image"]
        self.assertEqual(self.rgb.mode, "RGB")
        self.assertEqual(self.rgb.size, (8, 6))
        self.assertEqual(messages[0]["content"][1]["text"], EXTRACTION_PROMPT)
        return {"input_ids": self.ids, "attention_mask": self.mask, "pixel_values": self.pixels}

    def generate(self, **kwargs):
        self.assertTrue(self.torch.in_inference)
        self.model.eval.assert_called()
        return [Tensor([1, 2, 3, 9, 10])]

    def test_full_extraction_and_diagnostics(self):
        result = self.extractor.extract_result(self.image, source_key="eval/custom.png")
        self.assertIsInstance(result.chart, VerticalBarChart)
        self.assertEqual(result.chart.source_key, "eval/custom.png")
        self.assertEqual(result.chart.series[0].values, (4, 6))
        self.assertEqual(result.raw_response, self.raw)
        self.processor_class.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID, revision="test-revision", use_fast=False)
        self.model_class.from_pretrained.assert_called_once_with(
            DEFAULT_MODEL_ID, revision="test-revision", torch_dtype="float32")
        self.model.to.assert_called_once_with("cpu")
        self.processor.apply_chat_template.assert_called_once()
        self.assertEqual(self.processor.apply_chat_template.call_args.kwargs, {
            "tokenize": True, "return_dict": True, "return_tensors": "pt", "add_generation_prompt": True})
        self.assertEqual(self.processor.decode.call_args.args[0].values, [9, 10])
        self.assertEqual(self.processor.decode.call_args.kwargs,
                         {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})
        kwargs = self.model.generate.call_args.kwargs
        for key, value in result.metadata["generation_settings"].items():
            self.assertEqual(kwargs[key], value)
        self.ids.to.assert_called_once_with(device="cpu")
        self.mask.to.assert_called_once_with(device="cpu")
        self.pixels.to.assert_called_once_with(device="cpu", dtype="float32")
        self.assertEqual(result.metadata["resolved_revision"], "resolved-commit")
        self.assertEqual(result.metadata["requested_revision"], "test-revision")
        self.assertEqual(result.metadata["prompt_version"], PROMPT_VERSION)
        self.assertEqual(result.metadata["generated_token_count"], 2)
        self.assertTrue(result.metadata["output_budget_reached"])
        self.assertGreaterEqual(result.metadata["inference_seconds"], 0)
        self.assertEqual(result.metadata["completion_status"], "complete")
        json.dumps(result.metadata, allow_nan=False)
        with self.assertRaises(ValueError):
            self.rgb.getpixel((0, 0))  # closed even after successful inference

    def test_saved_end_of_turn_response_and_diagnostics(self):
        fixture = FIXTURES / "gemma_end_of_turn_response.txt"
        self.raw = fixture.read_bytes().decode("utf-8")
        result = self.extractor.extract_result(self.image)
        chart = result.chart
        self.assertIsInstance(chart, VerticalBarChart)
        self.assertEqual(chart.title, "AB")
        self.assertEqual(chart.x_axis_title, "X")
        self.assertEqual(chart.y_axis_title, "Y")
        self.assertEqual(chart.categories, ("A", "B"))
        self.assertEqual(len(chart.series), 1)
        self.assertEqual(chart.series[0].name, "N")
        self.assertEqual(chart.series[0].values, (3.5, 6))
        self.assertTrue(chart.series[0].has_bars)
        self.assertEqual(result.raw_response, self.raw)
        self.assertTrue(_save_artifacts(self.image.parent, result.metadata,
                                        result.raw_response, result.chart))
        self.assertEqual((self.image.parent / "raw_response.txt").read_bytes(), fixture.read_bytes())

    def test_terminal_turn_marker_and_whitespace(self):
        valid = self.raw
        for body in (valid, "```json\n" + valid, "```json\n" + valid + "\n```"):
            for suffix in ("", "<end_of_turn>", "<end_of_turn> \t\r\n"):
                with self.subTest(body=body, suffix=suffix):
                    self.raw = body + suffix
                    result = self.extractor.extract_result(self.image)
                    self.assertEqual(result.chart.series[0].values, (4, 6))
                    self.assertEqual(result.raw_response, self.raw)

    def test_turn_marker_inside_json_string_is_preserved(self):
        data = json.loads(self.raw)
        data["title"] = "Literal <end_of_turn>"
        self.raw = json.dumps(data) + "<end_of_turn>"
        self.assertEqual(self.extractor.extract(self.image).title, data["title"])

    def test_turn_marker_does_not_salvage_invalid_response(self):
        valid = self.raw
        for response in (valid + "<end_of_turn><end_of_turn>",
                         valid + "<end_of_turn> Done.", valid[:-3] + "<end_of_turn>",
                         json.dumps({"chart_type": "line"}) + "<end_of_turn>"):
            with self.subTest(response=response):
                self.raw = response
                error = self.assert_safe_failure("parsing")
                self.assertEqual(error.raw_response, response)
                self.assertTrue(_save_artifacts(self.image.parent, error.metadata, error.raw_response))
                self.assertEqual((self.image.parent / "raw_response.txt").read_bytes(), response.encode())

    def test_reuse_and_both_orientations(self):
        first = self.extractor.extract_result(self.image)
        self.raw = (FIXTURES / "horizontal_response.json").read_text()
        second = self.extractor.extract_result(self.image)
        self.assertIsInstance(second.chart, HorizontalBarChart)
        self.assertEqual(second.chart.x_axis_title, "Sales")
        self.assertEqual(second.chart.y_axis_title, "Y axis")
        self.assertEqual(second.metadata["placeholder_fields"], ["title", "y_axis_title"])
        self.assertEqual(first.metadata["placeholder_fields"], [])
        self.model_class.from_pretrained.assert_called_once()
        self.processor_class.from_pretrained.assert_called_once()
        self.assertEqual(self.model.generate.call_count, 2)

    def test_extract_delegates(self):
        with patch.object(self.extractor, "extract_result") as extract_result:
            self.assertIs(self.extractor.extract(self.image, source_key="eval/a.png"),
                          extract_result.return_value.chart)
        extract_result.assert_called_once_with(self.image, source_key="eval/a.png")

    def test_output_finishes_before_budget_and_unknown_commit(self):
        self.extractor = GemmaChartExtractor(max_new_tokens=10)
        self.model.config._commit_hash = None
        result = self.extractor.extract_result(self.image)
        self.assertFalse(result.metadata["output_budget_reached"])
        self.assertIsNone(result.metadata["resolved_revision"])

    def test_input_errors_precede_loading(self):
        for path, key in ((self.image, "invalid"), (self.image, ""),
                          (self.image.parent / "missing.png", None), (None, None)):
            with self.subTest(path=path, key=key), self.assertRaises(GemmaExtractionError) as caught:
                self.extractor.extract(path, source_key=key)
            self.assertEqual(caught.exception.stage, "input")
        self.dependencies.assert_not_called()
        self.image.write_text("not an image")
        with self.assertRaises(GemmaExtractionError) as caught:
            self.extractor.extract(self.image)
        self.assertEqual(caught.exception.stage, "input")
        self.model_class.from_pretrained.assert_not_called()

    def test_missing_dependencies(self):
        self.dependencies.side_effect = ImportError("secret-token")
        self.assert_safe_failure("dependencies")

    def assert_safe_failure(self, stage):
        with self.assertRaises(GemmaExtractionError) as caught:
            self.extractor.extract(self.image)
        error = caught.exception
        self.assertEqual(error.stage, stage)
        self.assertNotIn("secret-token", str(error))
        self.assertNotIn("secret-token", json.dumps(error.metadata))
        self.assertTrue(error.__suppress_context__)
        self.assertEqual(error.metadata["completion_status"], "failed")
        self.assertEqual(error.metadata["error_stage"], stage)
        return error

    def test_device_failure_precedes_model_loading(self):
        self.extractor = GemmaChartExtractor(device="cuda")
        self.assert_safe_failure("device")
        self.model_class.from_pretrained.assert_not_called()

    def test_wrapped_processor_access_failure_is_classified_safely(self):
        for status in (401, 403):
            with self.subTest(status=status):
                denied = RuntimeError("secret-token")
                denied.response = SimpleNamespace(status_code=status)
                wrapped = OSError("secret-token")
                wrapped.__cause__ = denied
                self.processor_class.from_pretrained.side_effect = wrapped
                error = self.assert_safe_failure("loading")
                self.assertIn("Cannot access", str(error))
                self.assertIn("hf auth login", str(error))
                self.model_class.from_pretrained.assert_not_called()

    def test_device_probe_errors_are_sanitized(self):
        self.torch.cuda.is_available.side_effect = ValueError("secret-token")
        self.assert_safe_failure("device")

    def test_loading_failure_does_not_publish_partial_cache(self):
        self.model.to.side_effect = RuntimeError("secret-token")
        self.assert_safe_failure("loading")
        self.assertIsNone(self.extractor._processor)
        self.assertIsNone(self.extractor._model)
        self.model.to.side_effect = None
        self.extractor.extract(self.image)
        self.assertEqual(self.model_class.from_pretrained.call_count, 2)

    def test_access_and_memory_loading_errors(self):
        class GatedRepoError(Exception):
            pass

        class OutOfMemoryError(Exception):
            pass

        for exception, expected in ((GatedRepoError("secret-token"), "hf auth login"),
                                    (OutOfMemoryError("secret-token"), "memory")):
            with self.subTest(exception=type(exception)):
                self.model_class.from_pretrained.side_effect = exception
                error = self.assert_safe_failure("loading")
                self.assertIn(expected, str(error))

    def test_inference_failure_preserves_loaded_model(self):
        self.model.generate.side_effect = RuntimeError("secret-token")
        self.assert_safe_failure("inference")
        self.model.generate.side_effect = self.generate
        self.extractor.extract(self.image)
        self.model_class.from_pretrained.assert_called_once()

    def test_inference_oom_and_preparation_and_decode_errors(self):
        for target, exception in ((self.model.generate, MemoryError("secret-token")),
                                  (self.model.generate, RuntimeError("MPS backend out of memory secret-token")),
                                  (self.processor.apply_chat_template, RuntimeError("secret-token")),
                                  (self.processor.decode, RuntimeError("secret-token"))):
            with self.subTest(target=target):
                original = target.side_effect
                target.side_effect = exception
                error = self.assert_safe_failure("inference")
                if isinstance(exception, MemoryError):
                    self.assertIn("memory", str(error))
                target.side_effect = original

    def test_parsing_failures_retain_raw_response_and_runtime(self):
        valid = self.raw
        for response in ('{"error":"secret-token"}', valid[:-3], "prose " + valid,
                         json.dumps({"chart_type": "line"})):
            with self.subTest(response=response):
                self.raw = response
                error = self.assert_safe_failure("parsing")
                self.assertEqual(error.raw_response, response)
                self.assertEqual(error.metadata["generated_token_count"], 2)
                self.assertIsNotNone(error.metadata["inference_seconds"])
        self.raw = valid
        self.extractor.extract(self.image)
        self.model_class.from_pretrained.assert_called_once()

    def test_accelerator_dtype_and_synchronization(self):
        for device, dtype in (("cuda", "bfloat16"), ("mps", "float32")):
            with self.subTest(device=device):
                self.torch.cuda.is_available.return_value = device == "cuda"
                self.torch.cuda.is_bf16_supported.return_value = True
                self.torch.backends.mps.is_available.return_value = device == "mps"
                self.extractor = GemmaChartExtractor()
                result = self.extractor.extract_result(self.image)
                self.assertEqual(result.metadata["resolved_device"], device)
                self.assertEqual(result.metadata["resolved_dtype"], dtype)
                self.assertEqual(self.ids.to.call_args.kwargs, {"device": device})
                self.assertEqual(self.pixels.to.call_args.kwargs, {"device": device, "dtype": dtype})
                getattr(self.torch, device).synchronize.assert_called()

    def test_prompt_contains_required_policies(self):
        for text in ("vbar_categorical", "hbar_categorical", "left to right", "top to bottom",
                     "x_axis_title is the", "Return null", "omit schema_version and source_key",
                     '"error"', "Omit rescued", "Do not guess", "has_bars:false"):
            self.assertIn(text, EXTRACTION_PROMPT)

    def test_extraction_imports_no_other_pipeline(self):
        script = """
import sys
from src.baseline.gemma import GemmaChartExtractor
assert not any(name in sys.modules for name in ('src.detection', 'src.ocr', 'src.reconstruction', 'src.tactile'))
"""
        subprocess.run([sys.executable, "-S", "-c", script], check=True, capture_output=True,
                       text=True, cwd=Path(__file__).resolve().parents[1])
