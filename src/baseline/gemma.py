"""Isolated, lazy Gemma inference for the categorical bar chart baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.baseline.vlm_response import parse_chart_response
from src.contracts import HorizontalBarChart, VerticalBarChart, validate_source_key


DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
PROMPT_VERSION = "gemma-chart-v1"
EXTRACTION_PROMPT = """Read this chart and return exactly one JSON object, without prose or Markdown.
Support only categorical vertical or horizontal bar charts, including grouped bars.
Return chart content only; omit schema_version and source_key (the caller supplies them).
Use this structure (replace the example labels and numbers with the image content):
{"chart_type":"vbar_categorical","title":null,"x_axis_title":null,
 "y_axis_title":null,"categories":["Category"],"series":[{"name":null,"values":[1]}]}
chart_type must be vbar_categorical or hbar_categorical. Do not represent stacked,
line, pie, or other unsupported charts as categorical bars.
Preserve the printed axis meanings. For horizontal charts, x_axis_title is the
value heading and y_axis_title is the category heading. For vertical charts,
x_axis_title is the category heading and y_axis_title is the value heading.
List categories left to right for vertical charts and top to bottom for horizontal
charts. Preserve printed category labels and series names. Align each visible
series' values with the categories in exactly that order. Values must be finite
JSON numbers, not strings, booleans, or null. Do not guess missing categories or values.
Return null for absent chart/axis titles and for an absent series name.
The optional chart field omit_legend is a boolean for a single series without a
printed legend. A series may optionally contain has_bars (boolean, default true)
and rescued (an array of booleans). For a named legend entry whose bars are not
visible, use has_bars:false and values:[]; do not substitute zeros. At least one
series must have visible bars. Omit rescued: it is reserved for pixel measurements
performed by another pipeline. Omit unknown fields.
If the chart is unreadable, unsupported, or required categories/values cannot be
extracted, return exactly {"error":"unreadable or unsupported chart"}.
"""


@dataclass(frozen=True)
class GemmaExtractionResult:
    chart: VerticalBarChart | HorizontalBarChart
    raw_response: str
    metadata: dict[str, Any]


class GemmaExtractionError(RuntimeError):
    """Safe failure details, with available diagnostics for a future CLI."""

    def __init__(self, stage: str, message: str, metadata: dict[str, Any],
                 raw_response: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.raw_response = raw_response
        self.metadata = {**metadata, "completion_status": "failed",
                         "error_stage": stage, "error_message": message}


def _load_dependencies():
    # Keep imports here: ordinary contract/tactile imports need no VLM packages.
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    return torch, Image, AutoProcessor, Gemma3ForConditionalGeneration


class _DeviceConfigurationError(ValueError):
    """A device error whose message is generated locally and safe to expose."""


def _resolve_runtime(torch, device: str, dtype: str) -> tuple[str, str]:
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    resolved = ("cuda" if cuda else "mps" if mps else "cpu") if device == "auto" else device
    if (resolved == "cuda" and not cuda) or (resolved == "mps" and not mps):
        raise _DeviceConfigurationError(f"Requested {resolved} device is unavailable; choose an available device or cpu.")
    bf16 = resolved == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    if dtype == "auto":
        dtype = "bfloat16" if bf16 else "float32"
    if ((dtype == "bfloat16" and not bf16)
            or (resolved == "cpu" and dtype != "float32")):
        raise _DeviceConfigurationError(f"{dtype} is unsupported on {resolved} for this baseline; use auto or float32.")
    return resolved, dtype


def _synchronize(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _loading_message(exc: Exception) -> str:
    # Classify without incorporating arbitrary server messages or credentials.
    names = {kind.__name__ for kind in type(exc).__mro__}
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403) or names & {"GatedRepoError", "RepositoryNotFoundError"}:
        return ("Cannot access the Gemma checkpoint. Accept its Hugging Face model terms "
                "and authenticate with hf auth login; check the model ID and revision.")
    if _out_of_memory(exc):
        return "Insufficient memory to load Gemma; free memory or choose another device or checkpoint."
    return ("Could not load the Gemma model/processor. Check the model ID, revision, network, "
            "and local cache. For gated models, accept the Hugging Face terms and run hf auth login.")


def _out_of_memory(exc: Exception) -> bool:
    # MPS reports OOM as RuntimeError, rather than torch.OutOfMemoryError.
    return (isinstance(exc, MemoryError)
            or "OutOfMemoryError" in {kind.__name__ for kind in type(exc).__mro__}
            or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()))


class GemmaChartExtractor:
    """Reuse one lazily loaded processor/model across sequential image requests.

    Construction performs no imports, downloads, or device initialization.
    Concurrent calls on the same extractor are not supported.
    """

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, revision: str | None = None,
                 device: str = "auto", dtype: str = "auto", max_new_tokens: int = 4096):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonblank string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a nonblank string or None")
        if not isinstance(device, str) or device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be auto, cuda, mps, or cpu")
        if not isinstance(dtype, str) or dtype not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be auto, float32, bfloat16, or float16")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model_id, self.revision = model_id, revision
        self.device, self.dtype = device, dtype
        self.max_new_tokens = max_new_tokens
        self._processor = self._model = None
        self._device = self._dtype = None

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "requested_revision": self.revision,
            "resolved_revision": None, "prompt_version": PROMPT_VERSION,
            "requested_device": self.device, "requested_dtype": self.dtype,
            "resolved_device": self._device, "resolved_dtype": self._dtype,
            "generation_settings": {"do_sample": False, "num_beams": 1,
                                    "num_return_sequences": 1, "max_new_tokens": self.max_new_tokens},
            "generated_token_count": None, "output_budget_reached": None,
            "inference_seconds": None, "placeholder_fields": [],
            "completion_status": "started",
        }

    def _ensure_loaded(self, torch, processor_class, model_class, metadata) -> None:
        if self._model is None:
            try:
                device, dtype = _resolve_runtime(torch, self.device, self.dtype)
            except Exception as exc:
                message = (str(exc) if isinstance(exc, _DeviceConfigurationError)
                           else "Could not initialize the requested device.")
                raise GemmaExtractionError("device", message, metadata) from None
            metadata.update(resolved_device=device, resolved_dtype=dtype)
            try:
                processor = processor_class.from_pretrained(
                    self.model_id, revision=self.revision, use_fast=False)
                model = model_class.from_pretrained(
                    self.model_id, revision=self.revision, torch_dtype=getattr(torch, dtype))
                model.to(device)
                model.eval()
            except Exception as exc:
                raise GemmaExtractionError("loading", _loading_message(exc), metadata) from None
            self._processor, self._model = processor, model
            self._device, self._dtype = device, dtype
        commit = getattr(self._model.config, "_commit_hash", None)
        metadata.update(resolved_device=self._device, resolved_dtype=self._dtype,
                        resolved_revision=commit if isinstance(commit, str) else None)

    def extract(self, image: str | Path, *, source_key: str | None = None
                ) -> VerticalBarChart | HorizontalBarChart:
        return self.extract_result(image, source_key=source_key).chart

    def extract_result(self, image: str | Path, *, source_key: str | None = None
                       ) -> GemmaExtractionResult:
        metadata = self._metadata()
        try:
            path = Path(image)
            validate_source_key(source_key if source_key is not None else f"vlm/{path.name}")
            if not path.is_file():
                raise ValueError("missing image")
        except (TypeError, ValueError, OSError):
            raise GemmaExtractionError("input", "Provide a local image file and a valid namespaced source_key.", metadata) from None
        try:
            torch, Image, processor_class, model_class = _load_dependencies()
        except Exception:
            raise GemmaExtractionError(
                "dependencies", "Install compatible VLM dependencies: python -m pip install -r requirements-vlm.txt",
                metadata) from None
        try:
            with Image.open(path) as original:
                rgb = original.convert("RGB")
        except Exception:
            raise GemmaExtractionError("input", "Could not open the image; provide a readable image file.", metadata) from None
        try:
            self._ensure_loaded(torch, processor_class, model_class, metadata)
            return self._infer(torch, rgb, path, source_key, metadata)
        finally:
            rgb.close()

    def _infer(self, torch, rgb, path, source_key, metadata) -> GemmaExtractionResult:
        raw_response = None
        try:
            _synchronize(torch, self._device)
            started = perf_counter()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": rgb}, {"type": "text", "text": EXTRACTION_PROMPT},
            ]}]
            inputs = self._processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
            inputs = {key: value.to(device=self._device, dtype=getattr(torch, self._dtype))
                      if value.is_floating_point() else value.to(device=self._device)
                      for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self._model.generate(**inputs, **metadata["generation_settings"])
            generated = output[0][input_length:]
            raw_response = self._processor.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            _synchronize(torch, self._device)
            metadata.update(inference_seconds=perf_counter() - started,
                            generated_token_count=len(generated),
                            output_budget_reached=len(generated) >= self.max_new_tokens)
        except Exception as exc:
            message = ("Insufficient memory during Gemma inference; reduce max_new_tokens or choose another device/checkpoint."
                       if _out_of_memory(exc) else "Gemma image preparation, generation, or decoding failed; check the runtime and image.")
            raise GemmaExtractionError("inference", message, metadata, raw_response) from None
        try:
            parsed = parse_chart_response(raw_response, image=path, source_key=source_key)
        except Exception:
            raise GemmaExtractionError(
                "parsing", "Gemma did not return a supported, complete chart JSON object; inspect raw_response.",
                metadata, raw_response) from None
        metadata.update(placeholder_fields=list(parsed.placeholder_fields), completion_status="complete")
        return GemmaExtractionResult(parsed.chart, raw_response, metadata)
