"""Isolated, lazy Gemma inference for the categorical bar chart baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable

from src.baseline.vlm_response import parse_chart_response
from src.contract_io import write_bar_chart
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
    """Safe failure details, with available diagnostics for the CLI."""

    def __init__(self, stage: str, message: str, metadata: dict[str, Any],
                 raw_response: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.raw_response = raw_response
        self.metadata = {**metadata, "completion_status": "failed",
                         "error_stage": stage, "error_message": message}


def _load_dependencies():
    # Keep imports here: ordinary contract/tactile imports need no VLM packages.
    import os
    # This baseline uses PyTorch; do not auto-import an unrelated TensorFlow install.
    os.environ.setdefault("USE_TF", "0")
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
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {kind.__name__ for error in chain for kind in type(error).__mro__}
    statuses = {getattr(getattr(error, "response", None), "status_code", None)
                for error in chain}
    if statuses & {401, 403} or names & {"GatedRepoError", "RepositoryNotFoundError"}:
        return ("Cannot access the Gemma checkpoint. Accept its Hugging Face model terms "
                "and authenticate or refresh your login with hf auth login; "
                "check token permissions, model ID, and revision.")
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


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Replace one artifact only after its temporary sibling is fully written."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        write(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    _atomic_write(output_dir / "run_metadata.json",
                  lambda path: path.write_text(text, encoding="utf-8"))


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("Output directory must be new or empty; choose a different directory.")


def _save_artifacts(output_dir: Path, metadata: dict[str, Any], raw_response: str | None,
                    chart: VerticalBarChart | HorizontalBarChart | None = None,
                    *, finalize: bool = True) -> bool:
    """Save extraction output; finalize metadata on request or write failure."""
    saved = True
    try:
        if raw_response is not None:
            _atomic_write(output_dir / "raw_response.txt",
                          lambda path: path.write_text(raw_response, encoding="utf-8", newline=""))
        if chart is not None:
            _atomic_write(output_dir / "chart.json", lambda path: write_bar_chart(path, chart))
    except Exception:
        saved = False
        metadata = {**metadata, "completion_status": "failed", "error_stage": "output",
                    "error_message": "Could not save extraction artifacts."}
        print("gemma: output: Could not save extraction artifacts.", file=sys.stderr)
    if saved and not finalize:
        return True
    try:
        _write_metadata(output_dir, metadata)
    except Exception:
        print("gemma: output: Could not save final run metadata; diagnostics are incomplete.", file=sys.stderr)
        return False
    return saved


def _render_chart(
    chart: VerticalBarChart | HorizontalBarChart, options: dict[str, Any],
) -> list[tuple[dict[str, Any], str, str]]:
    """Prepare all pages in memory, importing rendering only when requested."""
    from src.tactile.layout import layout_vertical_bar_chart, layout_vertical_bar_chart_pages
    from src.tactile.layout_hbar import layout_horizontal_bar_chart, layout_horizontal_bar_chart_pages
    from src.tactile.rules import check_layout, format_report
    from src.tactile.svg import render_svg

    if isinstance(chart, HorizontalBarChart):
        layouts = ([layout_horizontal_bar_chart(chart, labels=options["x_labels"])]
                   if options["one_page"] else
                   layout_horizontal_bar_chart_pages(chart, labels=options["x_labels"]))
    else:
        layouts = ([layout_vertical_bar_chart(chart, x_labels=options["x_labels"])]
                   if options["one_page"] else
                   layout_vertical_bar_chart_pages(chart, x_labels=options["x_labels"]))
    pages = []
    for number, layout in enumerate(layouts, 1):
        violations = check_layout(layout)
        record = {
            "filename": "chart.svg" if len(layouts) == 1 else f"chart-{number}.svg",
            "saved": False, "page_number": number, "page_kind": layout.page_kind,
            "section": layout.section, "notes": list(layout.notes),
            "needs_review": list(layout.needs_review),
            "violations": [{"rule_id": v.rule.id, "section": v.rule.section,
                            "message": v.message, "element_ids": list(v.element_ids)}
                           for v in violations],
        }
        svg = render_svg(layout, show_print=options["show_print"],
                         violations=violations if options["show_violations"] else None)
        pages.append((record, svg, format_report(violations)))
    return pages


def _cli_failure(metadata: dict[str, Any], stage: str, message: str) -> None:
    metadata.update(completion_status="failed", error_stage=stage, error_message=message)
    print(f"gemma: {stage}: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Extract chart JSON and diagnostics, then render tactile SVG pages."""
    parser = argparse.ArgumentParser(
        description="Extract a chart with Gemma and save JSON, diagnostics, and tactile SVG pages.")
    parser.add_argument("--image", type=Path, required=True, help="Local chart image")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--source-key", help="Namespaced identity (default: vlm/<image filename>)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--extract-only", action="store_true",
                        help="Save extraction artifacts only; renderer options have no effect")
    parser.add_argument("--show-print", action="store_true", help="Add print text under braille labels")
    parser.add_argument("--show-violations", action="store_true", help="Mark layout violations in SVGs")
    parser.add_argument("--x-labels", choices=("auto", "full", "abbreviation", "number"), default="auto")
    parser.add_argument("--one-page", action="store_true", help="Never divide the graph into sections")
    parser.add_argument("--strict", action="store_true", help="Fail if any layout rule is violated")
    args = parser.parse_args(argv)
    source_key = args.source_key if args.source_key is not None else f"vlm/{args.image.name}"
    try:
        extractor = GemmaChartExtractor(model_id=args.model_id, revision=args.revision,
                                       device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens)
        validate_source_key(source_key)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        _prepare_output_dir(args.output_dir)
    except ValueError as exc:
        print(f"gemma: output: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("gemma: output: Could not create or access the output directory; use a writable directory.", file=sys.stderr)
        return 1

    context = {"image": str(args.image), "source_key": source_key}
    options = {name: getattr(args, name) for name in
               ("show_print", "show_violations", "x_labels", "one_page", "strict")}
    metadata = {**extractor._metadata(), **context,
                "rendering": {"status": "skipped" if args.extract_only else "pending",
                              "options": options, "pages": []}}
    try:
        _write_metadata(args.output_dir, metadata)
    except Exception:
        print("gemma: output: Could not save initial run metadata; extraction was not started.", file=sys.stderr)
        return 1
    try:
        result = extractor.extract_result(args.image, source_key=args.source_key)
    except GemmaExtractionError as exc:
        saved = _save_artifacts(args.output_dir, {**metadata, **exc.metadata, **context}, exc.raw_response)
        location = f" Diagnostics saved in {args.output_dir}." if saved else " Diagnostics are incomplete."
        print(f"gemma: {exc.stage}: {exc}{location}", file=sys.stderr)
        return 1
    except Exception:
        metadata.update(completion_status="failed", error_stage="inference",
                        error_message="Unexpected extraction failure.")
        _save_artifacts(args.output_dir, metadata, None)
        print("gemma: inference: Unexpected extraction failure; inspect available run metadata.", file=sys.stderr)
        return 1
    metadata.update(result.metadata)
    metadata.update(context, completion_status="started")
    if not _save_artifacts(args.output_dir, metadata, result.raw_response, result.chart,
                           finalize=False):
        return 1
    if not args.extract_only:
        try:
            pages = _render_chart(result.chart, options)
        except Exception:
            metadata["rendering"]["status"] = "failed"
            _cli_failure(metadata, "rendering", "Could not render the extracted chart; inspect chart.json and renderer limitations.")
            _save_artifacts(args.output_dir, metadata, None)
            return 1
        metadata["rendering"]["pages"] = [record for record, _, _ in pages]
        try:
            for record, svg, report in pages:
                path = args.output_dir / record["filename"]
                _atomic_write(path, lambda pending: pending.write_text(svg, encoding="utf-8"))
                record["saved"] = True
                print(f"Wrote {path}")
                for note in record["notes"]:
                    print(f"Note: {note}")
                for reason in record["needs_review"]:
                    print(f"NEEDS REVIEW: {reason}")
                print(report)
        except Exception:
            metadata["rendering"]["status"] = "failed"
            _cli_failure(metadata, "output", "Could not save tactile SVG artifacts.")
            _save_artifacts(args.output_dir, metadata, None)
            return 1
        metadata["rendering"]["status"] = "complete"
        if args.strict and any(record["violations"] for record, _, _ in pages):
            _cli_failure(metadata, "validation", "Tactile SVGs saved, but layout rules failed in strict mode.")
    if metadata["completion_status"] != "failed":
        metadata["completion_status"] = "complete"
    if not _save_artifacts(args.output_dir, metadata, None):
        return 1
    if metadata["completion_status"] == "failed":
        return 1
    action = "Chart extraction" if args.extract_only else "Chart extraction and tactile rendering"
    print(f"{action} completed. Artifacts saved in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
