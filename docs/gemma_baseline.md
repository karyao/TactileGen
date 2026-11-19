# Gemma chart extraction baseline

The optional Gemma baseline reads a local chart image directly and returns a
`VerticalBarChart` or `HorizontalBarChart`. It reuses the chart parser and VLM
response adapter without calling detection, OCR, reconstruction, or tactile
rendering. The Python inference API is implemented; CLI arguments and diagnostic
artifact writing are planned for Stage 3.

## Optional installation

Use a separate Python 3.12 environment from the repository root:

```bash
python3.12 -m venv .venv-vlm
source .venv-vlm/bin/activate
python -m pip install -r requirements-vlm.txt
```

The default checkpoint is
[`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).
Accept the model's access/license terms on Hugging Face, then authenticate locally:

```bash
hf auth login
```

The first extraction loads the processor and model from the local Hugging Face
cache or downloads them if needed. Importing the module and constructing the
extractor do not import inference packages, download weights, or initialize a
device. Subsequent sequential calls on the same extractor reuse the loaded model.
Use one extractor per worker; concurrent calls on one instance are unsupported.

## Python API

```python
from src.baseline.gemma import GemmaChartExtractor

extractor = GemmaChartExtractor()
chart = extractor.extract("charts/sales.png")
assert chart.source_key == "vlm/sales.png"

# Optional: retain raw output and run details for inspection.
result = extractor.extract_result("charts/sales.png", source_key="eval/sales.png")
print(result.chart)
print(result.raw_response)
print(result.metadata)
```

Configuration is keyword-only:

| Option | Default | Behavior |
| --- | --- | --- |
| `model_id` | `google/gemma-3-4b-it` | Hugging Face checkpoint ID or local checkpoint directory. |
| `revision` | `None` | Optional revision passed to both processor and model. |
| `device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. Automatic selection prefers CUDA, then MPS, then CPU. |
| `dtype` | `auto` | `auto`, `float32`, `bfloat16`, or `float16`. Auto uses native BF16 on capable CUDA hardware and float32 elsewhere. |
| `max_new_tokens` | `4096` | Positive integer output token limit. |

Explicit BF16 requires native CUDA BF16 support. Explicit float16 supports CUDA
and MPS; CPU uses float32. Unsupported device/dtype choices raise errors rather
than silently falling back. Float32 uses more memory than reduced precision;
adequate memory is required for both weights and generation.

The image is opened locally as RGB. The processor handles resizing and
normalization. Generation is greedy (`do_sample=False`, one beam, one response)
under inference mode, and only newly generated tokens are decoded. Greedy
generation does not promise identical results across hardware/library versions.

The versioned prompt requests chart content only: chart type, printed titles,
ordered categories, and aligned series values. Missing chart/axis titles should
be `null`; the adapter supplies placeholders and records their field names.
The adapter also supplies `schema_version: 1` and caller-owned `source_key`.
Malformed or incomplete responses fail without repair, retries, or fallback.

`GemmaExtractionResult` contains the chart, decoded `raw_response`, and a
JSON-serializable metadata dictionary recording requested/resolved configuration,
available model commit, prompt version, generation settings, token count,
output-budget status, inference duration, placeholders, and completion status.
Inference timing covers processor preparation, generation, and decoding, with
accelerator synchronization; it excludes loading and chart parsing.

## Failure diagnostics

```python
from src.baseline.gemma import GemmaExtractionError

try:
    result = extractor.extract_result("charts/sales.png")
except GemmaExtractionError as error:
    print(error.stage, str(error))
    print(error.metadata)
    # Available when output was decoded, including a parsing failure:
    raw_response = error.raw_response
```

Invalid constructor configuration raises `ValueError`. Extraction errors use
stages `input`, `dependencies`, `device`, `loading`, `inference`, and `parsing`.
Public errors and metadata contain safe messages rather than unrestricted
upstream exception text. Raw model responses are retained separately for review.
Extraction does not write artifacts or generate SVGs.

## Validation

Run offline tests using an environment with the existing contract/tactile test
dependencies; processor/model calls are mocked and no weights are downloaded:

```bash
python -m unittest tests.test_gemma tests.test_vlm_response \
  tests.test_contract_io tests.test_contracts tests.test_tactile
```

A real-model smoke test is separate: extract one vertical and one horizontal
image with suitable hardware/model access, inspect the chart and raw response,
and record the checkpoint/revision, configuration, timing, and hardware. The
automated tests do not establish model extraction accuracy. Structural contract
validation cannot establish that Gemma read the printed chart correctly.
