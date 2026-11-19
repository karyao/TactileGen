# Gemma chart extraction baseline

The optional Gemma baseline reads a local chart image directly and returns a
`VerticalBarChart` or `HorizontalBarChart`. It reuses the chart parser and VLM
response adapter without calling detection, OCR, reconstruction, or tactile
rendering. Both the Python API and CLI support extraction and diagnostics.

## Optional installation

Use a separate Python 3.12 environment from the repository root:

```bash
python3.12 -m venv .venv-vlm
source .venv-vlm/bin/activate
python -m pip install -r requirements-vlm.txt
```

This includes SentencePiece, required by the slow Gemma tokenizer selected by
the processor's `use_fast=False` setting.

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

## Command-line extraction

```bash
python -m src.baseline.gemma \
  --image charts/sales.png \
  --output-dir output/gemma/sales \
  --extract-only
```

`--image` and `--output-dir` are required. The CLI accepts `--source-key`,
`--model-id`, `--revision`, `--device`, `--dtype`, and `--max-new-tokens`; defaults
match the Python configuration table below. Use `python -m src.baseline.gemma
--help` for usage; help works without inference dependencies installed.

Omit `--extract-only` to also render tactile SVG pages using the existing vertical
or horizontal bar layout. The Python extraction API continues to return chart
data only. Rendering does not call detection, OCR, or reconstruction.

```bash
python -m src.baseline.gemma --image charts/sales.png --output-dir output/gemma/rendered
```

Renderer options match `python -m src.tactile`: `--show-print` adds print text
under braille, `--show-violations` marks rule violations, `--x-labels
{auto,full,abbreviation,number}` selects category labeling (default `auto`),
`--one-page` disables division into sections, and `--strict` returns failure if
any layout rule fails. Boolean flags default to false. All five options are
accepted but have no effect with `--extract-only`.

The output directory must be new or empty, including hidden files and
subdirectories. Existing contents are never cleared. Choose a different output
directory for a retry. Use one invocation per output directory.

| Artifact | Contents |
| --- | --- |
| `raw_response.txt` | Exact decoded model response, including whitespace. Saved whenever decoding produced text, even if the response is empty or invalid. |
| `chart.json` | Validated chart serialized with the existing schema-version-1 writer. Absent after extraction or parsing failures. |
| `run_metadata.json` | Requested/resolved settings, input image and source identity, available timing and placeholder diagnostics, completion status, and safe failure details. |
| `chart.svg` | Tactile SVG when the rendered chart fits on one page. Absent in extraction-only mode. |
| `chart-1.svg`, `chart-2.svg`, … | Multi-page output, in layout order including any key page; replaces the single `chart.svg` filename. |

Initial metadata has `completion_status: "started"`; the CLI records `"complete"`
only after every requested stage and artifact save succeeds. Individual artifacts are written
through temporary files and replaced when complete. A failed or interrupted run
may retain partial sets of artifacts; check metadata and exit status. A valid
chart already saved may remain if a subsequent metadata write fails.

The CLI adds a `rendering` metadata object containing `status` (`pending`,
`skipped`, `complete`, or `failed`), resolved `options`, and ordered `pages`.
Each page records `filename` (relative to the output directory), `saved`,
`page_number`, `page_kind`, `section` (a pair or null), `notes`, `needs_review`,
and `violations`. Each violation records `rule_id`, `section`, `message`, and
`element_ids`. Extraction-only mode records `skipped` and an empty page list.

Rendering failures preserve `raw_response.txt` and `chart.json`. A later page
write failure also preserves earlier SVGs; page `saved` flags identify successful
writes when final metadata can be saved. Failed in-memory rendering publishes
no SVGs. A final metadata failure can leave an earlier `started` record even
when the SVGs exist; stderr reports incomplete diagnostics.

Layout findings are reported and saved without failing the default command.
With `--strict`, any rule violation returns failure **after saving SVGs**:
overall completion is `failed`, `error_stage` is `validation`, and rendering is
`complete`. Notes and review reasons do not independently fail the CLI, but the
existing checker translates review reasons into overflow violations. Divided
charts currently report `overview_first` because no overview is generated, so
they also fail strict mode. Passing checks covers implemented rules only and
does not establish full BANA compliance.

Negative values, more than five series, and text outside the supported braille
character set can fail rendering. The extracted chart is retained unchanged.
Horizontal charts currently remain one page even when crowded; findings report
layout problems rather than adding pagination.

Exit codes are `0` for success, `1` for extraction/rendering/output or strict validation failures, and `2` for
argument/configuration errors. Failures print concise messages to stderr. When
possible, failures retain metadata and any decoded response, without retries or
fallback analysis. Output-writing failures use `error_stage: "output"`; if
metadata cannot be written, stderr reports that diagnostics are incomplete.

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
The CLI additionally uses `rendering` for layout/checking/SVG-generation errors,
`validation` for strict rule failures, and `output` for filesystem failures.
An SVG write failure takes precedence over strict validation findings.
Public errors and metadata contain safe messages rather than unrestricted
upstream exception text. Raw model responses are retained separately for review.
The Python extraction API does not write artifacts or generate SVGs; the CLI
saves extraction artifacts as described above.

## Validation

Run offline tests using an environment with the existing contract/tactile test
dependencies; processor/model calls are mocked and no weights are downloaded:

```bash
python -m unittest tests.test_gemma_cli tests.test_gemma tests.test_vlm_response \
  tests.test_contract_io tests.test_contracts tests.test_tactile
```

CLI tests mock Gemma extraction and exercise real layouts, rule checks, SVG XML,
pagination, strict mode, import isolation, and partial-output failure handling.

In the VLM environment, also run the real offline processor initialization check:

```bash
python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest tests.test_gemma_processor
```

This check creates a tiny synthetic SentencePiece tokenizer and image processor
locally, then initializes the real `AutoProcessor` through the extractor. Only
model loading is mocked; no checkpoint access, weights, or GPU are needed. The
test skips when the entire optional VLM stack is absent, but a partial installation
(including a missing SentencePiece dependency) fails. This checks initialization
and tokenization, not model extraction accuracy.

A real-model smoke test is separate: run the default rendering command on one
vertical and one horizontal image with suitable hardware/model access, inspect
the chart, raw response, SVG pages, and layout findings,
and record the checkpoint/revision, configuration, timing, and hardware. The
automated tests do not establish model extraction accuracy. Structural contract
validation cannot establish that Gemma read the printed chart correctly.
