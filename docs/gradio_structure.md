# Finding your way around the Gradio app

Start with `src/gradio_app.py`. It reads command-line options, loads the detector,
calls `build_app()`, and starts the server. OCR engines are selected in the browser
and loaded on first use by `gradio_ui/ocr.py`. RapidOCR is the default;
`--ocr tesseract` changes the initial selection. The command is still:

```bash
python3.12 -m src.gradio_app
```

The rest of the application lives in `src/gradio_ui/`. Each module answers a
different question:

| File | Question it answers |
| --- | --- |
| `layout.py` | What appears on the page, and which action does each button trigger? |
| `callbacks.py` | What happens when someone analyzes, edits, generates, or clears? |
| `analysis.py` | How does an uploaded image become detections, text, and a chart specification? |
| `review.py` | How are label edits validated and applied before tactile generation? |
| `formatting.py` | How do results become readable status messages and reports? |
| `artifacts.py` | How are OCR and chart specifications saved for download? |
| `components.py` | Which page components receive each callback's results? |
| `settings.py` | What defaults, upload limits, and warning messages do we use? |
| `dependencies.py` | How is the optional Gradio dependency loaded? |

## Follow one action

Read these three functions in order to understand **Analyze chart**:

1. `layout.build_app()` creates the page and connects the Analyze button to a
   callback. A **callback** is a function the interface runs after a user action.
2. `callbacks.create_callbacks()` creates `analyze_event()`, which calls the
   analysis code, prepares downloads, and decides what the page should display.
3. `analysis.analyze_image()` validates the upload, detects chart elements,
   resolves the bar orientation with the shared resolver, recognizes text,
   matches legends, reconstructs values, and renders an overlay. The
   `orientation_resolution` diagnostics stage records why a direction was
   selected or why downstream stages were skipped. Its `AnalysisResult` holds
   both successful results and warnings.

For **Generate tactile graphic**, follow `generate_event()` into
`review.generate_review_submission()`. The review model keeps a draft of label
edits separate from the reconstructed specification. Numeric values remain
read-only. A corrected specification is committed only after successful generation.

## Make a small change

- To change a label or arrange page sections, edit `layout.py`.
- To change the wording of a report, edit `formatting.py`.
- To change which label edits are accepted, edit `review.py`.
- To change what happens after analysis succeeds or fails, edit `callbacks.py`.
- To change detection, OCR, or reconstruction algorithms, follow the imports in
  `analysis.py` to their existing domain modules; the interface calls those modules.

`components.py` gives callback outputs names because Gradio expects returned values
in the same order as the registered output components. For example,
`results.updates(summary_output=..., ...)` converts named values into that order
and rejects missing or unknown fields. This keeps a status message from
accidentally being sent to an image component.

When adding an analysis output, add its field to `AnalysisComponents`, supply the
component when constructing that group in `layout.py`, and provide its initial
update in `result_updates()` inside `callbacks.py`. The shared group determines
the order for both event wiring and returned updates.

## Keep the boundaries simple

`layout.py` uses callbacks; callbacks use analysis, review, formatting, and artifact
writers. Analysis and review do not import the page builder or Gradio. This lets
their logic run and be tested without starting a browser interface.

New Python code should import helpers from their owning module, for example:

```python
from src.gradio_ui.analysis import analyze_image
from src.gradio_ui.review import build_review_model
```

Existing imports from `src.gradio_app` continue to work. Tests that replace a
dependency with a mock must patch where the dependency is used: for example,
`src.gradio_ui.analysis.run_detection` or `src.gradio_ui.callbacks.analyze_image`.

## Run the focused tests

`tests/gradio/test_gradio_app.py` contains CLI startup, image analysis, review
rules, component output ordering, UI callbacks, theme, and diagnostics integration
tests, together with their shared chart fixtures.

```bash
python3.12 -m pytest -q tests/gradio/
```

The tests use small generated images and mocked detector results, so they do not
need a real model checkpoint or a running server.

## Gemma tab

The app always shows **Detector + OCR** and **Gemma** tabs. Each has its own upload,
review, downloads, and Clear action; switching tabs preserves both workflows.
Gemma loads its checkpoint on the first extraction and shares one serialized
extractor across browser sessions. Missing VLM packages or model access are shown
in the tab; installing Gemma is optional for the detector workflow.

```bash
python -m pip install -r requirements-vlm-app.txt
python -m src.gradio_app --workflow gemma
```

Gemma-only startup skips detector weights; it still needs the app dependencies.
The default `--workflow both` starts the normal detector and also provides Gemma.
Configure Gemma at startup with `--gemma-model-id`, `--gemma-revision`,
`--gemma-device`, `--gemma-dtype`, and `--gemma-max-new-tokens`. Defaults match the
[Gemma baseline](gemma_baseline.md). For gated checkpoints, accept the model terms
and authenticate locally with `hf auth login` before extracting.

In Gemma, upload → Extract chart → review labels and numeric values → Generate
Gemma tactile pages. Edit only the Text/Value columns. Category/series order and
counts stay fixed. Blank, boolean, NaN, and infinite values are rejected. Original
extraction files remain separate from the reviewed chart JSON. Placeholder titles
are identified in the review status. The gallery contains annotated preview SVGs;
the ZIP contains clean SVGs. Layout findings require review before production.

`gradio_ui/gemma.py` owns the Gemma backend, chart review, and callbacks. The shared
`generate_chart_tactile_artifacts(chart, artifact_root)` service renders chart
contracts directly; the detector continues using the existing spec-based API.
Run tickets and review revisions prevent late extraction/generation from restoring
cleared or edited results. Clear/unload discard session artifacts, not cached model
weights. Clearing while extraction runs discards its result without promising to
interrupt GPU inference.

Automated tab tests inject inference and exercise real Gradio file publication
and tactile rendering. A real-model acceptance check remains separate: extract
one vertical and one horizontal chart, correct a value, inspect the preview and
reviewed chart, and download the clean ZIP.

For the combined app, install `requirements-vlm-app.txt` in one pip command.
It selects Gradio 6.17.x, compatible with Transformers 4's Hugging Face Hub <1
requirement, and resolves TorchVision together with Torch. Restart the running
server after package installation. The Gemma loader defaults `USE_TF=0` because
it uses PyTorch and does not need an unrelated TensorFlow installation.
