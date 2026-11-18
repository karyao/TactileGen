# Bar orientation baseline — Step 1

Step 1 is complete: the corpus includes both chart orientations, genuine mixed
YOLO predictions, reproducible raw captures, and a production integration map.
This work does not add an orientation gate or change application behavior.

## Corpus and observed outcomes

Human expectations are in `tests/fixtures/orientation/cases.json`; observations
are in `capture.json` beside it. Each `*.detections.json` is the complete raw
`DetectionFrame` from real inference, including non-bar roles and low-confidence
detections. The hand-authored contract JSON elsewhere in the repository is not
used as detector output.

| Case | Human orientation | Raw vertical / horizontal bars | App completeness | CLI exit |
| --- | --- | --- | --- | --- |
| `clean_vertical` | Vertical | 3 / 0 | ready | 0 |
| `clean_horizontal` | Horizontal | 0 / 3 | ready | 0 |
| `mixed_predictions` | Horizontal | 1 / 3 | needs_input | 2 |
| `bundled_vertical` | Vertical | 8 / 0 | cannot_build | 3 |
| `stacked_horizontal` | Horizontal | 0 / 36 | cannot_build | 3 |

The clean authored charts show Alpha, Beta, Gamma with values 40, 70, 100.
The vertical reconstruction gives approximately 39.77, 69.86, 99.60; horizontal
gives 39.75, 70.03, 99.83, in the correct category order. Both preserve titles
and axis labels and generate CLI SVGs. The vertical result has `bars_rescued:1`:
a visually simple chart is not a claim of perfect detector performance.

The mixed case is the initial authored horizontal chart with a full plot border
and multiword category labels. YOLO identifies the right border as a vertical
bar at confidence 0.6783, alongside three horizontal bars at 0.9523–0.9877.
All four exceed 0.30. Their confidence sums are 2.9240 horizontal and 0.6783
vertical (81.17% horizontal). The current count helper selects horizontal;
the overlay still shows both orientations. Reconstruction also produces an
extra category `B`, so this case requires text review. An orientation gate
alone should not be expected to correct that category error.

The bundled contract image has four visible bars but eight raw bar predictions
and existing OCR/calibration/series ambiguity. Preserve this failure as an
additional detector baseline, not a successful end-to-end reference.

`stacked_horizontal.png` is a byte-for-byte snapshot of the user's root
`images.png`; that original remains untouched. Its external provenance is
unknown. It contains horizontal stacks, and the current helper selects
horizontal. Reconstruction has baseline, category and series blockers. Stacked
reconstruction is outside the orientation work; this case assesses orientation
only. No external images were downloaded.

## Reproduce and inspect

Run from the repository root with Python 3.12 and the application dependencies:

```bash
python3.12 -m venv /private/tmp/tactile-orientation-venv
/private/tmp/tactile-orientation-venv/bin/python -m pip install -r requirements-app.txt pytest
/private/tmp/tactile-orientation-venv/bin/python -m scripts.capture_orientation_baseline capture
```

The optional `prepare` action regenerates the three authored PNGs using
Matplotlib's packaged DejaVu Sans font. It retains the stacked snapshot when
present; its initial creation requires root `images.png`. Prefer the committed
PNG bytes when comparing model behavior: rendering/library changes can change
the pixels. Recapture deliberately replaces the committed observations.

Canonical settings: `models/yolo_bars_hv/best.pt`, confidence 0.25, NMS IoU 0.70,
requested size 640, CPU, RapidOCR PP-OCRv6-small, language `en`. The capture
records image/checkpoint hashes, repository revision and dirty state, all
installed package versions, OCR model hashes, and actual inference evidence.
This baseline used Python 3.12, Ultralytics 8.4.215, Torch 2.15.0, RapidOCR
3.10.0 and ONNX Runtime 1.32.0. Requirements have version ranges: consult the
recorded package versions before interpreting differences after an upgrade.

For 800×500 authored images, YOLO reports effective size `[640, 640]` and actual
tensor shape `[1, 3, 416, 640]`; the requested size does not mean a square input.

The script calls real app analysis with diagnostics, then the public CLI runner
using the real `analyse(..., ocr_engine="rapidocr")`, matching the entrypoint's
OCR selection. No model/backend is mocked. The app explicitly requests CPU;
the CLI has no device flag and uses its existing automatic device selection.
All five CLI detection frames matched the app frames exactly on this host.

`output/orientation_baseline/<case>/` contains annotated PNGs, diagnostic ZIPs,
CLI raw predictions, stdout/stderr and successful CLI SVGs. `capture.json`
identifies the exact diagnostic archive for each recorded run. Full archives
remain ignored; compact inputs, predictions and observations are committed.

To exercise the CLI independently:

```bash
/private/tmp/tactile-orientation-venv/bin/python -m src.tactile.pipeline \
  --image tests/fixtures/orientation/clean_horizontal.png --ocr rapidocr --yes \
  --output output/orientation_baseline/manual_horizontal.svg
```

## Production integration map

| Area | Current behavior and later gate integration |
| --- | --- |
| Raw detection | `src/detection.py:run_detection()` returns unfiltered predictions. Preserve its contract and role/ID ordering. |
| Orientation decision | `src/reconstruct_hbar.py:bar_orientation()` counts bars at the reconstruction confidence floor (0.30). Horizontal must strictly outnumber vertical; ties/no bars select vertical. The only production callers are app analysis and tactile CLI analysis. |
| App orchestration | `src/gradio_ui/analysis.py:analyze_image()` validates dimensions and snapshots raw detections, creates the OCR backend, runs OCR, then chooses matcher/reconstructor. A later gate belongs after detection/dimension validation and before backend construction. |
| Horizontal adaptation | `match_legend_to_hbars()` swaps bar roles for the vertical matcher. `reconstruct_hbar()` transposes geometry/image and maps the spec back to horizontal. These adapters consume an orientation decision; they should not independently choose one. Horizontal matching/reconstruction currently do not accept the diagnostics collector. |
| App consumers | The same current raw frame supplies OCR, matching, reconstruction, edge completeness checks, detection count, summary, overlay, and callback tables. A later resolved effective frame must reach all these consumers consistently. |
| Presentation | `src/presentation.py` renders both bar roles. `summarize_detections()` hardcodes “vertical bars” for legend assignment counts, even on horizontal results. |
| Diagnostics | `AnalysisDiagnostics` snapshots `outputs.detections`, captures source text crops and final annotation; `diagnostics_report.py` packages the report. Stage order currently has detection directly followed by OCR. Retain raw snapshots/crops when introducing a separate effective frame and orientation stage. |
| CLI analysis | `src/tactile/pipeline.py:analyse()` detects, creates OCR, reads text, then chooses orientation. The horizontal reconstructor computes its own horizontal legend mapping; the vertical branch computes matching explicitly. Returns `(spec, error, frame)`. `run()` handles blockers, text review, completeness and output. |
| Spec consumers | `spec_to_chart_with_notes()` in completeness chooses chart type from `spec.orientation`; both `generate_tactile_artifacts()` and CLI `run()` select the corresponding layout. Layout/rule code uses the resulting orientation. These consume the spec rather than recount detections. |

`src/baseline/opencv_charts.py` uses an explicit benchmark orientation, not the
YOLO count helper. Standalone YOLO `predict` uses `run_detection`; evaluation
uses raw batched model results and `detection_frame`; validation uses model
validation directly. Preserve these raw benchmark paths. Vertical-specific
evaluation/review scripts are not app/CLI automatic orientation routing.

## Review lifecycle and verification

- `layout.build_app()` wires image changes to `ready_results()`, which resets
  result components, review state, generation availability and visible downloads.
- Analyze calls `begin_analysis()`: controls are disabled and all prior results
  reset. `analyze_event()` builds a fresh review model only when a spec and
  completeness exist. Generation is enabled for ready/needs-input results, and
  disabled for cannot-build results. `finish_analysis()` re-enables controls.
- Clear restores existing setting defaults and calls `ready_results()`. Rerun
  keeps current settings but resets results before processing. Current setting
  controls do not have a general result-invalidation handler; the future
  orientation selector needs an explicit change handler.
- `review.py` holds an immutable source spec plus text drafts. Generation applies
  submitted text and writes reviewed artifacts. A later unresolved orientation
  result must leave no review model, spec, generation enablement, or old visible
  artifact links. Existing reset logic clears UI state; it does not delete every
  prior temporary file.
- Existing routing assertions/mocks live in `tests/test_reconstruct.py`,
  `tests/gradio/test_gradio_app.py` and `tests/test_tactile.py`. Tie/no-bar tests
  describe today's behavior and will need migration when the resolver replaces
  the helper. The capture utility also calls that helper to record the current
  decision; migrate that observer when the helper is removed. No new tests
  require the vertical fallback to survive.

The archive integrity tests load the existing detection contracts, verify image
and raw JSON hashes/dimensions/IDs, require genuine mixed evidence above 0.30,
and compare clean archived results against human truth (absolute value tolerance
1 book). They do not run inference or claim to validate a future resolver.

```bash
/private/tmp/tactile-orientation-venv/bin/python -m pytest -q tests/test_orientation_baseline.py
/private/tmp/tactile-orientation-venv/bin/python -m pytest -q
```

Verification on the completed corpus: **591 passed, 2 skipped, 304 subtests
passed**. All five diagnostic ZIPs passed archive-integrity checks. Both clean
CLI outputs reported all 24 automated BANA rules passed; this is not a tactile
readability assessment.

The mixed case was found in the initial local fixtures, so no candidate search
or fabricated predictions were necessary. Threshold calibration, a selector,
uncertainty handling, and behavioral changes remain subsequent steps.
