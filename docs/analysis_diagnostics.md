# Analysis diagnostics

Use this guide to run a chart analysis, download its diagnostics, and understand
warnings or failures. No previous runs or files under ignored `output/` are needed.
Run all commands from the repository root.

- [Start the app](#start-the-app)
- [Download and read diagnostics](#download-and-read-diagnostics)
- [Compare YOLO inference sizes](#compare-yolo-inference-sizes)
- [Understand missing evidence](#understand-missing-evidence)
- [Archive contents](#archive-contents)
- [Developer reference](#developer-reference)

## Start the app

If the app is already running, continue to [Download and read diagnostics](#download-and-read-diagnostics).

1. Create a Python 3.12 environment if needed and install the application dependencies:

   ```sh
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements-app.txt
   ```

2. Install the Tesseract executable and English language data using your system
   package manager. On macOS with Homebrew, run `brew install tesseract`; on
   Ubuntu/Debian, run `sudo apt-get install tesseract-ocr tesseract-ocr-eng`.
   Confirm that `tesseract --version` works and `tesseract --list-langs` lists `eng`.
3. Start the app with the included detector checkpoint:

   ```sh
   .venv/bin/python -m src.gradio_app --weights models/yolo_vbars/best.pt
   ```

4. Open the local URL printed in the terminal.

## Download and read diagnostics

1. Select the bundled example or upload
   `tests/fixtures/contracts/val/example.png`. You can also use your own chart.
   Select **Analyze chart**.
2. Expand **Analysis details**, select **Download analysis diagnostics (ZIP)**,
   and save the archive before clearing the form or starting another analysis.
   Warnings and reconstruction blockers can still produce a diagnostic download.
3. Extract the entire archive into one directory. Open `report.md` in a Markdown
   viewer so its relative links and image previews work.
4. Start with **Findings**. Follow a finding's evidence links to see the original
   OCR text, how it was trimmed or parsed, and the confidence or threshold used.
   For example, a dropped tick can show that OCR returned `"="` and numeric parsing
   rejected it.
5. Follow **OCR attempts** links to compare the source crop, images submitted for
   recognition, raw engine output, and selected result. Strings use JSON escapes
   to distinguish tabs, newlines, empty strings, and spaces while preserving Unicode.
6. If you need to match the download to the terminal logs, compare the run ID.
   It appears in the app, archive name, manifest, report, JSON evidence, and run log.

Reports explain recorded decisions; they do not verify recognition accuracy or
ground-truth text. Real OCR results may differ with the chart and engine version.
The app does not highlight issue regions interactively.

## Compare YOLO inference sizes

The default requested size is **640 pixels**. To start with a different value:

```sh
python3.12 -m src.gradio_app --imgsz 960
```

Change **YOLO inference size (pixels)** under **Advanced settings** before an
analysis to override that value for the next run. Clear restores the startup
value. Quick-select buttons offer **640**, **960**, and **1280** pixels. Custom
values must be integers from **1 to 1280**, enforced by the shared inference
validator, CLI, and UI. The maximum bounds per-request work; it does not guarantee
performance on every device. Sizes need not be multiples of the model stride.

In `diagnostics.json`, `settings.imgsz` is the requested size.
`metadata.detection` distinguishes three measurements:

| Field | Meaning |
| --- | --- |
| `requested_imgsz` | The scalar size submitted to YOLO |
| `effective_imgsz` | YOLO's checked target `[height, width]`, after size adjustment |
| `tensor_shape` | Observed inference input `[batch, channels, height, width]`, excluding warmup |
| `unavailable_reasons` | Reasons for measurements that remain null |

For example, requesting 641 with a stride-32 model can produce an effective
size of `[672, 672]`, while rectangular padding produces an actual tensor of
`[1, 3, 512, 672]` for a 320-by-240 chart. These dimensions describe model input;
detection boxes, OCR crops, and overlays still use original-image coordinates.
Read the same fields in the settings/metadata sections of `report.md` or
`report.txt`, or the detection measurement line in `analysis.log`.

Measurements remain null if the relevant stage has not executed or a custom model
does not expose observation hooks. Failed runs retain whatever was observed
before failure; original-image dimensions are never substituted for tensor shape.

To compare sizes manually:

1. Keep the chart image, checkpoint, device, confidence, NMS IoU, OCR engine, and
   OCR language fixed. Use the same running app and avoid other inference work.
2. Test **640, 960, and 1280**. At each size, run one analysis to warm up, then
   collect three more analyses. Download each diagnostics ZIP before starting the
   next analysis, and label the extracted folders with size and repetition.
3. Compare the median `stages.detection.duration_seconds` at each size. Compare
   `duration_seconds` separately for total analysis time. Detection-stage timing
   includes wrapper and evidence work; it is not pure model-forward latency.
4. Inspect the annotated detections and OCR results for known small labels on the
   same charts. Record missed labels, incorrect detections, and recognition errors;
   a larger detection count alone does not establish better accuracy.

This procedure enables exploratory comparisons; it does not add an automated
benchmark or establish accuracy without reference annotations.

## Understand missing evidence

- **An analysis failed early:** the ZIP contains what was collected before the
  failure. An upload validation failure has no detection or OCR output. Read
  `manifest.json` and the report for the reasons evidence is unavailable.
- **A warning has no detailed explanation:** some warnings have no linked decision
  evidence. The report says so. Checks skipped after an earlier blocker are not
  presented as executed.
- **OCR attempt details are missing:** legacy or custom backends may expose final
  results without internal attempts. Availability is recorded explicitly.
- **The download disappeared:** changing the upload or example, clearing the form,
  or starting another analysis resets it. Save each download before continuing.
  Temporary archives remain available only until the app shuts down.
- **Diagnostic export failed:** the download is hidden and the app shows an export
  error with the run ID. Analysis and review results remain available.

Later review edits do not change the original analysis archive.

## Archive contents

The filename is `analysis-diagnostics-<run_id>.zip`.

| File or directory | Contents |
| --- | --- |
| `report.md` | Readable findings, measurements, source/recognition images, and evidence links |
| `manifest.json` | Run identity, outcome, artifact availability, and reasons for unavailable evidence |
| `diagnostics.json` | Original analysis snapshot, including stages, warnings, events, and specification sources |
| `analysis.log` | Run-owned timestamped collector messages and recorded exception tracebacks |
| `ocr.json` | Available OCR result using the existing versioned OCR interchange contract |
| `detections.json` | Available detection frame using the existing versioned detection contract |
| `ocr_trace.json` | Available batch history, raw engine evidence, attempts, and selection |
| `ocr_crops/` | Source text crops with their coordinates in the diagnostic record |
| `ocr_inputs/` | Actual recognition API images or Tesseract TIFF-page pixels |
| `annotated.png` | Available detection overlay |
| Other output JSON files | Available legend mapping, specification, and completeness snapshots |
| `report.txt` | Existing text evidence dump retained for compatibility |

The first four files are always present in a successfully packaged archive.
Other files are included only when available; empty results are not invented for
failed stages. The full uploaded `input.png` is omitted. The annotated overlay is
a separate derived image. Legacy/custom backends may not expose internal attempts.

## Developer reference

This section describes the recorded fields, export behavior, and checks used to
investigate the implementation. For ordinary chart inspection, use the report
steps above.

- [OCR attempts](#ocr-attempts)
- [Source crops and recognition images](#source-crops-and-recognition-images)
- [Warning provenance](#warning-provenance)
- [Export and logging](#export-and-logging)
- [Generate local examples](#generate-local-examples)
- [Run automated checks](#run-automated-checks)

### OCR attempts

`run_ocr(..., diagnostics=collector)` adds `ocr_trace` to the existing
`AnalysisDiagnostics` record. The app passes its run collector automatically.
The existing ZIP contains the same section in `diagnostics.json` and a separate
`ocr_trace.json`; `ocr.json` retains its existing chart interchange contract.

The trace has its own `schema_version: 1` and three collections:

- `batches`: backend, detections, operation, outcome, original response fields,
  and any batch exception. RapidOCR also retains the classifier's returned
  metadata without inferring an orientation from unknown values.
- `attempts`: stable run-scoped ID, owning detection and batch, candidate order,
  preprocessing, submission operations, raw output, converted adapter result,
  and selection outcome. Tesseract token rows retain their original batch row
  indexes. Its assembled adapter text is explicitly derived.
- `results`: detection, batch links, selected attempt, original adapter result,
  and application text/confidence after existing unreadable-result handling.

Attempt IDs increase across OCR calls within a run. They do not incorporate
user-supplied detection IDs. Candidate order and the existing first-wins tie
rule are unchanged. Tracing never retries inference or splits an engine batch.

#### Attempt availability

Submission and output availability are independent. An attempt can be
`not_submitted`, submitted only to RapidOCR classification, or submitted to
recognition. An unavailable output includes a reason: not submitted, recognition
not submitted, recognition return unavailable, or batch response not mapped.
A batch failure belongs to the batch, rather than each crop.

RapidOCR fields distinguish `presence: absent` from a present null value.
Per-attempt strings distinguish empty, whitespace, and nonblank text. For
Tesseract, the original dictionary preserves absent keys, empty columns, token
strings, confidence strings, and original row order. Full batch evidence remains
authoritative when malformed output cannot be mapped safely to candidates.
If conversion aborts, selection remains unresolved even if an earlier candidate
was readable; no final adapter result was returned.

Non-finite numeric values use a `diagnostic_type: nonfinite_float` marker.
NumPy values retain a type marker where needed; unsupported objects retain an
explicit type/unavailable marker rather than being presented as engine text.
Strings, including Unicode and whitespace, are never normalized in raw evidence.

Both built-in adapters expose `recognize_many_traced`. Existing custom backends
continue receiving their original call signatures. Their adapter returns are
recorded, while internal attempts and raw engine evidence are marked unavailable.
Legacy batch success includes response validation: malformed containers, result
counts, entries, text, or nonblank confidence values fail the batch at
`interpret_output`, retaining the response and leaving final selection unresolved.
An exception from the backend call instead fails at `recognize_many`, with its
response unavailable. Existing exceptions and blank-text handling are unchanged.

### Source crops and recognition images

Diagnostics capture source crops and the actual image inputs used by both built-in
recognizers. Capture uses the original preprocessing pass; it adds no inference.
`OCRFrame`, chart contracts, backend signatures, and winner selection are unchanged.

`crop_text_bounds(image, detection, neighbors)` is the shared crop authority.
`crop_text_region` slices those bounds. Coordinates are source-image pixels with
top-left origin and exclusive right/bottom bounds. Titles retain the existing
neighbor-aware strip policy; other detections retain their padded boxes.

`diagnostics.json` has an additive `ocr_source_crops` collection. Each record has
a run-scoped `source_crop_id`, run and detection IDs, the original `detection_box`,
actual integer `crop_bounds`, crop policy, and an `image` descriptor. Collection
works from both the app and direct `run_ocr(..., diagnostics=collector)` calls.
Collecting the same image object and detection frame twice reuses the source
records; additional attempts still get their own input images.

Each existing `ocr_trace.attempts` entry adds `source_crop_id` and a
`recognition_image` descriptor. These descriptors record availability, PNG path,
width, height, original array shape, dtype, and channel layout. Generated numeric
paths live under `ocr_crops/` and `ocr_inputs/`; user-provided detection IDs never
enter filenames. Paths are relative to the existing diagnostic ZIP root.

- RapidOCR captures BGR images returned by `cls_and_rotate`, before
  `recognize_txt`. The batch's `classifier_output` explicitly snapshots available
  `cls_res` and elapsed time. Each attempt retains its classifier label/score in
  `preprocessing.classifier_orientation`, independently of the adapter's requested
  rotation. A classifier prediction alone is not proof that a rotation was applied.
- Tesseract captures grayscale/Otsu candidates used to create the lossless
  multipage TIFF. `tiff_page_number` links each attempt to its submitted page;
  rotation, variant, and upscale factor remain in `preprocessing`.

Engine input means the recognition API image or TIFF page, before internal engine
resizing/normalization. Custom backends retain their existing
unavailable-internal-attempt behavior and still receive source-crop evidence.

Capture errors record unavailable evidence and exceptions without replacing OCR
results. Prepared images remain available even if recognition later fails.
Classification failure leaves final recognition images unavailable. Submission
state remains separate from image availability. PNG capture accepts uint8/uint16
pixels; unsupported dtypes are explicitly unavailable instead of silently converted.

### Warning provenance

The run collector links OCR-related findings to the decisions that produced them.
Instrumentation observes the original analysis without changing recognition,
parsing, thresholds, calibration, braille support, or completeness results.

Every diagnostic event has a generated, run-scoped `event_id`.
`AnalysisDiagnostics.event()` returns that ID; callers that ignore its return
remain valid. Warning records retain code/stage deduplication and add
`evidence_ids` when evidence exists. Parameterized flags, such as
`legend_entry_skipped:low_confidence:Ghost`, remain intact on decision events.
Warnings outside this step may still have no evidence links.

The diagnostic-only `spec_sources` mapping connects `title`, `x_axis_title`,
`y_axis_title`, `category[j]`, `series_name[i]`, and `y_axis.ticks` to detection IDs
and producer events. Series mappings follow final legend ordering, including
series with no visible bars; partial mappings survive later failures. Successful
tick mappings describe the retained ticks in specification order. Calibration
input events retain all numeric candidates separately.

Join detection IDs to OCR attempts/results and `ocr_source_crops`
to inspect text and pixels. Absent detections and externally provided specs have
explicit unavailable source mappings rather than invented detection relationships.
`VerticalBarSpec`, `OCRFrame`, and completeness result contracts are unchanged.

#### Recorded decisions

- Confidence filtering records confidence and the floor. Deduplication records
  the first suppressing detection, measured IoU, and threshold. Tick spatial
  selection records candidate IDs, coordinates, accepted bounds, and exclusions.
- Consumer events preserve input strings before trimming, including Unicode and
  whitespace, for both reconstruction and legend matching. Missing consumer input
  remains distinct from present empty/whitespace text. Legacy mappings and region
  inputs are supported.
- Numeric parsing records each actual normalization step, normalized and numeric
  text, percent unit handling, suffix multiplier, converted value, and invalid or
  non-finite rejection. `raw_text` on the legacy tick event denotes the original
  consumer input, not the engine response; engine evidence stays in `ocr_trace`.
- Title decisions distinguish no detection, filtered detections, absent OCR, and
  empty consumed text. Existing cut-off checks record edge coordinates or ink
  counts and the actual thresholds.
- Category findings record the unreadable guard, median length/year comparisons,
  and supplied-text exemptions. Legend-entry decisions retain the first failing
  guard, confidence components, and any evaluated case-folded name comparison.
- Braille events capture the encoder's actual whitespace normalization. Rejections
  record unsupported characters and the existing encoder error. The original
  per-character validation remains unchanged; no additional encoder calls are made.
- Calibration events record inputs, tolerance calculation, candidate pair fits,
  residuals, scores, inliers, the selected candidate and actual competing inlier
  sets, then the least-squares result. Selection evidence is emitted before
  refitting, so exceptions retain it. The existing ambiguity semantics are preserved.

Completeness emits checks with field sources and earlier evidence links. Missing
items link to their checks; blockers preserve the existing early return and mark
later checks as unexecuted. Optional keyword-only `diagnostics` parameters on
consumer helpers, `check_completeness`, `cut_off_flags`, and `to_braille` preserve
existing calls and returns. The app forwards its collector only during analysis.

### Export and logging

Export builds the report, manifest, and JSON from one frozen snapshot and writes
the ZIP atomically. If formatting, serialization, or packaging fails, analysis and
review results remain available. The download is hidden and the app shows a
specific diagnostic-export error with the run ID.

Upload/example changes, clear, and analysis start reset the previous download and
run information. Archives remain in the app's existing temporary storage until
shutdown. Later review edits do not change the original analysis archive.
Logging is collector-owned, so concurrent runs do not capture each other's
messages. It does not collect arbitrary third-party engine stdout or global logs.

The terminal logs successful ZIP creation after the archive is finalized, then
one completion summary with outcome, duration, completeness status, warning count,
and JSON/ZIP paths (or `unavailable`). The warning count counts recorded warning
entries. These final messages remain in the collector's retained log but are not
inside the ZIP's `analysis.log`, which was captured before export completed.

### Generate local examples

These examples use deterministic fake OCR engines or supplied OCR strings. They
illustrate recorded inputs, selection, and warning provenance, rather than
real-engine recognition accuracy. No detector or OCR models, Tesseract executable,
or existing output files are needed.

1. Create a Python 3.12 environment if you do not already have one, then install
   the fixture and test dependencies:

   ```sh
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt Pillow pytest markdown-it-py
   ```

2. Generate both sets of examples:

   ```sh
   .venv/bin/python -m scripts.review_ocr_images
   .venv/bin/python -m scripts.review_parsing_provenance
   ```

3. Inspect OCR attempts and images: open `output/ocr-plan3-review/REVIEW.md` for an
   index, or open `output/ocr-plan3-review/rapidocr/gallery.html` and
   `output/ocr-plan3-review/tesseract/gallery.html` in a browser. Compare each
   source crop with its recognition inputs, raw output, and selected result.
   Each backend directory also contains diagnostic JSON and a ZIP archive.
4. Inspect warning provenance: open `output/ocr-plan4-review/REVIEW.md` and choose
   an example, such as `output/ocr-plan4-review/title_whitespace/review.html`.
   Open the HTML in a browser to inspect findings, text transformations, source
   crops, and decision evidence. In the same directory, read `verification.json`
   for comparisons with analysis run without diagnostics, or extract the ZIP
   and open `report.md` for the packaged report.

The scripts create these ignored directories. Rerun step 2 whenever you need the
examples; no generated files need to be committed or downloaded. The image
script compares saved recognition PNGs with submitted RapidOCR arrays or
Tesseract TIFF pages. The provenance script checks that diagnostics preserve
specifications, completeness results, and expected reconstruction errors.

### Run automated checks

1. Install the fixture and test dependencies from step 1 of
   [Generate local examples](#generate-local-examples). To include app integration
   checks, also install `requirements-app.txt` in the same environment.
2. Run the combined diagnostics checks:

   ```sh
   .venv/bin/python -m pytest \
     tests/test_diagnostics_bundle.py tests/test_analysis_evidence.py \
     tests/test_diagnostics.py tests/test_parsing_provenance.py \
     tests/test_ocr_trace.py tests/test_ocr_image_evidence.py \
     tests/test_ocr_crop.py tests/test_ocr.py tests/test_rapidocr_ocr.py \
     tests/test_tesseract_ocr.py tests/test_reconstruct.py \
     tests/test_completeness.py tests/test_legend_matcher.py -q
   ```

These tests cover complete and partial archives, links and image targets, exact
strings, OCR attempts and selected results, submitted pixels, parsing decisions,
warning attribution, and diagnostics-on/off parity. They also exercise malformed
responses, backend failures, export failures, immutable snapshots, run isolation,
and preserved app results.

Markdown rendering/link tests use `markdown-it-py` when installed; the production
reporter does not require a Markdown rendering dependency.
