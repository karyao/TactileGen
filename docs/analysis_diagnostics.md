# Downloading and reading analysis diagnostics

Upload a chart, select **Analyze chart**, then expand **Analysis details** and select
**Download analysis diagnostics (ZIP)**. Extract the entire archive and open
`report.md` in a Markdown viewer so its relative links and image previews work.
The run ID shown in the app also appears in the archive name, manifest, report,
JSON evidence, and run log.

This is evidence from the actual analysis, including real OCR when that stage ran.
A failed analysis also produces an archive containing whatever evidence was
collected. An early validation failure has no detection or OCR output; the report
and manifest explain which evidence is unavailable.

## Finding the cause

Start with **Findings** in `report.md`. Each supported finding explains the failed
check and links to recorded decision evidence. The report surfaces the original
consumer text, text after trimming and numeric parsing, measured confidences and
thresholds, and the source detection's crop. **OCR attempts** links lead directly
to the corresponding region, its recognition inputs, raw engine output, and
selected attempt. Strings use JSON escapes to distinguish tabs, newlines, empty
strings, and ordinary spaces while preserving Unicode.

For example, a dropped tick can show that OCR returned `"="`, that numeric parsing
rejected it, and which source crop and recognition attempts produced that text.
Missing-title evidence distinguishes no detection, filtered detections, missing
OCR, and empty consumed text. Calibration evidence includes the tick inputs,
candidate fits, residuals, selected inliers, competing sets, and final fit.

Some warnings outside the provenance coverage still have no detailed explanation;
the report marks this explicitly. Checks skipped by an earlier blocker are not
presented as executed. Reports describe recorded checks, not recognition accuracy
or verified ground truth. The app does not highlight issue regions interactively.

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

## Failure handling and lifecycle

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

## Validation and review checkpoint

```sh
.venv/bin/python -m pytest tests/test_diagnostics_bundle.py tests/test_analysis_evidence.py tests/test_diagnostics.py tests/test_parsing_provenance.py tests/test_ocr_trace.py tests/test_ocr_image_evidence.py -q
```

Markdown rendering/link tests use `markdown-it-py` when installed; the production
reporter does not require a Markdown rendering dependency. Tests cover complete
and partial archives, contract loaders, links and image targets, exact strings,
export failures, immutable snapshots, run-local logs, and preserved app results.

The [Plan 5 review checkpoint](../output/ocr-plan5-review/REVIEW.md) contains actual
UI downloads from real Tesseract analysis and an early validation failure, plus
real RapidOCR packaging evidence on the same detection frame. It distinguishes
browser downloads from direct backend verification and records outcomes and
archive checks. The real chart may contain reconstruction blockers; those are
part of the evidence, not suppressed for this checkpoint.

Earlier evidence formats are described in [OCR attempts](ocr_attempt_diagnostics.md),
[OCR images](ocr_image_diagnostics.md), and [warning provenance](ocr_warning_provenance.md).
