# OCR attempt diagnostics (Plan 2)

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

## Reading availability

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

## Validation and checkpoint

Run:

```sh
.venv/bin/python -m unittest tests.test_ocr_trace tests.test_ocr tests.test_rapidocr_ocr tests.test_tesseract_ocr tests.test_diagnostics
```

The fake-engine tests compare tracing on/off for results, errors, call counts,
input order, and submitted pixels. They exercise Unicode, whitespace, empty and
missing output, malformed responses, confidence conversion, ties, preprocessing
failures, batch exceptions, legacy backends, run isolation, and JSON/ZIP exports.

Local review examples are generated under `output/ocr-plan2-review/REVIEW.md`.
They compare the fake responses with every attempt and final selection for both
backends, and include batch-failure diagnostic JSON and ZIP files. Review them
before continuing to Plan 3. These examples are fake-engine evidence, not an
assessment of real-engine recognition accuracy.

### Real-engine checkpoint

The downloaded run `42d75f60-eb8d-4ea5-a4c4-d837361312a6` was used to compare
both backends on the same 1296 × 1098 chart and the same detection boxes.

| Backend | Text detections | Attempts | Selected readable results | Evidence checks |
|---|---:|---:|---:|---|
| RapidOCR 3.10.0 / ONNX Runtime 1.31.1 | 12 | 14 | 11 | Passed |
| Tesseract, previously reviewed downloaded run | 12 | 28 | 11 | Passed |

RapidOCR ran one classification call and one recognition call using cached
models. The real response was independently captured before conversion in that
same recognition call. Every raw string and confidence, candidate association,
adapter result, winner, and final application result matched the trace.
Tesseract's previously retained batch fields were rechecked against all token
rows, derived results, selected attempts, and application results; no fresh
Tesseract inference was performed.

The local [real-engine review](../output/ocr-plan2-review/real-engines/REVIEW.md)
contains every attempt, side-by-side final results, verification JSON, engine
responses, and the RapidOCR diagnostic ZIP. These files live under ignored
`output/`; this document retains the checkpoint summary in the repository.

The backends differ on `det-00011`: RapidOCR selected `0`, while Tesseract selected
`=` from a rotated candidate. Both selections faithfully follow their engine
responses and existing confidence rule. Changing that OCR behavior is outside
Plan 2. The checkpoint is ready for user sign-off before proceeding to Plan 3.
