# OCR parsing and warning provenance (Plan 4)

The user-facing report and diagnostic download are now described in
[Analysis diagnostics](analysis_diagnostics.md).

The run collector now links existing OCR-related findings to the decisions that
produced them. Instrumentation observes the original analysis. It does not change
recognition, numeric parsing, thresholds, warning wording, calibration selection,
braille support, or completeness results. Review edits are not written back into
the original run record.

## Following evidence

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

Join detection IDs to Plan 2 OCR attempts/results and Plan 3 `ocr_source_crops`
to inspect text and pixels. Absent detections and externally provided specs have
explicit unavailable source mappings rather than invented detection relationships.
`VerticalBarSpec`, `OCRFrame`, and completeness result contracts are unchanged.

## Recorded decisions

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

## Validation and review

```sh
.venv/bin/python -m pytest tests/test_parsing_provenance.py tests/test_analysis_evidence.py tests/test_reconstruct.py tests/test_completeness.py tests/test_legend_matcher.py tests/test_diagnostics.py -q
.venv/bin/python -m scripts.review_parsing_provenance
```

The focused tests verify attribution, measured thresholds, original strings and
transformation chains, numeric failures, ambiguous fits, partial reconstruction
failures, final series ordering, source availability, warning deduplication,
run isolation, and JSON/ZIP round trips. They compare diagnostic and ordinary
specs, completeness results, exceptions, and braille call counts.

The [review checkpoint](../output/ocr-plan4-review/REVIEW.md) links synthetic
examples with source crops, findings, navigable decision evidence, diagnostic
JSON/ZIPs, and parity verification. The generator uses fixture detections and
consumer OCR strings without loading ML models. These examples demonstrate
provenance, not real-engine accuracy. Artifacts are ignored and reproducible.

Plan 4 stops at this checkpoint. Unrelated warning expansion, CLI diagnostic
export, review-edit tracing, and Plan 5 report/download changes remain deferred.
