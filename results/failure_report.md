# TactileGen vertical-bar failure audit

Population: all 11242 PlotQA test vbar_categorical charts, cache of commit 2412bb2e (`--yes --strict`).

## Rates

- (a) emitted without blocking: 8574/11242 = 76.3%  <- the 76.3% baseline
- (b) of emitted, content matches ground truth: 7466/8574 = 87.1%
- (c) emitted AND correct, of all charts: 7466/11242 = 66.4%
- emitted but wrong: 1108 (9.9% of all); blocked: 2668 (23.7%); total not (emitted AND correct): 3776 (33.6%)
- 'correct' = series count, category count, category text (normalised case/whitespace), series names and every value within 1% of the GT y-axis maximum. Titles/axis titles are not compared. Also lenient variants below.

- lenient: emitted with values within 1% & series aligned (text ignored): 7602/8574 = 88.7%
- lenient: as above with category text compared alphanumerics-only (ignores hyphen/space/case): 7524/8574 = 87.8%

## Failure causes (one primary cause per chart)

Priority order (first match wins): no_spec_interpretation_error > detector_missing_bars > detector_extra_bars > ocr_category_text > ocr_title_or_axis_title > y_calibration > series_false_alarm_single_series > series_separation > category_pairing > bar_rescue_wrong > other_wrong_values > strict_bana_layout. `false_block` = blocked although content is correct (exit 1/3 only).

| primary_cause | charts | blocked | emitted_wrong | % of failures | % of all charts |
|---|---|---|---|---|---|
| detector_missing_bars | 1921 | 994 | 927 | 50.9 | 17.1 |
| series_separation | 770 | 770 | 0 | 20.4 | 6.8 |
| ocr_category_text | 421 | 327 | 94 | 11.1 | 3.7 |
| ocr_title_or_axis_title | 218 | 218 | 0 | 5.8 | 1.9 |
| false_block | 215 | 215 | 0 | 5.7 | 1.9 |
| other_wrong_values | 110 | 25 | 85 | 2.9 | 1.0 |
| y_calibration | 64 | 62 | 2 | 1.7 | 0.6 |
| category_pairing | 26 | 26 | 0 | 0.7 | 0.2 |
| detector_extra_bars | 10 | 10 | 0 | 0.3 | 0.1 |
| no_spec_interpretation_error | 9 | 9 | 0 | 0.2 | 0.1 |
| strict_bana_layout | 8 | 8 | 0 | 0.2 | 0.1 |
| series_false_alarm_single_series | 4 | 4 | 0 | 0.1 | 0.0 |

Because detector causes outrank the others, a chart blocked by `series_ambiguous` that also lost a bar is counted under `detector_missing_bars`. The block-reason view below is not subject to that ordering.

### Blocked charts by exact set of block reasons (multi-flag overlaps)

| block reasons (all raised, ';'-joined) | charts |
|---|---|
| category_count_mismatch;series_ambiguous | 841 |
| series_ambiguous | 684 |
| category[n] | 373 |
| bana:label_owner_distance | 235 |
| y_axis_title | 122 |
| title | 114 |
| value_clipped_at_axis | 61 |
| bana:bar_width;bana:label_owner_distance | 47 |
| category_ambiguous;category_count_mismatch | 41 |
| category_ambiguous;category_count_mismatch;series_ambiguous | 36 |
| bana:heading_placement | 24 |
| y_tick_leading_zero | 23 |
| series_name[n] | 12 |
| interpretation_error | 9 |
| category_count_mismatch;series_ambiguous;value_clipped_at_axis | 8 |

### Most common spec flag sets

| flag set (spec flags, any status) | charts |
|---|---|
| bars_rescued;value_inferred_zero | 1164 |
| legend_entry_skipped;series_ambiguous | 622 |
| category_count_mismatch;series_ambiguous | 503 |
| category_count_mismatch;legend_entry_skipped;series_ambiguous | 288 |
| y_ticks_dropped;y_ticks_not_arithmetic | 265 |
| bars_rescued | 167 |
| category_unreadable | 150 |
| legend_order_differs | 71 |
| y_ticks_dropped | 63 |
| y_axis_title_cut_off | 63 |
| bars_rescued;legend_order_differs;value_inferred_zero | 53 |
| y_ticks_not_arithmetic | 46 |

## series_ambiguous

- Verified: 1577 charts are blocked with the series_ambiguous tag (summary.md said 1577); the flag is on 1577 specs; all are exit 3.
- Rate: 16.8% of multi-series charts (1566/9334), 0.58% of single-series (11/1908).
- False alarms (GT is a single series): 11. Flagged charts whose reconstructed content is nevertheless correct: 0/1577.

Mechanism that raised the flag (measured, from legend-mapping statuses of the detected bars):

| mechanism | charts |
|---|---|
| unresolved:weak_match | 1068 |
| unresolved:ambiguous | 422 |
| unresolved:ambiguous+weak_match | 76 |
| unresolved:no_legend | 11 |

Primary reason by priority (single_series > extra bars > missing bars > conflicting swatch > legend not detected > legend count mismatch > similar GT colours > unresolved status):

| reason | charts | pct |
|---|---|---|
| missing_detected_bars | 639 | 40.5 |
| unresolved_bars:weak_match | 418 | 26.5 |
| similar_series_colors | 411 | 26.1 |
| extra_detected_bars | 93 | 5.9 |
| single_series_false_alarm | 11 | 0.7 |
| unresolved_bars:ambiguous | 3 | 0.2 |
| legend_count_mismatch | 2 | 0.1 |

Where the 1577 series_ambiguous charts land in the primary-cause groups (the priority order puts bar-detection first):

| group (primary cause) | series_ambiguous charts |
|---|---|
| series_legend_colour_match | 773 |
| tiny_bars_undetected | 657 |
| real_bars_undetected | 72 |
| category_label_ocr | 43 |
| y_axis_calibration | 22 |
| other_rare | 10 |

Co-factors (each over all flagged charts; not exclusive; association, not proven cause):

- detected legend swatches == GT legend entries: 1576/1577
- any GT bar not detected: 46.2% of flagged vs 23.6% of non-flagged multi-series charts
- any extra detected bar: 6.0% vs 0.5%
- closest pair of GT series colours < 80 RGB distance: 42.2% of flagged (n=1566); overall multi-series baseline not computed (unverified)

## Extra-bar detections

Charts with more surviving bars than GT bars (unmatched at IoU 0.5): 145/11242; false bars: 520.
- confidence of false bars (p10/p50/p90): [0.371, 0.951, 0.965] (n=520)
- confidence of true bars in those charts: [0.728, 0.947, 0.964] (n=1044)
- confidence of true bars in all charts: [0.912, 0.953, 0.968] (n=119107)

Split (seed 0 over sorted chart ids): tuning n=2246, report n=8987 (charts with a spec).

Rule: remove a surviving bar when `conf < tau` AND (A: its legend assignment is not `matched` / B: the chart is flagged series_ambiguous); legend matching + reconstruct are re-run from cached detections (no detector re-run) and a chart 'passes' when completeness is READY and content matches GT (pre-layout). fixed = was not passing, now passes; broke = was passing, now not. **Not enabled in the pipeline.**

Selected on tuning split (max net = fixed - broke): A_bar_unmatched_to_legend: tau=0.85, B_chart_series_ambiguous: tau=0.8

Reading the sweep: most fixes remove true bars (`fixed_true_bar_removed`), not false ones. Dropping unresolved bars clears `series_ambiguous`, and the rescue step re-measures the removed bars from pixels. Extra bars exist in only 145 charts and have high confidence, so a confidence threshold cannot target them.

All thresholds, 80% report split (columns tune_* = 20% tuning split):

| variant | tau | affected | fixed | broke | net | fixed_only_false_bars_removed | fixed_true_bar_removed | fixed_chart_had_extras | true_removed | extra_removed | tune_fixed | tune_broke | tune_net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_bar_unmatched_to_legend | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_bar_unmatched_to_legend | 0.35 | 77 | 14 | 0 | 14 | 3 | 11 | 3 | 59 | 31 | 5 | 0 | 5 |
| A_bar_unmatched_to_legend | 0.4 | 139 | 35 | 0 | 35 | 6 | 29 | 6 | 119 | 54 | 7 | 0 | 7 |
| A_bar_unmatched_to_legend | 0.45 | 188 | 54 | 0 | 54 | 9 | 45 | 13 | 174 | 77 | 15 | 0 | 15 |
| A_bar_unmatched_to_legend | 0.5 | 245 | 79 | 0 | 79 | 9 | 70 | 16 | 256 | 83 | 19 | 0 | 19 |
| A_bar_unmatched_to_legend | 0.55 | 312 | 107 | 0 | 107 | 9 | 98 | 20 | 371 | 88 | 24 | 0 | 24 |
| A_bar_unmatched_to_legend | 0.6 | 384 | 137 | 0 | 137 | 9 | 128 | 21 | 502 | 90 | 35 | 0 | 35 |
| A_bar_unmatched_to_legend | 0.65 | 450 | 180 | 0 | 180 | 9 | 171 | 24 | 640 | 92 | 45 | 0 | 45 |
| A_bar_unmatched_to_legend | 0.7 | 517 | 217 | 0 | 217 | 9 | 208 | 29 | 770 | 92 | 52 | 0 | 52 |
| A_bar_unmatched_to_legend | 0.75 | 580 | 254 | 0 | 254 | 9 | 245 | 29 | 859 | 93 | 63 | 0 | 63 |
| A_bar_unmatched_to_legend | 0.8 | 649 | 296 | 0 | 296 | 9 | 287 | 30 | 974 | 94 | 73 | 0 | 73 |
| A_bar_unmatched_to_legend | 0.85 | 712 | 324 | 0 | 324 | 9 | 315 | 30 | 1085 | 95 | 77 | 0 | 77 |
| A_bar_unmatched_to_legend | 0.9 | 818 | 325 | 0 | 325 | 9 | 316 | 30 | 1286 | 96 | 77 | 0 | 77 |
| A_bar_unmatched_to_legend | 0.95 | 1535 | 342 | 50 | 292 | 9 | 333 | 30 | 3888 | 105 | 84 | 9 | 75 |
| B_chart_series_ambiguous | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_chart_series_ambiguous | 0.35 | 86 | 14 | 0 | 14 | 3 | 11 | 3 | 70 | 31 | 5 | 0 | 5 |
| B_chart_series_ambiguous | 0.4 | 154 | 35 | 0 | 35 | 6 | 29 | 6 | 138 | 54 | 7 | 0 | 7 |
| B_chart_series_ambiguous | 0.45 | 204 | 54 | 0 | 54 | 9 | 45 | 13 | 202 | 77 | 15 | 0 | 15 |
| B_chart_series_ambiguous | 0.5 | 261 | 79 | 0 | 79 | 9 | 70 | 16 | 287 | 83 | 19 | 0 | 19 |
| B_chart_series_ambiguous | 0.55 | 324 | 107 | 0 | 107 | 9 | 98 | 20 | 407 | 88 | 24 | 0 | 24 |
| B_chart_series_ambiguous | 0.6 | 394 | 135 | 0 | 135 | 9 | 126 | 21 | 542 | 90 | 35 | 0 | 35 |
| B_chart_series_ambiguous | 0.65 | 454 | 174 | 0 | 174 | 9 | 165 | 24 | 700 | 92 | 45 | 0 | 45 |
| B_chart_series_ambiguous | 0.7 | 511 | 210 | 0 | 210 | 9 | 201 | 29 | 854 | 92 | 51 | 0 | 51 |
| B_chart_series_ambiguous | 0.75 | 569 | 240 | 0 | 240 | 8 | 232 | 29 | 1027 | 92 | 61 | 0 | 61 |
| B_chart_series_ambiguous | 0.8 | 623 | 270 | 0 | 270 | 8 | 262 | 30 | 1254 | 92 | 66 | 0 | 66 |
| B_chart_series_ambiguous | 0.85 | 685 | 276 | 0 | 276 | 8 | 268 | 30 | 1744 | 93 | 64 | 0 | 64 |
| B_chart_series_ambiguous | 0.9 | 765 | 253 | 0 | 253 | 7 | 246 | 28 | 2700 | 93 | 57 | 0 | 57 |
| B_chart_series_ambiguous | 0.95 | 1028 | 125 | 0 | 125 | 1 | 124 | 18 | 8784 | 94 | 26 | 0 | 26 |

## Failure groups

| group | charts | % of failures | why it fails | estimated fix | examples (typical/borderline/extreme) |
|---|---|---|---|---|---|
| tiny_bars_undetected | 1357 | 35.9 | Every missed bar is near-zero (<5% of axis max or <=5 px tall); YOLO does not find it and the rescue step either guesses 0 or the count mismatch/series flag blocks the chart. | detector retrain (small-bar data) or pixel heuristic | 28178/22720/29307 |
| real_bars_undetected | 564 | 14.9 | At least one clearly visible bar (not near-zero) has no surviving detection. | detector retrain | 26539/23382/30384 |
| series_legend_colour_match | 774 | 20.5 | Detected legend swatches do not resolve to bar colours (weak_match/ambiguous), so series are unnamed/ambiguous. | pixel/heuristic fix (colour matching) - hypothesis | 31400/22510/30986 |
| category_label_ocr | 421 | 11.1 | Bars are right but a category label is misread, hyphen-dropped or unreadable. | needs VLM (or better OCR) - hypothesis | 22752/24998/32560 |
| title_axis_title_ocr | 218 | 5.8 | Title / axis title / series name text could not be read, so the pipeline asks for input. | needs VLM (or better OCR) - hypothesis | 25500/27969/32032 |
| false_block_gate | 215 | 5.7 | Pipeline's answer is correct but a BANA/completeness gate blocked it. | pixel/heuristic fix (rule change) | 28769/26418/29450 |
| y_axis_calibration | 64 | 1.7 | Y ticks dropped / not arithmetic / leading zero / value clipped at axis end. | pixel/heuristic fix - hypothesis | 32420/30242/25334 |
| other_rare | 163 | 4.3 | Pairing ambiguity, extra bars, interpretation errors, layout errors, residual wrong values. | mixed; long tail | 30170/22838/28013 |

Image paths: `data/png_test/png/<id>.png`; contact sheets in `results/failure_groups/<group>/contact_sheet.png` (green dashed = GT bar boxes, red = surviving YOLO bars). Group assignment follows the primary-cause priority order; 'estimated fix' is a hypothesis, not tested.

## Upper bounds (no fix implemented; assumes each group is fully fixed and the chart then emits correctly)

| group | charts | sole-cause charts (strict floor) | gain if fixed alone, upper (pts) | gain, sole-cause floor (pts) | effort (my estimate) | upper gain per effort-step | (c) after | heuristic-fixable? | VLM-plausible? |
|---|---|---|---|---|---|---|---|---|---|
| tiny_bars_undetected | 1357 | 553 | 12.07 | 4.92 | M-L: detector retrain or pixel scan for near-zero bars | 3.02 | 78.5% | yes | no/unknown |
| series_legend_colour_match | 774 | 774 | 6.88 | 6.88 | M: colour matching / drop weak bars then rescue | 2.29 | 73.3% | yes | no/unknown |
| false_block_gate | 215 | 180 | 1.91 | 1.6 | S: rule change (key-runover fix exists on a branch) | 1.91 | 68.3% | yes | no/unknown |
| real_bars_undetected | 564 | 381 | 5.02 | 3.39 | L: detector retrain + data | 1.0 | 71.4% | no | no/unknown |
| category_label_ocr | 421 | 351 | 3.74 | 3.12 | M-L: OCR tuning or VLM text read | 0.94 | 70.2% | no | yes (hypothesis) |
| title_axis_title_ocr | 218 | 211 | 1.94 | 1.88 | M: OCR tuning or VLM text read | 0.48 | 68.4% | no | yes (hypothesis) |
| other_rare | 163 | 151 | 1.45 | 1.34 | L: long tail | 0.29 | 67.9% | no | no/unknown |
| y_axis_calibration | 64 | 41 | 0.57 | 0.36 | S-M: tick/clip heuristics | 0.28 | 67.0% | yes | no/unknown |

- All heuristic-fixable groups fixed: +21.4 pts -> (c) 87.8% (2410 charts).
- All groups fixed would reach 100% by construction; the long-tail group is `other_rare`.
- Needs-VLM groups (hypothesis): 639 charts = 16.9% of failures = +5.7 pts of (c).
- False blocks: 215/2668 blocked charts (8.1%) have correct content, = +1.9 pts of (c); 203 of them are BANA `label_owner_distance` (verified: with the fix commit 189c31c8 applied, all 203 exit 0, both on 2412bb2e + that commit and on the branch tip).
- Completeness-blocker (exit 3) false blocks: 4/1719.

- Of 421 category_label_ocr charts, 33 differ only by punctuation/space/case (labels_ok_loose) and 388 are real misreads or unreadable.