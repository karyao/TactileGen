# End-to-end audit

Checkpoint `models/yolo_bars_hv/best.pt`, OCR `rapidocr`, image size 640, annotations `data/plotqa/test/annotations.json`, `--yes --strict`.

## Vertical bars (4876 charts)

- (a) built (exit 0): 4462/4876 = **91.5%**
- (b) of built, content correct: 3895/4462 = 87.3%
- (c) built AND correct: 3895/4876 = **79.9%**
- built but wrong: 567; blocked: 414

- charts with a compared numeric value above tolerance: 221
- charts with bar_values_off as primary group: 4

### Failure groups

| group | charts | blocked | built wrong | % of failures | % of all | why | examples |
|---|---|---|---|---|---|---|---|
| tiny_bars_undetected | 351 | 38 | 313 | 35.8% | 7.2% | every missed bar is near-zero; the detector does not find it | 22419, 22464, 22478 |
| real_bars_undetected | 219 | 119 | 100 | 22.3% | 4.5% | at least one clearly visible bar has no detection | 22434, 22451, 22453 |
| false_block_gate | 97 | 97 | 0 | 9.9% | 2.0% | the content is correct but a BANA/completeness check blocked it | 22438, 22496, 22517 |
| title_axis_title_ocr | 96 | 96 | 0 | 9.8% | 2.0% | the title or an axis title could not be read | 22514, 22516, 22547 |
| category_label_ocr | 74 | 3 | 71 | 7.5% | 1.5% | bars are right but a category label is misread or unreadable | 22463, 22490, 22502 |
| series_name_mismatch | 71 | 0 | 71 | 7.2% | 1.5% | a matched series name differs from ground truth | 22426, 22567, 22600 |
| other_rare | 43 | 43 | 0 | 4.4% | 0.9% | pairing, extra bars, rescued values, interpretation errors, layout | 22437, 22596, 22767 |
| series_legend_colour_match | 11 | 11 | 0 | 1.1% | 0.2% | bars do not resolve to legend colours, so series are ambiguous | 22625, 23006, 23469 |
| bar_value_unread | 10 | 5 | 5 | 1.0% | 0.2% | an expected value could not be read | 24493, 24583, 25646 |
| y_axis_calibration | 5 | 2 | 3 | 0.5% | 0.1% | value-axis ticks dropped, uneven, or a value clipped at the axis | 22676, 23285, 25450 |
| bar_values_off | 4 | 0 | 4 | 0.4% | 0.1% | a compared numeric value is off by more than 1% of the axis maximum | 22738, 22838, 23509 |

### Primary causes

| cause | charts | % of all |
|---|---|---|
| detector_missing_bars | 570 | 11.7% |
| false_block | 97 | 2.0% |
| ocr_title_or_axis_title | 96 | 2.0% |
| ocr_category_text | 74 | 1.5% |
| series_name_mismatch | 71 | 1.5% |
| no_spec_interpretation_error | 29 | 0.6% |
| category_pairing | 14 | 0.3% |
| series_separation | 11 | 0.2% |
| bar_value_unread | 10 | 0.2% |
| y_calibration | 5 | 0.1% |
| other_wrong_values | 4 | 0.1% |

### Most common block reasons

| block reasons | charts |
|---|---|
| bars_mostly_missed | 194 |
| title | 59 |
| y_axis_title | 37 |
| value_clipped_at_axis | 29 |
| error:Error: tesseract is not installed or it's not in your PATH. See README file for  | 28 |
| category_ambiguous;category_count_mismatch | 18 |
| series_ambiguous | 10 |
| bana:heading_placement | 9 |
| bar_colour_unaccounted | 8 |
| title;y_axis_title | 4 |
| series_name[n] | 4 |
| bars_mostly_missed;category_count_mismatch;series_ambiguous;values_unread | 4 |

### Missed bars

- true bars: 52650, missed after filtering: 4343
- of the missed: near-zero 3384, under 4 px 3185
- why missed: never detected 4183, below confidence 43, removed as duplicate/outside plot 117
- charts with extra bars: 73

## Example sheets

Green = true bars, red = bars the pipeline used.

- vertical / bar_value_unread: `sheets/vertical_bar_value_unread.png`
- vertical / bar_values_off: `sheets/vertical_bar_values_off.png`
- vertical / category_label_ocr: `sheets/vertical_category_label_ocr.png`
- vertical / false_block_gate: `sheets/vertical_false_block_gate.png`
- vertical / other_rare: `sheets/vertical_other_rare.png`
- vertical / real_bars_undetected: `sheets/vertical_real_bars_undetected.png`
- vertical / series_legend_colour_match: `sheets/vertical_series_legend_colour_match.png`
- vertical / series_name_mismatch: `sheets/vertical_series_name_mismatch.png`
- vertical / tiny_bars_undetected: `sheets/vertical_tiny_bars_undetected.png`
- vertical / title_axis_title_ocr: `sheets/vertical_title_axis_title_ocr.png`
- vertical / y_axis_calibration: `sheets/vertical_y_axis_calibration.png`
