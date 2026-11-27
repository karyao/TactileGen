# End-to-end audit

Checkpoint `models/yolo_bars_hv/best.pt`, OCR `rapidocr`, image size 640, annotations `data/plotqa/test/annotations.json`, `--yes --strict`.

## Vertical bars (11242 charts)

- (a) built (exit 0): 10340/11242 = **92.0%**
- (b) of built, content correct: 8869/10340 = 85.8%
- (c) built AND correct: 8869/11242 = **78.9%**
- built but wrong: 1471; blocked: 902

- charts with a compared numeric value above tolerance: 574
- charts with bar_values_off as primary group: 10

### Failure groups

| group | charts | blocked | built wrong | % of failures | % of all | why | examples |
|---|---|---|---|---|---|---|---|
| tiny_bars_undetected | 973 | 124 | 849 | 41.0% | 8.7% | every missed bar is near-zero; the detector does not find it | 22419, 22464, 22478 |
| real_bars_undetected | 560 | 304 | 256 | 23.6% | 5.0% | at least one clearly visible bar has no detection | 22434, 22451, 22453 |
| title_axis_title_ocr | 180 | 180 | 0 | 7.6% | 1.6% | the title or an axis title could not be read | 22514, 22516, 22547 |
| category_label_ocr | 179 | 11 | 168 | 7.5% | 1.6% | bars are right but a category label is misread or unreadable | 22463, 22490, 22502 |
| series_name_mismatch | 175 | 0 | 175 | 7.4% | 1.6% | a matched series name differs from ground truth | 22426, 22567, 22600 |
| false_block_gate | 163 | 163 | 0 | 6.9% | 1.4% | the content is correct but a BANA/completeness check blocked it | 22438, 22496, 22517 |
| other_rare | 77 | 77 | 0 | 3.2% | 0.7% | pairing, extra bars, rescued values, interpretation errors, layout | 22437, 22596, 22767 |
| series_legend_colour_match | 25 | 25 | 0 | 1.1% | 0.2% | bars do not resolve to legend colours, so series are ambiguous | 22625, 23006, 23469 |
| y_axis_calibration | 16 | 13 | 3 | 0.7% | 0.1% | value-axis ticks dropped, uneven, or a value clipped at the axis | 22676, 23285, 25450 |
| bar_value_unread | 15 | 5 | 10 | 0.6% | 0.1% | an expected value could not be read | 24493, 24583, 25646 |
| bar_values_off | 10 | 0 | 10 | 0.4% | 0.1% | a compared numeric value is off by more than 1% of the axis maximum | 22738, 22838, 23509 |

### Primary causes

| cause | charts | % of all |
|---|---|---|
| detector_missing_bars | 1533 | 13.6% |
| ocr_title_or_axis_title | 180 | 1.6% |
| ocr_category_text | 179 | 1.6% |
| series_name_mismatch | 175 | 1.6% |
| false_block | 163 | 1.4% |
| no_spec_interpretation_error | 73 | 0.6% |
| series_separation | 25 | 0.2% |
| y_calibration | 16 | 0.1% |
| bar_value_unread | 15 | 0.1% |
| other_wrong_values | 10 | 0.1% |
| category_pairing | 4 | 0.0% |

### Most common block reasons

| block reasons | charts |
|---|---|
| bars_mostly_missed | 423 |
| title | 115 |
| y_axis_title | 77 |
| error:Error: tesseract is not installed or it's not in your PATH. See README file for  | 72 |
| value_clipped_at_axis | 67 |
| series_ambiguous | 29 |
| category_count_mismatch;series_ambiguous;values_unread | 28 |
| bana:heading_placement | 20 |
| bars_mostly_missed;category_count_mismatch;series_ambiguous;values_unread | 16 |
| series_name[n] | 15 |
| bar_colour_unaccounted | 14 |
| category[n] | 8 |

### Missed bars

- true bars: 131623, missed after filtering: 12376
- of the missed: near-zero 9945, under 4 px 9205
- why missed: never detected 11987, below confidence 150, removed as duplicate/outside plot 239
- charts with extra bars: 163

## Horizontal bars (11292 charts)

- (a) built (exit 0): 9784/11292 = **86.6%**
- (b) of built, content correct: 8500/9784 = 86.9%
- (c) built AND correct: 8500/11292 = **75.3%**
- built but wrong: 1284; blocked: 1508

- charts with a compared numeric value above tolerance: 863
- charts with bar_values_off as primary group: 47

### Failure groups

| group | charts | blocked | built wrong | % of failures | % of all | why | examples |
|---|---|---|---|---|---|---|---|
| tiny_bars_undetected | 908 | 310 | 598 | 32.5% | 8.0% | every missed bar is near-zero; the detector does not find it | 10005, 10014, 10017 |
| real_bars_undetected | 642 | 319 | 323 | 23.0% | 5.7% | at least one clearly visible bar has no detection | 10041, 10046, 10053 |
| false_block_gate | 470 | 470 | 0 | 16.8% | 4.2% | the content is correct but a BANA/completeness check blocked it | 10011, 10068, 10071 |
| title_axis_title_ocr | 226 | 226 | 0 | 8.1% | 2.0% | the title or an axis title could not be read | 10108, 10114, 10181 |
| series_name_mismatch | 205 | 0 | 205 | 7.3% | 1.8% | a matched series name differs from ground truth | 10138, 10301, 10443 |
| other_rare | 109 | 100 | 9 | 3.9% | 1.0% | pairing, extra bars, rescued values, interpretation errors, layout | 10038, 10242, 10284 |
| category_label_ocr | 84 | 7 | 77 | 3.0% | 0.7% | bars are right but a category label is misread or unreadable | 10204, 10221, 10457 |
| bar_values_off | 47 | 6 | 41 | 1.7% | 0.4% | a compared numeric value is off by more than 1% of the axis maximum | 10022, 11099, 11120 |
| series_legend_colour_match | 39 | 39 | 0 | 1.4% | 0.3% | bars do not resolve to legend colours, so series are ambiguous | 10205, 10259, 10300 |
| y_axis_calibration | 35 | 10 | 25 | 1.3% | 0.3% | value-axis ticks dropped, uneven, or a value clipped at the axis | 10384, 10593, 12009 |
| bar_value_unread | 27 | 21 | 6 | 1.0% | 0.2% | an expected value could not be read | 10288, 10323, 10337 |

### Primary causes

| cause | charts | % of all |
|---|---|---|
| detector_missing_bars | 1550 | 13.7% |
| false_block | 470 | 4.2% |
| ocr_title_or_axis_title | 226 | 2.0% |
| series_name_mismatch | 205 | 1.8% |
| detector_extra_bars | 85 | 0.8% |
| ocr_category_text | 84 | 0.7% |
| other_wrong_values | 47 | 0.4% |
| series_separation | 37 | 0.3% |
| y_calibration | 35 | 0.3% |
| bar_value_unread | 27 | 0.2% |
| category_pairing | 12 | 0.1% |
| bar_rescue_wrong | 8 | 0.1% |
| no_spec_interpretation_error | 4 | 0.0% |
| series_false_alarm_single_series | 2 | 0.0% |

### Most common block reasons

| block reasons | charts |
|---|---|
| category_ambiguous;category_count_mismatch | 279 |
| bars_mostly_missed | 276 |
| title | 246 |
| category_count_mismatch;series_ambiguous;values_unread | 145 |
| bar_colour_unaccounted | 142 |
| bar_colour_unaccounted;category_ambiguous;category_count_mismatch | 140 |
| bars_mostly_missed;category_count_mismatch;series_ambiguous;values_unread | 84 |
| error:The tactile layout rejected this chart: no category-label mode could be laid out | 78 |
| series_ambiguous | 23 |
| series_name[n] | 20 |
| category[n] | 17 |
| value_clipped_at_axis | 11 |

### Missed bars

- true bars: 131653, missed after filtering: 10823
- of the missed: near-zero 7906, under 4 px 6913
- why missed: never detected 10613, below confidence 195, removed as duplicate/outside plot 15
- charts with extra bars: 963

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
- horizontal / bar_value_unread: `sheets/horizontal_bar_value_unread.png`
- horizontal / bar_values_off: `sheets/horizontal_bar_values_off.png`
- horizontal / category_label_ocr: `sheets/horizontal_category_label_ocr.png`
- horizontal / false_block_gate: `sheets/horizontal_false_block_gate.png`
- horizontal / other_rare: `sheets/horizontal_other_rare.png`
- horizontal / real_bars_undetected: `sheets/horizontal_real_bars_undetected.png`
- horizontal / series_legend_colour_match: `sheets/horizontal_series_legend_colour_match.png`
- horizontal / series_name_mismatch: `sheets/horizontal_series_name_mismatch.png`
- horizontal / tiny_bars_undetected: `sheets/horizontal_tiny_bars_undetected.png`
- horizontal / title_axis_title_ocr: `sheets/horizontal_title_axis_title_ocr.png`
- horizontal / y_axis_calibration: `sheets/horizontal_y_axis_calibration.png`
