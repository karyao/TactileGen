# OCR baselines on PMC bar charts

Tesseract and EasyOCR, scored on ground-truth text crops from
`data/pmc_yolo_bars`. Scope is vertical bar charts only. This measures OCR
alone: the boxes are ground truth, so detector errors are not included.

## Protocol

Shared by every engine (`src/baseline/ocr_eval.py`):

- **Crops.** Each ground-truth text box of these classes: `x_title`, `y_title`,
  `chart_title`, `xlabel` (category labels), `ylabel` (value-axis labels, whatever the
  chart's orientation), `legend_label`, `legend_title`, `value_label` (numbers on bars).
  Text comes from the plots_detection COCO annotations. Only charts that
  `src/dataset/bar_orientation.py` calls vertical are used: 150 in val, 139 in test.
- **Preprocessing.** 4 px padding, grayscale, cubic upscale to a minimum height,
  optional Otsu binarization. Crops taller than 1.5x their width are also read
  rotated a quarter turn each way and the most confident read wins.
- **Tuning on val.** A sample of up to 300 val crops per class is read with each
  engine setting (height 32 or 64, binarize on/off, and the engine's reading mode:
  Tesseract line vs block mode, EasyOCR recognizer-only vs detector+recognizer). Numeric
  classes also try a digits-only allow list. Per class, the setting with the highest
  correct-parse rate (numeric) or lowest character error rate (text) is kept.
  `legend_title` has no val crops, so it uses the `legend_label` setting.
- **Test.** `bar_test` is scored once per engine with the chosen settings.
- **Metrics.** *Exact raw*: the read equals the ground truth after stripping
  whitespace. *Exact norm*: equal after normalization (NFKC so ligatures become
  letters, curly quotes and dashes to ASCII, whitespace collapsed, and a literal
  backslash-n treated as a space). *CER*: character edit distance over reference
  length on normalized text, summed over crops. *Parse*: the read parses as a number
  (a trailing % is dropped, thousands commas removed, a decimal comma accepted);
  *correct*: it parses to the ground-truth number. Parse rates count only crops whose
  ground truth parses.
- **Scope subset.** Single-series charts (fewer than two legend patches, an inferred
  proxy) with at most 10 bars: 47 test charts.

### Normalization note: added after seeing test results

The literal backslash-n rule was added to the normalized metric after the first
Tesseract test scoring. 87 test crops contain a literal backslash-n in the
ground truth (an annotation artifact for a line break: 61 `xlabel`, 13 `y_title`,
9 `legend_label`, 3 `chart_title`, 1 `legend_title`); no val crop does, so the val
tuning is unaffected. Tesseract was re-scored from its saved reads without running OCR
again, the raw metric is unchanged, and the same rule applies to EasyOCR. Before the
rule, Tesseract's normalized exact match was 0.546 on `xlabel`, 0.677 on `y_title`
and 0.784 on `legend_label` (equal to the raw values).

## Results on test, all 139 vertical charts

| Class | Crops | Tesseract exact raw | norm | CER | EasyOCR exact raw | norm | CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x_title | 54 | 0.907 | 0.907 | 0.011 | 0.815 | 0.815 | 0.030 |
| y_title | 127 | 0.677 | 0.724 | 0.047 | 0.551 | 0.583 | 0.109 |
| chart_title | 26 | 0.692 | 0.692 | 0.190 | 0.654 | 0.654 | 0.142 |
| xlabel | 1039 | 0.546 | 0.581 | 0.385 | 0.556 | 0.579 | 0.321 |
| ylabel | 1028 | 0.868 | 0.868 | 0.074 | 0.931 | 0.931 | 0.038 |
| legend_label | 245 | 0.784 | 0.820 | 0.030 | 0.735 | 0.755 | 0.050 |
| value_label | 208 | 0.663 | 0.663 | 0.197 | 0.615 | 0.615 | 0.258 |
| legend_title | 5 | 0.400 | 0.400 | 0.069 | 0.400 | 0.600 | 0.034 |

| Class | Ground truth parses | Tesseract parses | Tesseract correct | EasyOCR parses | EasyOCR correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| ylabel | 1028 | 0.925 | 0.887 | 0.999 | 0.936 |
| value_label | 192 | 0.891 | 0.766 | 0.844 | 0.677 |
| charts with every ylabel correct | | | 84/137 (61%) | | 112/137 (82%) |

### Scope subset: single series, up to 10 bars (47 charts)

| Class | Crops | Tesseract exact raw | norm | CER | EasyOCR exact raw | norm | CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x_title | 13 | 0.923 | 0.923 | 0.005 | 0.769 | 0.769 | 0.025 |
| y_title | 46 | 0.630 | 0.717 | 0.051 | 0.500 | 0.565 | 0.087 |
| chart_title | 15 | 0.867 | 0.867 | 0.019 | 0.867 | 0.867 | 0.026 |
| xlabel | 239 | 0.477 | 0.498 | 0.515 | 0.577 | 0.598 | 0.417 |
| ylabel | 320 | 0.878 | 0.878 | 0.063 | 0.950 | 0.950 | 0.021 |
| legend_label | 2 | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | 0.071 |
| value_label | 40 | 0.625 | 0.625 | 0.302 | 0.625 | 0.625 | 0.341 |

| Class | Ground truth parses | Tesseract parses | Tesseract correct | EasyOCR parses | EasyOCR correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| ylabel | 320 | 0.947 | 0.909 | 1.000 | 0.963 |
| value_label | 34 | 0.853 | 0.735 | 0.853 | 0.735 |
| charts with every ylabel correct | | | 30/46 (65%) | | 39/46 (85%) |

`legend_label` has 2 scope crops and `legend_title` 5 crops overall; ignore them.

## Tesseract vs EasyOCR

- **Value-axis labels (`ylabel`):** EasyOCR is clearly better. Correct-parse is 0.936
  against 0.887, and it parses 99.9% of reads against 92.5%. Tesseract returned an
  empty read for 26 of 1,028 labels and EasyOCR for none. Charts with every value-axis
  label correct: 112 of 137 (82%) against 84 of 137 (61%); in the scope subset 39 of
  46 against 30 of 46. Short boxes under 14 px are the hardest for both (EasyOCR 0.889,
  Tesseract 0.844 correct-parse).
- **Category labels (`xlabel`):** no real difference in exact match (0.579 vs 0.581
  normalized). EasyOCR has a lower CER (0.321 vs 0.385). Single-line labels are read
  correctly 70-77% of the time by both, with Tesseract ahead on small ones (0.769 vs
  0.702 under 14 px). Tall boxes, which may be rotated, are about equal (0.658 vs 0.649).
  Boxes 25 px or higher, mostly multi-line or long, are the weak spot for both, with
  EasyOCR ahead (0.374 vs 0.302), except labels that contain a line break in the ground
  truth (0.393 vs 0.607 for Tesseract). Neither engine is near usable for category names.
- **Rotated `y_title`:** Tesseract is better. Normalized exact match is 0.720 against
  0.584 on the 125 rotated titles, and CER is 0.047 against 0.109.
- **Other classes:** Tesseract is ahead on `x_title` (0.907 vs 0.815),
  `legend_label` (0.820 vs 0.755) and `value_label` correct-parse (0.766 vs 0.677).
  `chart_title` has only 26 crops: Tesseract exact is higher (0.692 vs 0.654), EasyOCR
  CER lower (0.142 vs 0.190).
- Neither engine is tuned on test; both used their val-chosen settings.

## Can the value axis be recovered from noisy tick reads?

`src/baseline/ocr_axis_fit.py` pairs each ground-truth `ylabel` box with its nearest
ground-truth `y_tick` box, and fits value = a + b y from tick pixel position to the
parsed read. The fit does not assume the reads are right: each pair of points proposes
a line, the line with the most inliers (within 2% of the implied value span, slope
required to fall as y grows) wins, and it is refit on its inliers. The reference line is a
least-squares fit to the ground-truth label text. A fit matches when slope is within 1% (relative) and offset within 1% of the value span; 0.5% to
5% give the same result within one chart. A chart is flagged for a human when fewer than 3 reads parse, fewer than 3 points are inliers,
inliers are under 60% of the ticks, or two best lines are supported by different point sets.

137 charts have value-axis labels; 5 are excluded because even the ground-truth labels
do not lie on a line (likely log or irregular axes), leaving 132.

| | Tesseract | EasyOCR |
| --- | ---: | ---: |
| Every label read correctly | 81 (61%) | 109 (83%) |
| **Fitted axis matches the reference** | **126 (95%)** | **126 (95%)** |
| Flagged for a human | 10 (8%) | 4 (3%) |
| Accepted and match | 121 of 122 | 125 of 128 |
| Accepted but wrong (silent errors) | 1 | 3 |
| Flagged, but would have matched | 5 | 1 |

- Fitting through noisy reads recovers the axis in 95% of charts for both engines,
  against 61% (Tesseract) and 83% (EasyOCR) when every read has to be right, so the
  engine gap on this task mostly disappears.
- No chart failed for having fewer than 3 ticks. Flag reasons for Tesseract: inliers
  under 60% of ticks 6, ambiguous best line 4, fewer than 3 parsed reads 3, fewer than 3
  inliers 1 (a chart can have several). EasyOCR: inliers under 60% 4, ambiguous 1,
  fewer than 3 inliers 1.
- **The silent errors are systematic**, not random: every one has a slope off by a factor
  of ten. A decimal separator is dropped (`4.5` read as `45`, `0,6` as `06`) or a `%` is read
  as `6` or `8` (`70%` as `708`). A clean line through consistent wrong reads passes every
  check here. Catching it needs other evidence, such as the tick-label format or
  agreement with printed value labels.
- Caveat on method: the ambiguity rule was first written to compare line slopes, which
  over-flagged (26 charts for Tesseract, 21 for EasyOCR) because several lines through the same
  inliers differ slightly in slope. I changed it to compare inlier sets after seeing
  this, on test. The fits and the 126 matches are identical either way. Ground-truth
  tick positions are used, so a real tick detector would add error.

## Checks for a wrong axis scale (factor-of-ten errors)

The fit accepts a chart whose reads are consistently wrong by a factor of ten.
`src/baseline/ocr_axis_checks.py` measures independent signals that flag such charts;
the fit itself is unchanged. Thresholds were chosen on val only (pooled over both
engines): the most sensitive setting whose false-flag rate on val charts the fit
already got right is at most 10%. Val has just 1 silent error (Tesseract) and none for
EasyOCR, so val sets the cost limit but cannot tune catch rates.

| Signal | Setting | Val false flags |
| --- | --- | ---: |
| Unit suffix: some tick reads end in `%`, others do not (a `%` read as 6 or 8) | none | 0.4% |
| Value labels: median \|log10(printed / fitted)\| over bars with a printed number | above 0.05 | 3.4% |
| Decimals: reads do not use one number of decimal places | lenient (strict is 13%, over the cap) | 6.1% |
| Percent title: y-axis title read mentions % but fitted ticks pass a cap | 100 | 8.0% |
| Nice step: fitted tick step not 1, 2, 2.5 or 5 times a power of ten | none | 1.9% |

Adopted: **unit suffix and value labels**. The nice-step check was dropped (a pure scale
error leaves a nice step nice, and it caught nothing); decimals and the percent title
are not adopted (cost with little or no catch, or over the cap).

On test, among charts the fit accepted (Tesseract 122, one silent error; EasyOCR 128,
three), the adopted pair caught **0 of 1** (Tesseract) and **2 of 3** (EasyOCR) silent
errors and newly flagged **5 of 121** and **1 of 125** correct charts. The value-label
check is only available where numbers are printed on bars (12-13 test charts per
engine); the unit-suffix check does the work. Not caught: decimals dropped from every
tick (`4.5` read as `45` with `4`, `3` kept) and `0,6` read as `06`; strict
decimals would catch the second. These rates rest on 4 silent errors in total, so
treat them as indicative only.

## Decimal points lost in value-axis reads (val only)

Test was not read for this. `src/baseline/ocr_decimal_diagnosis.py` counts, on the val
`ylabel` and `value_label` crops (150 vertical charts), how often the ground truth has a
decimal point (or decimal comma, not a thousands comma) and the read does not.

| Engine | Class | Crops with a decimal | Point dropped | Digits right, only the point missing | Point invented |
| --- | --- | ---: | ---: | ---: | ---: |
| Tesseract | ylabel | 248 | 14 (5.6%) | 13 | 0 of 881 |
| Tesseract | value_label | 205 | 27 (13.2%) | 8 | 0 of 150 |
| EasyOCR | ylabel | 248 | 26 (10.5%) | 19 | 0 of 881 |
| EasyOCR | value_label | 205 | 33 (16.1%) | 10 | 0 of 150 |

- **Neither engine invents a decimal point.** The loss is one-directional. A comma read
  for a point (`0,46` for `0.46`) happens (4 and 20 times for Tesseract, 11 and 11 for
  EasyOCR) and still counts as a decimal.
- **Failures are small crops.** The median failing crop is 12-13 px high and 22-25 px wide,
  against 14-15 px and 26-32 px for crops whose point survived. Of the ylabel and
  value_label crops that have a decimal, those under 14 px high lose it far more often
  (for example Tesseract value labels: 17 of 96 at 10-13 px against 4 of 76 at 14-19 px).
- **What the montage shows** (`decimal_failures_montage.png` in each engine's output
  folder): the point is a few pixels, often low contrast or on a coloured fill, touching a
  digit (`7.52` read as `72`) or crossed by a gridline (`28|.37` read as `28137`).
  Some failures are not only a missing point: `11.7` read as `4117`.

One change at a time, with each class's tuned setting plus the change, on all val crops of
both classes. The best setting was defined before running: the highest correct-parse rate
pooled over both classes (within 0.3 points, the lower drop rate).

| Change | Tesseract: pooled correct-parse / drop rate | EasyOCR: pooled correct-parse / drop rate |
| --- | ---: | ---: |
| Current setting | **82.3% / 9.1%** | **88.8% / 13.0%** |
| Upscale to 96 px | 81.7% / 8.8% | 88.8% / 13.2% |
| Upscale to 128 px | 80.5% / 9.5% | 88.5% / 15.0% |
| No binarization | 82.3% / 9.1% | 88.8% / 13.0% |
| Mild sharpening | 81.7% / 9.5% | 88.7% / 13.2% |
| Allow only `0-9 . , - %` | 82.0% / 9.1% | 88.4% / 13.0% |

**No change helped, so the current settings stay and the reads are unchanged.** The numeric
classes were already unbinarized, so "no binarization" is identical to the current setting.
EasyOCR resizes crops to 64 px internally, so 96 px changes little. The point is lost before
upscaling can help: at native resolution there is too little of it. Realistic next steps are
not preprocessing but evidence from outside the crop (consistent decimals across a chart's
ticks, the tick step, printed value labels), or a recogniser trained on small tick text.

## Recreating the environments

Nothing here is in `requirements*.txt`; both engines are optional. Neither changes the
`base` conda environment.

**Tesseract** (CPU), in its own conda environment (504 MB, because the conda-forge package
ships all 125 language models; only English is used):

```bash
conda create -y -n tactilegen-ocr -c conda-forge --override-channels tesseract
```

This installed tesseract 5.5.3 with leptonica 1.87.0. The script finds the binary through
`$TESSERACT_CMD`, then `PATH`, then `~/miniconda3/envs/tactilegen-ocr/bin/tesseract`.

**EasyOCR** (GPU), in a venv that reuses the `base` environment's `torch` (2.14.1+cu130,
Python 3.13.13, numpy 2.5.1) and installs only its own extras (about 235 MB):

```bash
python -m venv --system-site-packages ~/.venvs/tactilegen-easyocr
~/.venvs/tactilegen-easyocr/bin/pip install easyocr==1.7.2
```

That adds easyocr 1.7.2, opencv-python-headless 5.0.0.93, scikit-image 0.26.0,
shapely 2.2.0, pyclipper 1.4.0, python-bidi 0.6.12, ninja 1.13.2, imageio 2.38.1,
tifffile 2026.10.15 and lazy-loader 0.6. The first use downloads the English detector and
recognizer (about 94 MB) into `~/.venvs/tactilegen-easyocr/models` (override with
`EASYOCR_MODEL_DIR`) and needs network access. Total on disk: 329 MB.

Run (output goes under `output/`, which is gitignored):

```bash
python -m src.baseline.ocr_tesseract tune --out output/ocr/tesseract
python -m src.baseline.ocr_tesseract test --out output/ocr/tesseract      # refuses a second run
~/.venvs/tactilegen-easyocr/bin/python -m src.baseline.ocr_easyocr tune --out output/ocr/easyocr
~/.venvs/tactilegen-easyocr/bin/python -m src.baseline.ocr_easyocr test --out output/ocr/easyocr
python -m src.baseline.ocr_axis_fit --predictions output/ocr/<engine>/ocr_test_predictions.json
```

## Caveats

- Tuning picks the best of 8 settings per class (16 for the numeric classes, which also try
  the digits-only allow list) on at most 300 val crops, so val numbers are a little optimistic; test was scored once per
  engine.
- Crops come from ground-truth boxes. A detector's boxes will be looser or missing, so
  real end-to-end OCR will score lower.
- Ground-truth text contains markup (`H_{2}S`, `\alpha`) that no engine can produce, which
  caps exact match on some labels.
- The scope subset uses legend patches as a proxy for single series.
- Small classes (`legend_title`, `chart_title`) have too few crops for firm conclusions.
