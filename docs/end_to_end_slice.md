# End-to-end slice: vertical, single-series bar charts

Scope: vertical bar charts with one series and at most 10 bars. Stage 1 feeds the
interpreter **ground-truth boxes and ground-truth text**, so its errors come from the
interpreter's logic, not from detection or OCR. Stage 2 (ground-truth boxes, OCR reads)
is below; Stage 3 (detector boxes, OCR reads) is not run yet.

## What the interpreter does (`src/interpret_bars.py`)

1. **Scope.** No bars, more than 10 bars, or two or more legend entries: `rejected`.
2. **Categories.** Bars and `x_tick_label` boxes are sorted left to right and paired in
   order; the counts must be equal. Tick marks, when there is one per bar, anchor the
   check that labels line up with bars (a rotated label's centre sits well left of its
   tick, so label centres alone would be misleading). Category text is the OCR read with
   whitespace collapsed; an empty read leaves no chart.
3. **Axis.** Each value-axis label is paired with its tick mark (else its own centre,
   flagged), parsed to a number, and a robust line is fitted from pixel y to value
   (`src/axis_fit.py`: pair-proposed lines, inlier voting, refit; the same fit as
   `docs/ocr_baselines.md`). Reads that disagree with the line are flagged, not
   rewritten.
4. **Bar values.** Each bar's value is the fit at its top edge (its bottom edge for bars
   below zero). Values are rounded to the pixel resolution of the fit and carry
   +/- half a pixel of uncertainty.
5. **Axis fields.** `y_tick_values` are the inlier reads; `y_min` and `y_max` are the fit
   at the plot-area edges, widened to include the ticks (so they can run past the
   top tick); `y_scale` is linear. Log axes are not handled yet.
6. **Flags** (status `needs_review`): bar and label counts differ, an empty or
   unreadable text, too few or disagreeing tick labels, an ambiguous axis, labels that
   do not line up with bars, bars that do not share a baseline (stacked or floating),
   mixed `%` suffixes on tick labels, printed bar values that disagree with the fitted
   axis (median |log10 ratio| above 0.05), and, when a threshold is given, low OCR
   confidence. The last two are the checks adopted in `docs/ocr_baselines.md`.

## Stage 1 results (`python -m src.baseline.e2e_stage1 --split {val,test}`)

Nothing is tuned here; val was used to debug, test was run once after the logic was fixed.

| | val | test |
| --- | ---: | ---: |
| Scope charts | 58 | 47 |
| Accepted | 53 | 39 |
| Needs review | 5 | 8 |
| Chart built (bars = categories) | 55 | 43 |
| Bar and tick marks line up | 55 of 55 | 42 of 43 |
| Pairing agrees with an independent tick-based route | 55 of 55 | 41 of 43 |
| Bars read with printed numbers on every bar | 55 bars, 11 charts | 28 bars, 4 charts |
| ...of accepted charts: bars within 1% of axis span | 48 of 48 | 28 of 28 |
| ...median / max error as a fraction of axis span | 0.20% / 0.83% | 0.08% / 0.40% |
| ...factor-of-ten errors | 0 | 0 |

- **Value error is small where truth exists.** The one val chart with a large error
  (a printed 50,174 against a fitted 2,962) was flagged, not accepted; three of its tick labels
  disagree with the fitted line (possibly a log or broken axis; not verified).
- **Needs-review reasons, test:** bar and label counts differ (3 charts), tick labels
  disagree or are unparsable (4), bars without a shared baseline (2), labels not lined
  up with bars (1). Some charts have several.
- **Pairing check.** The independent route (nearest tick mark to each bar, then the
  label nearest that tick after removing the constant offset of rotated labels) agrees
  on every val chart and 41 of 43 on test. One of the two test disagreements was
  flagged by the interpreter; the other (`PMC3275625___g004`, accepted) has a short
  first label that sits off the offset pattern. I checked it by eye: the interpreter's
  order is right (`CFP-YFP` first, value 0.015). Boxes alone cannot independently
  verify pairing for rotated labels, so a visual spot-check is part of Stage 2.
- **Category-name accuracy is not a Stage 1 measurement.** The text is ground truth, so
  names match by construction; what Stage 1 checks is the pairing. Name accuracy
  starts to mean something in Stage 2.
- **Truth is thin.** Value error is only measured where every bar has a printed number
  (15 charts across both splits). `src/dataset/handread_sheet.py` makes a blank sheet
  for reading about 25 more charts by eye.

## Stage 2: ground-truth boxes, OCR reads (`python -m src.baseline.e2e_stage2`)

The text of every box comes from the engine's read instead of the annotation. The OCR
settings are the tuned ones from `docs/ocr_baselines.md`; the decimal-point fixes tried
there did not help, so the reads are unchanged. The reference for each chart is the Stage 1
result on ground-truth text; only charts Stage 1 accepted have a trusted reference
(53 in val, 39 in test).

- **Wrong chart:** no chart was built, a category name differs from the reference
  (after normalization), or a bar value is more than 1% of the axis span off.
- **Flagged chart:** the status is not `accepted`. Flag precision is the share of flagged
  charts that are wrong; recall is the share of wrong charts that are flagged.
- **Confidence threshold:** a read below it is flagged. It was chosen **on val only**, per
  engine, as the best F1 of the flags over 0.3 to 0.95 (ties to the higher value): **0.8**
  for Tesseract and **0.6** for EasyOCR. **Test was scored once** with those values.

**Val numbers** (used to choose the threshold, so optimistic) and **test numbers** (reporting):

| | Tess val | Easy val | Tess **test** | Easy **test** |
| --- | ---: | ---: | ---: | ---: |
| Scope charts (reference charts) | 58 (53) | 58 (53) | 47 (39) | 47 (39) |
| Accepted / needs review / rejected | 5 / 53 / 0 | 18 / 40 / 0 | **6 / 41 / 0** | **16 / 31 / 0** |
| Chart built, bars = categories (of reference) | 51 | 49 | **37** | **37** |
| Category names exactly right | 149 of 277 | 159 of 266 | **89 of 185** | **108 of 182** |
| Charts with every category right | 9 | 14 | **14 of 39** | **14 of 39** |
| Charts with every value within 1% of span | 50 | 49 | **37 of 39** | **37 of 39** |
| Factor-of-ten errors (charts) | 1 (flagged) | 0 | **0** | **0** |
| Printed-number truth, accepted charts: bars within 1% | 13 of 13 | 13 of 13 | none accepted | **4 of 4** |
| Charts without a trusted reference that were flagged | 5 of 5 | 5 of 5 | **8 of 8** | **8 of 8** |

**Do the flags find the wrong charts?** "Wrong" counts category names as well as values.

| Flag quality, any wrong output | Tess val | Easy val | Tess **test** | Easy **test** |
| --- | ---: | ---: | ---: | ---: |
| Wrong / flagged / caught | 44 / 48 / 44 | 39 / 35 / 32 | **25 / 33 / 24** | **25 / 23 / 20** |
| Precision | 0.92 | 0.91 | **0.73** | **0.87** |
| Recall | 1.00 | 0.82 | **0.96** | **0.80** |
| Silent errors (wrong, not flagged) | 0 | 7 | **1** | **5** |
| False flags (flagged, not wrong) | 4 | 3 | **9** | **3** |

- **Values are safe; names are the problem.** Only 2 test charts (3 in val for Tesseract,
  4 for EasyOCR) have a wrong value or no chart, and every one is flagged (recall 1.00 on
  values alone, for both engines and both splits). No value error went unflagged and no
  factor-of-ten error reached a test chart. All 6 silent errors on test (1 Tesseract, 5
  EasyOCR) are category-name mistakes only (1 to 3 wrong names in a chart).
- **Category names are poor**: right 48% (Tesseract) and 59% (EasyOCR) of the time on test,
  and every category in only 14 of 39 charts. Part of that cannot be fixed by OCR: ground
  truth contains markup such as `\alpha` and `\beta`.
- **Review burden is high.** At the val-chosen thresholds Tesseract sends 41 of 47 test
  charts to review and EasyOCR 31 of 47, mostly because of low-confidence category reads.
  Tesseract's threshold is the stricter one (0.8), which buys recall but costs 9 false flags
  on test against 3.
- **Silent errors with EasyOCR** are the worry: 5 of 25 wrong charts were not flagged at the
  val threshold (0.6); the val sweep shows higher thresholds trade silent errors for review
  load (0.9 flags all 53 val reference charts).
- **Bar count** was right whenever a chart was built (37 of 37): boxes are ground truth, so
  the failures here are empty category reads, which leave no chart (2 of 39 in test).

**Rotated category labels.** `python -m src.baseline.e2e_spotcheck` draws red numbers on
bars and blue numbers on the label each bar was paired with, with ground truth and both
engines' reads underneath (4 val and 4 test charts with slanted labels; I viewed 4: val
`PMC5456925...g005` and `PMC5449974...g005`, test `PMC3614921...g003` and
`PMC5964208...g003`). Pairing by order was right in all four, even where neighbouring
rotated labels overlap heavily. The reads of slanted names are the weak point: Tesseract
returns mostly noise and EasyOCR often returns a mix of the label and its neighbours,
because the boxes of 45-degree text overlap. Crops are only tried at 0 and plus or minus
90 degrees, so slanted text needs deskewing before reading.


## Not done

- Stage 3; deskewing slanted labels; log axes; multi-series, stacked or horizontal charts.
- `y_min`/`y_max` come from the plot area and are not checked against any printed axis end.
- The review page and the corrections file that rebuilds a chart.
