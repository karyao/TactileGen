# End-to-end slice: vertical, single-series bar charts

Scope: vertical bar charts with one series and at most 10 bars. Stage 1 feeds the
interpreter **ground-truth boxes and ground-truth text**, so its errors come from the
interpreter's logic, not from detection or OCR. Stage 2 (ground-truth boxes, OCR reads)
and Stage 3 (detector boxes, OCR reads) are not run yet.

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

## Not done

- Stage 2 and 3; log axes; multi-series, stacked or horizontal charts.
- `y_min`/`y_max` come from the plot area and are not checked against any printed axis end.
- The review page and the corrections file that rebuilds a chart.
