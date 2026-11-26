# End-to-end audit (vertical and horizontal bars)

`scripts/audit_e2e.py` puts PlotQA bar charts through the whole pipeline
(`src.tactile.pipeline --yes --strict`), the way a user runs it, and compares
what comes out with the PlotQA ground truth. Each failed chart gets one primary
cause and a failure group. It is a port of Chloe's vertical audit
(`analysis/vbar-failure-audit-v2`) to the refactored code, extended to
horizontal bars.

Detector benchmarks (`src.baseline.yolo_bars evaluate`) say how many bars are
found; this says how many charts come out right, and why the rest do not.

## Run on the training host

Run from the repository root in the project Python environment. Install the
pipeline dependencies with `python -m pip install -r requirements-app.txt`.
Check that `tesseract --list-langs` lists `eng`. The vertical category re-read
can use Tesseract even when RapidOCR is selected.

Use the original PlotQA test annotations and PNGs. If they are missing:

```bash
python -m pip install -r requirements-dataset.txt
python -m src.dataset.download plotqa --splits test
```

Start with a smoke test of 10 charts per orientation. Run the commands below
in Bash so the same settings are reused for the full audit.

```bash
audit_root="output/audit/horizontal_endpoint_$(git rev-parse --short HEAD)_640"
audit_args=(
  --annotations data/plotqa/test/annotations.json
  --images data/plotqa/test/png
  --weights models/yolo_bars_hv/best.pt
  --imgsz 640 --ocr rapidocr --seed 0
  --orientations vertical horizontal
)

python scripts/audit_e2e.py run "${audit_args[@]}" \
  --workers 1 --limit 10 --cache "$audit_root/smoke/cache" &&
python scripts/audit_e2e.py report \
  --cache "$audit_root/smoke/cache" --out "$audit_root/smoke/report"
```

Check `$audit_root/smoke/report/report.md` for 10 charts per orientation and
inspect any setup errors under **Most common block reasons**. Then run the
full test split. Omitting `--limit` processes every eligible chart.

```bash
python scripts/audit_e2e.py run "${audit_args[@]}" \
  --workers 4 --cache "$audit_root/full/cache" &&
python scripts/audit_e2e.py report \
  --cache "$audit_root/full/cache" --out "$audit_root/full/report"
```

The report directory contains `report.md`, `audit.csv`, and image sheets.
The cache records the run settings and completed charts. Repeat the same
commands to resume. Use a separate cache when changing the checkpoint, OCR,
inference size, code revision, or dataset selection. The tracked checkpoint
and 640 px inference match the historical audit settings.

To measure the endpoint fix, run this same audit script on the revision
immediately before the fix, using a fresh cache and identical inputs. Join
the two CSVs by `chart_id` and compare horizontal built-and-correct rates,
the number of charts with numeric errors, and unread values. The historical
results below use a sampled run and predate layout and classifier changes.

## What counts as correct

Over all audited charts of one orientation:

- **built**: exit 0, a tactile page was written
- **built and correct**: built, and the series count, category count, category
  text, series names and every value (within 1% of the ground-truth axis
  maximum) match. Titles are not compared.

Horizontal ground truth is reordered top to bottom, as the pipeline reports it.
Failure groups, first match wins:

| group | meaning |
|---|---|
| `tiny_bars_undetected` | every missed bar is near-zero (under 5% of the axis or at most 5 px thick) |
| `real_bars_undetected` | at least one clearly visible bar has no detection |
| `false_block_gate` | content is correct but a BANA or completeness check blocked it |
| `title_axis_title_ocr` | the title or an axis title could not be read |
| `y_axis_calibration` | value-axis ticks dropped or uneven, or a value clipped at the axis |
| `series_legend_colour_match` | bars do not resolve to legend colours |
| `category_label_ocr` | bars are right but a category label is misread |
| `bar_values_off` | a compared numeric value is off by more than 1% of the axis maximum |
| `bar_value_unread` | an expected value is unread |
| `series_name_mismatch` | values pass but a matched series name differs |
| `unclassified_failure` | no earlier cause explains the failure |
| `other_rare` | pairing, extra bars, rescued values, interpretation errors, layout |

The CSV retains `n_bad_values` and adds `n_numeric_value_errors` and
`n_unread_values` for matched series when category counts agree. A blank means
the comparison was unavailable; a zero does not rule out errors in unmatched
series or values outside the compared pairs. Earlier primary causes, including
missing detections and incorrect rescued values, can take priority over numeric
errors. The report separately counts charts with compared numeric errors and
charts assigned `bar_values_off` as their primary group.

To compare classifier versions, run both reports against the same cache and
join their CSV rows by chart ID. Check that status and correctness are unchanged,
then tabulate the new groups for rows previously assigned `bar_values_off`.
If the original cache is unavailable, recreate the run below and generate both
reports from that new cache; results need not reproduce the historical run exactly.

## Baseline (19 Nov 2026)

Code at `2a807c8` (before #42), `models/yolo_bars_hv/best.pt`, image size 640,
RapidOCR, 2,000 PlotQA test charts per orientation, seed 0.

| | vertical | horizontal |
|---|---|---|
| built | 79.7% | 38.2% |
| built and correct | **65.9%** | **7.5%** |

Failure groups, as a share of all charts:

These historical groups use the old classifier: `bar_values_off` also included
unread values and unexplained failures, so its percentage is not a numeric-error rate.

| group | vertical | horizontal |
|---|---|---|
| `tiny_bars_undetected` | 10.5% | 14.8% |
| `real_bars_undetected` | 4.9% | 8.6% |
| `title_axis_title_ocr` | 4.5% | 4.4% |
| `false_block_gate` | 3.9% | 8.9% |
| `bar_values_off` | 3.5% | 40.9% |
| `series_legend_colour_match` | 3.3% | 6.2% |
| `category_label_ocr` | 2.0% | 0.9% |
| `y_axis_calibration` | 0.3% | 4.1% |
| `other_rare` | 1.1% | 3.7% |

Notes:

- Vertical built (79.7%) is up from 76.3% in Chloe's audit of 11,242 charts;
  near-zero bars are still the largest cause.
- 618 horizontal charts were blocked by `bana:bar_width` (bars squeezed under
  the minimum thickness). #42 and the horizontal page split fix most of them.
- 11 vertical charts stopped with "Install the OCR dependencies": the category
  re-read uses Tesseract, which was not installed for this run.
