# End-to-end audit (vertical and horizontal bars)

`scripts/audit_e2e.py` puts PlotQA bar charts through the whole pipeline
(`src.tactile.pipeline --yes --strict`), the way a user runs it, and compares
what comes out with the PlotQA ground truth. Each failed chart gets one primary
cause and a failure group. It is a port of Chloe's vertical audit
(`analysis/vbar-failure-audit-v2`) to the refactored code, extended to
horizontal bars.

Detector benchmarks (`src.baseline.yolo_bars evaluate`) say how many bars are
found; this says how many charts come out right, and why the rest do not.

## Run

Download PlotQA test with `python -m src.dataset.download plotqa --splits test` (see
[dataset_setup.md](dataset_setup.md)), then:

```bash
# 1. run the pipeline on 2,000 random test charts per orientation and cache each one
python scripts/audit_e2e.py run \
  --annotations data/plotqa/test/annotations.json --images data/plotqa/test/png \
  --cache output/audit/baseline_640/cache --limit 2000 --seed 0 --workers 4

# 2. compare with the ground truth and write audit.csv, report.md and example sheets
python scripts/audit_e2e.py report \
  --cache output/audit/baseline_640/cache --out output/audit/baseline_640/report
```

`run` is resumable: cached charts are skipped. Use a new `--cache` folder for
every checkpoint or code version, and keep `--limit` and `--seed` the same so
runs compare the same charts. `--weights`, `--imgsz` and `--ocr` (default
RapidOCR) are recorded in `run_settings.json`. About 1–2 s per chart per worker
on a laptop CPU.

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
| `bar_values_off` | bars, labels and series are right but a value is off by more than 1% |
| `other_rare` | pairing, extra bars, rescued values, interpretation errors, layout |

## Baseline (19 Nov 2026)

Code at `2a807c8` (before #42), `models/yolo_bars_hv/best.pt`, image size 640,
RapidOCR, 2,000 PlotQA test charts per orientation, seed 0.

| | vertical | horizontal |
|---|---|---|
| built | 79.7% | 38.2% |
| built and correct | **65.9%** | **7.5%** |

Failure groups, as a share of all charts:

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
- In `bar_values_off`, labels, ticks and series are right but some bar ends
  are measured short, typically 2–12% of the axis.
- 11 vertical charts stopped with "Install the OCR dependencies": the category
  re-read uses Tesseract, which was not installed for this run.
