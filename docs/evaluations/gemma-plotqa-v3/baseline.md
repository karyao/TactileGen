# Gemma 3 4B PlotQA accuracy baseline

We evaluated `google/gemma-3-4b-it` on **22,534 PlotQA test bar charts**. Gemma returned a chart that passed schema validation for **10,378 inputs (46.05%)**. Across all **263,276** annotated numeric values, **20,550 (7.81%)** were recovered within the evaluation tolerance. Use this all-values score as the headline for future runs with the same inputs and scoring rules.

## Evaluation scope

The run processed every PNG in `dataset_plotqa_bars_eval/images/test`. The dataset preparation includes both categorical bar orientations. The existing evaluation records for this dataset report [11,242 vertical charts](https://github.com/karyao/TactileGen/blob/3ba7ae6326354a6b81224bca65e2fa8d62807587/docs/with_chartqa_plotqa_vertical_bar/metrics.json) and [11,292 horizontal charts](https://github.com/karyao/TactileGen/blob/3ba7ae6326354a6b81224bca65e2fa8d62807587/docs/with_chartqa_plotqa_horizontal_bar/metrics.json). The saved Gemma metrics are pooled across both orientations.

The evaluator scores structured chart extraction against PlotQA annotations. The earlier [OCR baseline](../../ocr_baseline.md) measures text recognition on 11,242 leakage-cleaned vertical charts. A comparison between Gemma and the existing extraction pipeline needs shared chart IDs and the same scoring rules. This run does not measure tactile rendering or fallback recovery.

## Results

| Metric | Count | Rate |
| --- | ---: | ---: |
| Chart extraction passed schema validation | 10,378 / 22,534 | 46.05% |
| Category labels matched | 21,815 / 96,508 | 22.60% |
| Series labels matched | 10,478 / 62,475 | 16.77% |
| Charts with exact bar count among eligible charts | 4,563 / 22,532 | 20.25% |
| Annotated values aligned to predictions | 44,862 / 263,276 | 17.04% |
| Values correct among aligned pairs | 20,550 / 44,862 | 45.81% |
| **Values correct across all annotated numeric values** | **20,550 / 263,276** | **7.81%** |

Extraction success indicates that a response met the chart schema. For scoring, the evaluator requires the predicted orientation to match the annotation. It aligns values by unique category and series labels after case and whitespace normalization, with a positional series match when both charts have exactly one series. A value is correct when its absolute error is at most 5% of the chart's largest absolute annotated value. Failed extractions and unaligned numeric values remain in the denominator of the all-values score. Five missing annotations across two charts were excluded from value totals, and those two charts were excluded from bar-count accuracy.

The largest measured gap was alignment coverage. Only 17.04% of annotated numeric values reached the numeric comparison, and 45.81% of those aligned values passed the tolerance. Low coverage can reflect extraction failures, orientation mismatches, or label matching failures. The aggregate files do not attribute the gap to individual causes.

## Reproducing and comparing runs

Use the [scoring-v3 evaluator](https://github.com/karyao/TactileGen/blob/20c917bd477dab8bb4cc1ce0bcdd765f9a1be72d/scripts/eval_gemma.py) as the scoring reference. The [run manifest](manifest.json) preserves the exact image IDs, annotation SHA-256, model ID, requested device and dtype settings, 4,096-token generation limit, and prompt version `gemma-chart-v1`. The archived manifest expresses the two input paths relative to the repository root. The original used host-specific absolute paths. The run requested the default model revision.

To compare a future run, use the same image IDs, annotation snapshot, and scoring version. Report both alignment coverage and all-values accuracy so changes in the number of matched values are visible. For a fallback evaluation, also report recovery on the subset blocked by the existing pipeline using shared chart IDs.

The original aggregate outputs are [metrics.json](metrics.json) and the [generated report](report.md).
