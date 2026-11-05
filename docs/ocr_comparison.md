# OCR baseline comparison

## Scope

This report compares the published, title-fixed Tesseract baseline with the
completed, title-fixed RapidOCR PP-OCRv6 baseline on the leakage-cleaned PlotQA
vertical-bar test split. Both reports cover 11,242 images and 180,731 unique
text regions with the same detector checkpoint and scoring thresholds.

The title-fix tables use each engine's own pre-fix run as its reference. The
final comparison uses Tesseract as the established reference, so every delta
is `RapidOCR - Tesseract`. Accuracy increases are favorable; error-rate
decreases are favorable. RapidOCR's pre-fix title CER comes from the preserved
`output/rapidocr_test_pre_title_fix/metrics.json` artifact; the remaining
displayed values are reported in the two baseline documents.

## Effect of the shared chart-title fix

The shared full-width title strip produced a much larger change than the
remaining differences between OCR engines.

| Engine | Track | Exact accuracy before | Exact accuracy after | Change | CER before | CER after | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract | Oracle | 96.41% | 99.25% | +2.84 pp | 1.90% | 0.48% | -1.42 pp |
| Tesseract | End-to-end | 95.57% | 98.68% | +3.11 pp | 2.58% | 1.02% | -1.56 pp |
| RapidOCR | Oracle | 96.61% | 99.21% | +2.60 pp | 1.87% | 0.40% | -1.47 pp |
| RapidOCR | End-to-end | 96.13% | 99.04% | +2.91 pp | 2.50% | 0.81% | -1.69 pp |

| Engine | Track | Title accuracy before | Title accuracy after | Change | Title CER before | Title CER after | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract | Oracle | 53.55% | 98.92% | +45.37 pp | 4.28% | 0.12% | -4.16 pp |
| Tesseract | End-to-end | 49.00% | 98.93% | +49.93 pp | 5.70% | 1.14% | -4.56 pp |
| RapidOCR | Oracle | 53.68% | 98.23% | +44.55 pp | 5.12% | 0.75% | -4.37 pp |
| RapidOCR | End-to-end | 48.92% | 98.32% | +49.40 pp | 6.59% | 1.58% | -5.01 pp |

| Engine | Track | All-label image accuracy before | All-label image accuracy after | Change |
| --- | --- | ---: | ---: | ---: |
| Tesseract | Oracle | 47.15% | 89.04% | +41.89 pp |
| Tesseract | End-to-end | 39.35% | 81.12% | +41.77 pp |
| RapidOCR | Oracle | 50.01% | 88.42% | +38.41 pp |
| RapidOCR | End-to-end | 41.49% | 80.98% | +39.49 pp |

## Final engine comparison

| Metric | Tesseract | RapidOCR | RapidOCR - Tesseract |
| --- | ---: | ---: | ---: |
| Oracle normalized exact accuracy ↑ | **99.25%** | 99.21% | -0.04 pp |
| End-to-end normalized exact accuracy ↑ | 98.68% | **99.04%** | +0.36 pp |
| Oracle character error rate ↓ | 0.48% | **0.40%** | -0.08 pp |
| End-to-end character error rate ↓ | 1.02% | **0.81%** | -0.21 pp |
| Oracle word error rate ↓ | **0.56%** | 0.81% | +0.25 pp |
| End-to-end word error rate ↓ | 1.51% | **1.44%** | -0.07 pp |
| Oracle all-label image accuracy ↑ | **89.04%** | 88.42% | -0.62 pp |
| End-to-end all-label image accuracy ↑ | **81.12%** | 80.98% | -0.14 pp |

Tesseract has slightly higher oracle exact accuracy and strict whole-image
accuracy. RapidOCR has higher end-to-end region accuracy and lower character
error rates. The differences between the final engines are small compared with
the improvements produced by the shared title fix.

## Final accuracy by text role

| Text role | Track | Tesseract | RapidOCR | RapidOCR - Tesseract |
| --- | --- | ---: | ---: | ---: |
| Chart title | Oracle | **98.92%** | 98.23% | -0.69 pp |
| Chart title | End-to-end | **98.93%** | 98.32% | -0.61 pp |
| Legend label | Oracle | **99.76%** | 99.01% | -0.75 pp |
| Legend label | End-to-end | **99.72%** | 98.25% | -1.47 pp |
| X-axis title | Oracle | **99.98%** | 99.89% | -0.09 pp |
| X-axis title | End-to-end | **99.96%** | 99.78% | -0.18 pp |
| X tick label | Oracle | 98.29% | **99.13%** | +0.84 pp |
| X tick label | End-to-end | 98.46% | **98.88%** | +0.42 pp |
| Y-axis title | Oracle | **98.45%** | 97.09% | -1.36 pp |
| Y-axis title | End-to-end | 97.87% | **98.09%** | +0.22 pp |
| Y tick label | Oracle | **99.75%** | **99.75%** | 0.00 pp |
| Y tick label | End-to-end | 98.29% | **99.64%** | +1.35 pp |

## Final character error rate by text role

| Text role | Track | Tesseract | RapidOCR | RapidOCR - Tesseract |
| --- | --- | ---: | ---: | ---: |
| Chart title | Oracle | **0.12%** | 0.75% | +0.63 pp |
| Chart title | End-to-end | **1.14%** | 1.58% | +0.44 pp |
| Legend label | Oracle | **0.06%** | 0.28% | +0.22 pp |
| Legend label | End-to-end | **0.13%** | 0.60% | +0.47 pp |
| X-axis title | Oracle | **0.01%** | 0.02% | +0.01 pp |
| X-axis title | End-to-end | **0.05%** | 0.08% | +0.03 pp |
| X tick label | Oracle | 1.92% | **0.29%** | -1.63 pp |
| X tick label | End-to-end | 1.87% | **0.41%** | -1.46 pp |
| Y-axis title | Oracle | 0.34% | **0.20%** | -0.14 pp |
| Y-axis title | End-to-end | 0.76% | **0.23%** | -0.53 pp |
| Y tick label | Oracle | **0.09%** | 0.12% | +0.03 pp |
| Y tick label | End-to-end | 1.00% | **0.55%** | -0.45 pp |

## Interpretation and limitations

- Tesseract is stronger on chart titles, legends, and most axis-title exact
  matches.
- RapidOCR is substantially stronger on X tick labels and has lower
  end-to-end CER for Y-axis titles and Y tick labels.
- The results are descriptive measurements on one synthetic test split. Small
  differences are not claims of statistical significance or expected
  performance on real charts.
- These are comparisons of the two published baseline reports, not a paired
  rerun over identical detector rows. The Tesseract report recorded 1,060
  extra detector boxes and the RapidOCR report recorded 1,065, so small
  end-to-end differences may include detector-run variation.

See the individual [Tesseract](ocr_baseline.md) and
[RapidOCR](rapidocr_baseline.md) reports for methodology, reproduction
commands, and detailed caveats.
