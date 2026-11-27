# PlotQA end-to-end audits

The **latest full audit** covers all 22,534 PlotQA test bar charts with the
`models/yolo_bars_hv/best.pt` checkpoint, RapidOCR, image size 640, and strict
checks. Its original outputs are the [report](plotqa_both_49e6620_640/report.md)
and [chart-level CSV](plotqa_both_49e6620_640/audit.csv).

| Latest full audit | Charts | Built | Correct among built | Built and correct |
| --- | ---: | ---: | ---: | ---: |
| Vertical | 11,242 | 10,245 (91.1%) | 8,783 / 10,245 (85.7%) | **8,783 / 11,242 (78.1%)** |
| Horizontal | 11,292 | 9,750 (86.3%) | 8,475 / 9,750 (86.9%) | **8,475 / 11,292 (75.1%)** |
| Combined | 22,534 | 19,995 (88.7%) | 17,258 / 19,995 (86.3%) | **17,258 / 22,534 (76.6%)** |

"Built and correct" is the end-to-end success measure: the chart must both
build and pass the content checks. The CSV's `correct` flag alone also includes
some blocked charts, so it is not the numerator for this measure.

An [earlier vertical subset report](plotqa_full_49e6620_20261126_134204/report.md)
and its [chart-level CSV](plotqa_full_49e6620_20261126_134204/audit.csv) cover
4,876 charts. In that subset, 3,895 charts were built and correct (79.9%). All
4,876 chart IDs and their outcomes match the latest full audit. The subset and
full-run percentages use different denominators and should not be treated as a
before-and-after accuracy change.

The original reports reference example sheets that are not included in this
archive. The reports and CSVs are preserved as generated.
