# PlotQA end-to-end audits

The **latest full audit** covers all 22,534 PlotQA test bar charts with the
`models/yolo_bars_hv/best.pt` checkpoint, RapidOCR, image size 640, and strict
checks. Its original outputs are the [report](plotqa_both_e71b0b8_640/report.md)
and [chart-level CSV](plotqa_both_e71b0b8_640/audit.csv).

| Latest full audit | Charts | Built | Correct among built | Built and correct |
| --- | ---: | ---: | ---: | ---: |
| Vertical | 11,242 | 10,340 (92.0%) | 8,869 / 10,340 (85.8%) | **8,869 / 11,242 (78.9%)** |
| Horizontal | 11,292 | 9,784 (86.6%) | 8,500 / 9,784 (86.9%) | **8,500 / 11,292 (75.3%)** |
| Combined | 22,534 | 20,124 (89.3%) | 17,369 / 20,124 (86.3%) | **17,369 / 22,534 (77.1%)** |

"Built and correct" is the end-to-end success measure: the chart must both
build and pass the content checks. The CSV's `correct` flag alone also includes
some blocked charts, so it is not the numerator for this measure.

## Change from the previous full audit

The [previous full report](plotqa_both_49e6620_640/report.md) and
[chart-level CSV](plotqa_both_49e6620_640/audit.csv) used the same 22,534
chart IDs, checkpoint, OCR engine, image size, and strict checks. The new run
built 129 more charts and built **111 more charts correctly**: 86 vertical and
25 horizontal. Combined end-to-end success rose from 17,258 / 22,534 (76.6%)
to 17,369 / 22,534 (77.1%), a gain of 0.49 percentage points.

At chart level, 130 previously blocked charts now build: 109 are correct and
21 are wrong. Two charts that previously built incorrectly are now correct.
One previously incorrect horizontal chart is now blocked. No chart that was
previously built and correct regressed. The number built but wrong rose from
2,737 to 2,755 (+18).

The improvement is concentrated in blocking and category pairing: primary
false-block cases fell from 694 to 633, and primary category-pairing failures
fell from 65 to 16. Missed-bar totals are unchanged at 12,376 vertical and
10,823 horizontal; bar detection remains the largest failure source. The 21
newly built but wrong charts warrant review when assessing the updated gates.

An [earlier vertical subset report](plotqa_full_49e6620_20261126_134204/report.md)
and its [chart-level CSV](plotqa_full_49e6620_20261126_134204/audit.csv) cover
4,876 charts. In that subset, 3,895 charts were built and correct (79.9%). All
4,876 chart IDs and their outcomes match the previous full audit. The subset and
full-run percentages use different denominators and should not be treated as a
before-and-after accuracy change.

The original reports reference example sheets that are not included in this
archive. The reports and CSVs are preserved as generated.
