# Chart stage contracts (version 1)

These contracts let detection, OCR, chart interpretation, and tactile rendering
work against the same image and chart data. They do not add a user command or
change the existing dataset and OpenCV CLI output.

The Python types and 15 YOLO roles live in `src/contracts.py`. Normal stages
pass these objects in memory. `src/contract_io.py` reads and writes JSON for
small fixtures or optional inspection; the pipeline does not need a JSON file
between every stage.

## Handoffs

| Result | Producer | Consumer | Contents |
| --- | --- | --- | --- |
| `DetectionFrame` | Detector | OCR and interpreter | Image size and identified chart elements with role, pixel box, and confidence. |
| `OCRFrame` | OCR | Interpreter | Text and confidence keyed by detection ID; role and location come from the matching detection. |
| `VerticalBarChart` | Interpreter | Tactile renderer | Title, axis titles, ordered categories, and one or more bar series with numeric values. |

Every result has a `source_key` such as `val/22410.png`. PlotQA reuses numeric
image IDs across splits, so the split or another namespace is required. The
keys in a set of stage results must agree. Detection IDs are unique within one
image; `join_text_regions()` checks that each OCR reference points to a text
role in the matching `DetectionFrame`.

`Box(x1, y1, x2, y2)` uses image pixels, a top-left origin, and exclusive right
and bottom edges. A frame's boxes must have positive area and lie within its
declared width and height. Confidence is a finite number from 0 to 1. The
role strings and their order match the existing `classes.txt` format; see
`CLASS_NAMES` and `CHART_MARK_ROLE`.

OCR may contain an empty text string when a crop was attempted but unreadable.
The interpreter must either produce a complete `VerticalBarChart` or raise
`ChartInterpretationError`; it must not invent missing values. A series name
may be `None` when the source chart has no legend. Each series has one value
per category in the same order.

## Example fixtures

`tests/fixtures/contracts/` contains a small synthetic chart image plus
matching `detections.json`, `ocr.json`, and `vertical_bar_chart.json`. The PNG
lets future OCR work test cropping without downloading PlotQA. The JSON proves
the data handoffs; it does not claim Tesseract can read the synthetic text.

```python
from pathlib import Path
from src.contract_io import load_detection_frame, load_ocr_frame, load_vertical_bar_chart
from src.contracts import join_text_regions

root = Path("tests/fixtures/contracts")
detections = load_detection_frame(root / "detections.json")
ocr = load_ocr_frame(root / "ocr.json")
text_with_boxes = join_text_regions(detections, ocr)
chart = load_vertical_bar_chart(root / "vertical_bar_chart.json")
assert detections.source_key == ocr.source_key == chart.source_key
```

All three JSON files use `schema_version: 1`. JSON validation happens when a
fixture or optional interchange file is loaded. In-process stages use the
typed objects directly and should not serialize whole datasets or write a
file per image by default. Representative throughput and memory measurements
belong to the model and end-to-end pipeline tasks.

## VerticalBarSpec (schema 0.2)

`src/reconstruct.py` turns a `DetectionFrame`, OCR text and the image into a
`VerticalBarSpec` with `reconstruct_vbar(frame, ocr_texts, image)`. Unlike
`VerticalBarChart`, titles may be `None`, series values may be `None`, and
anything the interpreter could not resolve is listed in `flags` instead of
raising. `ChartInterpretationError` is raised only when there are no bars, no
x tick labels, or fewer than two consistent y ticks. `y_axis.min/max` span the
tick values and the calibrated plot edges; K/M/B tick suffixes are expanded
into the numbers and only `%` is kept as `unit`.

Flags (each is "could not resolve this", not "wrong"):

| Flag | Meaning |
| --- | --- |
| `title_missing`, `x_axis_title_missing`, `y_axis_title_missing` | No title box detected, or its text was empty. The field is `None`. |
| `title_touches_right_edge` | The title box reaches the right image edge; the text may be clipped. |
| `y_ticks_dropped:N` | N y tick labels were unreadable or inconsistent with the fitted line. |
| `y_tick_leading_zero` | A surviving y tick label starts with a zero followed by a digit (e.g. `04`); a decimal point was probably lost, so the calibration may fit misread numbers. |
| `y_ticks_not_arithmetic`, `y_calibration_ambiguous` | Surviving ticks are not evenly spaced, or two tick subsets fit equally well. |
| `baseline_outside_plot`, `bar_baseline_offset` | The calibrated zero line is outside the plot, or bar bases sit far from it. |
| `value_clipped_at_axis` | A bar reaches or passes the plot edge beyond the last tick, so its value is a lower bound. |
| `category_count_mismatch`, `category_ambiguous` | Bars and x labels disagree in number, or two bars of a series map to one label. |
| `category_unreadable` | An x label is empty or has no letter or digit (e.g. a lone `|`); OCR confidence is not used. That category's confidence is set to 0.1 (`confidence["category[j]"]`). |
| `series_ambiguous` | Colour groups and legend entries could not be paired one-to-one (a legend swatch with no bars of its colour becomes a series when it can be measured; see `series_from_swatch`). |
| `bars_mostly_missed` | Fewer than 60% of the category x series slots (counting only series with at least one detected bar; very short bars count as detected) had a detected bar, so most values were measured from pixels. A blocker for tactile output that `--force` cannot skip: bars drawn below an axis that starts at zero are invisible to the probe and read as 0. |
| `short_bars_remeasured:N` | N detected bars shorter than max(6 px, 3% of the plot height) were left out of colour clustering and re-measured from pixels (`value_rescued` true), because their box is mostly outline and antialiasing. Only when the zero line is inside the plot and rescue is on. Note, not a blocker. |
| `bar_edge_corrected:N` | N positive bars' top edge, measured from pixels with the rescue's run-length probe, differed from the detector box edge by 5 px or more; the pixel value was used. Only when the zero line is inside the plot and rescue is on. Note, not a blocker. |
| `series_from_swatch:N` | N legend swatches matched no bar colour. Each became a series named by its legend label, with values measured from pixels at the slots implied by the detected series' spacing (`value_rescued` true). Note, not a blocker. |
| `plot_area_missing` | No plot area detected; position filters were skipped. |
| `bars_rescued:N` | N values (including those of swatch-built series) had no detected bar and were measured from pixels (`SeriesSpec.value_rescued`, low `value_confidence`). |
| `value_inferred_zero` | At least one rescued value found no bar pixels and was set to 0.0. |

Series colours are seeded from the legend swatches, so near-identical series colours stay separate when a legend is detected. Bar rescue and the category guard can be switched off with `rescue_bars=False` and `category_guard=False`.

`python -m scripts.eval_reconstruct` measures the stage on PlotQA test charts.
