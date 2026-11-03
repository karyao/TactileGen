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
| `VerticalBarChart` | Interpreter | Tactile renderer | Optional title and axis titles, ordered categories, one or more bar series with numeric values, and an optional description of the value axis. |

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

`title`, `x_axis_title` and `y_axis_title` are `None` when the source chart
has no such text. The renderer omits them; the interpreter must not make one up
(for example from the file name or a category). An empty string is rejected, so
OCR that read nothing should produce `None` only when no title box was
detected, and otherwise raise `ChartInterpretationError`. In JSON a missing key
and `null` both mean `None`.

### Value axis (optional)

These fields describe the printed value (y) axis. All are optional; a chart
without them is still valid and the renderer must then scale from the data.

| Field | Default | Meaning |
| --- | --- | --- |
| `y_min`, `y_max` | `None` | Ends of the printed axis, which may extend past the data. `y_min` must be less than `y_max`. |
| `y_tick_values` | `()` | Numbers printed at the ticks, strictly increasing, inside `y_min`/`y_max` when those are set. |
| `y_scale` | `"linear"` | `"linear"` or `"log"`. A log axis requires every limit, tick and bar value to be positive. |

Adding these fields keeps `schema_version` 1: existing JSON files load with
the defaults, and new files write the extra keys.

## Roles beyond the 15 YOLO classes

`CLASS_NAMES` stays the 15 PlotQA classes so existing class ids, `classes.txt` files and
the PlotQA model checks keep working. Three more roles are valid for a `Detection`
(`ROLES = CLASS_NAMES + EXTRA_ROLES`) and have no YOLO class id in the PlotQA model:

| Role | Meaning |
| --- | --- |
| `x_tick`, `y_tick` | The small tick mark on an axis (about 10 px square in the PMC data). Pixel positions for the axis fit come from here; label centres are only a fallback. |
| `value_label` | A number printed on or next to a bar. It is a text role, so OCR may reference it. Used to cross-check the fitted axis. |

## Interpretation result

The interpreter (`src/interpret_bars.py`) returns an `InterpretationResult` instead of a
bare chart, so that a person can see what to check. `VerticalBarChart` is unchanged.

| Field | Meaning |
| --- | --- |
| `status` | `accepted`: a chart and nothing flagged. `needs_review`: something is flagged; `chart` is a provisional chart when one could be built, else `None` (for example the bar and label counts differ). `rejected`: out of scope (no bars, more than 10 bars, or more than one series), no chart. |
| `chart` | The `VerticalBarChart`, or `None`. |
| `items` | One `ReviewItem` per element a person may check: category labels (left to right), value-axis tick labels, titles, value labels, and each bar with its computed value and uncertainty. |
| `reasons` | Chart-level reasons, such as "6 bars but 5 category labels". |

A `ReviewItem` carries `kind`, `index`, `detection_id`, the raw OCR `text` and its
`confidence`, the `box`, a computed `value` with `uncertainty` for bars, and `flagged`
with a `reason`. A flagged item must have a reason. The raw read is kept even when the
chart holds a cleaned version, so staff correct what the OCR actually said.

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
