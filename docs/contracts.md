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
