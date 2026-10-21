# TactileGen

ML pipeline that automatically converts statistical charts into accessible, embosser-ready tactile graphics for blind and low-vision students.

## Set up
1. Download images and train annotations from [PlotQA Dataset](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md)
2. Run `python3 src/dataset/prepare_plotqa.py` from the repository root to get data ready for YOLO.

To prepare only vertical categorical bar charts, use a new output directory:

```bash
python3 src/dataset/prepare_plotqa.py --chart-types vbar_categorical --output dataset_yolo_vbars
```

Pass multiple types after `--chart-types` to include each of them, for example
`--chart-types vbar_categorical hbar_categorical`. Supported types are
`hbar_categorical`, `vbar_categorical`, `dot_line`, and `line`. Without the option,
all four types are included. Filtered runs require an output directory with no
existing images or labels in the selected split. Use `--split val` or `--split test`
with the corresponding PlotQA image directory and annotation JSON. Conversion
also writes `chart_types_<split>.csv` for image-level chart-type ground truth.

## OpenCV vertical-bar baseline

Install the baseline dependencies in a virtual environment:

```bash
python3 -m pip install -r requirements.txt
```

The current predictor detects vertical bars from pixels only. For a single PNG:

```bash
python3 -m src.baseline.opencv_bars predict --image data/png_validation/22410.png --output-dir output/prediction
```

The existing `dataset_yolo_vbars` directory has train images and YOLO labels.
Use it to check that the pipeline runs, then convert PlotQA validation data into
the same directory for a provisional score:

```bash
python3 src/dataset/prepare_plotqa.py --annotations data/annotations_validation.json --images data/png_validation --output dataset_yolo_vbars --split val --chart-types vbar_categorical
python3 -m src.baseline.opencv_bars evaluate --data-root dataset_yolo_vbars --split val --output-dir output/validation
```

Evaluation reads YOLO labels only after prediction. It reports precision,
recall, F1, and exact bar-count accuracy at box IoU 0.5. A train-split run is
only a smoke test, not an unbiased accuracy estimate. After downloading the
official test annotation JSON, score the untouched test split with:

```bash
python3 src/dataset/prepare_plotqa.py --annotations data/annotations_test.json --images data/png_test/png --output dataset_yolo_vbars --split test --chart-types vbar_categorical
python3 -m src.baseline.opencv_bars evaluate --data-root dataset_yolo_vbars --split test --output-dir output/test_vertical_full
```

PlotQA reuses numeric image IDs across splits. Keep each PNG directory paired
with its own annotation JSON; copying validation PNGs over `data/png` makes the
training annotations incorrect.

For the experimental four-type baseline, convert the full validation split and
run the chart predictor. It returns one of the four PlotQA types and the
corresponding data-mark boxes. The report includes chart-type accuracy and
mark detection with predicted type and known type; the known-type result is a
diagnostic only, since it uses annotation metadata to select a detector.

```bash
python3 src/dataset/prepare_plotqa.py --annotations data/annotations_validation.json --images data/png_validation --output dataset_yolo_all --split val
python3 -m src.baseline.opencv_charts predict --image data/png_validation/5571.png --output-dir output/chart_prediction
python3 -m src.baseline.opencv_charts evaluate --data-root dataset_yolo_all --split val --limit-per-type 100 --output-dir output/chart_validation_sample
```

`--limit-per-type` selects a reproducible random sample (seed 0) from each
chart type. Remove it to evaluate the complete validation split. The
line-segment boxes in PlotQA are often extremely thin; interpret their box-IoU
score separately from bar and dot scores. The conversion command prints how
many zero-area annotations could not be encoded as YOLO boxes.

Prepare the all-type test split with `--annotations data/annotations_test.json`,
`--images data/png_test/png`, and `--split test`, then run
`opencv_charts evaluate --data-root dataset_yolo_all --split test`. Omit
`--limit-per-type` to score all test charts.

### Measured baseline

On the complete official vertical-bar test split (11,242 charts), the OpenCV
bar detector scored precision **0.954**, recall **0.763**, F1 **0.848**, and
exact bar-count accuracy **0.474** at IoU 0.5. On a fixed random sample of 250
test charts per type (1,000 total), chart-type accuracy was **0.983** and the
end-to-end data-mark F1 scores were:

| Horizontal bars | Vertical bars | Dots | Line segments |
| ---: | ---: | ---: | ---: |
| 0.851 | 0.843 | 0.917 | 0.395 |

These are PlotQA synthetic-chart results. Zero-area annotations are excluded
from box-IoU scoring, and line-segment boxes often have only a few pixels of
height. The all-type scores above are for a test sample, not the full test set.

## Datasets 

PlotQA: https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md
- Use its official training, validation, and test splits.
