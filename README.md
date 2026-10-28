# TactileGen

TactileGen is a research project exploring how to turn statistical charts into
accessible tactile graphics for blind and low-vision students. The long-term
goal is to identify chart elements, recover their meaning, and produce files
that can be embossed.

## Current status

The project currently prepares [PlotQA](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md)
charts for object detection and includes OpenCV and trained YOLO baselines for
finding chart elements. The vertical-bar detector produces boxes and labels
that later stages can use. OCR, chart interpretation, and tactile rendering are
still in development; the project does not yet generate complete tactile
graphics.

## Early results

Vertical-bar detection was evaluated on 11,242 images from a cleaned PlotQA
test split. Boxes were matched at IoU 0.5; YOLO used a confidence threshold of
0.25.

| Detector | Precision | Recall | F1 | Exact bar count |
| --- | ---: | ---: | ---: | ---: |
| OpenCV | 0.9537 | 0.7630 | 0.8477 | 0.4745 |
| YOLO baseline | 0.9917 | 0.9416 | 0.9660 | 0.7917 |
| YOLO, no mosaic | **0.9924** | **0.9431** | **0.9671** | **0.7941** |
| YOLO, no HSV | 0.9898 | 0.9421 | 0.9653 | 0.7903 |

The YOLO checkpoints used the same model and training settings except for the
named augmentation change. The no-mosaic checkpoint disabled mosaic while
retaining HSV augmentation; the no-HSV checkpoint retained mosaic while
disabling HSV augmentation. The ablation mAP evaluations are still pending.
See the [ablation benchmark](docs/ablation_benchmark.md) for the experimental
plan and reproduction commands.

These are detection results, not measures of chart understanding or tactile
output. See the [experiment guide](docs/experiments.md) for methods and
reproduction steps.

## Try the detector

From the repository root, install the dependencies and run the included
checkpoint on a chart PNG:

```bash
python3.12 -m pip install -r requirements.txt -r requirements-model.txt
python3.12 -m src.baseline.yolo_bars predict --image path/to/chart.png --output-dir output/demo
```

The command writes a detection JSON file and a PNG overlay to `output/demo`.
It uses [`models/yolo_vbars/best.pt`](models/yolo_vbars/best.pt) by default.
Replace `path/to/chart.png` with your image path. For an example image already
in the repository, use `tests/fixtures/contracts/val/example.png`.

## Contributing and documentation

The [experiment guide](docs/experiments.md) covers data preparation, model
evaluation, and the OpenCV benchmarks. The [stage contracts](docs/contracts.md)
describe how detection, OCR, interpretation, and rendering will exchange data.
The original PlotQA dataset must be obtained separately from its
[project page](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md).
