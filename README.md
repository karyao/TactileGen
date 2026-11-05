# TactileGen

TactileGen is a research project exploring how to turn statistical charts into
accessible tactile graphics for blind and low-vision students. The long-term
goal is to identify chart elements, recover their meaning, and produce files
that can be embossed.

## Current status

The project currently prepares [PlotQA](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md)
charts for object detection and includes OpenCV and trained YOLO baselines for
finding chart elements. The application uses YOLO to detect chart elements and
Tesseract to read detected text. Chart interpretation and tactile rendering are
still in development, so the project does not yet generate complete tactile
graphics.

```text
Chart image -> YOLO boxes -> Tesseract text -> frontend results
```

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

Tesseract achieved **99.25%** normalized exact accuracy with authoritative
PlotQA boxes and **98.68%** through the end-to-end YOLO and OCR pipeline. See
the [OCR baseline report](docs/ocr_baseline.md) for the full results.

RapidOCR with PP-OCRv6-small achieved **99.21%** normalized exact accuracy
with authoritative boxes and **99.04%** end to end. See the
[RapidOCR baseline report](docs/rapidocr_baseline.md) for results and
reproduction details. The [OCR comparison](docs/ocr_comparison.md) shows both
engines side by side and quantifies the effect of the shared title fix.

Boxes were also matched at IoU 0.75 and 0.90 where no-mosaic model still scored highly at scores of 0.9595 and 0.9329 respectively. Refer to [experiment guide](docs/experiments.md) for full results.

The YOLO checkpoints used the same model and training settings except for the
named augmentation change. The no-mosaic checkpoint disabled mosaic while retaining HSV augmentation; the no-HSV checkpoint retained mosaic while disabling HSV augmentation. 
See the [ablation benchmark](docs/ablation_benchmark.md) for the experimental plan and reproduction commands.

The detector table and ablation results measure detection, not chart
understanding or tactile output. See the
[experiment guide](docs/experiments.md) for methods and reproduction steps.

## Try the detector

From the repository root, install the detector dependencies and run the
included checkpoint on a chart PNG:

```bash
python3.12 -m pip install -r requirements.txt -r requirements-model.txt
python3.12 -m src.baseline.yolo_bars predict --image path/to/chart.png --output-dir output/demo
```

The detector command writes a detection JSON file and a PNG overlay to
`output/demo`. It uses [`models/yolo_vbars/best.pt`](models/yolo_vbars/best.pt)
by default. Replace `path/to/chart.png` with your image path. For an example
image already in the repository, use
`tests/fixtures/contracts/val/example.png`.

To install the detector together with the Gradio application dependencies,
use the dedicated application requirements file instead:

```bash
python3.12 -m pip install -r requirements-app.txt
python3.12 -m src.gradio_app
```

With the Tesseract executable installed, this starts the frontend with YOLO
detection and Tesseract OCR. Gradio remains optional for users who only need
the command-line detector.

## Read detected text with Tesseract

The standalone OCR command uses the text labels and boxes in YOLO's detection
JSON to crop the original image. The frontend performs the same handoff in
memory without requiring an intermediate JSON file. Install the optional
Python dependency plus the Tesseract executable, then run:

```bash
python3.12 -m pip install -r requirements-ocr.txt
python3.12 -m src.ocr \
  --image path/to/chart.png \
  --detections output/demo/chart_detections.json \
  --output output/demo/chart_ocr.json \
  --language eng
```

The OCR JSON stores recognized text and confidence linked by detection ID.
YOLO's detection JSON remains the sole source of roles and box coordinates.

## Try PP-OCRv6 with RapidOCR

RapidOCR provides a local ONNX Runtime build of PP-OCRv6. The integration uses
only its recognition and orientation stages: YOLO remains the sole source of
text boxes. Like Tesseract, it expands `chart_title` boxes into the shared,
neighbor-aware title strip before recognition, while all other roles retain
their padded YOLO crops. Install the optional backend and select it on the same
OCR command:

```bash
python3.12 -m pip install -r requirements-rapidocr.txt
python3.12 -m src.ocr \
  --engine rapidocr \
  --image path/to/chart.png \
  --detections output/demo/chart_detections.json \
  --output output/demo/chart_rapidocr.json
```

The backend uses the multilingual PP-OCRv6-small recognition model. The
`--language` option remains accepted because it is part of the shared OCR
interface, but PP-OCRv6 does not use Tesseract language-pack codes. Tesseract
remains the default engine, and the Gradio application continues to use it.

## Contributing and documentation

The [experiment guide](docs/experiments.md) covers data preparation, model
evaluation, and the OpenCV benchmarks. The
[OCR baseline report](docs/ocr_baseline.md) records the Tesseract accuracy
results, and the [RapidOCR baseline report](docs/rapidocr_baseline.md) records
the PP-OCRv6 results. The [OCR comparison](docs/ocr_comparison.md) summarizes
their differences. The [stage contracts](docs/contracts.md) describe how
detection, OCR, interpretation, and rendering exchange data. The original
PlotQA dataset must be obtained separately from its
[project page](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md).

## Real-chart data 

PlotQA charts are synthetic. For a real-chart check we use the refined PubMed
Central (PMC) set from ChartDete: real charts from open-access papers, with
18 element classes in COCO format. The data is **not stored in this repository**
(`data/` is gitignored). Download it yourself:

```bash
pip install huggingface_hub
hf download Pengyu965/ChartDete-PMC2022 pmc_coco.zip --repo-type dataset --local-dir data/pmc_2022
unzip -oq data/pmc_2022/pmc_coco.zip -d data/pmc_2022/
```

You should get `data/pmc_2022/pmc_coco/element_detection/` (train, val and
split3_test images plus `.json` files) and `plots_detection/` (bar-chart subset).
If `hf` is not found, run `pip install -U huggingface_hub`. A Google Drive copy
is linked from the [ChartDete repository](https://github.com/pengyu965/ChartDete).

Notes:
- These are scientific-paper charts, not textbook charts.
- The 18 ChartDete classes do not match our 15 roles one-to-one; see the class
  mapping before converting to YOLO format. There are no bar or line classes in
  `element_detection`.
- **License:** the ChartDete README states the PMC data comes from the ICPR 2022
  CHART-Infographics competition and is licensed CC BY-NC-SA 4.0. Do not
  redistribute it in this repository. Credit the sources below if you publish
  results that use it.

Credit: Yan, Ahmed and Doermann, "Context-Aware Chart Element Detection," ICDAR
2023 ([arXiv 2305.04151](https://arxiv.org/abs/2305.04151)); ICPR 2022
CHART-Infographics competition.
