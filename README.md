# TactileGen

TactileGen is a research project that turns vertical and horizontal bar charts
into tactile SVGs for blind and low-vision students. It combines YOLO detection,
OCR, chart reconstruction, and braille layouts checked against BANA rules.
Generated graphics still need review before embossing.

```text
Chart image → YOLO detections → OCR → chart data → tactile SVG
```

## Run the app

Use Python 3.12 and run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m src.gradio_app
```

Open the local URL printed in the terminal. Upload a chart, analyze it, review
its text and values, and generate tactile SVGs.

- **OCR:** RapidOCR is the default and may download models on first use. Choose
  RapidOCR or Tesseract in the **OCR engine** dropdown. Tesseract requires its
  executable and language packs installed; start with `--ocr tesseract` to select
  it by default. **Tesseract language** applies only to Tesseract.
- **Detection size:** choose **640**, **960**, or **1280** under **Advanced settings**,
  or start with `--imgsz 960`. The default is 640; the maximum is 1280. Larger
  sizes may help with small elements but use more memory and take longer.
- **Diagnostics:** download the ZIP under **Analysis details** and open `report.md`.
  Save it before starting another analysis or clearing the form.

The app uses [the vertical + horizontal checkpoint](models/yolo_bars_hv/best.pt).
See the [diagnostics guide](docs/analysis_diagnostics.md) for setup,
troubleshooting, and size comparisons.

## Command-line tools

For CLI-only use, install `requirements-ocr.txt` and `requirements-model.txt`;
Gradio is optional. Replace `chart.png` with your image, or use
`tests/fixtures/contracts/val/example.png` to try the detector.

**Detect chart elements:** writes detection JSON and an annotated PNG.
The standalone detector defaults to [the vertical-bar checkpoint](models/yolo_vbars/best.pt);
pass `--weights models/yolo_bars_hv/best.pt` for both bar orientations.

```bash
python -m src.baseline.yolo_bars predict --image chart.png \
  --output-dir output/demo --imgsz 960
```

`predict`, `evaluate`, and `validate` accept `--imgsz` (default 640, maximum 1280).
Evaluation records the requested size in `metrics.json`.

**Read detected text:** standalone OCR defaults to Tesseract; use
`--engine rapidocr` to select RapidOCR, as below.

```bash
python -m src.ocr --engine rapidocr --image chart.png \
  --detections output/demo/chart_detections.json --output output/demo/chart_ocr.json
```

**Generate tactile SVGs:** the image pipeline uses the vertical + horizontal
checkpoint and defaults to Tesseract. Add `--ocr rapidocr` to use RapidOCR.
It prompts for missing text or reports why a chart cannot be reconstructed.

```bash
python -m src.tactile.pipeline --image chart.png --ocr rapidocr \
  --output output/tactile/chart.svg
```

To try the layout without detection or OCR dependencies:

```bash
python -m src.tactile --chart tests/fixtures/tactile/books_by_grade.json \
  --output output/tactile/books_by_grade.svg --show-print
```

## Results and documentation

Experiments use [PlotQA](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md),
which must be downloaded separately. Detection and OCR scores measure individual
stages, not the accuracy or tactile readability of the final graphic.

- [Experiments and reproduction](docs/experiments.md) — datasets, detection results,
  training, and evaluation commands.
- [YOLO ablation benchmark](docs/ablation_benchmark.md) — model comparisons.
- [OCR comparison](docs/ocr_comparison.md) — Tesseract and RapidOCR results.
- [Tactile output](docs/tactile.md) — layouts, BANA references, and limitations.
- [Analysis diagnostics](docs/analysis_diagnostics.md) — reports and troubleshooting.
- [Stage contracts](docs/contracts.md) and [Gradio structure](docs/gradio_structure.md)
  — developer documentation.
- [Real-chart dataset](docs/real_chart_data.md) — ChartDete-PMC downloads, class
  differences, and licensing.
