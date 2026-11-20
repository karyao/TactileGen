# Evaluate PlotQA and ChartQA bar-chart OCR

The four-group runner measures image → YOLO text boxes → OCR on the official
test splits of both datasets, with separate results for vertical and horizontal
bars. It also scores oracle crops and matched-box OCR using the same definitions
as the original PlotQA vertical benchmark. It does not evaluate reconstructed
numeric values or tactile SVGs.

Run from the repository root. Install `requirements-model.txt` and
`requirements-ocr.txt`. The default inputs match the preparation commands below:

```bash
python -m src.dataset.prepare_plotqa \
  --annotations data/plotqa_eval/test/annotations.json \
  --images data/plotqa_eval/test/png \
  --output dataset_plotqa_bars_eval --split test \
  --chart-types vbar_categorical hbar_categorical

python -m src.dataset.prepare_chartqa \
  --annotations data/chartqa/test/annotations \
  --images data/chartqa/test/png \
  --output dataset_chartqa_bars_eval --split test \
  --chart-types v_bar h_bar
```

Keep the generated `chart_types_test.csv` files with their prepared datasets.
They select the correct orientation before applying a smoke-test limit. The
original annotation files supply text; YOLO labels alone do not contain it.
Use new conversion output directories if a split is already populated.

First run ten images **per group**:

```bash
python -m scripts.eval_ocr_bars \
  --weights models/best_with_chartqa.pt \
  --limit 10 --workers 1 --batch 4 \
  --output-dir output/ocr_bars_smoke
```

Then run every prepared test image, with no sampling or limit:

```bash
python -m scripts.eval_ocr_bars \
  --weights models/best_with_chartqa.pt \
  --workers 8 --batch 16 --imgsz 640 \
  --output-dir output/ocr_bars_test
```

Add `--device 0` for NVIDIA GPU inference, or `--device cpu`. RapidOCR uses CPU
ONNX Runtime sessions; `--device` selects the YOLO device. Reduce batch size for
GPU-memory pressure and worker count for CPU/RAM pressure. Use the inference
size, software versions, and engine of the reference experiment for comparisons;
larger image sizes are not guaranteed to improve scores.

The default OCR engine is RapidOCR, confidence is 0.25, NMS IoU is 0.70, and
ground-truth matching IoU is 0.50. Override input locations with
`--plotqa-root`, `--plotqa-annotations`, `--chartqa-root`, and
`--chartqa-annotations`. These arguments point to the prepared YOLO roots and
the original PlotQA JSON / ChartQA annotation directory, respectively.

Outputs:

```text
output/ocr_bars_test/
  summary.csv
  plotqa_vertical/{metrics.json,regions.csv}
  plotqa_horizontal/{metrics.json,regions.csv}
  chartqa_vertical/{metrics.json,regions.csv}
  chartqa_horizontal/{metrics.json,regions.csv}
```

`summary.csv` contains each group's end-to-end region exact accuracy, character
and word error rates, and strict all-label image accuracy. Scores are fractions;
multiply by 100 for percentages. Group percentages are not averaged together.
Reports include actual image counts, dataset format, selected chart types,
checkpoint path, inference size, and thresholds. Four groups contain more than
the original 11,242-image PlotQA vertical test benchmark; counts come from the
prepared test data. Never substitute training images for missing test images.

Missing annotations, mismatched orientation manifests, conflicting gold text,
and unpaired text/boxes fail explicitly rather than silently dropping images.
ChartQA sections with no text box (such as an absent axis title) do not create
gold regions. Scores measure the available annotated text; incomplete upstream
annotations can make a correct unannotated prediction count as spurious.

The original evaluator remains available for individual groups. For example:

```bash
python -m src.baseline.ocr evaluate \
  --dataset-format chartqa --chart-types h_bar \
  --annotations data/chartqa/test/annotations \
  --data-root dataset_chartqa_bars_eval --split test \
  --weights models/best_with_chartqa.pt --engine rapidocr \
  --output-dir output/chartqa_horizontal
```

The original CLI defaults still select PlotQA vertical bars. This implementation
is verified with synthetic annotation fixtures; a full checkpoint benchmark
requires your downloaded datasets and checkpoint. Its score is not implied by
the older checkpoint's reported accuracy or by test-data preparation.
