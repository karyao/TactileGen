# Tesseract OCR baseline

## Result summary

Tesseract was evaluated on all 11,242 images in the leakage-cleaned PlotQA
vertical-bar test split. There are two primary evaluation modes:

| Evaluation | Input to Tesseract | Normalized exact accuracy | Character error rate | Word error rate |
| --- | --- | ---: | ---: | ---: |
| Oracle OCR | Authoritative PlotQA text boxes | **96.41%** | **1.90%** | **2.30%** |
| End-to-end | Text boxes predicted by YOLO | **95.57%** | **2.58%** | **3.46%** |

The **end-to-end result represents the frontend pipeline**. Both use the same
path:

```text
Image -> YOLO text boxes -> Tesseract -> recognized text
```

The evaluator compares the frontend's OCR results with the correct PlotQA
text. It does not change the frontend pipeline or its output.

The oracle test skips YOLO and gives Tesseract the correct PlotQA boxes. This
shows how accurately Tesseract reads text when the boxes are correct.

## Supporting diagnostics

These diagnostics explain the difference between the two primary scores; they
are not additional pipelines.

- YOLO text detection: **99.41% precision**, **99.92% recall**, and
  **99.67% F1** at box IoU 0.50.
- OCR on matched YOLO boxes only: **95.65% normalized exact accuracy**,
  **2.16% CER**, and **2.84% WER**.
- YOLO produced 180,584 matched text boxes, 147 missed boxes, and 1,065 extra
  boxes.
- A complete image had every label correct in 47.15% of oracle cases and
  39.35% of end-to-end cases. This is deliberately strict: one incorrect label
  makes the entire image incorrect.

The matched-box score uses OCR results already produced by the end-to-end run,
but excludes missed and extra detections. It is retained only for diagnosing
whether errors arise from recognition, crop localization, or missing boxes.

## Accuracy by text role

| Text role | Oracle exact accuracy | End-to-end exact accuracy | Oracle CER | End-to-end CER |
| --- | ---: | ---: | ---: | ---: |
| Chart title | 53.55% | 49.00% | 4.28% | 5.70% |
| Legend label | 99.76% | 99.72% | 0.06% | 0.13% |
| X-axis title | 99.98% | 99.96% | 0.01% | 0.05% |
| X tick label | 98.29% | 98.46% | 1.92% | 1.86% |
| Y-axis title | 98.45% | 97.87% | 0.34% | 0.75% |
| Y tick label | 99.71% | 98.27% | 0.10% | 1.01% |

Chart titles have much lower exact-match accuracy than the other roles, while
their character error rate remains relatively low. This indicates that many
title failures are small transcription differences rather than completely
unreadable titles. Because the oracle result already shows this behavior, it
primarily reflects Tesseract recognition rather than YOLO localization.

## Dataset and scoring details

- Source annotations: `data/annotations_test.json`
- Selected images: `dataset_yolo_vbars_clean/images/test`
- Model: `models/yolo_vbars/best.pt`
- Detector confidence: 0.25
- Detector NMS IoU: 0.70
- Ground-truth matching IoU: 0.50
- OCR language: English (`eng`)
- Unique scored regions: 180,731
- Duplicate PlotQA annotations removed: 117,573
- Wholly off-image annotations excluded: 1

Normalized exact matching applies Unicode NFKC normalization, case folding,
trimming, and whitespace collapsing. Punctuation, decimal points, percent
signs, and minus signs remain significant. CER and WER are micro-averaged over
the complete test set; lower values are better.

PlotQA is synthetic and visually consistent. These scores establish the
baseline for this test split, but they do not prove equivalent performance on
real charts with different fonts, backgrounds, resolutions, rotations, or
annotation conventions.

## Reproduction and artifacts

Run the full evaluation from the repository root:

```bash
OMP_THREAD_LIMIT=1 .venv/bin/python -m src.baseline.ocr evaluate \
  --annotations data/annotations_test.json \
  --data-root dataset_yolo_vbars_clean \
  --weights models/yolo_vbars/best.pt \
  --workers 8 \
  --batch 16 \
  --output-dir output/ocr_test
```

The complete machine-readable results are in
[`output/ocr_test/metrics.json`](../output/ocr_test/metrics.json). Per-region
comparisons are in
[`output/ocr_test/regions.csv`](../output/ocr_test/regions.csv).

