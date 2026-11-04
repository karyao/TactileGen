# Tesseract OCR baseline

## Result summary

Tesseract was evaluated on all 11,242 images in the leakage-cleaned PlotQA
vertical-bar test split. There are two primary evaluation modes:

| Evaluation | Input to Tesseract | Normalized exact accuracy | Character error rate | Word error rate |
| --- | --- | ---: | ---: | ---: |
| Oracle OCR | Authoritative PlotQA text boxes | **96.41%** | **1.90%** | **2.30%** |
| End-to-end | Text boxes predicted by YOLO | **95.57%** | **2.58%** | **3.46%** |

The **end-to-end result represents the frontend pipeline**:

```text
Image -> YOLO text boxes -> Tesseract -> recognized text
```

The evaluator compares the frontend's OCR results with the correct PlotQA
text. It does not change the frontend pipeline or its output.

The **oracle test uses a different evaluation path**: it skips YOLO and gives
Tesseract the correct PlotQA boxes. This shows how accurately Tesseract reads
text when the boxes are correct.

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

## Chart title clipping

> The headline numbers in this document and in the README were measured before
> this change and are unchanged. They will be updated after the full-split
> rerun; the chart-title results below come from a 300-image sample only.

**Problem.** PlotQA's `chart_title` annotation boxes end short of the drawn
text (for example "...Agricultural sector" is boxed as "...Agricultural"). The
oracle track was clipped too, so this is not a YOLO or Tesseract fault alone.
YOLO was trained on those boxes and inherits the short ones; end-to-end also
clipped the left end ("Amount spent" read as "unt spent").

**Fix.** For `chart_title` only, `src/tesseract_ocr.py` OCRs a full-width strip:
x from 0 to the image width, y from the box top to the box bottom plus vertical
padding. Text detected in the same vertical band is skipped: the strip starts
after text to the left of the title (y tick labels, the rotated y-axis title)
and stops before text to the right (a legend at the top right). Text
overlapping the title box itself is ignored. Upscaling, grayscale/Otsu
candidates and `--psm 7` are unchanged, and all other roles crop as before.
`_TITLE_FULL_WIDTH_STRIP` turns the behavior off. The 40 px margin kept clear
of legend labels (to avoid the legend swatch and frame) was derived from three
charts' annotations and is unverified on others.

**Results** on the same 300 test images (`--limit 300`, before -> after):

| Measure | Oracle | End-to-end |
| --- | --- | --- |
| Title normalized exact accuracy | 0.8667 -> 0.9733 | 0.7867 -> 0.9733 |
| Title character error rate | 1.33% -> 0.28% | 3.60% -> 1.82% |
| Images with all labels correct | 0.78 -> 0.8767 | 0.6533 -> 0.8033 |

All other roles are unchanged to the digit.

**Remaining misses** (20 rows in `output/ocr_title_fix/regions.csv`):

- 8 charts whose long titles are cut off by the image's right edge in the
  source PNG, counted in both tracks. In `test/22514.png` the title ends
  "...Central Eu"; the missing letters are not in the pixels. This was
  confirmed by viewing 22514.png; the other 7 charts were not individually
  viewed.
- 4 extra YOLO `chart_title` detections, scored as spurious although their
  text is correct.

**Caveats.** Only 300 images were measured, not the full 11,242; the full-split
rerun is pending. The 40 px legend margin needs checking on more charts.

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

The command generates machine-readable results at
`output/ocr_test/metrics.json` and per-region comparisons at
`output/ocr_test/regions.csv`. The `output/` directory is intentionally ignored
by Git, so these generated files are available locally after running the
evaluation but are not published in this repository.

