# Tesseract OCR baseline

## Result summary

Tesseract was evaluated on all 11,242 images in the leakage-cleaned PlotQA
vertical-bar test split. There are two primary evaluation modes:

| Evaluation | Input to Tesseract | Normalized exact accuracy | Character error rate | Word error rate |
| --- | --- | ---: | ---: | ---: |
| Oracle OCR | Authoritative PlotQA text boxes | **99.25%** | **0.48%** | **0.56%** |
| End-to-end | Text boxes predicted by YOLO | **98.68%** | **1.02%** | **1.51%** |

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
- OCR on matched YOLO boxes only: **98.77% normalized exact accuracy**,
  **0.55% CER**, and **0.85% WER**.
- YOLO produced 180,584 matched text boxes, 147 missed boxes, and 1,060 extra
  boxes. The extra-box count was 1,065 in the earlier run; the small difference
  is not caused by the title change, and its cause is unknown.
- A complete image had every label correct in 89.04% of oracle cases and
  81.12% of end-to-end cases (47.15% and 39.35% before the title fix). This is
  deliberately strict: one incorrect label
  makes the entire image incorrect.

The matched-box score uses OCR results already produced by the end-to-end run,
but excludes missed and extra detections. It is retained only for diagnosing
whether errors arise from recognition, crop localization, or missing boxes.

## Accuracy by text role

| Text role | Oracle exact accuracy | End-to-end exact accuracy | Oracle CER | End-to-end CER |
| --- | ---: | ---: | ---: | ---: |
| Chart title | 98.92% | 98.93% | 0.12% | 1.14% |
| Legend label | 99.76% | 99.72% | 0.06% | 0.13% |
| X-axis title | 99.98% | 99.96% | 0.01% | 0.05% |
| X tick label | 98.29% | 98.46% | 1.92% | 1.87% |
| Y-axis title | 98.45% | 97.87% | 0.34% | 0.76% |
| Y tick label | 99.75% | 98.29% | 0.09% | 1.00% |

Chart titles were previously far below the other roles (53.55% oracle, 49.00%
end-to-end exact accuracy). The cause was that PlotQA's title boxes end short of
the drawn text, so both tracks read clipped titles. The full-width strip fixes
this; see [Chart title clipping](#chart-title-clipping).

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

These results were reproduced on a test split rebuilt from the raw PlotQA
annotations with `prepare_plotqa.py` (`dataset_yolo_vbars`). The leakage cleaner
keeps every test image, so the image and region counts are identical (11,242
images, 180,731 unique regions).

Normalized exact matching applies Unicode NFKC normalization, case folding,
trimming, and whitespace collapsing. Punctuation, decimal points, percent
signs, and minus signs remain significant. CER and WER are micro-averaged over
the complete test set; lower values are better.

PlotQA is synthetic and visually consistent. These scores establish the
baseline for this test split, but they do not prove equivalent performance on
real charts with different fonts, backgrounds, resolutions, rotations, or
annotation conventions.

## Chart title clipping

> The full-split results above include this fix. The 300-image table below
> was the first sample, measured before the full-split rerun.

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

**Full-split results** (11,242 images; before -> after):

| Measure | Oracle | End-to-end |
| --- | --- | --- |
| Overall normalized exact accuracy | 96.41% -> 99.25% | 95.57% -> 98.68% |
| Overall character error rate | 1.90% -> 0.48% | 2.58% -> 1.02% |
| Title normalized exact accuracy | 53.55% -> 98.92% | 49.00% -> 98.93% |
| Title character error rate | 4.28% -> 0.12% | 5.70% -> 1.14% |
| Images with all labels correct | 47.15% -> 89.04% | 39.35% -> 81.12% |

**First sample: 300 test images** (`--limit 300`, before -> after):

| Measure | Oracle | End-to-end |
| --- | --- | --- |
| Title normalized exact accuracy | 0.8667 -> 0.9733 | 0.7867 -> 0.9733 |
| Title character error rate | 1.33% -> 0.28% | 3.60% -> 1.82% |
| Images with all labels correct | 0.78 -> 0.8767 | 0.6533 -> 0.8033 |

All other roles are unchanged to the digit.

**Remaining misses in the 300-image sample** (20 rows in
`output/ocr_title_fix/regions.csv`):

- 8 charts whose long titles are cut off by the image's right edge in the
  source PNG, counted in both tracks. In `test/22514.png` the title ends
  "...Central Eu"; the missing letters are not in the pixels. This was
  confirmed by viewing 22514.png; the other 7 charts were not individually
  viewed.
- 4 extra YOLO `chart_title` detections, scored as spurious although their
  text is correct.

**Caveat.** The 40 px margin kept clear of legend labels needs checking on more
charts.

### Remaining title errors

About 1.1% of titles still fail. At least one confirmed case
(`test/22514.png`) is a title cut off by the image's right edge, so the missing
letters are not in the pixels; the other cases have not been individually
checked. The end-to-end versus oracle title CER gap (1.14% against 0.12%) is
probably the 83 extra title detections scored as spurious; this was not tested.

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

