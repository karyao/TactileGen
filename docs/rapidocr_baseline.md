# RapidOCR PP-OCRv6 baseline

## Result summary

RapidOCR with the multilingual PP-OCRv6-small recognition model was evaluated
on all 11,242 images in the leakage-cleaned PlotQA vertical-bar test split.
RapidOCR's text detector was disabled: the oracle track used authoritative
PlotQA boxes, while the end-to-end track used boxes predicted by YOLO.

| Evaluation | Input to RapidOCR | Normalized exact accuracy | Character error rate | Word error rate |
| --- | --- | ---: | ---: | ---: |
| Oracle OCR | Authoritative PlotQA text boxes | **99.21%** | **0.40%** | **0.81%** |
| End-to-end | Text boxes predicted by YOLO | **99.04%** | **0.81%** | **1.44%** |

See the [OCR comparison](ocr_comparison.md) for a side-by-side view of these
results and the published Tesseract baseline.

The end-to-end result represents this pipeline:

```text
Image -> YOLO text boxes -> RapidOCR PP-OCRv6 -> recognized text
```

The oracle result isolates recognition quality by skipping YOLO localization.
Both tracks use the same normalization and scoring rules as the Tesseract
baseline.

## Supporting diagnostics

- YOLO text detection achieved **99.41% precision**, **99.92% recall**, and
  **99.67% F1** at box IoU 0.50.
- OCR on matched YOLO boxes only achieved **99.12% normalized exact accuracy**,
  **0.36% CER**, and **0.79% WER**.
- YOLO produced 180,584 matched text boxes, 147 missed boxes, and 1,065 extra
  boxes.
- Every label in an image was correct for **88.42%** of oracle images and
  **80.98%** of end-to-end images. This is deliberately strict: one incorrect
  label makes the entire image incorrect.

The matched-box score reuses end-to-end OCR results but excludes missed and
extra detections. It is a diagnostic for separating recognition errors from
localization errors, not a third pipeline.

## Accuracy by text role

| Text role | Oracle exact accuracy | End-to-end exact accuracy | Oracle CER | End-to-end CER |
| --- | ---: | ---: | ---: | ---: |
| Chart title | 98.23% | 98.32% | 0.75% | 1.58% |
| Legend label | 99.01% | 98.25% | 0.28% | 0.60% |
| X-axis title | 99.89% | 99.78% | 0.02% | 0.08% |
| X tick label | 99.13% | 98.88% | 0.29% | 0.41% |
| Y-axis title | 97.09% | 98.09% | 0.20% | 0.23% |
| Y tick label | 99.75% | 99.64% | 0.12% | 0.55% |

The end-to-end Y-axis-title exact score is higher than the oracle score because
the detector's slightly different crops can be easier for the recognizer. The
end-to-end aggregate still counts missed and extra detections.

## Dataset and scoring details

- Source annotations: `data/annotations_test.json`
- Selected images: `dataset_yolo_vbars_clean/images/test`
- Detector: `models/yolo_vbars/best.pt`
- Detector confidence: 0.25
- Detector NMS IoU: 0.70
- Ground-truth matching IoU: 0.50
- OCR engine: RapidOCR 3.10.0 with ONNX Runtime 1.31.1
- Recognition model: multilingual PP-OCRv6-small
- RapidOCR text detection: disabled
- Unique scored regions: 180,731
- Duplicate PlotQA annotations removed: 117,573
- Wholly off-image annotations excluded: 1

Normalized exact matching applies Unicode NFKC normalization, case folding,
trimming, and whitespace collapsing. Punctuation, decimal points, percent
signs, and minus signs remain significant. CER and WER are micro-averaged over
the complete test set; lower values are better.

PlotQA is synthetic and visually consistent. These results establish the
baseline for this test split but do not imply equivalent performance on real
charts with different fonts, backgrounds, resolutions, rotations, or
annotation conventions.

## Chart-title correction

The canonical results above use the shared, neighbor-aware full-width title
strip in `src/ocr_crop.py`. PlotQA title boxes commonly end before the rendered
text, so ordinary padded crops truncate titles.

An earlier full run made before RapidOCR adopted the shared title crop is
preserved locally at `output/rapidocr_test_pre_title_fix`. The correction
changed the principal measurements as follows:

| Measure | Oracle | End-to-end |
| --- | --- | --- |
| Overall normalized exact accuracy | 96.61% -> 99.21% | 96.13% -> 99.04% |
| Overall character error rate | 1.87% -> 0.40% | 2.50% -> 0.81% |
| Title normalized exact accuracy | 53.68% -> 98.23% | 48.92% -> 98.32% |
| Images with every label correct | 50.01% -> 88.42% | 41.49% -> 80.98% |

## Reproduction and artifacts

From the repository root, install the detector and RapidOCR dependencies, then
run the full evaluation:

```bash
python3.12 -m pip install -r requirements-model.txt -r requirements-ocr.txt
python3.12 -m src.baseline.ocr evaluate \
  --engine rapidocr \
  --annotations data/annotations_test.json \
  --data-root dataset_yolo_vbars_clean \
  --weights models/yolo_vbars/best.pt \
  --workers 8 \
  --batch 16 \
  --output-dir output/rapidocr_test
```

Each backend instance uses single-threaded ONNX Runtime sessions, allowing the
eight evaluator workers to use separate CPU cores without oversubscribing the
machine. Worker count affects throughput, not scoring.

The command writes aggregate results to `output/rapidocr_test/metrics.json`
and per-region diagnostics to `output/rapidocr_test/regions.csv`. The
`output/` directory is intentionally ignored by Git, so generated artifacts
remain local.
