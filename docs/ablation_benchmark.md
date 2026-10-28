# YOLO augmentation ablation benchmark

## Plan

Compare the existing vertical-bar baseline with checkpoints trained without
mosaic augmentation and without HSV augmentation. Use the same cleaned PlotQA
test split (`dataset_yolo_vbars_clean`, 11,242 images) for every model.

1. Verify that the two checkpoint files are distinct and that their embedded
   training arguments identify the expected ablation.
2. Smoke-test each checkpoint on eight test images with the project's evaluator.
3. Run full `evaluate` for each checkpoint: vertical-bar precision, recall, F1,
   and exact bar-count accuracy at confidence 0.25 and box IoU 0.5.
4. Run full `validate` for each checkpoint: overall mAP50, overall mAP50–95,
   and vertical-bar mAP50–95.
5. Compare the reports with the existing baseline reports using the same test
   split. Record the commands and measured values below.

The baseline reports already exist at `output/yolo_test/metrics.json` and
`output/yolo_map_test/metrics.json`. Each ablation has its own output directory
to preserve the baseline and prevent report overwrites. A training argument
pointing to a cleaned dataset is useful provenance but cannot, by itself,
prove that the test images were excluded from training.

The three checkpoints record the same cleaned-dataset path, YOLOv8s pretrained
model, 50 epochs, seed 0, image size 640, and batch size 16. Their saved
augmentation settings differ as intended: the baseline has mosaic 1.0 and HSV
(0.015, 0.7, 0.4); `no_mosaic` has mosaic 0 with the baseline HSV settings;
`no_hsv` has mosaic 1.0 with all three HSV values at 0.

## Commands

Run from the repository root with the environment specified in
`requirements-model.txt`. On this machine, it is
`/private/tmp/tactile-yolo-venv/bin/python`.

```bash
PY=/private/tmp/tactile-yolo-venv/bin/python
$PY -m src.baseline.yolo_bars evaluate --weights models/best_no_mosaic.pt --data-root dataset_yolo_vbars_clean --split test --limit 8 --output-dir output/no_mosaic_smoke
$PY -m src.baseline.yolo_bars evaluate --weights models/best_no_hsv.pt --data-root dataset_yolo_vbars_clean --split test --limit 8 --output-dir output/no_hsv_smoke

$PY -m src.baseline.yolo_bars evaluate --weights models/best_no_mosaic.pt --data-root dataset_yolo_vbars_clean --split test --output-dir output/no_mosaic_test
$PY -m src.baseline.yolo_bars validate --weights models/best_no_mosaic.pt --data-root dataset_yolo_vbars_clean --split test --output-dir output/no_mosaic_map_test
$PY -m src.baseline.yolo_bars evaluate --weights models/best_no_hsv.pt --data-root dataset_yolo_vbars_clean --split test --output-dir output/no_hsv_test
$PY -m src.baseline.yolo_bars validate --weights models/best_no_hsv.pt --data-root dataset_yolo_vbars_clean --split test --output-dir output/no_hsv_map_test
```

## Results

Pending benchmark runs.
