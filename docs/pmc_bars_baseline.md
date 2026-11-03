# Bar detection baselines on PMC (`data/pmc_yolo_bars/test`)

Both models were run on the 192 `bar_test` images (3,059 bar boxes) without any
fine-tuning. Boxes are matched one-to-one at the stated IoU. The confidence
threshold was chosen on the 205 val images (best F1 at IoU 0.5 over
0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8); the test split was predicted and
scored once, at that value. Reproduce with
`python -m src.baseline.pmc_bars_baseline --output-dir <dir>`.

**Class handling.**
- *PlotQA YOLO, no mosaic* (`models/best_no_mosaic.pt`, 15 classes): only its
  `vertical_bar` detections count as `bar`. Its `horizontal_bar` class is
  ignored, as specified.
- *COCO zero-shot* (`yolov8s.pt`, 80 COCO classes, none of them a chart
  element): every detection of any class counts as a bar. This is the most
  generous mapping; a strict one predicts nothing and scores exactly 0.

**Orientation** comes from `src/dataset/bar_orientation.py` using the
ground-truth boxes: `ylabel` (the value-axis labels) below the plot area means
horizontal, left of it means vertical, with bar edge alignment as fallback.
Test: 139 vertical images (2,136 bars), 53 horizontal (923 bars), none
undecided.

## Results on the test split

| Model | Subset | Images | IoU | Precision | Recall | F1 | Exact bar count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO zero-shot (conf 0.05) | all | 192 | 0.50 | 0.150 | 0.014 | 0.026 | 0.016 |
| | | | 0.75 | 0.105 | 0.010 | 0.018 | |
| | | | 0.90 | 0.034 | 0.003 | 0.006 | |
| | vertical | 139 | 0.50 | 0.185 | 0.020 | 0.036 | 0.022 |
| | | | 0.75 | 0.134 | 0.015 | 0.026 | |
| | | | 0.90 | 0.043 | 0.005 | 0.008 | |
| | horizontal | 53 | 0.50 | 0.016 | 0.001 | 0.002 | 0.000 |
| | | | 0.75 | 0.000 | 0.000 | 0.000 | |
| | | | 0.90 | 0.000 | 0.000 | 0.000 | |
| PlotQA YOLO, no mosaic (conf 0.70) | all | 192 | 0.50 | 0.688 | 0.468 | 0.557 | 0.125 |
| | | | 0.75 | 0.564 | 0.384 | 0.457 | |
| | | | 0.90 | 0.327 | 0.222 | 0.265 | |
| | vertical | 139 | 0.50 | 0.722 | 0.632 | 0.674 | 0.173 |
| | | | 0.75 | 0.590 | 0.516 | 0.551 | |
| | | | 0.90 | 0.349 | 0.305 | 0.326 | |
| | horizontal | 53 | 0.50 | 0.392 | 0.090 | 0.146 | 0.000 |
| | | | 0.75 | 0.335 | 0.077 | 0.125 | |
| | | | 0.90 | 0.132 | 0.030 | 0.049 | |

Exact bar count is the share of images where the predicted and true bar counts
are equal; it does not depend on IoU.

Val-set F1 at IoU 0.5 for the PlotQA model peaked at 0.516 (conf 0.70); the
COCO model's best was 0.046 (conf 0.05, the lowest value in the grid).

## Synthetic-to-real gap

The PlotQA model on its own synthetic test split (vertical bars only, 11,242
images, conf 0.25, from [experiments.md](experiments.md)) against the same
model on real PMC charts:

| Measure | PlotQA test | PMC test, all | PMC test, vertical only |
| --- | ---: | ---: | ---: |
| F1 at IoU 0.50 | 0.967 | 0.557 | 0.674 |
| F1 at IoU 0.75 | 0.960 | 0.457 | 0.551 |
| F1 at IoU 0.90 | 0.933 | 0.265 | 0.326 |
| Recall at IoU 0.50 | 0.943 | 0.468 | 0.632 |
| Precision at IoU 0.50 | 0.992 | 0.688 | 0.722 |
| Exact bar count | 0.794 | 0.125 | 0.173 |

Caveats: the PlotQA numbers use a different confidence (0.25 vs 0.70 chosen on
PMC val), and PMC charts include grouped and stacked bars and many more bars
per chart, so exact count is a harder target there.

## Fine-tuning on `data/pmc_yolo_bars` (19 classes)

Two variants were fine-tuned with identical settings and compared on val; the
winner was then scored once on test. Reproduce with
`python -m src.baseline.pmc_bars_finetune train --variant {coco,plotqa}`,
then `select`, then `test --variant <winner>`.

- **A (coco):** initial weights `yolov8s.pt` (COCO).
- **B (plotqa):** initial weights `models/best_no_mosaic.pt` (synthetic PlotQA).
- **C (PlotQA + PMC mix):** not run; it needs `dataset_yolo_vbars_clean`, which
  was not available on the training machine.
- **Shared settings**, copied from the PlotQA no-mosaic checkpoint: YOLOv8s,
  50 epochs, image size 640, batch 16, seed 0, mosaic 0, MuSGD, lr 0.01,
  deterministic. Training set: the 1,612 train images.
- **Hardware:** NVIDIA RTX 2000 Ada (16 GB), 32 CPU threads, Python 3.13.13,
  torch 2.14.1 (CUDA 13), ultralytics 8.4.183.
- **Run times:** A 14.7 min, B 14.5 min, run one after the other.

### Val comparison (205 images, confidence 0.40 chosen on val for both)

| Variant | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | F1 at 0.75 | F1 at 0.9 | Exact bar count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (coco) | 0.935 | 0.883 | 0.908 | 0.838 | 0.534 | 0.717 |
| B (plotqa) | 0.899 | 0.850 | 0.874 | 0.793 | 0.530 | 0.678 |

A scored higher than B on every val measure: by 0.034 F1 at IoU 0.5 and 0.039
exact bar count, and by only 0.004 F1 at IoU 0.9. Each variant is one training
run, so these gaps are suggestive rather than conclusive. The IoU 0.9 gap is
within what a different seed could change, and the IoU 0.5 and exact-count gaps
are modest. We cannot say from this whether COCO initialization is generally
better than PlotQA initialization for PMC bars. A was used for the test run
because it led on all three selection criteria.

### The ladder on test (192 images)

All three rows are scored on the same `bar_test` images. The first two rows use
the confidence chosen on val for that model (0.05, 0.70, 0.40).

| Model | Subset | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | F1 at 0.75 | F1 at 0.9 | Exact bar count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO zero-shot | all | 0.150 | 0.014 | 0.026 | 0.018 | 0.006 | 0.016 |
| | vertical | 0.185 | 0.020 | 0.036 | 0.026 | 0.008 | 0.022 |
| | horizontal | 0.016 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| PlotQA model as-is | all | 0.688 | 0.468 | 0.557 | 0.457 | 0.265 | 0.125 |
| | vertical | 0.722 | 0.632 | 0.674 | 0.551 | 0.326 | 0.173 |
| | horizontal | 0.392 | 0.090 | 0.146 | 0.125 | 0.049 | 0.000 |
| Fine-tuned A | all | 0.939 | 0.840 | 0.887 | 0.787 | 0.495 | 0.635 |
| | vertical | 0.951 | 0.840 | 0.892 | 0.783 | 0.498 | 0.691 |
| | horizontal | 0.911 | 0.840 | 0.874 | 0.794 | 0.488 | 0.491 |

Fine-tuning closes most of the gap at IoU 0.5 and removes the vertical versus
horizontal difference in F1. Tight localization (F1 at IoU 0.9 of 0.495, against
0.933 for the PlotQA model on its synthetic test) and exact counting (63.5%
overall, 49.1% on horizontal charts) remain weak.

### Caveats

- **Single seed.** Each variant is one run with seed 0. There is no estimate of
  run-to-run variance.
- **Val chose three things:** the checkpoint (Ultralytics picks the best epoch
  by its val fitness), the confidence threshold, and the winner. Val numbers are
  therefore optimistic for A and B; test was not used for any choice.
- **Test was scored once**, for the winner only. B was never scored on test.
- **PMC is a harder target** than PlotQA: grouped and stacked bars, many more
  bars per chart, and varied real-world styling, so exact count is a stricter
  measure here than on the synthetic set.
- **Orientation** comes from the heuristic in `src/dataset/bar_orientation.py`;
  val has 54 horizontal images and test 53, so per-orientation numbers are noisy.
- The mapping of PlotQA's `vertical_bar` to `bar` applies only to the as-is
  PlotQA row; the fine-tuned models predict `bar` directly.
