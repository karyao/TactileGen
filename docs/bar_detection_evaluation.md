# Compare bar detectors on PlotQA and ChartQA

`src.baseline.yolo_bars evaluate` reads prepared images and standard YOLO labels.
It supports PlotQA numeric filenames and ChartQA filenames such as
`two_col_995.png`, and maps `v_bar`/`h_bar` manifest types to the shared
vertical/horizontal chart types. Original annotation JSONs are not needed for
this detector benchmark.

Use the test datasets prepared with `prepare_plotqa` and `prepare_chartqa`,
keeping their `classes.txt`, labels, and `chart_types_test.csv` files. Run from
the repository root:

```bash
for dataset in plotqa chartqa; do
  for role in vertical_bar horizontal_bar; do
    python -m src.baseline.yolo_bars evaluate \
      --weights models/best_with_chartqa.pt \
      --data-root "dataset_${dataset}_bars_eval" \
      --split test --bar-role "$role" \
      --confidence 0.25 --imgsz 640 --batch 16 \
      --match-iou 0.50 --match-iou 0.75 --match-iou 0.90 \
      --output-dir "output/with_chartqa_${dataset}_${role}" || break 2
  done
done
```

Add `--device 0` for NVIDIA GPU inference. For a first check, add `--limit 8`
and use separate smoke output directories. Omit the limit for the full test
sets. The loop stops at a failed group, preserving completed reports.

Each group's `metrics.json` contains `matching_results` for the three IoUs.
Each result includes precision, recall, F1, exact bar-count accuracy, and actual
image counts. Inference confidence remains 0.25 and NMS IoU remains 0.70, as in
the earlier detector benchmark. Matching and metrics calculations are unchanged;
the compatibility update changes filename ordering and manifest selection only.

Repeat with each checkpoint, changing weights and output directories while
keeping datasets and settings fixed. Compare old published PlotQA vertical
scores against the PlotQA vertical group. ChartQA and horizontal groups extend
that benchmark; compare models only within identical groups. These metrics
measure detection, not OCR or chart-value reconstruction. Standard mAP remains
a separate `validate` run and requires a valid dataset `data.yaml`.
