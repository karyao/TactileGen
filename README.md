# TactileGen

ML pipeline that automatically converts statistical charts into accessible, embosser-ready tactile graphics for blind and low-vision students.

## Set up
1. Download images and train annotations from [PlotQA Dataset](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md)
2. Run `python3 src/dataset/prepare_plotqa.py` from the repository root to get data ready for YOLO.

To prepare only vertical categorical bar charts, use a new output directory:

```bash
python3 src/dataset/prepare_plotqa.py --chart-types vbar_categorical --output dataset_yolo_vbars
```

Pass multiple types after `--chart-types` to include each of them, for example
`--chart-types vbar_categorical hbar_categorical`. Supported types are
`hbar_categorical`, `vbar_categorical`, `dot_line`, and `line`. Without the option,
all four types are included. Filtered runs require an output directory with no
existing images or labels.

## Datasets 

PlotQA: https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md
- Already has a training / validation / test split for us.
