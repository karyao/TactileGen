# Download and prepare datasets

Run commands from the repository root with Python 3.12. Dataset assets stay in
ignored `data/`; model/OCR dependencies are separate from download dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dataset.txt
python -m src.dataset.download plotqa --splits val
python -m src.dataset.download chartqa --splits val
```

Omit `--splits` for train, val, and test. Use `--output /path/to/dataset` to
change the destination, or `--download-only` to cache assets without extracting.
ChartQA ships as one ZIP: selecting val still downloads the full ZIP, but only
extracts val. PlotQA downloads only the requested splits. No account or token
is required for the public sources as checked on 2026-11-17.

## Sources, versions, and licenses

The checked-in [source manifest](../src/dataset/sources.json) records exact asset
identifiers. These downloads are dataset files, not GitHub source checkouts.

| Dataset | Official source and version | Dataset terms |
| --- | --- | --- |
| PlotQA | [Authors' download page](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md); official split image archives and detection JSONs, identified by Drive IDs | [Authors' README](https://github.com/NiteshMethani/PlotQA#license) specifies CC-BY-4.0 for datasets and MIT for code/models |
| ChartQA | [Authors' repository](https://github.com/vis-nlp/ChartQA) links the full annotated [Hugging Face release](https://huggingface.co/datasets/ahmed-masry/ChartQA); `ChartQA Dataset.zip` at revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` | Hugging Face dataset card declares GPL-3.0; retain upstream attribution and review original chart-source terms when redistributing |

ChartQA's ZIP is 875,370,872 bytes; the manifest includes its published SHA-256.
PlotQA's detection download page has no numbered asset release or published
checksums. Drive IDs identify files but their contents can change; local
receipts record SHA-256 and size on first download, not an upstream authenticity
guarantee. PlotQA QA-v1/v2 collections are excluded because the detector does
not use question-answer pairs.

Allow space for downloaded archives plus extracted files. ChartQA also keeps
a Hugging Face local cache, so budget at least twice the ZIP size plus extracted
splits. PlotQA includes 224,377 images across all splits; its archive sizes are
not published on the download page. Start with val and check disk space before
requesting all splits. YOLO conversion/cleanup uses hard links on the same
filesystem, avoiding another full PNG copy.

## Output and reruns

```text
data/plotqa/
  downloads/                   # original files and download receipts
  train/                       # similarly val/, test/
    png/<image_index>.png
    annotations.json
    .setup.json                # source identity and extracted file sizes

data/chartqa/
  downloads/                   # ZIP, receipts, Hugging Face local cache
  train/                       # similarly val/, test/
    png/<chart>.png
    annotations/<chart>.json
    tables/<chart>.csv
    train_human.json
    train_augmented.json
    .setup.json
```

Repeat the same command to verify and reuse cached downloads. Cache verification
hashes the downloaded bytes, so a rerun can take time even without network
activity. Interrupted transfers restart on rerun (Hugging Face may reuse its own
partial cache). Corrupt managed downloads are fetched again. Extraction stages
in a temporary directory and publishes a split only after validation.

Existing split directories are reused only when their setup receipt and file
sizes still match. Modified, incomplete, or manually installed splits cause an
error; choose a new output directory to preserve them. Extracted files are
checked by path and size, not rehashed. Run only one downloader per output at a
time. Archives cannot extract absolute/traversal paths, links, or device files.

If Drive returns an access/quota error, wait and retry or inspect the official
link. HTML error pages are rejected. Network and disk failures leave completed
cached assets available for the next attempt. If an official link disappears,
update the manifest after checking the authors' source; do not silently switch
releases or mirrors.

## PlotQA: preparation and evaluation from a fresh clone

Install the detector runtime, download all official splits, then prepare vertical
bars. Keep raw data and YOLO output on the same filesystem for hard links.
Use new output directories for conversion and cleanup.

```bash
python -m pip install -r requirements-model.txt
python -m src.dataset.download plotqa
for split in train val test; do
  python -m src.dataset.prepare_plotqa \
    --annotations "data/plotqa/$split/annotations.json" \
    --images "data/plotqa/$split/png" \
    --output dataset_yolo_vbars --split "$split" \
    --chart-types vbar_categorical
done
python -m src.dataset.remove_split_leakage \
  --source dataset_yolo_vbars --output dataset_yolo_vbars_clean
python -m src.baseline.yolo_bars evaluate \
  --data-root dataset_yolo_vbars_clean --split test \
  --weights models/yolo_vbars/best.pt --imgsz 640 \
  --confidence 0.25 --nms-iou 0.70 --match-iou 0.5 \
  --limit 8 --output-dir output/plotqa_setup_smoke
```

Remove `--limit 8` and choose a new output directory for full evaluation. The
tracked vertical-bar checkpoint SHA-256 is
`fc7869ec5ccbdffadaa8bc247323b56bc291283c4301bcfe8678bba0561cf83e`.
Cleanup retains exact duplicates in test before val before train. Conversion and
cleanup do not sample, so no random seed is needed. For training/subset runs,
use explicit `--seed 0`; see [experiments](experiments.md) and the
[ablation benchmark](ablation_benchmark.md) for training settings and provenance
limitations of existing checkpoints. Do not infer held-out accuracy merely
from successfully running evaluation.

## ChartQA: setup verification

The full annotated ZIP includes PNGs, chart annotations, underlying CSV tables,
and human/augmented QA files. Setup verifies required directories and checks QA
image references. Rerun the download command to check the completed split.
The authors describe bounding-box annotations as noisy; inspect them before
using them as evaluation ground truth. Convert the annotated bar charts to the
same 15 YOLO classes as PlotQA with:

```bash
python -m src.dataset.download chartqa
for split in train val test; do
  python -m src.dataset.prepare_chartqa \
    --annotations "data/chartqa/$split/annotations" \
    --images "data/chartqa/$split/png" \
    --output dataset_yolo_chartqa --split "$split"
done
```

Vertical and horizontal bars are included by default. Pie charts are skipped
because the shared class list has no pie-slice class. Output images and labels
use the same split layout and class IDs as PlotQA, so both image roots can be
listed in an Ultralytics dataset YAML for joint training. Keep each dataset's
official splits and check for duplicate images across sources before training.
