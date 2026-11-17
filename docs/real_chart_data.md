# Real-chart data

PlotQA charts are synthetic. For a real-chart check we use the refined PubMed
Central (PMC) set from ChartDete: real charts from open-access papers, with
18 element classes in COCO format. The data is **not stored in this repository**
(`data/` is gitignored). Download it yourself:

```bash
pip install huggingface_hub
hf download Pengyu965/ChartDete-PMC2022 pmc_coco.zip --repo-type dataset --local-dir data/pmc_2022
unzip -oq data/pmc_2022/pmc_coco.zip -d data/pmc_2022/
```

You should get `data/pmc_2022/pmc_coco/element_detection/` (train, val and
split3_test images plus `.json` files) and `plots_detection/` (bar-chart subset).
If `hf` is not found, run `pip install -U huggingface_hub`. A Google Drive copy
is linked from the [ChartDete repository](https://github.com/pengyu965/ChartDete).

Notes:
- These are scientific-paper charts, not textbook charts.
- The 18 ChartDete classes do not match our 15 roles one-to-one; see the class
  mapping before converting to YOLO format. There are no bar or line classes in
  `element_detection`.
- **License:** the ChartDete README states the PMC data comes from the ICPR 2022
  CHART-Infographics competition and is licensed CC BY-NC-SA 4.0. Do not
  redistribute it in this repository. Credit the sources below if you publish
  results that use it.

Credit: Yan, Ahmed and Doermann, "Context-Aware Chart Element Detection," ICDAR
2023 ([arXiv 2305.04151](https://arxiv.org/abs/2305.04151)); ICPR 2022
CHART-Infographics competition.
