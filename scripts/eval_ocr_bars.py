"""Run the four PlotQA/ChartQA orientation groups with one checkpoint.

Run from the repository root: python -m scripts.eval_ocr_bars --weights models/best_with_chartqa.pt
Use --limit 10 --output-dir output/ocr_bars_smoke for a short first run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.baseline.ocr import ENGINE_FACTORIES, evaluate
from src.detection import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--plotqa-annotations", type=Path,
                        default=Path("data/plotqa_eval/test/annotations.json"))
    parser.add_argument("--plotqa-root", type=Path, default=Path("dataset_plotqa_bars_eval"))
    parser.add_argument("--chartqa-annotations", type=Path, default=Path("data/chartqa/test/annotations"))
    parser.add_argument("--chartqa-root", type=Path, default=Path("dataset_chartqa_bars_eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ocr_bars_test"))
    parser.add_argument("--engine", choices=tuple(ENGINE_FACTORIES), default="rapidocr")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, help="Maximum images per group; omit for the full test sets")
    args = parser.parse_args()
    model = load_model(args.weights)
    summary = []
    for dataset, annotations, root, types in (
        ("plotqa", args.plotqa_annotations, args.plotqa_root, ("vbar_categorical", "hbar_categorical")),
        ("chartqa", args.chartqa_annotations, args.chartqa_root, ("v_bar", "h_bar")),
    ):
        for orientation, chart_type in zip(("vertical", "horizontal"), types, strict=True):
            group = f"{dataset}_{orientation}"
            print(f"Evaluating {group}", flush=True)
            report = evaluate(
                model, ENGINE_FACTORIES[args.engine], weights=args.weights,
                annotations=annotations, data_root=root, dataset_format=dataset,
                chart_types=[chart_type], output_dir=args.output_dir / group,
                engine=args.engine, language=args.language, workers=args.workers,
                batch=args.batch, imgsz=args.imgsz, device=args.device, limit=args.limit,
            )
            metrics = report["end_to_end"]["overall"]
            summary.append({
                "group": group, "images": report["dataset"]["images"],
                "text_regions": metrics["regions"],
                "normalized_exact_accuracy": metrics["normalized_exact_accuracy"],
                "character_error_rate": metrics["character_error_rate"],
                "word_error_rate": metrics["word_error_rate"],
                "all_labels_image_accuracy": report["end_to_end"]["all_labels_image_accuracy"],
            })
    path = args.output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Completed all four groups. Summary: {path}", flush=True)


if __name__ == "__main__":
    main()
