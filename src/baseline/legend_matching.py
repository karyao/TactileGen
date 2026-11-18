"""Evaluate in-memory legend-to-bar matching on PlotQA validation data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2

from src.baseline.ocr import match_regions
from src.contracts import (
    Box,
    Detection,
    DetectionFrame,
    OCRFrame,
    RecognizedText,
    validate_unit_interval,
)
from src.detection import DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU, load_model, run_detection
from src.dataset.prepare_plotqa import png_size
from src.legend_matcher import (
    _classify_legend_scores,
    _score_legend_to_bars,
    match_legend_to_bars,
)
from src.ocr import OCRBackend, run_ocr


DEFAULT_ANNOTATIONS = Path("data/annotations_validation.json")
DEFAULT_DATA_ROOT = Path("dataset_yolo_vbars_clean")
DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
DEFAULT_MATCH_IOU = 0.5
ERROR_GRID = (0.03, 0.05, 0.075, 0.10)
MARGIN_GRID = (0.01, 0.02, 0.03)
CSV_FIELDS = (
    "track", "source_key", "truth_bar_id", "predicted_bar_id", "iou",
    "expected_series", "predicted_series", "status", "match_score",
    "ocr_confidence", "correct",
)


@dataclass(frozen=True)
class TruthChart:
    path: Path
    frame: DetectionFrame
    ocr: OCRFrame
    series_by_bar: dict[str, str]


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _clipped_box(raw: dict[str, Any], width: int, height: int) -> Box | None:
    try:
        x, y, box_width, box_height = (
            float(raw[name]) for name in ("x", "y", "w", "h")
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, box_width, box_height)):
        return None
    x1, x2 = sorted((x, x + box_width))
    y1, y2 = sorted((y, y + box_height))
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def load_truth_charts(
    annotations: Path,
    data_root: Path,
    split: str,
    limit: int | None = None,
) -> list[TruthChart]:
    """Load positive-area vertical bars and their authoritative legend labels."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if not annotations.is_file():
        raise FileNotFoundError(f"Missing annotations: {annotations}")
    images = data_root / "images" / split
    if not images.is_dir():
        raise FileNotFoundError(f"Missing image split: {images}")
    available = {path.stem: path for path in images.glob("*.png")}
    raw_annotations = json.loads(annotations.read_text(encoding="utf-8"))
    selected = sorted(
        (
            item for item in raw_annotations
            if item.get("type") == "vbar_categorical"
            and item.get("general_figure_info", {}).get("legend")
            and str(item.get("image_index")) in available
        ),
        key=lambda item: int(item["image_index"]),
    )
    if limit is not None:
        selected = selected[:limit]
    charts = []
    for annotation in selected:
        image_id = str(annotation["image_index"])
        path = available[image_id]
        width, height = png_size(path)
        detections = []
        regions = []
        series_by_bar: dict[str, str] = {}
        legend = annotation["general_figure_info"]["legend"]
        parent_box = _clipped_box(legend["bbox"], width, height)
        if parent_box is not None:
            detections.append(Detection("legend", "legend", parent_box, 1.0))
        labels_by_model = {}
        for index, item in enumerate(legend["items"]):
            swatch_box = _clipped_box(item["preview"]["bbox"], width, height)
            label_box = _clipped_box(item["label"]["bbox"], width, height)
            if swatch_box is None or label_box is None:
                continue
            swatch_id, label_id = f"swatch-{index}", f"label-{index}"
            text = str(item["label"]["text"])
            detections.extend((
                Detection(swatch_id, "legend_swatch", swatch_box, 1.0),
                Detection(label_id, "legend_label", label_box, 1.0),
            ))
            regions.append(RecognizedText(label_id, text, 1.0))
            labels_by_model[str(item["model"])] = text
        for model_index, model in enumerate(annotation["models"]):
            expected = labels_by_model.get(str(model["name"]))
            if expected is None:
                continue
            for bar_index, raw_box in enumerate(model["bboxes"]):
                bar_box = _clipped_box(raw_box, width, height)
                if bar_box is None:
                    continue
                bar_id = f"bar-{model_index}-{bar_index}"
                detections.append(Detection(bar_id, "vertical_bar", bar_box, 1.0))
                series_by_bar[bar_id] = expected
        if not series_by_bar:
            continue
        source_key = f"{split}/{path.name}"
        frame = DetectionFrame(source_key, width, height, tuple(detections))
        charts.append(TruthChart(
            path, frame, OCRFrame(source_key, tuple(regions)), series_by_bar
        ))
    if not charts:
        raise ValueError("No vertical-bar charts with usable legends were found")
    return charts


def _empty_counts() -> dict[str, Any]:
    return {
        "truth_bars": 0, "emitted": 0, "emitted_on_truth": 0,
        "correct": 0, "incorrect": 0, "exact_charts": 0,
        "charts": 0, "statuses": Counter(),
        "bar_tp": 0, "bar_fp": 0, "bar_fn": 0,
    }


def _report(counts: dict[str, Any]) -> dict[str, Any]:
    precision = counts["correct"] / counts["emitted"] if counts["emitted"] else 0.0
    recall = counts["correct"] / counts["truth_bars"] if counts["truth_bars"] else 0.0
    coverage = (
        counts["emitted_on_truth"] / counts["truth_bars"]
        if counts["truth_bars"] else 0.0
    )
    accepted_accuracy = (
        counts["correct"] / counts["emitted_on_truth"]
        if counts["emitted_on_truth"] else 0.0
    )
    bar_precision = (
        counts["bar_tp"] / (counts["bar_tp"] + counts["bar_fp"])
        if counts["bar_tp"] + counts["bar_fp"] else 0.0
    )
    bar_recall = (
        counts["bar_tp"] / (counts["bar_tp"] + counts["bar_fn"])
        if counts["bar_tp"] + counts["bar_fn"] else 0.0
    )
    return {
        "charts": counts["charts"],
        "truth_bars": counts["truth_bars"],
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "emitted_assignments": counts["emitted"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "coverage": coverage,
        "accuracy_among_accepted_truth_bars": accepted_accuracy,
        "all_bars_correct_chart_accuracy": (
            counts["exact_charts"] / counts["charts"] if counts["charts"] else 0.0
        ),
        "statuses": dict(sorted(counts["statuses"].items())),
        "bar_detection": {
            "true_positive": counts["bar_tp"],
            "false_positive": counts["bar_fp"],
            "false_negative": counts["bar_fn"],
            "precision": bar_precision,
            "recall": bar_recall,
            "f1": (
                2 * bar_precision * bar_recall / (bar_precision + bar_recall)
                if bar_precision + bar_recall else 0.0
            ),
        },
    }


def _assignment_outcome(assignment, expected: str) -> tuple[bool, bool]:
    emitted = assignment.status == "matched"
    correct = (
        emitted
        and _normalized(assignment.series_name or "") == _normalized(expected)
    )
    return emitted, correct


def _count_assignment(
    counts: dict[str, Any], assignment, expected: str, *, on_truth: bool
) -> bool:
    emitted, correct = _assignment_outcome(assignment, expected)
    counts["statuses"][assignment.status] += 1
    counts["emitted"] += emitted
    counts["emitted_on_truth"] += emitted and on_truth
    counts["correct"] += correct
    counts["incorrect"] += emitted and not correct
    return correct


def _score_oracle_mapping(chart: TruthChart, mapping):
    counts = _empty_counts()
    counts["charts"] = 1
    counts["truth_bars"] = len(chart.series_by_bar)
    counts["bar_tp"] = len(chart.series_by_bar)
    rows = []
    all_correct = True
    for assignment in mapping.assignments:
        expected = chart.series_by_bar[assignment.bar_id]
        correct = _count_assignment(counts, assignment, expected, on_truth=True)
        all_correct = all_correct and correct
        rows.append({
            "track": "oracle", "source_key": chart.frame.source_key,
            "truth_bar_id": assignment.bar_id, "predicted_bar_id": assignment.bar_id,
            "iou": 1.0, "expected_series": expected,
            "predicted_series": assignment.series_name or "",
            "status": assignment.status,
            "match_score": assignment.match_score if assignment.match_score is not None else "",
            "ocr_confidence": 1.0, "correct": correct,
        })
    counts["exact_charts"] = int(all_correct)
    return counts, rows


def _score_oracle(chart: TruthChart, image, max_error: float, min_margin: float):
    mapping = match_legend_to_bars(
        image, chart.frame, chart.ocr, max_error=max_error, min_margin=min_margin
    )
    return _score_oracle_mapping(chart, mapping)


def _score_end_to_end(
    chart: TruthChart, image, predicted: DetectionFrame, ocr: OCRFrame,
    match_iou: float, max_error: float, min_margin: float,
):
    mapping = match_legend_to_bars(
        image, predicted, ocr, max_error=max_error, min_margin=min_margin
    )
    predicted_bars = [item for item in predicted.detections if item.role == "vertical_bar"]
    truth_bars = [item for item in chart.frame.detections if item.role == "vertical_bar"]
    matches, false_predictions, false_truth = match_regions(
        predicted_bars, truth_bars, match_iou
    )
    assignment_by_bar = {item.bar_id: item for item in mapping.assignments}
    ocr_confidence = {item.detection_id: item.confidence for item in ocr.regions}
    by_truth = {truth.id: (prediction, overlap) for prediction, truth, overlap in matches}
    counts = _empty_counts()
    counts.update({
        "charts": 1, "truth_bars": len(truth_bars), "bar_tp": len(matches),
        "bar_fp": len(false_predictions), "bar_fn": len(false_truth),
    })
    rows = []
    all_correct = not false_predictions and not false_truth
    for truth in truth_bars:
        expected = chart.series_by_bar[truth.id]
        matched = by_truth.get(truth.id)
        if matched is None:
            all_correct = False
            rows.append({
                "track": "end_to_end", "source_key": chart.frame.source_key,
                "truth_bar_id": truth.id, "predicted_bar_id": "", "iou": "",
                "expected_series": expected, "predicted_series": "",
                "status": "missed_detection", "match_score": "",
                "ocr_confidence": "", "correct": False,
            })
            continue
        prediction, overlap = matched
        assignment = assignment_by_bar[prediction.id]
        correct = _count_assignment(counts, assignment, expected, on_truth=True)
        all_correct = all_correct and correct
        rows.append({
            "track": "end_to_end", "source_key": chart.frame.source_key,
            "truth_bar_id": truth.id, "predicted_bar_id": prediction.id,
            "iou": overlap, "expected_series": expected,
            "predicted_series": assignment.series_name or "",
            "status": assignment.status,
            "match_score": assignment.match_score if assignment.match_score is not None else "",
            "ocr_confidence": (
                ocr_confidence.get(assignment.label_id, "")
                if assignment.label_id is not None else ""
            ),
            "correct": correct,
        })
    for prediction in false_predictions:
        assignment = assignment_by_bar[prediction.id]
        _count_assignment(counts, assignment, "", on_truth=False)
        rows.append({
            "track": "end_to_end", "source_key": chart.frame.source_key,
            "truth_bar_id": "", "predicted_bar_id": prediction.id, "iou": "",
            "expected_series": "", "predicted_series": assignment.series_name or "",
            "status": assignment.status, "match_score": (
                assignment.match_score if assignment.match_score is not None else ""
            ), "ocr_confidence": (
                ocr_confidence.get(assignment.label_id, "")
                if assignment.label_id is not None else ""
            ),
            "correct": False,
        })
    counts["exact_charts"] = int(all_correct)
    return counts, rows


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "truth_bars", "emitted", "emitted_on_truth", "correct", "incorrect",
        "exact_charts", "charts", "bar_tp", "bar_fp", "bar_fn",
    ):
        target[key] += source[key]
    target["statuses"].update(source["statuses"])


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_evaluation(
    output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output_dir,
        prefix=".assignments.", suffix=".csv", delete=False,
    ) as stream:
        csv_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        csv_path.replace(output_dir / "assignments.csv")
        _write_json_atomic(output_dir / "metrics.json", report)
    finally:
        csv_path.unlink(missing_ok=True)


def _backend(engine: str) -> OCRBackend:
    if engine == "tesseract":
        from src.tesseract_ocr import TesseractBackend
        return TesseractBackend()
    if engine == "rapidocr":
        from src.rapidocr_ocr import RapidOCRBackend
        return RapidOCRBackend()
    raise ValueError(f"Unsupported OCR engine: {engine}")


def evaluate(
    *, annotations: Path, data_root: Path, split: str, output_dir: Path,
    track: str = "both", weights: Path = DEFAULT_WEIGHTS,
    engine: str = "tesseract", language: str = "eng", limit: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE, nms_iou: float = DEFAULT_NMS_IOU,
    match_iou: float = DEFAULT_MATCH_IOU, max_error: float = 0.03,
    min_margin: float = 0.01, device: str | None = None,
) -> dict[str, Any]:
    if track not in {"oracle", "end_to_end", "both"}:
        raise ValueError("track must be oracle, end_to_end, or both")
    if engine not in {"tesseract", "rapidocr"}:
        raise ValueError("engine must be tesseract or rapidocr")
    confidence = validate_unit_interval(confidence, "confidence")
    nms_iou = validate_unit_interval(nms_iou, "nms_iou")
    match_iou = validate_unit_interval(match_iou, "match_iou")
    max_error = validate_unit_interval(max_error, "max_error")
    min_margin = validate_unit_interval(min_margin, "min_margin")
    charts = load_truth_charts(annotations, data_root, split, limit)
    run_oracle = track in {"oracle", "both"}
    run_end = track in {"end_to_end", "both"}
    model = load_model(weights) if run_end else None
    backend = _backend(engine) if run_end else None
    totals = {name: _empty_counts() for name in ("oracle", "end_to_end")}
    rows = []
    for index, chart in enumerate(charts, 1):
        image = cv2.imread(str(chart.path), cv2.IMREAD_COLOR)
        if run_oracle:
            counts, current_rows = _score_oracle(
                chart, image, max_error, min_margin
            )
            _merge(totals["oracle"], counts)
            rows.extend(current_rows)
        if run_end:
            if backend is None:
                raise RuntimeError("OCR backend was not initialized")
            raw_frame = run_detection(
                model, chart.path, confidence=confidence, nms_iou=nms_iou,
                device=device,
            )
            frame = DetectionFrame(
                chart.frame.source_key, raw_frame.width, raw_frame.height,
                raw_frame.detections,
            )
            recognized = run_ocr(image, frame, backend, language)
            counts, current_rows = _score_end_to_end(
                chart, image, frame, recognized, match_iou, max_error, min_margin
            )
            _merge(totals["end_to_end"], counts)
            rows.extend(current_rows)
        if index % 100 == 0 or index == len(charts):
            print(f"Scored {index}/{len(charts)} images", flush=True)
    report = {
        "schema_version": 1, "annotations": str(annotations),
        "data_root": str(data_root), "split": split, "track": track,
        "weights": str(weights), "engine": engine, "language": language,
        "limit": limit, "confidence": confidence, "nms_iou": nms_iou,
        "match_iou": match_iou, "max_error": max_error,
        "min_margin": min_margin,
    }
    if run_oracle:
        report["oracle"] = _report(totals["oracle"])
    if run_end:
        report["end_to_end"] = _report(totals["end_to_end"])
    _write_evaluation(output_dir, report, rows)
    return report


def calibrate(
    *, annotations: Path, data_root: Path, split: str, output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    charts = load_truth_charts(annotations, data_root, split, limit)
    settings = [
        (max_error, min_margin)
        for max_error in ERROR_GRID
        for min_margin in MARGIN_GRID
    ]
    totals = {setting: _empty_counts() for setting in settings}
    for chart in charts:
        image = cv2.imread(str(chart.path), cv2.IMREAD_COLOR)
        scored = _score_legend_to_bars(image, chart.frame, chart.ocr)
        for max_error, min_margin in settings:
            mapping = _classify_legend_scores(
                scored, max_error=max_error, min_margin=min_margin
            )
            counts, _ = _score_oracle_mapping(chart, mapping)
            _merge(totals[(max_error, min_margin)], counts)
    candidates = [
        {
            "max_error": max_error,
            "min_margin": min_margin,
            **_report(totals[(max_error, min_margin)]),
        }
        for max_error, min_margin in settings
    ]
    candidates.sort(key=lambda item: (
        -round(item["f1"], 3), -item["precision"],
        item["max_error"], -item["min_margin"],
    ))
    report = {
        "schema_version": 1, "split": split, "limit": limit,
        "selection": (
            "highest F1 rounded to 3 decimals; ties prefer precision, "
            "lower error, larger margin"
        ),
        "chosen": candidates[0], "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "metrics.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "calibrate"):
        command = commands.add_parser(name)
        command.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
        command.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        command.add_argument("--split", choices=("train", "val", "test"), default="val")
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--limit", type=int)
    scoring = commands.choices["evaluate"]
    scoring.add_argument("--track", choices=("oracle", "end_to_end", "both"), default="both")
    scoring.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    scoring.add_argument("--engine", choices=("tesseract", "rapidocr"), default="tesseract")
    scoring.add_argument("--language", default="eng")
    scoring.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    scoring.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    scoring.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    scoring.add_argument("--max-error", type=float, default=0.03)
    scoring.add_argument("--min-margin", type=float, default=0.01)
    scoring.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = dict(
        annotations=args.annotations, data_root=args.data_root, split=args.split,
        output_dir=args.output_dir, limit=args.limit,
    )
    if args.command == "calibrate":
        report = calibrate(**common)
    else:
        report = evaluate(
            **common, track=args.track, weights=args.weights, engine=args.engine,
            language=args.language, confidence=args.confidence, nms_iou=args.nms_iou,
            match_iou=args.match_iou, max_error=args.max_error,
            min_margin=args.min_margin, device=args.device,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
