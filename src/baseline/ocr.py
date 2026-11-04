"""Benchmark interchangeable OCR engines against authoritative PlotQA text.

The oracle track uses PlotQA boxes and measures recognition alone.  The
end-to-end track uses YOLO boxes and therefore also measures localization.
Run ``python -m src.baseline.ocr evaluate --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import threading
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baseline.opencv_bars import intersection_over_union
from src.contracts import Box, Detection, DetectionFrame, RecognizedText, TEXT_ROLES
from src.dataset.prepare_plotqa import iter_annotations, png_size
from src.detection import (
    DEFAULT_CONFIDENCE,
    DEFAULT_NMS_IOU,
    _validate_threshold,
    check_model_classes,
    detection_frame,
    load_model,
)
from src.ocr import OCRBackend, _read_image, run_ocr


DEFAULT_ANNOTATIONS = Path("data/annotations_test.json")
DEFAULT_DATA_ROOT = Path("dataset_yolo_vbars_clean")
DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
DEFAULT_MATCH_IOU = 0.5
CSV_FIELDS = (
    "track", "source_key", "role", "status", "ground_truth_id", "prediction_id",
    "iou", "reference_text", "predicted_text", "ocr_confidence", "raw_exact",
    "normalized_exact", "character_edits", "reference_characters", "word_edits",
    "reference_words", "gt_x1", "gt_y1", "gt_x2", "gt_y2",
    "pred_x1", "pred_y1", "pred_x2", "pred_y2",
)


@dataclass(frozen=True)
class TruthImage:
    image_id: str
    path: Path
    frame: DetectionFrame
    text_by_id: dict[str, str]
    duplicate_regions: int


@dataclass(frozen=True)
class Comparison:
    reference: str
    prediction: str
    raw_exact: bool
    normalized_exact: bool
    character_edits: int
    reference_characters: int
    word_edits: int
    reference_words: int


@dataclass(frozen=True)
class ImageResult:
    rows: tuple[dict[str, Any], ...]
    oracle: tuple[tuple[str, Comparison], ...]
    matched: tuple[tuple[str, Comparison], ...]
    end_to_end: tuple[tuple[str, Comparison, bool], ...]
    detection: tuple[tuple[str, str], ...]
    oracle_image_exact: bool
    end_to_end_image_exact: bool


class TextMetrics:
    def __init__(self) -> None:
        self.reference_regions = 0
        self.readable_regions = 0
        self.raw_exact = 0
        self.normalized_exact = 0
        self.character_edits = 0
        self.reference_characters = 0
        self.word_edits = 0
        self.reference_words = 0
        self.spurious_regions = 0

    def add(self, comparison: Comparison, *, is_reference: bool = True) -> None:
        if is_reference:
            self.reference_regions += 1
            self.readable_regions += bool(normalize_text(comparison.prediction))
            self.raw_exact += comparison.raw_exact
            self.normalized_exact += comparison.normalized_exact
        else:
            self.spurious_regions += 1
        self.character_edits += comparison.character_edits
        self.reference_characters += comparison.reference_characters
        self.word_edits += comparison.word_edits
        self.reference_words += comparison.reference_words

    def report(self) -> dict[str, int | float]:
        count = self.reference_regions
        return {
            "regions": count,
            "spurious_regions": self.spurious_regions,
            "readable_rate": self.readable_regions / count if count else 0.0,
            "raw_exact_accuracy": self.raw_exact / count if count else 0.0,
            "normalized_exact_accuracy": self.normalized_exact / count if count else 0.0,
            "character_edits": self.character_edits,
            "reference_characters": self.reference_characters,
            "character_error_rate": (
                self.character_edits / self.reference_characters
                if self.reference_characters else 0.0
            ),
            "word_edits": self.word_edits,
            "reference_words": self.reference_words,
            "word_error_rate": (
                self.word_edits / self.reference_words if self.reference_words else 0.0
            ),
        }


class DetectionMetrics:
    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        )

    def add(self, role: str, status: str) -> None:
        self.counts[role][status] += 1
        self.counts["overall"][status] += 1

    @staticmethod
    def _report(counts: dict[str, int]) -> dict[str, int | float]:
        tp, fp, fn = (counts[name] for name in
                      ("true_positive", "false_positive", "false_negative"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
        }

    def report(self) -> dict[str, dict[str, int | float]]:
        return {role: self._report(counts) for role, counts in sorted(self.counts.items())}


def normalize_text(value: str) -> str:
    """Normalize case and whitespace while preserving meaningful punctuation."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return Levenshtein distance using memory linear in the shorter input."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def compare_text(reference: str, prediction: str) -> Comparison:
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)
    reference_words = normalized_reference.split()
    prediction_words = normalized_prediction.split()
    return Comparison(
        reference=reference,
        prediction=prediction,
        raw_exact=reference == prediction,
        normalized_exact=normalized_reference == normalized_prediction,
        character_edits=edit_distance(normalized_reference, normalized_prediction),
        reference_characters=len(normalized_reference),
        word_edits=edit_distance(reference_words, prediction_words),
        reference_words=len(reference_words),
    )


def _plotqa_box(raw: Any, width: int, height: int, context: str) -> Box | None:
    try:
        x, y, box_width, box_height = (float(raw[name]) for name in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PlotQA box for {context}") from exc
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        raise ValueError(f"Non-finite PlotQA box for {context}")
    x1, x2 = sorted((x, x + box_width))
    y1, y2 = sorted((y, y + box_height))
    box = Box(
        min(max(x1, 0.0), width), min(max(y1, 0.0), height),
        min(max(x2, 0.0), width), min(max(y2, 0.0), height),
    )
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        # PlotQA contains a small number of labels drawn wholly beyond the PNG
        # edge. They cannot be cropped or scored; dataset preparation likewise
        # omits boxes that are empty after clipping.
        return None
    return box


def _annotation_regions(annotation: dict[str, Any]) -> Iterator[tuple[str, str, Any]]:
    info = annotation["general_figure_info"]
    title = info["title"]
    yield "chart_title", str(title["text"]), title["bbox"]
    for axis in ("x", "y"):
        axis_info = info[f"{axis}_axis"]
        label = axis_info["label"]
        yield f"{axis}_axis_title", str(label["text"]), label["bbox"]
        values = axis_info["major_labels"]["values"]
        boxes = axis_info["major_labels"]["bboxes"]
        if not isinstance(values, list) or not isinstance(boxes, list) or len(values) != len(boxes):
            raise ValueError(f"{axis}-axis major-label values and boxes differ in length")
        for value, box in zip(values, boxes, strict=True):
            yield f"{axis}_tick_label", str(value), box
    legend = info.get("legend")
    if legend is not None:
        for item in legend["items"]:
            label = item["label"]
            yield "legend_label", str(label["text"]), label["bbox"]


def load_truth_images(
    annotations: Path, data_root: Path, split: str, limit: int | None = None
) -> tuple[list[TruthImage], dict[str, int]]:
    """Join a PlotQA annotation stream to images selected by a prepared split."""
    images_dir = data_root / "images" / split
    paths = sorted(images_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not paths:
        raise ValueError(f"No PNG images found in {images_dir}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        paths = paths[:limit]
    requested = {path.stem: path for path in paths}
    found: dict[str, TruthImage] = {}
    raw_regions = unique_regions = duplicates = excluded_empty_boxes = 0
    for annotation in iter_annotations(annotations):
        image_id = str(annotation.get("image_index"))
        path = requested.get(image_id)
        if path is None:
            continue
        if image_id in found:
            raise ValueError(f"Duplicate annotation for selected image {image_id}")
        if annotation.get("type") != "vbar_categorical":
            raise ValueError(f"Expected vbar_categorical annotation for image {image_id}")
        width, height = png_size(path)
        detections: list[Detection] = []
        text_by_id: dict[str, str] = {}
        seen: set[tuple[str, str, Box]] = set()
        text_for_box: dict[tuple[str, Box], str] = {}
        image_raw = image_excluded = 0
        for role, text, raw_box in _annotation_regions(annotation):
            image_raw += 1
            if not text:
                raise ValueError(f"Empty authoritative text for image {image_id} role {role}")
            box = _plotqa_box(raw_box, width, height, f"image {image_id} role {role}")
            if box is None:
                excluded_empty_boxes += 1
                image_excluded += 1
                continue
            key = (role, text, box)
            if key in seen:
                continue
            seen.add(key)
            previous = text_for_box.setdefault((role, box), text)
            if previous != text:
                raise ValueError(
                    f"Conflicting authoritative text for image {image_id} role {role} box {box}"
                )
            detection_id = f"gt-{len(detections) + 1:05d}"
            detections.append(Detection(detection_id, role, box, 1.0))
            text_by_id[detection_id] = text
        source_key = f"{split}/{path.name}"
        image_duplicates = image_raw - image_excluded - len(detections)
        found[image_id] = TruthImage(
            image_id, path, DetectionFrame(source_key, width, height, tuple(detections)),
            text_by_id, image_duplicates,
        )
        raw_regions += image_raw
        unique_regions += len(detections)
        duplicates += image_duplicates
    missing = requested.keys() - found.keys()
    if missing:
        sample = ", ".join(sorted(missing, key=int)[:5])
        raise ValueError(f"Annotations are missing selected image IDs: {sample}")
    ordered = [found[path.stem] for path in paths]
    return ordered, {
        "images": len(ordered), "raw_regions": raw_regions,
        "unique_regions": unique_regions, "duplicate_regions_removed": duplicates,
        "empty_boxes_excluded": excluded_empty_boxes,
    }


def match_regions(
    predicted: Sequence[Detection], actual: Sequence[Detection], threshold: float
) -> tuple[list[tuple[Detection, Detection, float]], list[Detection], list[Detection]]:
    """Greedily pair same-role boxes by descending IoU without reuse."""
    candidates = []
    for prediction in predicted:
        for truth in actual:
            if prediction.role != truth.role:
                continue
            overlap = intersection_over_union(prediction.box, truth.box)
            if overlap >= threshold:
                candidates.append((overlap, prediction, truth))
    candidates.sort(key=lambda item: (-item[0], item[1].id, item[2].id))
    used_predictions: set[str] = set()
    used_truth: set[str] = set()
    matches = []
    for overlap, prediction, truth in candidates:
        if prediction.id in used_predictions or truth.id in used_truth:
            continue
        used_predictions.add(prediction.id)
        used_truth.add(truth.id)
        matches.append((prediction, truth, overlap))
    unmatched_predictions = [item for item in predicted if item.id not in used_predictions]
    unmatched_truth = [item for item in actual if item.id not in used_truth]
    return matches, unmatched_predictions, unmatched_truth


def _box_columns(prefix: str, box: Box | None) -> dict[str, float | str]:
    return {
        f"{prefix}_x1": "" if box is None else box.x1,
        f"{prefix}_y1": "" if box is None else box.y1,
        f"{prefix}_x2": "" if box is None else box.x2,
        f"{prefix}_y2": "" if box is None else box.y2,
    }


def _row(
    track: str, source_key: str, role: str, status: str,
    comparison: Comparison, truth: Detection | None, prediction: Detection | None,
    confidence: float, overlap: float | None,
) -> dict[str, Any]:
    return {
        "track": track, "source_key": source_key, "role": role, "status": status,
        "ground_truth_id": truth.id if truth else "",
        "prediction_id": prediction.id if prediction else "",
        "iou": "" if overlap is None else overlap,
        "reference_text": comparison.reference, "predicted_text": comparison.prediction,
        "ocr_confidence": confidence, "raw_exact": comparison.raw_exact,
        "normalized_exact": comparison.normalized_exact,
        "character_edits": comparison.character_edits,
        "reference_characters": comparison.reference_characters,
        "word_edits": comparison.word_edits, "reference_words": comparison.reference_words,
        **_box_columns("gt", truth.box if truth else None),
        **_box_columns("pred", prediction.box if prediction else None),
    }


def _score_image(
    truth_image: TruthImage, predicted_frame: DetectionFrame, backend: OCRBackend,
    language: str, match_iou: float,
) -> ImageResult:
    image = _read_image(truth_image.path)
    oracle_ocr = run_ocr(image, truth_image.frame, backend, language)
    oracle_by_id = {item.detection_id: item for item in oracle_ocr.regions}
    predicted_text = [item for item in predicted_frame.detections if item.role in TEXT_ROLES]
    # A detector can reproduce an authoritative box exactly. In that case both
    # tracks would send the same pixels, orientation candidates, role, and
    # language to the backend. Reuse that result instead of starting duplicate
    # Tesseract subprocesses. Differently localized crops are always recognized
    # independently so the end-to-end score remains meaningful.
    oracle_for_crop = {
        (detection.role, detection.box): oracle_by_id[detection.id]
        for detection in truth_image.frame.detections
    }
    uncached = [
        detection for detection in predicted_text
        if (detection.role, detection.box) not in oracle_for_crop
    ]
    uncached_frame = DetectionFrame(
        predicted_frame.source_key, predicted_frame.width, predicted_frame.height,
        tuple(uncached),
    )
    uncached_ocr = run_ocr(image, uncached_frame, backend, language)
    predicted_by_id = {item.detection_id: item for item in uncached_ocr.regions}
    for detection in predicted_text:
        cached = oracle_for_crop.get((detection.role, detection.box))
        if cached is not None:
            predicted_by_id[detection.id] = RecognizedText(
                detection.id, cached.text, cached.confidence
            )
    matches, false_positives, false_negatives = match_regions(
        predicted_text, truth_image.frame.detections, match_iou
    )
    by_truth = {truth.id: (prediction, overlap) for prediction, truth, overlap in matches}
    rows: list[dict[str, Any]] = []
    oracle_metrics = []
    matched_metrics = []
    end_metrics = []
    oracle_exact = True
    end_exact = not false_positives
    for truth in truth_image.frame.detections:
        reference = truth_image.text_by_id[truth.id]
        oracle_region = oracle_by_id[truth.id]
        oracle_comparison = compare_text(reference, oracle_region.text)
        oracle_metrics.append((truth.role, oracle_comparison))
        oracle_exact = oracle_exact and oracle_comparison.normalized_exact
        rows.append(_row(
            "oracle", truth_image.frame.source_key, truth.role, "authoritative_box",
            oracle_comparison, truth, truth, oracle_region.confidence, 1.0,
        ))
        matched = by_truth.get(truth.id)
        if matched is None:
            comparison = compare_text(reference, "")
            end_exact = False
            rows.append(_row(
                "end_to_end", truth_image.frame.source_key, truth.role, "false_negative",
                comparison, truth, None, 0.0, None,
            ))
            end_metrics.append((truth.role, comparison, True))
            continue
        prediction, overlap = matched
        region = predicted_by_id[prediction.id]
        comparison = compare_text(reference, region.text)
        matched_metrics.append((truth.role, comparison))
        end_metrics.append((truth.role, comparison, True))
        end_exact = end_exact and comparison.normalized_exact
        rows.append(_row(
            "end_to_end", truth_image.frame.source_key, truth.role, "true_positive",
            comparison, truth, prediction, region.confidence, overlap,
        ))
    for prediction in false_positives:
        region = predicted_by_id[prediction.id]
        comparison = compare_text("", region.text)
        end_metrics.append((prediction.role, comparison, False))
        rows.append(_row(
            "end_to_end", truth_image.frame.source_key, prediction.role, "false_positive",
            comparison, None, prediction, region.confidence, None,
        ))
    detection = (
        tuple((truth.role, "true_positive") for _, truth, _ in matches)
        + tuple((item.role, "false_positive") for item in false_positives)
        + tuple((item.role, "false_negative") for item in false_negatives)
    )
    return ImageResult(
        tuple(rows), tuple(oracle_metrics), tuple(matched_metrics), tuple(end_metrics),
        detection, oracle_exact, end_exact,
    )


def _text_report(overall: TextMetrics, roles: dict[str, TextMetrics]) -> dict[str, Any]:
    return {
        "overall": overall.report(),
        "by_role": {role: metrics.report() for role, metrics in sorted(roles.items())},
    }


def evaluate(
    model: Any,
    backend_factory: Callable[[], OCRBackend],
    *,
    annotations: Path = DEFAULT_ANNOTATIONS,
    data_root: Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    weights: Path = DEFAULT_WEIGHTS,
    output_dir: Path,
    engine: str = "tesseract",
    language: str = "eng",
    limit: int | None = None,
    batch: int = 16,
    workers: int = 1,
    confidence: float = DEFAULT_CONFIDENCE,
    nms_iou: float = DEFAULT_NMS_IOU,
    match_iou: float = DEFAULT_MATCH_IOU,
    device: str | None = None,
) -> dict[str, Any]:
    """Run detector and both OCR tracks, then atomically write benchmark artifacts."""
    for value, name in ((confidence, "--confidence"), (nms_iou, "--nms-iou"),
                        (match_iou, "--match-iou")):
        _validate_threshold(value, name)
    if batch < 1:
        raise ValueError("--batch must be positive")
    if workers < 1:
        raise ValueError("--workers must be positive")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("--language must be nonempty")
    if not annotations.is_file():
        raise FileNotFoundError(f"Missing annotations: {annotations}")
    check_model_classes(model)
    truth_images, dataset_report = load_truth_images(
        annotations, data_root, split, limit
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_fd, csv_name = tempfile.mkstemp(prefix=".regions.", suffix=".csv", dir=output_dir)
    os.close(csv_fd)
    csv_temporary = Path(csv_name)
    metrics_temporary = output_dir / ".metrics.json.tmp"
    overall = {name: TextMetrics() for name in ("oracle", "matched", "end_to_end")}
    by_role: dict[str, dict[str, TextMetrics]] = {
        name: defaultdict(TextMetrics) for name in overall
    }
    detection_metrics = DetectionMetrics()
    oracle_image_exact = end_image_exact = processed = last_reported = 0
    local = threading.local()

    def worker(job: tuple[TruthImage, DetectionFrame]) -> ImageResult:
        if not hasattr(local, "backend"):
            local.backend = backend_factory()
        return _score_image(job[0], job[1], local.backend, language, match_iou)

    try:
        with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(truth_images), batch):
                    current = truth_images[start:start + batch]
                    paths = [str(item.path) for item in current]
                    predictions = model.predict(
                        source=paths, stream=True, batch=batch, conf=confidence,
                        iou=nms_iou, device=device, verbose=False,
                    )
                    jobs = []
                    count = 0
                    for truth_image, prediction in zip(current, predictions, strict=True):
                        if Path(prediction.path).resolve() != truth_image.path.resolve():
                            raise ValueError(
                                f"Prediction order differs from input images: {prediction.path}"
                            )
                        frame = detection_frame(prediction, truth_image.frame.source_key)
                        jobs.append((truth_image, frame))
                        count += 1
                    if count != len(current):
                        raise ValueError(f"Expected {len(current)} model results, got {count}")
                    for result in executor.map(worker, jobs):
                        writer.writerows(result.rows)
                        for name, items in (
                            ("oracle", result.oracle), ("matched", result.matched)
                        ):
                            for role, comparison in items:
                                overall[name].add(comparison)
                                by_role[name][role].add(comparison)
                        for role, comparison, is_reference in result.end_to_end:
                            overall["end_to_end"].add(comparison, is_reference=is_reference)
                            by_role["end_to_end"][role].add(
                                comparison, is_reference=is_reference
                            )
                        for role, status in result.detection:
                            detection_metrics.add(role, status)
                        oracle_image_exact += result.oracle_image_exact
                        end_image_exact += result.end_to_end_image_exact
                        processed += 1
                    if processed - last_reported >= 100 or processed == len(truth_images):
                        print(
                            f"Scored {processed}/{len(truth_images)} images",
                            file=sys.stderr, flush=True,
                        )
                        last_reported = processed
        report = {
            "schema_version": 1,
            "engine": engine,
            "language": language,
            "model": str(weights),
            "annotations": str(annotations),
            "data_root": str(data_root),
            "split": split,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "match_iou": match_iou,
            "batch": batch,
            "workers": workers,
            "device": device,
            "limit": limit,
            "dataset": dataset_report,
            "oracle": {
                **_text_report(overall["oracle"], by_role["oracle"]),
                "all_labels_image_accuracy": oracle_image_exact / processed,
            },
            "text_detection": detection_metrics.report(),
            "matched_box_ocr": _text_report(overall["matched"], by_role["matched"]),
            "end_to_end": {
                **_text_report(overall["end_to_end"], by_role["end_to_end"]),
                "all_labels_image_accuracy": end_image_exact / processed,
            },
        }
        metrics_temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        csv_temporary.replace(output_dir / "regions.csv")
        metrics_temporary.replace(output_dir / "metrics.json")
        return report
    finally:
        csv_temporary.unlink(missing_ok=True)
        metrics_temporary.unlink(missing_ok=True)


def _tesseract_factory() -> OCRBackend:
    from src.tesseract_ocr import TesseractBackend

    return TesseractBackend()


def _rapidocr_factory() -> OCRBackend:
    from src.rapidocr_ocr import RapidOCRBackend

    return RapidOCRBackend()


ENGINE_FACTORIES: dict[str, Callable[[], OCRBackend]] = {
    "tesseract": _tesseract_factory,
    "rapidocr": _rapidocr_factory,
}


def register_engine(name: str, factory: Callable[[], OCRBackend]) -> None:
    """Register another backend factory for programmatic benchmark use."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError("engine name must be lowercase letters, digits, underscores, or hyphens")
    if not callable(factory):
        raise TypeError("engine factory must be callable")
    ENGINE_FACTORIES[name] = factory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("evaluate", help="Score OCR on a PlotQA split")
    scoring.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    scoring.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    scoring.add_argument("--split", choices=("train", "val", "test"), default="test")
    scoring.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--engine", choices=tuple(ENGINE_FACTORIES), default="tesseract")
    scoring.add_argument("--language", default="eng")
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--batch", type=int, default=16)
    scoring.add_argument("--workers", type=int, default=1)
    scoring.add_argument("--device")
    scoring.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    scoring.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    scoring.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = load_model(args.weights)
    report = evaluate(
        model, ENGINE_FACTORIES[args.engine], annotations=args.annotations,
        data_root=args.data_root, split=args.split, weights=args.weights,
        output_dir=args.output_dir, engine=args.engine, language=args.language,
        limit=args.limit, batch=args.batch, workers=args.workers,
        confidence=args.confidence, nms_iou=args.nms_iou,
        match_iou=args.match_iou, device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
