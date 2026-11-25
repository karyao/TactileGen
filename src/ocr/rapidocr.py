"""RapidOCR PP-OCRv6 adapter for YOLO-guided chart text recognition."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import cv2
import numpy as np

from src.contracts import Detection
from src.ocr import OCRBackendResult
from src.ocr.crop import crop_text_region
from src.ocr.trace import BatchTrace, field, text_state


_TALL_ASPECT_RATIO = 1.5
_SUPPORTED_IMAGE_ERROR = (
    "RapidOCR backend supports grayscale, BGR, or BGRA images"
)


def _load_rapidocr():
    try:
        import rapidocr
    except ImportError as exc:
        raise RuntimeError(
            "Install the RapidOCR dependencies: "
            "python -m pip install -r requirements-ocr.txt"
        ) from exc
    return rapidocr


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return a contiguous three-channel BGR image without changing its dtype."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(_SUPPORTED_IMAGE_ERROR)
    channels = image.shape[2]
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    if channels == 3:
        return np.ascontiguousarray(image)
    if channels == 4:
        return np.ascontiguousarray(image[:, :, :3])
    raise ValueError(_SUPPORTED_IMAGE_ERROR)


def _recognition_candidates(
    image: np.ndarray,
    detection: Detection,
    neighbors: Sequence[Detection] = (),
) -> tuple[np.ndarray, ...]:
    crop = _as_bgr(crop_text_region(image, detection, neighbors))
    rotations = _candidate_rotations(detection)
    return tuple(crop if rotation == 0 else cv2.rotate(
        crop, cv2.ROTATE_90_CLOCKWISE if rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
    ) for rotation in rotations)


def _candidate_rotations(detection: Detection) -> tuple[int, ...]:
    box = detection.box
    return ((0, 90, -90) if detection.role in ("x_axis_title", "y_axis_title")
            or box.y2 - box.y1 > _TALL_ASPECT_RATIO * (box.x2 - box.x1)
            else (0,))


def _result(output) -> OCRBackendResult | None:
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if texts is None or scores is None:
        return None
    if not isinstance(texts, (tuple, list)) or not isinstance(scores, (tuple, list)):
        raise RuntimeError("RapidOCR returned malformed recognition output")
    if len(texts) != 1 or len(scores) != 1:
        raise RuntimeError("RapidOCR recognition-only output must contain one result")
    text = texts[0]
    score = scores[0]
    if not isinstance(text, str):
        raise RuntimeError("RapidOCR returned non-text recognition output")
    text = text.strip()
    if not text:
        return None
    if (isinstance(score, bool) or not isinstance(score, numbers.Real)
            or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
        raise RuntimeError("RapidOCR returned an invalid confidence")
    return OCRBackendResult(text, float(score))


class RapidOCRBackend:
    """Recognize text in authoritative YOLO boxes with PP-OCRv6-small ONNX."""

    def diagnostic_metadata(self, *, language: str) -> dict:
        """Describe the configured multilingual recognizer without inference."""
        return {
            "language_behavior": "multilingual model; requested language is not applied",
            "effective_language": "multilingual",
            "model": "PP-OCRv6-small", "runtime": "onnxruntime",
            "text_score": 0.0, "intra_op_num_threads": 1, "inter_op_num_threads": 1,
        }

    def __init__(self) -> None:
        rapidocr = _load_rapidocr()
        try:
            self._engine = rapidocr.RapidOCR(params={
                "Global.text_score": 0.0,
                "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
                "Rec.model_type": rapidocr.ModelType.SMALL,
                "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV6,
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        except ImportError as exc:
            raise RuntimeError(
                "Install the RapidOCR dependencies: "
                "python -m pip install -r requirements-ocr.txt"
            ) from exc

    def recognize(
        self,
        image: np.ndarray,
        detection: Detection,
        *,
        language: str,
        neighbors: Sequence[Detection] = (),
    ) -> OCRBackendResult | None:
        """Recognize one crop; PP-OCRv6 uses one multilingual model."""
        best = None
        for candidate in _recognition_candidates(image, detection, neighbors):
            output = self._engine(
                candidate, use_det=False, use_cls=True, use_rec=True
            )
            current = _result(output)
            if current is not None and (
                best is None or current.confidence > best.confidence
            ):
                best = current
        return best

    def recognize_many(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        *,
        language: str,
    ) -> tuple[OCRBackendResult | None, ...]:
        return self._recognize_many(image, detections, language=language)

    def recognize_many_traced(self, image, detections, *, language, diagnostics):
        """Return ordinary batch results and retain evidence before conversion."""
        trace = BatchTrace(diagnostics, "rapidocr", detections)
        with trace.capture():
            return self._recognize_many(image, detections, language=language, trace=trace)

    def _recognize_many(self, image, detections, *, language, trace=None):
        if not detections:
            return ()
        candidates = []
        owners = []
        if trace is not None:
            for detection in detections:
                rotations = _candidate_rotations(detection)
                for index, rotation in enumerate(rotations):
                    trace.attempt(detection, index, color="BGR", rotation_degrees_clockwise=rotation,
                                  classifier_orientation="unavailable")
        for detection_index, detection in enumerate(detections):
            current = _recognition_candidates(image, detection, detections)
            candidates.extend(current)
            owners.extend([detection_index] * len(current))

        if trace is not None:
            trace.submit("cls_and_rotate")
        oriented, orientation = self._engine.cls_and_rotate(candidates)
        if trace is not None:
            from src.ocr.trace import snapshot
            labels = getattr(orientation, "cls_res", None)
            trace.batch["classifier_output"] = {
                "cls_res": snapshot(labels),
                "availability": "available" if labels is not None else "unavailable",
                "elapse": snapshot(getattr(orientation, "elapse", None)),
            }
            for index, candidate in enumerate(oriented):
                if index >= len(trace.attempts):
                    break
                trace.input_image(index, candidate, channel_layout="BGR")
                if isinstance(labels, (list, tuple)) and index < len(labels):
                    trace.attempts[index]["preprocessing"]["classifier_orientation"] = snapshot(labels[index])
            trace.submit("recognize_txt")
        output = self._engine.recognize_txt(oriented)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if trace is not None:
            fields = {key: getattr(output, key) for key in ("txts", "scores") if hasattr(output, key)}
            trace.raw({key: field(fields, key) for key in ("txts", "scores")})
        if not isinstance(texts, (tuple, list)) or not isinstance(scores, (tuple, list)):
            raise RuntimeError("RapidOCR returned malformed batched recognition output")
        if len(texts) != len(candidates) or len(scores) != len(candidates):
            raise RuntimeError("RapidOCR batched output differs from its input crops")

        if trace is not None:
            # Capture every response before any candidate can fail conversion.
            for index, (text, score) in enumerate(zip(texts, scores, strict=True)):
                trace.output(index, {"text": text, "text_state": text_state(text), "confidence": score})
        best: list[OCRBackendResult | None] = [None] * len(detections)
        winners = {}
        for index, (owner, text, score) in enumerate(zip(owners, texts, scores, strict=True)):
            if trace is not None:
                trace.attempts[index]["processing"] = "processing"
            result = _result(type("Output", (), {"txts": (text,), "scores": (score,)})())
            if trace is not None:
                trace.converted(index, result)
            if result is not None and (
                best[owner] is None or result.confidence > best[owner].confidence
            ):
                best[owner] = result
                if trace is not None:
                    winners[detections[owner].id] = trace.attempts[index]["attempt_id"]
        if trace is not None:
            trace.selected(winners)
        return tuple(best)
