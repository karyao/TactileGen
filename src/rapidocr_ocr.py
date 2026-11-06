"""RapidOCR PP-OCRv6 adapter for YOLO-guided chart text recognition."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import cv2
import numpy as np

from src.contracts import Detection
from src.ocr import OCRBackendResult
from src.ocr_crop import crop_text_region


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
    box = detection.box
    if box.y2 - box.y1 <= _TALL_ASPECT_RATIO * (box.x2 - box.x1):
        return (crop,)
    return (
        crop,
        cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


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
        if not detections:
            return ()
        candidates = []
        owners = []
        for detection_index, detection in enumerate(detections):
            current = _recognition_candidates(image, detection, detections)
            candidates.extend(current)
            owners.extend([detection_index] * len(current))

        oriented, _ = self._engine.cls_and_rotate(candidates)
        output = self._engine.recognize_txt(oriented)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if not isinstance(texts, (tuple, list)) or not isinstance(scores, (tuple, list)):
            raise RuntimeError("RapidOCR returned malformed batched recognition output")
        if len(texts) != len(candidates) or len(scores) != len(candidates):
            raise RuntimeError("RapidOCR batched output differs from its input crops")

        best: list[OCRBackendResult | None] = [None] * len(detections)
        for owner, text, score in zip(owners, texts, scores, strict=True):
            result = _result(type("Output", (), {"txts": (text,), "scores": (score,)})())
            if result is not None and (
                best[owner] is None or result.confidence > best[owner].confidence
            ):
                best[owner] = result
        return tuple(best)
