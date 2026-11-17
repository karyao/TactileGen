"""Framework-independent YOLO inference for one chart image."""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from pathlib import Path

from src.contracts import CLASS_NAMES, Box, Detection, DetectionFrame


DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS_IOU = 0.70
DEFAULT_IMGSZ = 640
# Bound per-request inference work; this is an application limit, not a YOLO limit.
MAX_IMGSZ = 1280


def validate_imgsz(imgsz: int) -> int:
    """Validate the requested scalar size; YOLO handles stride adjustment."""
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if imgsz > MAX_IMGSZ:
        raise ValueError(f"imgsz must be at most {MAX_IMGSZ} pixels")
    return imgsz


@contextmanager
def _inference_evidence(model, diagnostics, evidence):
    """Observe this prediction without retaining tensors or changing model outputs."""
    if diagnostics is None:
        yield
        return
    registrations = []
    handles = []

    def prediction_started(predictor):
        try:
            evidence["effective_imgsz"] = [int(n) for n in predictor.imgsz]
            evidence["unavailable_reasons"].pop("effective_imgsz", None)

            def input_shape(backend, inputs):
                # Some versions warm up through __call__; others call forward directly.
                try:
                    if not predictor.done_warmup:
                        return
                    evidence["tensor_shape"] = [int(n) for n in inputs[0].shape]
                    evidence["unavailable_reasons"].pop("tensor_shape", None)
                except Exception as exc:
                    evidence["unavailable_reasons"]["tensor_shape"] = str(exc)

            if not handles:
                handles.append(predictor.model.register_forward_pre_hook(input_shape))
        except Exception as exc:
            for key in ("effective_imgsz", "tensor_shape"):
                if evidence[key] is None:
                    evidence["unavailable_reasons"][key] = f"Observation unavailable: {exc}"

    try:
        # Ultralytics normally shares these lists. Also support an existing predictor
        # with its own callback list, without registering twice on a shared list.
        for owner in (model, getattr(model, "predictor", None)):
            callbacks = getattr(owner, "callbacks", None)
            if isinstance(callbacks, dict):
                entries = callbacks.get("on_predict_start")
                if isinstance(entries, list) and not any(entries is item for item in registrations):
                    entries.append(prediction_started)
                    registrations.append(entries)
        if not registrations:
            reason = "Model does not expose prediction callbacks"
            evidence["unavailable_reasons"] = dict.fromkeys(("effective_imgsz", "tensor_shape"), reason)
        yield
    finally:
        for entries in registrations:
            if prediction_started in entries:
                entries.remove(prediction_started)
        for handle in handles:
            handle.remove()
        diagnostics.log("detection requested_imgsz=%s effective_imgsz=%s tensor_shape=%s",
                        evidence["requested_imgsz"], evidence["effective_imgsz"],
                        evidence["tensor_shape"])


def _class_names(names: dict[int, str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(names, dict):
        try:
            return tuple(names[index] for index in range(len(names)))
        except KeyError as exc:
            raise ValueError("Model class IDs must be contiguous from zero") from exc
    return tuple(names)


def check_model_classes(model) -> None:
    if _class_names(model.names) != CLASS_NAMES:
        raise ValueError("Model class names/order differ from the chart contract")


def load_model(weights: Path):
    """Load and validate a YOLO checkpoint without importing Ultralytics eagerly."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Install the model dependencies: python -m pip install -r requirements-model.txt"
        ) from exc
    if not weights.is_file():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {weights}")
    model = YOLO(str(weights))
    check_model_classes(model)
    return model


def _validate_threshold(value: float, name: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite number between 0 and 1")


def _safe_filename(name: str) -> str:
    suffix = Path(name).suffix
    if not re.fullmatch(r"\.[A-Za-z0-9]+", suffix):
        suffix = ""
    stem = name[:-len(suffix)] if suffix else name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return f"{stem or 'upload'}{suffix}"


def detection_frame(result, source_key: str) -> DetectionFrame:
    """Adapt an Ultralytics result to the shared pixel-coordinate contract."""
    height, width = result.orig_shape
    boxes = result.boxes
    detections = []
    for index, (xyxy, class_id, confidence) in enumerate(zip(
        boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
        strict=True,
    )):
        if int(class_id) != class_id or not 0 <= int(class_id) < len(CLASS_NAMES):
            raise ValueError(f"Invalid model class ID: {class_id}")
        x1, y1, x2, y2 = xyxy
        box = Box(max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                  max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        detections.append(Detection(f"det-{index + 1:05d}", CLASS_NAMES[int(class_id)],
                                    box, float(confidence)))
    return DetectionFrame(source_key, width, height, tuple(detections))


def run_detection(
    model,
    image_path: str | Path,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    nms_iou: float = DEFAULT_NMS_IOU,
    device: str | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    diagnostics=None,
) -> DetectionFrame:
    """Run one-image inference and return the shared detection contract."""
    evidence = diagnostics.detection_request(imgsz) if diagnostics is not None else None
    validate_imgsz(imgsz)
    _validate_threshold(confidence, "confidence")
    _validate_threshold(nms_iou, "nms_iou")
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    check_model_classes(model)
    with _inference_evidence(model, diagnostics, evidence):
        results = list(model.predict(source=str(path), conf=confidence, iou=nms_iou,
                                     device=device, verbose=False, imgsz=imgsz))
    if len(results) != 1:
        raise ValueError(f"Expected one model result, got {len(results)}")
    result = results[0]
    result_path = getattr(result, "path", None)
    if result_path and Path(result_path).resolve() != path.resolve():
        raise ValueError("Prediction result does not match the requested image")
    return detection_frame(result, f"input/{_safe_filename(path.name)}")
