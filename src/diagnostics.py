"""Run-local analysis evidence, independent of chart interchange contracts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from zipfile import ZipFile, ZIP_DEFLATED
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import uuid


LOGGER = logging.getLogger("src.gradio_app")
STAGES = (
    "validation", "detection", "ocr", "legend_matching", "reconstruction",
    "presentation", "result_preparation",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_event_evidence(value):
    """Preserve non-finite evidence as strings compatible with strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize_event_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_event_evidence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_event_evidence(item) for item in value)
    return value


def configure_analysis_logging() -> None:
    """Make app summaries visible without configuring third-party loggers."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _model_metadata(model) -> dict:
    """Read only known fields; never traverse/serialize an arbitrary model."""
    result = {"class": f"{type(model).__module__}.{type(model).__qualname__}",
              "checkpoint": None, "task": None, "class_names": None,
              "configured_prediction_options": None}
    for attribute, key in (("ckpt_path", "checkpoint"), ("task", "task"),
                           ("names", "class_names")):
        try:
            value = getattr(model, attribute, None)
            if isinstance(value, Path):
                value = str(value)
            if value is None or isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, (dict, list, tuple)):
                # Reject non-JSON values rather than falling back to repr(model).
                result[key] = json.loads(json.dumps(value, allow_nan=False))
        except Exception:
            result[key] = None
    try:
        overrides = getattr(model, "overrides", None)
        if isinstance(overrides, dict):
            options = {key: overrides[key] for key in (
                "imgsz", "half", "agnostic_nms", "max_det", "rect", "augment",
            ) if key in overrides}
            result["configured_prediction_options"] = json.loads(
                json.dumps(options, allow_nan=False)
            )
    except Exception:
        pass
    return result


class AnalysisDiagnostics:
    """Mutable evidence owned by exactly one event handler/caller."""

    def __init__(self, *, settings: dict):
        self.assets = {}
        self._started = time.monotonic()
        self.record = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "started_at": _utc_now(), "ended_at": None, "duration_seconds": None,
            "outcome": None, "settings": {
                key: str(value) if isinstance(value, float) and not math.isfinite(value)
                else value for key, value in settings.items()
            }, "metadata": {},
            "stages": {name: {"outcome": None, "started_at": None,
                              "ended_at": None, "duration_seconds": None,
                              "reason": None} for name in STAGES},
            "counts": {"detections": None, "text_regions": None,
                       "ocr_returned_regions": None, "ocr_readable": None,
                       "ocr_empty": None},
            "warnings": [], "exceptions": [], "spec_flags": [],
            "completeness_status": None, "events": [], "outputs": {},
        }
        self.log("analysis started")

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    def log(self, message: str, *args, level: int = logging.INFO) -> None:
        LOGGER.log(level, "[run %s] " + message, self.run_id, *args)

    def capture_metadata(self, model, backend, language: str) -> None:
        packages = {}
        for name in ("numpy", "opencv-python", "Pillow", "gradio", "ultralytics",
                     "torch", "pytesseract", "rapidocr", "onnxruntime"):
            try:
                packages[name] = metadata.version(name)
            except Exception:
                packages[name] = None
        backend_info = {
            "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "requested_language": language, "language_behavior": "unknown",
            "effective_language": None,
        }
        try:
            describe = getattr(backend, "diagnostic_metadata", None)
            if callable(describe):
                details = describe(language=language)
                # Ensure metadata cannot break later persistence.
                backend_info.update(json.loads(json.dumps(details, allow_nan=False)))
        except Exception as exc:
            backend_info["metadata_error"] = f"{type(exc).__name__}: {exc}"
        self.record["metadata"] = {
            "packages": packages, "model": _model_metadata(model), "ocr": backend_info,
        }

    def exception(self, stage: str, exc: Exception) -> None:
        evidence = {"stage": stage, "type": type(exc).__name__, "message": str(exc),
                    "traceback": "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__, chain=True))}
        if evidence not in self.record["exceptions"]:
            self.record["exceptions"].append(evidence)

    def fail(self, exc: Exception) -> None:
        """Capture a fatal event error unless a stage already recorded its origin."""
        frames = traceback.extract_tb(exc.__traceback__)
        origin = frames[-1] if frames else None
        captured = any(
            item["type"] == type(exc).__name__ and item["message"] == str(exc)
            and (origin is None or f'File "{origin.filename}", line {origin.lineno}'
                 in item["traceback"])
            for item in self.record["exceptions"]
        )
        if not captured:
            self.exception("analysis", exc)
            self.log("analysis failed: %s: %s", type(exc).__name__,
                     str(exc).replace("\n", " ")[:240], level=logging.ERROR)

    @contextmanager
    def stage(self, name: str, *, blocked_errors: tuple[type[Exception], ...] = ()):
        item = self.record["stages"][name]
        item["started_at"] = _utc_now()
        started = time.monotonic()
        try:
            yield
        except Exception as exc:
            item["outcome"] = "blocked" if isinstance(exc, blocked_errors) else "failed"
            item["reason"] = f"{type(exc).__name__}: {exc}"
            self.exception(name, exc)
            raise
        else:
            item["outcome"] = "succeeded"
        finally:
            item["ended_at"] = _utc_now()
            item["duration_seconds"] = time.monotonic() - started
            reason = (" reason=" + item["reason"].replace("\n", " ")[:240]
                      if item["reason"] else "")
            self.log("stage=%s outcome=%s duration=%.3fs%s", name, item["outcome"],
                     item["duration_seconds"], reason,
                     level=logging.ERROR if item["outcome"] == "failed" else logging.INFO)

    def skip(self, name: str, reason: str) -> None:
        self.record["stages"][name].update(outcome="skipped", reason=reason)
        self.log("stage=%s outcome=skipped reason=%s", name, reason)

    def warning(self, code: str, stage: str) -> None:
        item = {"code": code, "stage": stage}
        if item not in self.record["warnings"]:
            self.record["warnings"].append(item)
            self.log("warning=%s stage=%s", code, stage, level=logging.WARNING)

    def finish(self, *, failed: bool = False) -> None:
        if self.record["ended_at"] is not None:
            return
        failures = [name for name, item in self.record["stages"].items()
                    if item["outcome"] in ("failed", "blocked")]
        for name, item in self.record["stages"].items():
            if item["outcome"] is None:
                self.skip(name, "upstream failure: " + (", ".join(failures) or "analysis")
                          if failed else "not executed by this caller")
        self.record["outcome"] = (
            "failed" if failed else "completed_with_warnings"
            if failures or self.record["warnings"]
            or self.record["completeness_status"] in ("needs_input", "cannot_build")
            else "succeeded"
        )
        self.record["ended_at"] = _utc_now()
        self.record["duration_seconds"] = time.monotonic() - self._started

    def event(self, stage: str, code: str, **evidence) -> None:
        """Capture facts; non-finite floats become 'inf', '-inf', or 'nan'."""
        self.record["events"].append({
            "stage": stage, "code": code, **_normalize_event_evidence(evidence),
        })

    def snapshot(self, name: str, value) -> None:
        if is_dataclass(value):
            self.record["outputs"][name] = asdict(value)

    def image(self, name: str, image) -> None:
        """Encode visual evidence best-effort; never replace a pipeline failure."""
        try:
            import cv2
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("PNG encoding failed")
            self.assets[name] = encoded.tobytes()
        except Exception as exc:
            self.exception("diagnostic_images", exc)

    def source_images(self, image, frame) -> None:
        from src.contracts import TEXT_ROLES
        from src.ocr_crop import crop_text_region
        neighbors = tuple(d for d in frame.detections if d.role in TEXT_ROLES)
        for index, detection in enumerate(neighbors):
            # Use generated filenames: detection IDs may contain path separators.
            filename = f"ocr_crops/{index:05d}_{detection.role}.png"
            try:
                self.image(filename, crop_text_region(image, detection, neighbors))
                self.event("ocr", "source_crop", detection_id=detection.id,
                           filename=filename, preprocessing="before backend preprocessing")
            except Exception as exc:
                self.exception("diagnostic_images", exc)

    def export_zip(self, artifact_root: Path) -> Path | None:
        """Export the complete evidence atomically, retaining JSON persistence."""
        path = artifact_root / "diagnostics" / f"{self.run_id}.zip"
        temporary_path = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
            with ZipFile(temporary_path, "w", ZIP_DEFLATED) as archive:
                archive.writestr("diagnostics.json", json.dumps(
                    self.record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                if "ocr_trace" in self.record:
                    archive.writestr("ocr_trace.json", json.dumps(
                        self.record["ocr_trace"], indent=2, ensure_ascii=False,
                        allow_nan=False) + "\n")
                lines = [f"Analysis run: {self.run_id}", f"Outcome: {self.record['outcome']}",
                         "Coordinates: pixels, origin at top left; y increases downward.",
                         "OCR crops are source crops, before backend preprocessing.",
                         "Warnings show evidence, not verified ground truth.", ""]
                for name, item in self.record["stages"].items():
                    lines.append(f"{name}: {item['outcome']} ({item['duration_seconds']} seconds) {item['reason'] or ''}")
                for key in ("settings", "metadata", "warnings", "events", "outputs", "exceptions"):
                    lines.extend(["", key.upper(), json.dumps(self.record[key], indent=2,
                                  ensure_ascii=False, allow_nan=False)])
                archive.writestr("report.txt", "\n".join(lines) + "\n")
                for name, value in self.record["outputs"].items():
                    archive.writestr(f"{name}.json", json.dumps(value, indent=2,
                                     ensure_ascii=False, allow_nan=False) + "\n")
                for name, data in self.assets.items():
                    archive.writestr(name, data)
            os.replace(temporary_path, path)
            return path
        except Exception as exc:
            self.exception("diagnostic_export", exc)
            self.log("diagnostic ZIP export failed: %s", exc, level=logging.ERROR)
            return None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    self.log("diagnostic ZIP cleanup failed: %s", exc, level=logging.ERROR)

    def persist(self, artifact_root: Path) -> Path | None:
        """Persist atomically; storage failures must not replace analysis results."""
        path = artifact_root / "diagnostics" / f"{self.run_id}.json"
        temporary_path = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(self.record, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary_path, path)
        except Exception as exc:
            self.log("diagnostics storage failed: %s: %s", type(exc).__name__,
                     str(exc).replace("\n", " "), level=logging.ERROR)
            path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.log("analysis outcome=%s duration=%.3fs record=%s", self.record["outcome"],
                 self.record["duration_seconds"], path or "unavailable")
        return path
