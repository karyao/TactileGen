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
import weakref


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
        self._log_entries = []
        self.export_error = None
        self._source_frames = []
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
        rendered = message % args if args else message
        self._log_entries.append(f"{_utc_now()} {logging.getLevelName(level)} [run {self.run_id}] {rendered}")
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

    def detection_request(self, imgsz) -> dict:
        """Initialize run-owned size evidence, including runs rejected before detection."""
        requested = _normalize_event_evidence(imgsz)
        self.record["settings"]["imgsz"] = requested
        evidence = {
            "requested_imgsz": requested, "effective_imgsz": None, "tensor_shape": None,
            "unavailable_reasons": {
                "effective_imgsz": "Prediction setup has not been observed",
                "tensor_shape": "Inference input has not been observed",
            },
        }
        self.record["metadata"]["detection"] = evidence
        return evidence

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
        self.log("stage=%s outcome=started", name)
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

    def evidence_for(self, code: str, *, flag: str | None = None) -> list[str]:
        """Find producer events, preserving exact parameterized flag identities."""
        if flag is not None:
            return [event["event_id"] for event in self.record["events"]
                    if event.get("flag") == flag or event["code"] == flag]
        return [event["event_id"] for event in self.record["events"]
                if event["code"] == code or (event.get("flag") or "").split(":", 1)[0] == code]

    def warning(self, code: str, stage: str, *, flag: str | None = None) -> None:
        item = next((w for w in self.record["warnings"]
                     if w["code"] == code and w["stage"] == stage), None)
        if item is None:
            item = {"code": code, "stage": stage}
            self.record["warnings"].append(item)
            self.log("warning=%s stage=%s", code, stage, level=logging.WARNING)
        evidence = self.evidence_for(code, flag=flag)
        if evidence:
            links = item.setdefault("evidence_ids", [])
            links.extend(link for link in evidence if link not in links)

    def field_source(self, field: str, detection_ids=(), evidence_ids=()) -> dict:
        source = {"detection_ids": list(detection_ids), "evidence_ids": list(evidence_ids),
                  "availability": "available" if detection_ids else "unavailable"}
        self.record.setdefault("spec_sources", {})[field] = source
        return source

    def field_evidence(self, field: str) -> dict:
        return self.record.get("spec_sources", {}).get(field, {
            "availability": "unavailable", "detection_ids": [], "evidence_ids": [],
        })

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

    def event(self, stage: str, code: str, **evidence) -> str:
        """Capture facts; non-finite floats become 'inf', '-inf', or 'nan'."""
        event_id = f"{self.run_id}:event-{len(self.record['events']) + 1:05d}"
        self.record["events"].append({
            "stage": stage, "code": code, **_normalize_event_evidence(evidence),
            "event_id": event_id,
        })
        return event_id

    def snapshot(self, name: str, value) -> None:
        if is_dataclass(value):
            self.record["outputs"][name] = asdict(value)

    def image(self, name: str, image) -> bool:
        """Encode visual evidence best-effort; never replace a pipeline failure."""
        try:
            import cv2
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("PNG encoding failed")
            self.assets[name] = encoded.tobytes()
            return True
        except Exception as exc:
            self.exception("diagnostic_images", exc)
            return False

    def image_evidence(self, filename, image, *, channel_layout):
        """Describe a lossless PNG, explicitly retaining capture failures."""
        evidence = {"filename": None, "availability": "unavailable",
                    "channel_layout": channel_layout}
        try:
            evidence.update(width=int(image.shape[1]), height=int(image.shape[0]),
                            shape=list(image.shape), dtype=str(image.dtype))
            if str(image.dtype) not in ("uint8", "uint16"):
                raise ValueError("PNG evidence requires uint8 or uint16 pixels")
            if not self.image(filename, image):
                evidence["reason"] = "PNG encoding failed"
                return evidence
            evidence.update(filename=filename, availability="available")
        except Exception as exc:
            evidence["reason"] = f"{type(exc).__name__}: {exc}"
            self.exception("diagnostic_images", exc)
        return evidence

    def source_images(self, image, frame) -> None:
        from dataclasses import asdict
        from src.contracts import TEXT_ROLES
        from src.ocr_crop import crop_text_bounds, crop_text_region, crop_policy
        if any(source() is image and saved == frame for source, saved in self._source_frames):
            return
        self._source_frames.append((weakref.ref(image), frame))
        neighbors = tuple(d for d in frame.detections if d.role in TEXT_ROLES)
        sources = self.record.setdefault("ocr_source_crops", [])
        for detection in neighbors:
            index = len(sources) + 1
            filename = f"ocr_crops/{index:05d}.png"
            item = {"source_crop_id": f"{self.run_id}:ocr-source-{index:05d}",
                    "run_id": self.run_id, "detection_id": detection.id,
                    "detection_box": asdict(detection.box),
                    "crop_policy": crop_policy(detection),
                    "coordinate_convention": "source pixels; xyxy; right/bottom exclusive",
                    "image": {"availability": "unavailable", "filename": None}}
            sources.append(item)
            try:
                item["crop_bounds"] = list(crop_text_bounds(image, detection, neighbors))
                crop = crop_text_region(image, detection, neighbors)
                layout = "GRAY" if crop.ndim == 2 or crop.shape[2] == 1 else (
                    "BGRA" if crop.shape[2] == 4 else "BGR")
                item["image"] = self.image_evidence(filename, crop, channel_layout=layout)
                self.event("ocr", "source_crop", detection_id=detection.id,
                           filename=item["image"]["filename"], source_crop_id=item["source_crop_id"],
                           preprocessing="before backend preprocessing")
            except Exception as exc:
                item["image"]["reason"] = f"{type(exc).__name__}: {exc}"
                self.exception("diagnostic_images", exc)

    def export_zip(self, artifact_root: Path) -> Path | None:
        """Package one immutable analysis snapshot; errors never replace results."""
        path = artifact_root / "diagnostics" / f"analysis-diagnostics-{self.run_id}.zip"
        temporary_path = None
        operation = "snapshot serialization"
        self.export_error = None
        try:
            from src.diagnostics_report import build_bundle

            # Freeze before report rendering so all members describe the same run.
            record = json.loads(json.dumps(self.record, ensure_ascii=False, allow_nan=False))
            assets = dict(self.assets)
            log_lines = list(self._log_entries)
            for error in record["exceptions"]:
                log_lines.extend([f"[run {self.run_id}] exception stage={error['stage']} "
                                  f"type={error['type']}: {error['message']}", error["traceback"]])
            operation = "report generation"
            members = build_bundle(record, assets, "\n".join(log_lines) + "\n")
            operation = "archive writing"
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
            with ZipFile(temporary_path, "w", ZIP_DEFLATED) as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            operation = "archive finalization"
            os.replace(temporary_path, path)
            self.log("diagnostic ZIP created path=%s", path)
            return path
        except Exception as exc:
            self.export_error = f"{operation}: {type(exc).__name__}: {exc}"
            self.exception("diagnostic_export", exc)
            self.log("diagnostic ZIP export failed during %s: %s", operation, exc, level=logging.ERROR)
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
        return path

    def log_summary(self, *, record_path: Path | None, archive_path: Path | None) -> None:
        """Summarize a finished analysis after persistence and export complete."""
        self.log("analysis outcome=%s duration=%.3fs completeness=%s warnings=%d record=%s zip=%s",
                 self.record["outcome"], self.record["duration_seconds"],
                 self.record["completeness_status"] or "unavailable",
                 len(self.record["warnings"]), record_path or "unavailable",
                 archive_path or "unavailable")
