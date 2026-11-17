"""Run-local OCR evidence. This module never interprets recognition results."""

from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np


def snapshot(value):
    """Copy engine evidence into strict JSON without stripping or coercing text."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {
            "diagnostic_type": "nonfinite_float", "value": str(value),
        }
    if isinstance(value, np.generic):
        return {"diagnostic_type": str(value.dtype), "value": snapshot(value.item())}
    if isinstance(value, np.ndarray):
        return {"diagnostic_type": "ndarray", "dtype": str(value.dtype),
                "value": snapshot(value.tolist())}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: snapshot(item) for key, item in value.items()}
    return {"diagnostic_type": "unsupported", "python_type":
            f"{type(value).__module__}.{type(value).__qualname__}"}


def field(mapping, key):
    return ({"presence": "present", "value": snapshot(mapping[key])}
            if key in mapping else {"presence": "absent"})


def text_state(value):
    if value is None:
        return "null"
    if not isinstance(value, str):
        return "non_text"
    return "empty" if value == "" else "whitespace" if not value.strip() else "text"


def trace_record(diagnostics):
    return diagnostics.record.setdefault("ocr_trace", {
        "schema_version": 1, "batches": [], "attempts": [], "results": [],
    })


class BatchTrace:
    """Own one adapter call and retain evidence even if that call raises."""

    def __init__(self, diagnostics, backend, detections, *, internal=True):
        self.run_id = diagnostics.run_id
        self.record = trace_record(diagnostics)
        self.batch = {
            "batch_id": f"{diagnostics.run_id}:ocr-batch-{len(self.record['batches']) + 1:05d}",
            "backend": backend, "detection_ids": [d.id for d in detections],
            "internal_attempts": "available" if internal else "unavailable",
            "operation": "preprocessing", "outcome": "pending", "raw_output": None,
            "raw_output_availability": "unavailable",
            "raw_output_kind": "engine_fields" if internal else "adapter_result",
            "selection_policy": "highest confidence; first candidate wins ties" if internal else "unknown",
        }
        self.record["batches"].append(self.batch)
        self.attempts = []

    def attempt(self, detection, candidate_index, **preprocessing):
        item = {
            "attempt_id": f"{self.run_id}:ocr-attempt-{len(self.record['attempts']) + 1:05d}",
            "batch_id": self.batch["batch_id"], "backend": self.batch["backend"],
            "detection_id": detection.id, "candidate_index": candidate_index,
            "preprocessing": preprocessing, "submission": "not_submitted",
            "submitted_operations": [], "recognition_submission": "not_submitted",
            "output_availability": "unavailable", "processing": "not_processed",
            "output_unavailability_reason": "not_submitted",
            "selection": "unresolved", "raw_output": None, "adapter_result": None,
        }
        self.record["attempts"].append(item)
        self.attempts.append(item)
        return item

    def submit(self, operation):
        self.batch["operation"] = operation
        for item in self.attempts:
            item["submission"] = "submitted"
            item["submitted_operations"].append(operation)
            item["output_unavailability_reason"] = "recognition_not_submitted"
            if operation in ("recognize_txt", "image_to_data"):
                item["recognition_submission"] = "submitted"
                item["output_unavailability_reason"] = "recognition_return_unavailable"

    def raw(self, value):
        self.batch["raw_output"] = snapshot(value)
        self.batch["raw_output_availability"] = "available"
        self.batch["operation"] = "interpret_output"
        for item in self.attempts:
            item["output_unavailability_reason"] = "batch_response_not_mapped"

    def output(self, index, value):
        self.attempts[index].update(raw_output=snapshot(value), output_availability="available",
                                    output_unavailability_reason=None)

    def converted(self, index, result, *, derived=True):
        self.attempts[index].update(
            processing="completed", adapter_result=None if result is None else {
                "text": result.text, "text_kind": "derived" if derived else "adapter_produced",
                "confidence": snapshot(result.confidence),
            })

    def selected(self, winners):
        for item in self.attempts:
            winner = winners.get(item["detection_id"])
            item["selection"] = (
                "selected" if winner == item["attempt_id"] else
                "unreadable" if item["adapter_result"] is None else "not_selected")
        self.batch["outcome"] = "succeeded"

    @contextmanager
    def capture(self):
        try:
            yield self
        except Exception as exc:
            for item in self.attempts:
                if item["processing"] == "processing":
                    item["processing"] = "invalid_output"
            self.batch.update(outcome="failed", error={
                "type": type(exc).__name__, "message": str(exc),
            })
            raise
        else:
            self.batch["outcome"] = "succeeded"


def adapter_evidence(result):
    if result is None:
        return None
    # Do not validate here: orchestration must retain its original error behavior.
    from src.ocr import OCRBackendResult
    if isinstance(result, OCRBackendResult):
        return {"text": snapshot(result.text), "text_state": text_state(result.text),
                "confidence": snapshot(result.confidence)}
    return snapshot(result)
