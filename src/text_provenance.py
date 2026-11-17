"""Small diagnostic helpers; consumers remain authoritative for all decisions."""

from src.ocr_trace import text_state


def consumer_text(diagnostics, stage, detection_id, original, trimmed, confidence):
    if diagnostics is None:
        return None
    return diagnostics.event(stage, "consumer_text", detection_id=detection_id,
        input_presence="present", input_text=original, input_state=text_state(original),
        consumer_text=trimmed, ocr_confidence=confidence,
        transformations=[{"operation": "strip", "before": original, "after": trimmed}])


def text_evidence(diagnostics, detection_id, *, stage="reconstruction"):
    """Find the last consumption of this text, without reconstructing its value."""
    event = next((e for e in reversed(diagnostics.record["events"])
                  if e["stage"] == stage and e["code"] == "consumer_text"
                  and e["detection_id"] == detection_id), None)
    if event is None:
        return {"input_presence": "absent", "input_text": None,
                "consumer_text": None, "evidence_ids": []}
    return {key: event[key] for key in ("input_presence", "input_text", "consumer_text")} | {
        "evidence_ids": [event["event_id"]]}
