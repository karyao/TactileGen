"""Render saved analysis evidence without importing or rerunning the pipeline."""

from collections import deque
import html
import json
import re
from urllib.parse import quote

from src.contract_io import SCHEMA_VERSION


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def literal(value):
    """Display exact strings safely, including whitespace and Markdown delimiters."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    fence = "`" * max(1, max((len(m) + 1 for m in re.findall(r"`+", text)), default=1))
    return fence + " " + text + " " + fence


def prose(value):
    return re.sub(r"([\\`*_{}\[\]()#+.!|>])", r"\\\1", html.escape(str(value))).replace("\n", " ")


def block(value):
    text = json_bytes(value).decode().rstrip()
    fence = "`" * max(3, max((len(m) + 1 for m in re.findall(r"`+", text)), default=3))
    return f"{fence}json\n{text}\n{fence}"


def references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_ids" or key.endswith("_evidence_ids"):
                yield from item
            else:
                yield from references(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from references(item)


def detection_references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_id") and isinstance(item, str):
                yield item
            elif key.endswith("_ids") and isinstance(item, list):
                yield from (entry for entry in item if isinstance(entry, str))
            else:
                yield from detection_references(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from detection_references(item)


def explanation(event):
    code = event["code"]
    if code == "title_selection":
        return {
            "no_detection": "No detection for this title role was present in the detection frame.",
            "filtered_out": "Title detections existed, but none survived confidence filtering and deduplication.",
            "absent_ocr": "A title detection was selected, but it had no OCR entry.",
            "empty_text": "The selected title's text was empty after consumer trimming.",
            "read": "The selected title supplied the recorded text.",
        }.get(event.get("reason"), "Title selection evidence is available below.")
    if code == "category_text_check":
        return ("The category failed the recorded unreadable-text guard."
                if event["unreadable"] else "The category passed the unreadable-text guard.")
    if code == "legend_entry_decision":
        return {
            "unreadable_text": "The legend entry was skipped because its consumed label was empty.",
            "low_confidence": "The legend entry was skipped because a swatch, label, or OCR confidence was below the recorded threshold.",
            "duplicate_name": "The legend entry was skipped because its case-folded name already existed.",
            "bars_not_settled": "The legend entry was skipped before text guards because bar assignments were not settled.",
            "outside_legend": "The legend entry was skipped because its swatch was outside the legend bounds.",
            "unreadable_swatch": "The legend entry was skipped because its swatch color was unavailable.",
            None: "The legend entry was added.",
        }.get(event.get("first_failing_guard"), "Legend decision evidence is available below.")
    if code == "numeric_parse":
        return {"parsed": "Numeric conversion succeeded using the recorded transformations.",
                "invalid_number": "The transformed text did not match the existing numeric grammar.",
                "nonfinite": "Numeric conversion produced a non-finite value and was rejected."}[event["reason"]]
    if code == "calibration_selection":
        if event["outcome"] == "rejected":
            return "Calibration could not find two consistent ticks. Selection evidence was retained before the error."
        return ("The existing calibration rule marked the scale ambiguous because multiple inlier sets remained."
                if event["ambiguous"] else "Calibration selected the recorded inlier set for least-squares refitting.")
    if code == "leading_zero_check":
        return ("A retained tick matched the leading-zero pattern, which can indicate a lost decimal point."
                if event["matched"] else "The retained tick did not match the leading-zero pattern.")
    if code == "y_ticks_dropped":
        return "Tick labels were dropped for parsing failure or disagreement with the selected numeric fit; the two causes are listed separately."
    if code == "y_ticks_not_arithmetic":
        return "Retained tick values failed the existing spacing check at the recorded tolerance."
    if code in ("title_edge_check", "title_cutoff_check"):
        return "The title-edge check used the recorded coordinates or ink counts and threshold."
    if code == "reconstruction_blocked":
        return "Reconstruction stopped at this recorded prerequisite check."
    if code == "bar_orientation_resolved":
        return (f"The shared orientation resolver selected {event.get('orientation')} "
                f"from preference {event.get('preference')} ({event.get('source')}); "
                "opposite-orientation detections were removed before OCR.")
    if code == "bar_orientation_uncertain":
        return ("The shared orientation resolver did not select an orientation from the "
                "recorded scores. OCR, legend matching, and reconstruction were skipped.")
    if code == "completeness_check":
        rule = event["rule"]
        if rule == "braille_encoding":
            return ("The braille encoder rejected the recorded characters."
                    if event["triggered"] else "The text passed the braille encoder.")
        if rule == "single_character_category":
            return "The category was compared with the median label length; any supplied-text exemption is recorded."
        if rule == "year_category_outlier":
            return "The category year was compared with the median year and existing distance threshold."
        if event.get("reason"):
            return event["reason"]
        if rule == "blocked_early_return":
            return "Completeness returned early because of blockers. Later text and braille checks did not execute."
    return "Recorded pipeline evidence; the exact values are shown below."


ORIENTATION_FACTS = (
    "preference", "status", "source", "reason", "orientation",
    "vertical_score", "horizontal_score", "min_bar_confidence",
    "min_orientation_score", "min_orientation_share", "suppressed_ids",
    "raw_detections", "effective_detections",
)


def decision_facts(event):
    """Surface relevant measurements without requiring the reader to inspect JSON."""
    fields = {
        "title_selection": ("reason", "candidate_ids", "retained_ids", "detection_id", "text"),
        "detection_confidence": ("detection_id", "confidence", "threshold", "outcome"),
        "detection_deduplication": ("detection_id", "outcome", "retained_detection_id", "measured_iou", "duplicate_iou"),
        "numeric_parse": ("detection_id", "consumer_text", "normalized_text", "numeric_text", "multiplier", "unit", "reason"),
        "category_text_check": ("detection_id", "text", "rule", "unreadable", "ocr_confidence"),
        "legend_entry_decision": ("label_id", "text", "first_failing_guard", "swatch_confidence",
                                  "label_confidence", "ocr_confidence", "confidence_threshold", "duplicate_name_key"),
        "calibration_selection": ("outcome", "inlier_indexes", "competing_inlier_sets", "selected_score", "ambiguous"),
        "y_ticks_dropped": ("count", "unparseable_ids", "numeric_outlier_ids"),
        "y_ticks_not_arithmetic": ("tick_values", "differences", "relative_tolerance", "absolute_tolerance"),
        "leading_zero_check": ("detection_id", "text", "pattern", "matched"),
        "title_cutoff_check": ("detection_id", "dark_pixel_count", "dark_threshold", "required_ink_pixels", "y1", "threshold", "triggered"),
        "title_edge_check": ("detection_id", "x2", "accepted_maximum", "threshold_pixels"),
        "completeness_check": ("rule", "field", "text", "length", "median_length", "median_length_threshold",
                               "median_year", "distance", "threshold", "supplied_exemption", "rejected_characters"),
        "bar_orientation_resolved": ORIENTATION_FACTS,
        "bar_orientation_uncertain": ORIENTATION_FACTS,
    }.get(event["code"], ())
    return [f"- {prose(key.replace('_', ' '))}: {literal(event[key])}" for key in fields if key in event]


class DiagnosticReport:
    def __init__(self, record, files):
        self.record, self.files = record, files
        self.events = {event["event_id"]: event for event in record["events"]}
        self.anchors = {event_id: f"evidence-{index:05d}"
                        for index, event_id in enumerate(self.events, 1)}
        self.sources = record.get("ocr_source_crops", [])
        self.region_anchors = {source["source_crop_id"]: f"text-region-{index:05d}"
                               for index, source in enumerate(self.sources, 1)}

    def link(self, path, label):
        return (f"[{prose(label)}]({quote(path, safe='/')})" if path in self.files
                else f"{prose(label)} — unavailable")

    def event_link(self, event_id):
        if event_id not in self.events:
            return f"Evidence {literal(event_id)} — unavailable"
        event = self.events[event_id]
        return f"[{prose(event['code'])}](#{self.anchors[event_id]}) ({literal(event_id)})"

    def related(self, ids):
        queue, seen, result = deque(ids), set(), []
        while queue:
            event_id = queue.popleft()
            if event_id in seen or event_id not in self.events:
                continue
            seen.add(event_id)
            event = self.events[event_id]
            result.append(event)
            queue.extend(references(event))
        return result

    def issue(self, label, ids, reason=None):
        lines = [f"### {prose(label)}", ""]
        if reason:
            lines.extend([prose(reason), ""])
        related = self.related(ids)
        if not related:
            lines.extend(["Detailed provenance is unavailable for this finding.", ""])
        for event_id in ids:
            if event_id in self.events:
                lines.extend([prose(explanation(self.events[event_id])), ""])
            lines.extend(["- " + self.event_link(event_id), ""])
        detections = set()
        for event in related:
            detections.update(detection_references(event))
            if event["code"] in ("y_ticks_not_arithmetic", "calibration_selection"):
                detections.update(self.record.get("spec_sources", {}).get("y_axis.ticks", {}).get("detection_ids", []))
        # Consumer and parser decisions are also joined through their authoritative
        # detection IDs, including warnings that predate explicit event references.
        known = {event["event_id"] for event in related}
        related.extend(event for event in self.events.values()
                       if event["event_id"] not in known and event.get("detection_id") in detections
                       and event["code"] in ("consumer_text", "numeric_parse"))
        measured = [event for event in related if decision_facts(event)
                    and event.get("rule") not in ("missing_item", "flag_note", "blocking_flag")]
        if measured:
            lines.extend(["Recorded checks and measurements:", ""])
            for event in measured:
                lines.extend([self.event_link(event["event_id"]), "", *decision_facts(event), ""])
        for source in self.sources:
            if source["detection_id"] in detections:
                lines.extend(["Source detection " + literal(source["detection_id"]) + ": " +
                              self.image_link(source["image"], "source crop") + "; " +
                              f"[OCR attempts](#{self.region_anchors[source['source_crop_id']]}).", ""])
                results = [result for result in self.record.get("ocr_trace", {}).get("results", [])
                           if result["detection_id"] == source["detection_id"]]
                if not results:
                    lines.extend(["Application OCR result unavailable for this detection.", ""])
                for result in results:
                    lines.extend(["Application OCR result: " + literal(result.get("application_result")) +
                                  "; selected attempt: " + literal(result.get("selected_attempt_id")) + ".", ""])
        for event in related:
            if event["code"] == "consumer_text":
                lines.extend([f"{prose(event['stage'])} input for {literal(event['detection_id'])}: "
                              f"{literal(event['input_text'])}; consumed text: {literal(event['consumer_text'])}.", ""])
        return lines

    def image_link(self, image, label):
        path = image.get("filename")
        if image.get("availability") == "available" and path in self.files:
            return self.link(path, label)
        return prose(label) + " — unavailable: " + prose(image.get("reason", "not captured"))

    def render(self):
        record = self.record
        lines = ["# Analysis diagnostics", "", f"Run: {literal(record['run_id'])}", "",
                 f"Outcome: **{prose(record['outcome'])}**. Completeness: {literal(record.get('completeness_status'))}.", "",
                 "Extract the entire ZIP and open this file in a Markdown viewer. Start with Findings, then follow the evidence and image links. "
                 "Quoted strings use JSON escapes so empty strings, tabs, spaces, and newlines remain distinguishable.", "",
                 "Coordinates use source-image pixels, origin at top left, with exclusive right/bottom crop bounds. "
                 "Recognition images are the API inputs before internal engine tensor preprocessing.", "",
                 "## Pipeline stages", "", "| Stage | Outcome | Reason |", "| --- | --- | --- |"]
        for name, stage in record["stages"].items():
            lines.append(f"| {prose(name)} | {prose(stage['outcome'])} | {prose(stage.get('reason') or '')} |")
        lines.extend(["", "## Findings", ""])
        completeness = record.get("outputs", {}).get("completeness", {})
        missing = completeness.get("missing", [])
        blockers = completeness.get("blockers", [])
        if not record["warnings"] and not missing and not blockers:
            lines.extend(["No warnings or completeness findings were recorded. This is not an accuracy assessment.", ""])
        for warning in record["warnings"]:
            stage = record["stages"].get(warning["stage"], {})
            reason = stage.get("reason") if not warning.get("evidence_ids") else None
            lines.extend(self.issue(warning["code"], warning.get("evidence_ids", []), reason))
        for item in missing:
            ids = [e["event_id"] for e in record["events"] if e["code"] == "completeness_check"
                   and e.get("rule") == "missing_item" and e.get("field") == item["field"]]
            lines.extend(self.issue("Input needed: " + item["field"], ids, item["reason"]))
        if blockers:
            ids = [e["event_id"] for e in record["events"] if e["code"] in
                   ("reconstruction_blocked", "calibration_selection") or
                   e["code"] == "completeness_check" and e.get("rule") in ("blocking_flag", "blocked_early_return")]
            for reason in blockers:
                lines.extend(self.issue("Build blocked", ids, reason))
        lines.extend(self.orientation())
        if record["exceptions"]:
            lines.extend(["## Recorded errors", ""])
            for error in record["exceptions"]:
                lines.extend([f"- {prose(error['stage'])}: {literal(error['type'])} — {prose(error['message'])}", ""])
            lines.extend(["Full exception evidence: " + self.link("diagnostics.json", "diagnostics.json") +
                          " and " + self.link("analysis.log", "analysis.log") + ".", ""])
        lines.extend(["## Source crops and OCR attempts", ""])
        trace = record.get("ocr_trace", {})
        attempts = trace.get("attempts", [])
        if not self.sources:
            lines.extend(["Source crops are unavailable; see the stage outcomes and manifest for the reason.", ""])
        for index, source in enumerate(self.sources, 1):
            lines.extend([f'<a id="{self.region_anchors[source["source_crop_id"]]}"></a>', "",
                          f"### Text region {index}: {literal(source['detection_id'])}", "",
                          "Detection box: " + literal(source["detection_box"]) + "; crop bounds: " +
                          literal(source.get("crop_bounds")) + "; policy: " + literal(source["crop_policy"]) + ".", "",
                          self.image_link(source["image"], "Source crop"), ""])
            path = source["image"].get("filename")
            if path in self.files:
                lines.extend([f"![Source crop {index}]({quote(path, safe='/')})", ""])
            owned = [a for a in attempts if a.get("source_crop_id") == source["source_crop_id"]]
            if not owned:
                lines.extend(["Internal OCR attempts are unavailable or none were prepared for this source crop.", ""])
            for attempt in owned:
                lines.extend(self.attempt(attempt))
        # Direct adapter tracing may have no source-crop records.
        source_ids = {s["source_crop_id"] for s in self.sources}
        for attempt in attempts:
            if attempt.get("source_crop_id") not in source_ids:
                lines.extend(self.attempt(attempt))
        if not trace:
            lines.extend(["OCR attempt history and application results are unavailable. See the stage outcomes.", ""])
        else:
            lines.extend(["### Adapter and application results", "", block(trace.get("results", [])), "",
                          "### Batch status and orientation", "",
                          block([{key: value for key, value in batch.items() if key != "raw_output"}
                                 for batch in trace.get("batches", [])]), ""])
        lines.extend(["## Calibration", ""])
        calibration = [e for e in record["events"] if e["code"] in (
            "calibration_tick_sources", "calibration_inputs", "calibration_selection", "y_calibration")]
        if not calibration:
            lines.extend(["Calibration evidence is unavailable; the calibration stage may not have executed.", ""])
        for event in calibration:
            lines.extend([self.event_link(event["event_id"]), "", prose(explanation(event)), "", block(event), ""])
        lines.extend(["## Decision evidence", ""])
        for event_id, event in self.events.items():
            lines.extend([f'<a id="{self.anchors[event_id]}"></a>', "",
                          "### " + prose(event["code"]), "", prose(explanation(event)), "", block(event), ""])
            for link in dict.fromkeys(references(event)):
                lines.extend(["- " + self.event_link(link), ""])
        lines.extend(["## Files and unavailable evidence", "",
                      "Open " + self.link("manifest.json", "manifest.json") + " for artifact availability and reasons. "
                      "The full uploaded image is omitted; any annotated overlay is a separate derived image.", "",
                      "Settings and backend metadata:", "", block({"settings": record["settings"], "metadata": record["metadata"]}), ""])
        for path in sorted(self.files):
            lines.append("- " + self.link(path, path))
        return "\n".join(lines) + "\n"

    def orientation(self):
        """Short section: why the orientation gate decided as it did."""
        event = next((e for e in self.record["events"]
                      if e["code"] in ("bar_orientation_resolved",
                                       "bar_orientation_uncertain")), None)
        lines = ["## Bar orientation", ""]
        if event is None:
            lines.extend(["Bar orientation evidence is unavailable; see the stage outcomes.", ""])
            return lines
        resolved = event["code"] == "bar_orientation_resolved"
        decision = (f"resolved **{prose(event['orientation'])}**" if resolved
                    else "**uncertain**")
        lines.append(f"- Decision: {decision}; preference {literal(event['preference'])}, "
                     f"source {literal(event['source'])}, reason {literal(event['reason'])}.")
        lines.append(
            f"- Evidence scores: vertical {literal(event['vertical_score'])}, "
            f"horizontal {literal(event['horizontal_score'])}; thresholds: bar confidence at or above "
            f"{literal(event['min_bar_confidence'])}, winning score at or above "
            f"{literal(event['min_orientation_score'])}, share at or above "
            f"{literal(event['min_orientation_share'])}.")
        suppressed = event.get("suppressed_ids") or []
        shown = ", ".join(literal(identifier) for identifier in suppressed) or "none"
        effective = event.get("effective_detections")
        counts = (f"raw {literal(event.get('raw_detections'))}, effective {literal(effective)}"
                  if resolved and effective is not None
                  else f"raw {literal(event.get('raw_detections'))}, "
                       "effective none (orientation unresolved)")
        lines.append(f"- Detections: {counts}; suppressed IDs: {shown}.")
        raw_frame = self.link("detections.json", "detections.json")
        if resolved:
            effective_frame = self.link("effective_detections.json", "effective_detections.json")
            lines.append(f"- Frames: raw {raw_frame}; effective {effective_frame}.")
        else:
            lines.append(f"- Frames: raw {raw_frame}; "
                         "effective: unavailable (orientation unresolved).")
        lines.append("")
        return lines

    def attempt(self, attempt):
        lines = ["#### Attempt " + literal(attempt["attempt_id"]), "",
                 "Submission: " + literal(attempt["recognition_submission"]) + "; selection: " + literal(attempt["selection"]) + ".", "",
                 "Preprocessing: " + literal(attempt["preprocessing"]) + ".", ""]
        image = attempt.get("recognition_image", {"availability": "unavailable", "reason": "not recorded"})
        lines.extend([self.image_link(image, "Recognition input"), ""])
        path = image.get("filename")
        if path in self.files:
            lines.extend([f"![Recognition input]({quote(path, safe='/')})", ""])
        if attempt["output_availability"] != "available":
            lines.extend(["Engine output unavailable: " + prose(attempt.get("output_unavailability_reason") or "not recorded") + ".", ""])
        lines.extend(["Raw engine evidence:", "", block(attempt["raw_output"]), "",
                      "Adapter-produced result:", "", block(attempt["adapter_result"]), ""])
        return lines


def legacy_report(record):
    lines = [f"Analysis run: {record['run_id']}", f"Outcome: {record['outcome']}",
             "Start with report.md for readable explanations and evidence links.", ""]
    for name, stage in record["stages"].items():
        lines.append(f"{name}: {stage['outcome']} ({stage['duration_seconds']} seconds) {stage['reason'] or ''}")
    for key in ("settings", "metadata", "warnings", "events", "outputs", "exceptions"):
        lines.extend(["", key.upper(), json_bytes(record[key]).decode().rstrip()])
    return "\n".join(lines) + "\n"


def build_bundle(record, assets, log_text):
    """Build all members from an already-frozen snapshot before writing the ZIP."""
    files = {name: data for name, data in assets.items() if name != "input.png"}
    files["diagnostics.json"] = json_bytes(record)
    if "ocr_trace" in record:
        files["ocr_trace.json"] = json_bytes(record["ocr_trace"])
    for name, value in record["outputs"].items():
        # Match the existing standalone contract downloads, including their envelope.
        payload = ({"schema_version": SCHEMA_VERSION, **value}
                   if name in ("ocr", "detections", "effective_detections") else value)
        files[f"{name}.json"] = json_bytes(payload)
    files["analysis.log"] = log_text.encode("utf-8")
    files["report.txt"] = legacy_report(record).encode("utf-8")
    available = set(files) | {"manifest.json", "report.md"}
    artifacts = [{"path": path, "availability": "available"} for path in sorted(available)]
    for path, stage in (("ocr.json", "ocr"), ("detections.json", "detection"),
                        ("ocr_trace.json", "ocr"),
                        ("effective_detections.json", "orientation_resolution")):
        if path not in available:
            status = record["stages"][stage]
            reason = status.get("reason") or f"No output recorded; {stage} stage: {status['outcome']}"
            if path == "effective_detections.json" and status.get("outcome") == "succeeded":
                reason = "Bar orientation was not resolved; no effective frame was recorded"
            artifacts.append({"path": path, "availability": "unavailable",
                              "reason": reason})
    unavailable = []
    if not record.get("ocr_source_crops"):
        unavailable.append({"kind": "source_crops", "reason": "No source crops recorded; see detection and OCR stage outcomes"})
    for batch in record.get("ocr_trace", {}).get("batches", []):
        if batch.get("internal_attempts") == "unavailable":
            unavailable.append({"kind": "internal_attempts", "owner_id": batch["batch_id"],
                                "reason": "Backend does not expose internal OCR attempts"})
    evidence_images = [("source_crop", s["source_crop_id"], s["image"]) for s in record.get("ocr_source_crops", [])]
    evidence_images += [("recognition_input", a["attempt_id"], a.get("recognition_image", {}))
                        for a in record.get("ocr_trace", {}).get("attempts", [])]
    for kind, owner_id, image in evidence_images:
        if image.get("availability") != "available" or image.get("filename") not in available:
            unavailable.append({"kind": kind, "owner_id": owner_id,
                                "reason": image.get("reason", "Image not captured or asset missing")})
    manifest = {"schema_version": 1, "run_id": record["run_id"], "outcome": record["outcome"],
                "started_at": record["started_at"], "ended_at": record["ended_at"],
                "artifacts": artifacts, "unavailable_evidence": unavailable}
    files["manifest.json"] = json_bytes(manifest)
    files["report.md"] = DiagnosticReport(record, available).render().encode("utf-8")
    return files
