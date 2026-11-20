"""Turn analysis and generation results into readable browser messages."""

from __future__ import annotations

from src.gradio_ui.review import ManualGenerationResult
from src.tactile.completeness import Completeness
from src.tactile.rules import format_report
from src.tactile.pipeline import review_reasons


def generation_report(result: ManualGenerationResult) -> str:
    """Format generated-page metadata as an in-page review report."""
    lines = [
        "### Tactile generation complete",
        "",
        "The gallery contains annotated review copies. The ZIP contains only "
        "clean SVG pages for embossing.",
        "",
        "**Applied corrections**",
    ]
    if result.override_notes:
        lines.extend(f"- {note}" for note in result.override_notes)
    else:
        lines.append("- No text corrections were needed.")
    if result.generation.notes:
        lines.extend(("", "**Generation notes**"))
        lines.extend(f"- {note}" for note in result.generation.notes)

    transcriber_review = False
    for page in result.generation.pages:
        lines.extend(("", f"#### {page.caption}"))
        if page.notes:
            lines.extend(f"- Note: {note}" for note in page.notes)
        if page.needs_review:
            transcriber_review = True
            lines.extend(f"- Needs review: {note}" for note in page.needs_review)
        if page.violations:
            transcriber_review = True
        lines.extend(("", "```text", format_report(list(page.violations)), "```"))
    if transcriber_review:
        lines.extend((
            "",
            "**Transcriber review is required before production.**",
        ))
    else:
        lines.extend((
            "",
            "Automated checks found no rule violations; this does not replace "
            "professional transcriber review.",
        ))
    return "\n".join(lines)


def ready_status() -> str:
    return '<div role="status" aria-live="polite">Ready to analyze</div>'


def analyzing_status() -> str:
    return '<div role="status" aria-live="polite">Analyzing chart…</div>'


def complete_status(detection_count: int = 0, ocr_count: int | None = None) -> str:
    return (
        '<div role="status" aria-live="polite">Analysis complete</div>'
    )


def alert_message(message: str) -> str:
    return f'<div role="alert">{message}</div>'


def completeness_summary(result: Completeness, spec=None) -> str:
    """Render a compact, readable reconstruction assessment."""
    labels = {
        "ready": "Ready",
        "needs_input": "Needs input",
        "cannot_build": "Cannot build",
    }
    lines = [f"### Reconstruction: {labels.get(result.status, result.status)}"]
    if spec is not None:
        reasons = review_reasons(spec)
        if reasons:
            lines.extend(("", "**Needs review**", *(f"- {item['detail']}" for item in reasons)))
    if result.notes:
        lines.extend(("", "**Notes**", *(f"- {note}" for note in result.notes)))
    if result.missing:
        lines.extend((
            "",
            "**Missing or unconfirmed text**",
            *(f"- `{item.field}`: {item.reason}" for item in result.missing),
        ))
    if result.blockers:
        lines.extend(("", "**Blockers**", *(f"- {item}" for item in result.blockers)))
    return "\n".join(lines)
