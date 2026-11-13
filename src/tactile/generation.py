"""Create review and embossable tactile SVG artifacts from a ready spec."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.contracts import VerticalBarSpec
from src.tactile.completeness import READY, check_completeness, spec_to_chart_with_notes
from src.tactile.layout import layout_vertical_bar_chart_pages
from src.tactile.rules import Violation, check_layout
from src.tactile.svg import render_svg


_JOB_NAME = re.compile(r"generation-[0-9a-f]{32}")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM_LENGTH = 120


class GenerationError(RuntimeError):
    """A generation failure with a stable stage for UI-safe error handling."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        cleanup_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cleanup_error = cleanup_error


@dataclass(frozen=True)
class GeneratedPage:
    """Files and review information for one generated tactile page."""

    page_number: int
    page_count: int
    page_kind: str
    section: tuple[int, int] | None
    review_svg: Path
    clean_svg: Path
    caption: str
    notes: tuple[str, ...]
    needs_review: tuple[str, ...]
    violations: tuple[Violation, ...]


@dataclass(frozen=True)
class GenerationResult:
    """All artifacts produced by one successful generation attempt."""

    output_dir: Path
    pages: tuple[GeneratedPage, ...]
    zip_path: Path
    notes: tuple[str, ...]


def _unique(items) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _safe_stem(source_key: str) -> str:
    stem = _UNSAFE_FILENAME.sub("_", Path(source_key).stem).strip("._-")
    return (stem[:_MAX_STEM_LENGTH].rstrip("._-") or "chart")


def _caption(layout, page_number: int, page_count: int) -> str:
    if page_count == 1:
        return "Tactile graphic"
    if layout.page_kind == "key":
        return f"Page {page_number} of {page_count} — key"
    if layout.section is not None:
        part, total = layout.section
        return f"Page {page_number} of {page_count} — part {part} of {total}"
    return f"Page {page_number} of {page_count}"


def _remove_job_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _failure(stage: str, message: str, cause: Exception, job_dir: Path | None):
    cleanup_error = None
    if job_dir is not None:
        try:
            _remove_job_dir(job_dir)
        except Exception as exc:  # Preserve the generation failure as the primary error.
            cleanup_error = exc
    error = GenerationError(stage, message, cleanup_error=cleanup_error)
    raise error from cause


def generate_tactile_artifacts(
    spec: VerticalBarSpec,
    artifact_root: Path | str,
) -> GenerationResult:
    """Generate review SVGs, clean SVGs, and a clean-SVG ZIP for ``spec``.

    ``artifact_root`` is caller-owned and must already be a directory. The
    service creates one UUID-named child and removes it if any later stage
    fails.
    """
    root = Path(artifact_root)
    try:
        completeness = check_completeness(spec)
        if completeness.status != READY:
            details = completeness.blockers or [
                f"{item.field}: {item.reason}" for item in completeness.missing
            ]
            raise ValueError(
                f"spec is not ready ({completeness.status}): " + "; ".join(details)
            )
        if not root.exists() or not root.is_dir():
            raise ValueError("artifact_root must be an existing directory")
        root = root.resolve()
    except Exception as exc:
        _failure("validation", "Tactile generation input is invalid.", exc, None)

    job_dir = root / f"generation-{uuid.uuid4().hex}"
    try:
        job_dir.mkdir()
    except Exception as exc:
        _failure("write", "Could not create the generation directory.", exc, job_dir)

    try:
        chart, conversion_notes = spec_to_chart_with_notes(spec)
        layouts = layout_vertical_bar_chart_pages(chart, x_labels="auto")
        page_violations = [tuple(check_layout(layout)) for layout in layouts]
    except Exception as exc:
        _failure("layout", "Could not lay out the tactile graphic.", exc, job_dir)

    stem = _safe_stem(spec.source_key)
    page_count = len(layouts)
    pages: list[GeneratedPage] = []
    for index, (layout, violations) in enumerate(
        zip(layouts, page_violations, strict=True), 1
    ):
        suffix = "" if page_count == 1 else f"-{index}"
        review_path = job_dir / f"{stem}{suffix}-review.svg"
        clean_path = job_dir / f"{stem}{suffix}.svg"
        try:
            review_svg = render_svg(layout, show_print=True, violations=list(violations))
            clean_svg = render_svg(layout, show_print=False, violations=None)
        except Exception as exc:
            _failure("render", f"Could not render tactile page {index}.", exc, job_dir)
        try:
            review_path.write_text(review_svg, encoding="utf-8")
            clean_path.write_text(clean_svg, encoding="utf-8")
        except Exception as exc:
            _failure("write", f"Could not write tactile page {index}.", exc, job_dir)
        pages.append(GeneratedPage(
            page_number=index,
            page_count=page_count,
            page_kind=layout.page_kind,
            section=layout.section,
            review_svg=review_path,
            clean_svg=clean_path,
            caption=_caption(layout, index, page_count),
            notes=tuple(layout.notes),
            needs_review=tuple(layout.needs_review),
            violations=violations,
        ))

    zip_path = job_dir / f"{stem}-tactile-svg.zip"
    try:
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for page in pages:
                archive.write(page.clean_svg, arcname=page.clean_svg.name)
    except Exception as exc:
        _failure("zip", "Could not package the clean tactile SVGs.", exc, job_dir)

    return GenerationResult(
        output_dir=job_dir,
        pages=tuple(pages),
        zip_path=zip_path,
        notes=_unique((*completeness.notes, *conversion_notes)),
    )


def cleanup_generation(result: GenerationResult, artifact_root: Path | str) -> None:
    """Idempotently remove one verified generation directory."""
    root = Path(artifact_root).resolve()
    target = Path(result.output_dir)
    if target.parent.resolve() != root or not _JOB_NAME.fullmatch(target.name):
        raise ValueError("generation output must be a UUID-named child of artifact_root")
    _remove_job_dir(target)
