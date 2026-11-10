"""PlotQA vertical-bar chart image in, tactile SVG out.

    python -m src.tactile.pipeline --image chart.png --output output/tactile/chart.svg

Exit status: 0 written, 1 error, 2 text still needed, 3 chart cannot be
built, 4 categories not confirmed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from src.contracts import ChartInterpretationError, VerticalBarSpec
from src.tactile.completeness import (
    CANNOT_BUILD, FORCEABLE_FLAGS, NEEDS_INPUT, READY, Completeness, Overrides,
    apply_overrides, check_completeness, completeness_from_error, cut_off_flags,
    spec_to_chart_with_notes,
)
from src.tactile.layout import X_LABEL_MODES, layout_vertical_bar_chart_pages
from src.tactile.rules import check_layout, format_report
from src.tactile.svg import render_svg

DEFAULT_WEIGHTS = Path("models/yolo_vbars/best.pt")
OVERVIEW_RULE = "overview_first"
EXIT_OK, EXIT_ERROR, EXIT_NEEDS_INPUT, EXIT_CANNOT_BUILD, EXIT_UNCONFIRMED = 0, 1, 2, 3, 4


def analyse(image_path: Path, weights: Path):
    """Run YOLO, OCR and reconstruction. Returns ``(spec, None, frame)`` or ``(None, error, frame)``."""
    import cv2

    from src.detection import load_model, run_detection
    from src.ocr import run_ocr
    from src.reconstruct import reconstruct_vbar
    from src.tesseract_ocr import TesseractBackend

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    frame = run_detection(load_model(weights), image_path)
    ocr = run_ocr(image, frame, TesseractBackend(), "eng")
    try:
        spec = reconstruct_vbar(frame, ocr, image)
        return replace(spec, flags=[*spec.flags, *cut_off_flags(frame, image)]), None, frame
    except ChartInterpretationError as error:
        return None, error, frame


def _indexed(pairs: list[str], option: str) -> dict[int, str]:
    result = {}
    for pair in pairs:
        index, sep, text = pair.partition("=")
        if not sep or not index.strip().isdigit() or not text.strip():
            raise ValueError(f"{option} expects INDEX=TEXT, got {pair!r}")
        result[int(index)] = text.strip()
    return result


def _file_overrides(path: Path) -> Overrides:
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - {"title", "x_axis_title", "y_axis_title", "categories", "series_names"}
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    return Overrides(
        data.get("title"), data.get("x_axis_title"), data.get("y_axis_title"),
        {int(k): v for k, v in data.get("categories", {}).items()},
        {int(k): v for k, v in data.get("series_names", {}).items()})


def build_overrides(args) -> Overrides:
    """File overrides first; command-line values replace them."""
    base = _file_overrides(args.overrides) if args.overrides else Overrides()
    return Overrides(
        args.title or base.title, args.x_title or base.x_axis_title,
        args.y_title or base.y_axis_title,
        {**base.categories, **_indexed(args.category, "--category")},
        {**base.series_names, **_indexed(args.series, "--series")})


def _merge(first: Overrides, second: Overrides) -> Overrides:
    return Overrides(second.title or first.title, second.x_axis_title or first.x_axis_title,
                     second.y_axis_title or first.y_axis_title,
                     {**first.categories, **second.categories},
                     {**first.series_names, **second.series_names})


def _ask_missing(missing, input_fn, out) -> tuple[Overrides, list]:
    """Prompt for each missing item. Returns the answers and the items left blank."""
    answers = Overrides()
    skipped = []
    for item in missing:
        print(f"\n{item.reason}", file=out)
        reply = input_fn(f"Enter text for {item.field} (empty to skip): ").strip()
        if not reply:
            skipped.append(item)
            continue
        name, _, index = item.field.partition("[")
        index = int(index.rstrip("]")) if index else None
        answers = _merge(answers, Overrides(
            reply if name == "title" else None,
            reply if name == "x_axis_title" else None,
            reply if name == "y_axis_title" else None,
            {index: reply} if name == "category" else {},
            {index: reply} if name == "series_name" else {}))
    return answers, skipped


def _print_missing(result: Completeness, out) -> None:
    print("This chart needs text that could not be read from the image:", file=out)
    for item in result.missing:
        print(f"  {item.field}: {item.reason} Supply it with {item.cli_flag}", file=out)
    print("Or put the same fields in a JSON file and pass --overrides FILE.", file=out)


def _print_blockers(result: Completeness, out, *, forceable_hint: bool) -> None:
    print("This chart cannot be turned into a tactile graphic:", file=out)
    for blocker in result.blockers:
        print(f"  - {blocker}", file=out)
    if forceable_hint and result.forceable:
        print("  --force continues past these ambiguity warnings at your own risk.", file=out)


def _print_summary(spec: VerticalBarSpec, out) -> None:
    print("\nCategories as read:", file=out)
    for j, name in enumerate(spec.categories):
        print(f"  {j}: {name}", file=out)
    print("Series:", file=out)
    for i, series in enumerate(spec.series):
        print(f"  {i}: {series.name or '(single series, unnamed)'}", file=out)


def run(args, *, analyse_fn=analyse, input_fn=input, stdin_is_tty=None,
        out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    tty = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    try:
        overrides = build_overrides(args)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=err)
        return EXIT_ERROR

    try:
        spec, error, frame = analyse_fn(args.image, args.weights)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=err)
        return EXIT_ERROR
    if error is not None:
        _print_blockers(completeness_from_error(error, frame), out, forceable_hint=False)
        return EXIT_CANNOT_BUILD

    try:
        spec, provenance = apply_overrides(spec, overrides)
    except ValueError as exc:
        print(f"Error: {exc}", file=err)
        return EXIT_ERROR
    result = check_completeness(spec)

    forced: list[str] = []
    if result.status == CANNOT_BUILD:
        if not (args.force and result.forceable):
            _print_blockers(result, out, forceable_hint=True)
            return EXIT_CANNOT_BUILD
        forced = [f for f in spec.flags if f.split(":", 1)[0] in FORCEABLE_FLAGS]
        print("WARNING: --force is building this chart despite: " + ", ".join(forced)
              + ". Bars may be paired with the wrong categories or series, and values "
              "may be wrong. Check the output against the original.", file=err)
        spec = replace(spec, flags=[f for f in spec.flags if f not in forced])
        result = check_completeness(spec)

    if result.status == NEEDS_INPUT:
        if not tty:
            _print_missing(result, out)
            return EXIT_NEEDS_INPUT
        answers, skipped = _ask_missing(result.missing, input_fn, out)
        spec, more = apply_overrides(spec, answers)
        provenance += more
        if skipped:
            print("\nStill missing: " + ", ".join(m.field for m in skipped), file=out)
            return EXIT_NEEDS_INPUT
        result = check_completeness(spec)
        if result.status != READY:
            _print_missing(result, out)
            return EXIT_NEEDS_INPUT

    _print_summary(spec, out)
    if not args.yes:
        if tty:
            if input_fn("Confirm these are correct? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Not confirmed; nothing written.", file=out)
                return EXIT_UNCONFIRMED
        else:
            print("WARNING: categories and series names were not confirmed "
                  "(not a terminal and --yes not given).", file=err)

    try:
        chart, adapter_notes = spec_to_chart_with_notes(spec)
        pages = layout_vertical_bar_chart_pages(chart, x_labels=args.x_labels)
    except ValueError as exc:
        print(f"The tactile layout rejected this chart: {exc}", file=err)
        return EXIT_ERROR
    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_transcriber = False
    blocking: list = []
    for number, layout in enumerate(pages, 1):
        path = args.output if len(pages) == 1 else args.output.with_name(
            f"{args.output.stem}-{number}{args.output.suffix}")
        found = check_layout(layout)
        # The overview page is not generated yet, so a person has to add it.
        needs_transcriber = needs_transcriber or any(v.rule.id == OVERVIEW_RULE for v in found)
        violations = [v for v in found if v.rule.id != OVERVIEW_RULE]
        blocking += violations
        path.write_text(render_svg(layout, show_print=args.show_print), encoding="utf-8")
        print(f"\nWrote {path}", file=out)
        for note in layout.notes:
            print(f"Note: {note}", file=out)
        for reason in layout.needs_review:
            print(f"NEEDS REVIEW: {reason}", file=out)
        print(format_report(violations), file=out)

    if needs_transcriber:
        print("\nNeeds a transcriber: overview page (BANA 3.9.1) not generated", file=out)
    for note in (*result.notes, *adapter_notes):
        print(f"Note: {note}", file=out)
    if forced:
        print("Built with --force past: " + ", ".join(forced), file=out)
    for note in provenance:
        print(f"Supplied by user: {note}", file=out)
    if args.strict and blocking:
        print(f"--strict: {len(blocking)} BANA violation(s) found.", file=err)
        return EXIT_ERROR
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="PlotQA vertical-bar chart image")
    parser.add_argument("--output", type=Path, required=True, help="SVG file to write")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"YOLO checkpoint (default {DEFAULT_WEIGHTS})")
    parser.add_argument("--title", help="chart title, when it could not be read")
    parser.add_argument("--x-title", help="x-axis title")
    parser.add_argument("--y-title", help="y-axis title")
    parser.add_argument("--category", action="append", default=[], metavar="J=TEXT",
                        help="text for category J (0-based); repeatable")
    parser.add_argument("--series", action="append", default=[], metavar="I=NAME",
                        help="name for series I (0-based); repeatable")
    parser.add_argument("--overrides", type=Path,
                        help="JSON with title, x_axis_title, y_axis_title, "
                             "categories {\"J\": text}, series_names {\"I\": name}")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
    parser.add_argument("--force", action="store_true",
                        help="build despite ambiguity flags (never missing bars or ticks)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any BANA rule fails (the missing overview page "
                             "is reported separately and does not count)")
    parser.add_argument("--show-print", action="store_true",
                        help="add small print text under each braille label")
    parser.add_argument("--x-labels", choices=("auto", *X_LABEL_MODES), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
