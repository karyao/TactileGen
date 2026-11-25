"""PlotQA bar chart image (vertical or horizontal bars) in, tactile SVG out.

    python -m src.tactile.pipeline --image chart.png --output output/tactile/chart.svg

Bar orientation comes from the shared resolver (``src.bar_orientation``),
the same policy as the Gradio app: automatic selection needs enough
consistent bar evidence, otherwise the chart cannot be built and you can
retry with ``--bar-orientation vertical`` or ``--bar-orientation
horizontal``. Detecting horizontal bars needs a checkpoint trained on them
(``--weights``).

Exit status: 0 written, 1 error, 2 text still needed, 3 chart cannot be
built, 4 categories not confirmed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

from src.bar_orientation import BarOrientationResult, resolve_bar_orientation
from src.contracts import (
    ChartInterpretationError,
    VerticalBarSpec,
)
from src.tactile.completeness import (
    CANNOT_BUILD, FORCEABLE_FLAGS, NEEDS_INPUT, READY, Completeness, Overrides,
    apply_overrides, check_completeness, completeness_from_error, cut_off_flags,
    spec_to_chart_with_notes,
)
from src.tactile.category_reread import reread_categories
from src.tactile.layout import X_LABEL_MODES, layout_vertical_bar_chart_pages
from src.tactile.layout_hbar import layout_horizontal_bar_chart_pages
from src.tactile.rules import check_layout, format_report
from src.tactile.svg import render_svg

# Fine-tuned on vertical and horizontal bars (notebooks/kaggle_train_bars.ipynb).
DEFAULT_WEIGHTS = Path("models/yolo_bars_hv/best.pt")
OVERVIEW_RULE = "overview_first"
EXIT_OK, EXIT_ERROR, EXIT_NEEDS_INPUT, EXIT_CANNOT_BUILD, EXIT_UNCONFIRMED = 0, 1, 2, 3, 4


OCR_ENGINES = ("tesseract", "rapidocr", "hybrid")


def _ocr_backend(name: str):
    """The OCR backend and its language code."""
    if name == "hybrid":
        from src.ocr.hybrid import HybridBackend
        return HybridBackend(), "eng"
    if name == "rapidocr":
        from src.ocr.rapidocr import RapidOCRBackend
        return RapidOCRBackend(), "en"
    from src.ocr.tesseract import TesseractBackend
    return TesseractBackend(), "eng"


def _orientation_error(result: BarOrientationResult) -> ChartInterpretationError:
    """Explain an uncertain orientation decision, with a retry hint."""
    if result.reason == "requested_orientation_unavailable":
        reason = f"No usable {result.preference} bars were detected."
    else:
        reason = {
            "no_qualifying_bars": "No usable bars were detected.",
            "insufficient_evidence": "The detected bars provide too little evidence.",
            "ambiguous_orientation":
                "The detected bars provide conflicting orientation evidence.",
        }[result.reason]
    return ChartInterpretationError(
        f"Bar orientation is unclear. {reason} "
        "Pass --bar-orientation vertical or --bar-orientation horizontal to choose."
    )


def analyse(image_path: Path, weights: Path, ocr_engine: str = "tesseract",
            *, bar_orientation: str = "auto"):
    """Run YOLO, resolve bar orientation, then OCR and reconstruction.

    Returns ``(spec, None, frame)`` or ``(None, error, frame)``. On success the
    frame is the resolver's effective frame with opposite-orientation bars
    removed. On uncertainty it is the raw detection frame.
    """
    import cv2

    from src.detection import load_model, run_detection
    from src.bar_filter import drop_unmatched_bars
    from src.reconstruction.legend_matcher import filter_bars_by_legend, match_legend_to_bars
    from src.ocr import run_ocr
    from src.reconstruction.vertical import reconstruct_vbar
    from src.reconstruction.horizontal import reconstruct_hbar

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    raw_frame = run_detection(load_model(weights), image_path)
    resolution = resolve_bar_orientation(raw_frame, bar_orientation)
    if resolution.status != "resolved":
        return None, _orientation_error(resolution), raw_frame
    frame = resolution.effective_frame
    assert frame is not None
    backend, language = _ocr_backend(ocr_engine)
    ocr = run_ocr(image, frame, backend, language)
    from src.reconstruction.bottom_boundary import filter_vertical_bottom
    bottom_boundary = (filter_vertical_bottom(frame)
                       if resolution.orientation == "vertical" else None)
    if bottom_boundary is not None:
        frame = bottom_boundary.frame
    from src.reconstruction.axis_titles import resolve_axis_titles
    frame = resolve_axis_titles(frame, ocr, orientation=resolution.orientation)
    try:
        reduced = frame
        if resolution.orientation != "horizontal":
            # Which bars to drop is decided from the template match alone; the colour resolution
            # that names the series would otherwise keep low-confidence bars this rule removes.
            reduced = drop_unmatched_bars(
                frame, match_legend_to_bars(image, frame, ocr, resolve_by_colour=False)) or frame
        colour_filter = filter_bars_by_legend(
            image, reduced, ocr,
            bar_role="horizontal_bar" if resolution.orientation == "horizontal" else "vertical_bar")
        legend_mapping = colour_filter.mapping
        mismatched_bar_ids = colour_filter.mismatched_bar_ids
        if resolution.orientation == "horizontal":
            spec = reconstruct_hbar(frame, ocr, image, legend_mapping=legend_mapping,
                                    mismatched_bar_ids=mismatched_bar_ids)
        else:
            spec = reread_categories(
                reconstruct_vbar(reduced, ocr, image, legend_mapping=legend_mapping,
                                 mismatched_bar_ids=mismatched_bar_ids,
                                 **({"provisional_baseline": bottom_boundary.baseline_y}
                                    if bottom_boundary is not None else {})),
                frame, image)
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
    for note in result.notes:
        print(f"Note: {note}", file=out)


def _print_blockers(result: Completeness, out, *, forceable_hint: bool) -> None:
    print("This chart cannot be turned into a tactile graphic:", file=out)
    for blocker in result.blockers:
        print(f"  - {blocker}", file=out)
    for note in result.notes:
        print(f"Note: {note}", file=out)
    if forceable_hint and result.forceable:
        print("  --force continues past these ambiguity warnings at your own risk.", file=out)


def _print_summary(spec: VerticalBarSpec, out) -> None:
    print("\nCategories as read:", file=out)
    for j, name in enumerate(spec.categories):
        print(f"  {j}: {name}", file=out)
    reread = sorted(int(f.split(":", 1)[1]) for f in spec.flags if f.startswith("category_reread_index:"))
    if reread:
        print("re-read by fallback: " + ", ".join(f"{j}: {spec.categories[j]!r}" for j in reread),
              file=out)
    print("Series:", file=out)
    for i, series in enumerate(spec.series):
        shown = series.name or '(single series, unnamed)'
        print(f"  {i}: {shown}" + (" (no visible bars)" if series.no_visible_bars else ""), file=out)


REVIEW_CODE = "possible_hidden_bars"


def review_reasons(spec) -> list[dict[str, str]]:
    """Reasons a person should check the output against the original; never blocks a chart.

    A bar that is not drawn may be zero, tiny, or outside the axis range (PlotQA does not draw a
    negative bar when the y-axis starts at 0), and the picture cannot tell these apart.
    """
    reasons = []
    hidden = [series.name or "(unnamed)" for series in spec.series if series.no_visible_bars]
    if hidden:
        reasons.append({"code": REVIEW_CODE, "detail": "no bars are visible for "
                        + ", ".join(hidden) + "; their values may be zero, tiny or outside the axis range"})
    inferred = sum(1 for series in spec.series for value, rescued
                   in zip(series.values, series.value_rescued) if rescued and value == 0.0)
    if inferred:
        reasons.append({"code": REVIEW_CODE, "detail": f"{inferred} bar(s) could not be seen and were "
                        "read as zero; they may be tiny or outside the axis range"})
    return reasons


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
        spec, error, frame = analyse_fn(args.image, args.weights,
                                        bar_orientation=args.bar_orientation)
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
        if spec.orientation == "horizontal":
            pages = layout_horizontal_bar_chart_pages(chart, labels=args.x_labels)
        else:
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

    review = review_reasons(spec)
    for reason in review:
        print(f"NEEDS REVIEW: {reason['code']}: {reason['detail']}", file=out)
    if review:
        sidecar = args.output.with_name(args.output.stem + ".review.json")
        sidecar.write_text(json.dumps({"image": str(args.image), "review": review}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"Review notes written to {sidecar}", file=out)
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
    parser.add_argument("--image", type=Path, required=True,
                        help="PlotQA bar chart image, vertical or horizontal bars")
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
    parser.add_argument("--x-labels", choices=("auto", *X_LABEL_MODES), default="auto",
                        help="category labels (beside the bars for horizontal charts)")
    parser.add_argument("--ocr", choices=OCR_ENGINES, default="tesseract",
                        help="OCR engine (default tesseract)")
    parser.add_argument("--bar-orientation", choices=("auto", "vertical", "horizontal"),
                        default="auto",
                        help="bar direction: auto picks it from the detections when "
                             "the evidence is clear; vertical or horizontal forces "
                             "that direction (default auto)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args, analyse_fn=partial(analyse, ocr_engine=args.ocr))


if __name__ == "__main__":
    sys.exit(main())
