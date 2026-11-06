"""Render a vertical bar chart JSON file as a tactile SVG and check BANA rules.

    python -m src.tactile --chart tests/fixtures/tactile/rainfall.json \
        --output output/tactile/rainfall.svg --show-print
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.contract_io import load_vertical_bar_chart
from src.tactile.layout import X_LABEL_MODES, layout_vertical_bar_chart
from src.tactile.rules import check_layout, format_report
from src.tactile.svg import render_svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chart", type=Path, required=True,
                        help="VerticalBarChart JSON (schema_version 1)")
    parser.add_argument("--output", type=Path, required=True, help="SVG file to write")
    parser.add_argument("--show-print", action="store_true",
                        help="add small print text under each braille label")
    parser.add_argument("--show-violations", action="store_true",
                        help="draw red boxes around elements that break a rule")
    parser.add_argument("--x-labels", choices=("auto", *X_LABEL_MODES), default="auto",
                        help="category labels: auto follows the overflow order "
                             "full -> abbreviation keys -> numeric keys -> review")
    parser.add_argument("--strict", action="store_true",
                        help="exit with status 1 when any rule fails")
    args = parser.parse_args(argv)

    chart = load_vertical_bar_chart(args.chart)
    layout = layout_vertical_bar_chart(chart, x_labels=args.x_labels)
    violations = check_layout(layout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(
        layout, show_print=args.show_print,
        violations=violations if args.show_violations else None), encoding="utf-8")
    print(f"Wrote {args.output}")
    for note in layout.notes:
        print(f"Note: {note}")
    for reason in layout.needs_review:
        print(f"NEEDS REVIEW: {reason}")
    print(format_report(violations))
    return 1 if args.strict and violations else 0


if __name__ == "__main__":
    sys.exit(main())
