"""Write a tactile layout as an SVG page in millimetres.

Black marks are the raised parts when printed on swell (microcapsule) paper
or imported into graphics-embosser software. Braille is drawn as dots at
standard spacing, so the file does not depend on a braille font.
``show_print`` adds small grey print text for sighted teachers and
reviewers; ``violations`` draws red boxes around elements that break a rule.
"""

from __future__ import annotations

from html import escape

from src.tactile.layout import DOT_DIAMETER, TactileLayout
from src.tactile.rules import Violation

_PATTERNS = {
    "diagonal": '<path d="M0,4 L4,0" stroke="#000" stroke-width="0.6"/>',
    # 0.8 mm dots, smaller than 1.45 mm braille dots (BANA §6.8.3.3).
    "dots": '<circle cx="1.5" cy="1.5" r="0.4" fill="#000"/>',
    "horizontal": '<path d="M0,2 L4,2" stroke="#000" stroke-width="0.6"/>',
    "vertical": '<path d="M2,0 L2,4" stroke="#000" stroke-width="0.6"/>',
    "crosshatch": '<path d="M0,4 L4,0 M0,0 L4,4" stroke="#000" stroke-width="0.5"/>',
}


def _fill(texture: str) -> str:
    return f"url(#tex-{texture})"


def render_svg(layout: TactileLayout, *, show_print: bool = False,
               violations: list[Violation] | None = None) -> str:
    w, h = layout.page_width, layout.page_height
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" '
        f'viewBox="0 0 {w:.2f} {h:.2f}">',
        f"<title>Tactile graphic: {escape(layout.source_key)}</title>",
        "<defs>",
    ]
    for name, body in _PATTERNS.items():
        size = 3 if name == "dots" else 4
        out.append(f'<pattern id="tex-{name}" width="{size}" height="{size}" '
                   f'patternUnits="userSpaceOnUse">{body}</pattern>')
    out += ["</defs>", f'<rect width="{w:.2f}" height="{h:.2f}" fill="#fff"/>']

    def line_svg(line, dash=""):
        return (f'<line id="{line.id}" x1="{line.x1:.2f}" y1="{line.y1:.2f}" '
                f'x2="{line.x2:.2f}" y2="{line.y2:.2f}" stroke="#000" '
                f'stroke-width="{line.width}"{dash}/>')

    out.append('<g id="grid">')
    out += [line_svg(l, ' stroke-dasharray="1.5 1.5"') for l in layout.lines if l.kind == "grid"]
    out.append('</g><g id="bars">')
    for bar in layout.bars:
        rect = (f'x="{bar.x:.2f}" y="{bar.y:.2f}" width="{bar.width:.2f}" '
                f'height="{bar.height:.2f}"')
        # A white base hides grid lines inside bars so the texture stays clean.
        out.append(f'<rect {rect} fill="#fff"/>')
        out.append(f'<rect id="{bar.id}" {rect} fill="{_fill(bar.texture)}" stroke="#000" '
                   f'stroke-width="1"><title>{bar.value:g}</title></rect>')
    out.append('</g><g id="key">')
    for s in layout.key_samples:
        out.append(f'<rect id="{s.id}" x="{s.x:.2f}" y="{s.y:.2f}" width="{s.width:.2f}" '
                   f'height="{s.height:.2f}" fill="{_fill(s.texture)}" stroke="#000" '
                   'stroke-width="1"/>')
    out.append('</g><g id="axes">')
    out += [line_svg(l) for l in layout.lines if l.kind != "grid"]
    out.append('</g><g id="braille">')
    radius = DOT_DIAMETER / 2
    for label in layout.labels:
        out.append(f'<g id="{label.id}"><title>{escape(label.text)}</title>')
        out += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.3f}" fill="#000"/>'
                for x, y in label.dot_centres()]
        out.append("</g>")
    out.append("</g>")

    if show_print:
        # Review aid only: blue italic text and a banner, so it cannot be
        # mistaken for the tactile content. Omit --show-print for embossing.
        out.append('<g id="print" fill="#1f5fbf" font-family="sans-serif" font-size="2.6" '
                   'font-style="italic">')
        out.append(f'<text x="{w / 2:.2f}" y="6" text-anchor="middle" font-size="3.5">'
                   "PRINT LABELS FOR REVIEW ONLY - do not emboss this version</text>")
        for label in layout.labels:
            b = label.bounds
            out.append(f'<text x="{b.x1:.2f}" y="{b.y2 + 2.8:.2f}">{escape(label.text)}</text>')
        out.append("</g>")

    if violations:
        out.append('<g id="violations" fill="none" stroke="#d00" stroke-width="0.6" '
                   'stroke-dasharray="2 1">')
        for v in violations:
            for element_id in v.element_ids:
                element = layout.by_id(element_id)
                if element is None:
                    continue
                b = element.bounds
                out.append(f'<rect x="{b.x1 - 1:.2f}" y="{b.y1 - 1:.2f}" '
                           f'width="{b.x2 - b.x1 + 2:.2f}" height="{b.y2 - b.y1 + 2:.2f}">'
                           f"<title>{escape(v.rule.id)} {escape(v.rule.section)}: "
                           f"{escape(v.message)}</title></rect>")
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"
