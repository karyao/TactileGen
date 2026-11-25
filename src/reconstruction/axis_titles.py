"""Resolve printed axis-title roles using source-image geometry."""
from dataclasses import replace
from statistics import median

from src.contracts import DetectionFrame


def resolve_axis_titles(frame, ocr, *, orientation='vertical', diagnostics=None):
    """Return a frame with geometrically plausible title roles.

    Titles without sufficient axis evidence are left for manual review.
    """
    ds = frame.detections
    if not any(d.role in ('x_axis_title', 'y_axis_title') for d in ds):
        return frame
    plots = [d for d in ds if d.role == 'plot_area' and d.confidence >= .3]
    source = 'detected'
    if plots:
        p = max(plots, key=lambda d: d.confidence).box
        left, top, right, bottom = p.x1, p.y1, p.x2, p.y2
    else:
        bars = [d.box for d in ds if d.role == f'{orientation}_bar' and d.confidence >= .3]
        xt = [d.box for d in ds if d.role == 'x_tick_label' and d.confidence >= .3]
        yt = [d.box for d in ds if d.role == 'y_tick_label' and d.confidence >= .3]
        if len(bars) < 2 or len(xt) < 2 or len(yt) < 2:
            left = top = right = bottom = None
            source = 'unavailable'
        elif orientation == 'vertical':
            endpoints = [b.y2 for b in bars]
            aligned = [b.x2 for b in yt]
            if max(endpoints) - min(endpoints) > frame.height * .025 or max(aligned) - min(aligned) > frame.width * .04:
                left = top = right = bottom = None
                source = 'unavailable'
            else:
                left, bottom = median(aligned), median(endpoints)
                right = max(b.x2 for b in bars)
                top = min(min(b.y1 for b in bars), min(b.y1 for b in yt))
        else:
            endpoints = [b.x1 for b in bars]
            aligned = [b.y1 for b in xt]
            if max(endpoints) - min(endpoints) > frame.width * .025 or max(aligned) - min(aligned) > frame.height * .04:
                left = top = right = bottom = None
                source = 'unavailable'
            else:
                left, bottom = median(endpoints), median(aligned)
                right = max(max(b.x2 for b in bars), max(b.x2 for b in xt))
                top = min(b.y1 for b in bars)
        if left is not None:
            source = 'inferred'
    readable = {r.detection_id: r.confidence for r in ocr.regions if r.text.strip()}
    candidates = [d for d in ds if d.role in ('x_axis_title', 'y_axis_title') and d.id in readable and d.confidence >= .3]
    choices = {'x_axis_title': [], 'y_axis_title': []}
    plausible = set()
    for d in candidates:
        if left is None:
            continue
        b = d.box
        cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        if left <= cx <= right and cy > bottom:
            choices['x_axis_title'].append((cy - bottom, d))
            plausible.add(d.id)
        if top <= cy <= bottom and cx < left:
            choices['y_axis_title'].append((left - cx, d))
            plausible.add(d.id)
    selected = {}
    for role, options in choices.items():
        options.sort(key=lambda item: (item[0], item[1].role != role,
                                       -readable[item[1].id], -item[1].confidence))
        if options and (len(options) == 1 or options[1][0] - options[0][0] > 2):
            selected[options[0][1].id] = role
    # Remove rejected title detections from the effective frame, so downstream
    # reconstruction and cutoff checks use the same assignment.
    output = tuple(replace(d, role=selected[d.id]) if d.id in selected else d
                   for d in ds if d.role not in choices or d.id in selected)
    if diagnostics is not None:
        rejected = {d.id: ('axis_geometry_unavailable' if left is None else
                           'competing_title' if d.id in plausible else
                           'outside_axis_regions') for d in candidates if d.id not in selected}
        diagnostics.event('reconstruction', 'axis_title_resolution', geometry=source,
                          accepted=selected, rejected=list(rejected), rejection_reasons=rejected)
    return DetectionFrame(frame.source_key, frame.width, frame.height, output)
