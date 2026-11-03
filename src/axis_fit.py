"""Robust fit from tick pixel position to tick value, and bar-to-label pairing.

Reads of tick labels may be wrong, so the line is not a plain least-squares fit:
every pair of points proposes a line, the line supported by the most points wins
(inliers lie within INLIER_TOL of the implied value span), and the winner is refit
on its inliers. Lines that rise with y are rejected because pixel y grows downward
and the value axis of a vertical bar chart grows upward.
"""
import itertools

INLIER_TOL = 0.02


def ols(points):
    """(a, b) of v = a + b*y; None when y does not vary."""
    n = len(points)
    my = sum(y for y, _ in points) / n
    mv = sum(v for _, v in points) / n
    sxx = sum((y - my) ** 2 for y, _ in points)
    if sxx == 0:
        return None
    b = sum((y - my) * (v - mv) for y, v in points) / sxx
    return mv - b * my, b


def robust_fit(points, strict_ambiguity=False):
    """-> (a, b, inliers, unique) or None when no admissible line exists.

    ``unique`` is False when two lines with the most inliers are supported by
    different sets of points. (strict_ambiguity=True is the first version of this
    rule, which also counted lines with the same inliers but slightly different
    slopes; it over-flags and is kept only to report that comparison.)
    """
    span = max(y for y, _ in points) - min(y for y, _ in points)
    best, ties = None, []
    for (y1, v1), (y2, v2) in itertools.combinations(points, 2):
        if abs(y2 - y1) < 1:
            continue
        b = (v2 - v1) / (y2 - y1)
        if b >= 0:
            continue
        a = v1 - b * y1
        tol = INLIER_TOL * abs(b) * span
        inliers = [(y, v) for y, v in points if abs(v - (a + b * y)) <= tol]
        score = (len(inliers), -sum((v - (a + b * y)) ** 2 for y, v in inliers) / max(tol, 1e-12) ** 2)
        if best is None or score[0] > best[0][0]:
            best, ties = (score, inliers, b), [(b, frozenset(inliers))]
        elif score[0] == best[0][0]:
            ties.append((b, frozenset(inliers)))
            if score[1] > best[0][1]:
                best = (score, inliers, b)
    if best is None:
        return None
    fit = ols(best[1]) if len(best[1]) >= 2 else None
    if fit is None:
        return None
    if strict_ambiguity:
        unique = all(abs(b - best[2]) <= 0.02 * abs(best[2]) for b, _ in ties)
    else:
        unique = len({inl for _, inl in ties}) == 1
    return fit[0], fit[1], best[1], unique


def pair_by_overlap(labels, bars):
    """{bar index: label index} by largest horizontal overlap, one label per bar.

    ``labels`` and ``bars`` are (x, y, width, height) boxes.
    """
    options = sorted(((min(l[0] + l[2], b[0] + b[2]) - max(l[0], b[0]), li, bi)
                      for li, l in enumerate(labels) for bi, b in enumerate(bars)), reverse=True)
    pairs, used = {}, set()
    for overlap, li, bi in options:
        if overlap > 0 and li not in used and bi not in pairs:
            pairs[bi] = li
            used.add(li)
    return pairs
