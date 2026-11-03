"""Text helpers shared by the OCR scoring and the chart interpreter."""

import math
import re
import unicodedata

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                         "\u2212": "-", "\u2013": "-", "\u2014": "-"})


def clean_text(text: str) -> str:
    """Text as shown to people: a literal backslash-n (an annotation artifact for a line
    break) becomes a space and whitespace runs collapse. Unicode is kept as read."""
    return " ".join(text.replace("\\n", " ").split())


def normalize(text: str) -> str:
    """Normalized form used by the OCR normalized metrics (the raw metric only strips).

    clean_text, then NFKC (ligatures to letters) and curly quotes and dashes to ASCII.
    The backslash-n rule was added after the first Tesseract test scoring.
    """
    return " ".join(unicodedata.normalize("NFKC", clean_text(text)).translate(_QUOTES).split())


def parse_number(text: str) -> float | None:
    """Parse a tick or value label; trailing % is dropped, thousands commas removed."""
    t = normalize(text).replace(" ", "")
    t = t[:-1] if t.endswith("%") else t
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    elif re.fullmatch(r"[-+]?\d+,\d+", t):
        t = t.replace(",", ".")
    try:
        value = float(t)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
