"""Uncontracted (grade 1) Unified English Braille for short chart labels.

This is a small, explicit subset meant for graph labels: letters, capitals,
digits with the numeric indicator, and common punctuation. It is not a full
transcriber. Text with characters outside the subset raises ``ValueError``
instead of producing a guess. Contracted braille and Nemeth Code are out of
scope; liblouis is the planned route for full translation.

Each braille cell is a ``frozenset`` of dot numbers 1-6.
"""

from __future__ import annotations

Cell = frozenset

_LETTER_DOTS = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4),
    "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}
_DIGIT_LETTER = dict(zip("1234567890", "abcdefghij"))

# Transcriber's note indicators, UEB Rule 3.27: dots 4, 46, 126 and 4, 46, 345.
TN_OPEN = (Cell({4}), Cell({4, 6}), Cell({1, 2, 6}))
TN_CLOSE = (Cell({4}), Cell({4, 6}), Cell({3, 4, 5}))
NUMERIC_INDICATOR = Cell({3, 4, 5, 6})
CAPITAL_LETTER = Cell({6})
GRADE1_SYMBOL = Cell({5, 6})
BLANK = Cell()
# Guide dot (dot 5) leading from a row label to the graph, BANA §6.6.6.6 and §6.7.6.
GUIDE_DOT = Cell({5})

# Punctuation and signs in UEB. Multi-cell symbols are listed cell by cell.
_SYMBOLS = {
    ".": ((2, 5, 6),), ",": ((2,),), "-": ((3, 6),), "'": ((3,),),
    ":": ((2, 5),), ";": ((2, 3),), "?": ((2, 3, 6),), "!": ((2, 3, 5),),
    "(": ((5,), (1, 2, 6)), ")": ((5,), (3, 4, 5)), "/": ((4, 5, 6), (3, 4)),
    "%": ((4, 6), (3, 5, 6)), "$": ((4,), (2, 3, 4)), "&": ((4,), (1, 2, 3, 4, 6)),
    "+": ((5,), (2, 3, 5)), "=": ((5,), (2, 3, 5, 6)),
}
_MINUS = ((5,), (3, 6))


def _cells(*groups) -> list[Cell]:
    return [Cell(group) for group in groups]


def to_braille(text: str) -> tuple[Cell, ...]:
    """Translate ``text`` to uncontracted UEB cells.

    Rules applied:
    - a digit run starts with the numeric indicator; ``.`` and ``,`` between
      digits stay in numeric mode (UEB decimal point and numeric comma);
    - a letter a-j directly after a digit gets the grade 1 symbol indicator;
    - one capital gets dot 6; a run of two or more capitals gets dot 6 twice
      (capitals word indicator);
    - a hyphen directly before a digit, at the start or after a space, is a minus sign.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("braille text must be a nonempty string")
    text = " ".join(text.split())
    out: list[Cell] = []
    numeric = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch.isdigit():
            if not numeric:
                out.append(NUMERIC_INDICATOR)
                numeric = True
            out.append(Cell(_LETTER_DOTS[_DIGIT_LETTER[ch]]))
        elif ch in ".," and numeric and nxt.isdigit():
            out.extend(_cells(*_SYMBOLS[ch]))
        elif ch.isascii() and ch.isalpha():
            if ch.isupper():
                run = 0
                while i + run < len(text) and text[i + run].isascii() and text[i + run].isalpha():
                    run += 1
                word = text[i:i + run]
                if run >= 2 and word.isupper():
                    out.extend([CAPITAL_LETTER, CAPITAL_LETTER])
                    out.extend(Cell(_LETTER_DOTS[c.lower()]) for c in word)
                    numeric = False
                    i += run
                    continue
                out.append(CAPITAL_LETTER)
            elif numeric and ch in "abcdefghij":
                out.append(GRADE1_SYMBOL)
            out.append(Cell(_LETTER_DOTS[ch.lower()]))
            numeric = False
        elif ch == " ":
            out.append(BLANK)
            numeric = False
        elif ch == "-" and nxt.isdigit() and (i == 0 or text[i - 1] == " "):
            out.extend(_cells(*_MINUS))
            numeric = False
        elif ch in _SYMBOLS:
            out.extend(_cells(*_SYMBOLS[ch]))
            numeric = False
        else:
            raise ValueError(f"character {ch!r} is outside the supported braille subset")
        i += 1
    return tuple(out)


def to_unicode(cells: tuple[Cell, ...]) -> str:
    """Render cells as Unicode braille patterns (U+2800 block), for display only."""
    return "".join(chr(0x2800 + sum(1 << (dot - 1) for dot in cell)) for cell in cells)


def has_numeric_indicator(cells: tuple[Cell, ...]) -> bool:
    return NUMERIC_INDICATOR in cells
