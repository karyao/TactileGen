"""Key symbols for labels that do not fit on the graphic (BANA §5.8).

Two kinds, tried in this order by the layout:

- **Abbreviation keys** (§5.8.1, literary): two or three lower-case letters,
  two preferred (§5.8.1.1), starting with the first letter of the item
  (recommended, §5.8.1.1), at least one letter with dot 3 and/or dot 6
  (§5.8.1.2), never ``aa``, ``ac``, ``cc`` or ``ca`` (§5.8.1.2).
- **Numeric keys** (§5.8.1.6): the number with the numeric indicator and no
  period.

The braille here is uncontracted. §5.8.1.4 says a single-cell contraction
should be used when it occurs in the keyed word, which would shrink a key
such as ``st`` to one cell and break the two-cell minimum, so letter groups
that UEB writes as a single cell are never used as keys.
"""

from __future__ import annotations

from itertools import combinations

from src.tactile.braille import to_braille
from src.tactile.constants import LETTER_KEY_MAX_CELLS, LETTER_KEY_MIN_CELLS

# Letters whose braille cell contains dot 3 and/or dot 6 (k-z), §5.8.1.2.
_DOT_3_OR_6 = frozenset(c for c in "abcdefghijklmnopqrstuvwxyz"
                        if {3, 6} & set(next(iter(to_braille(c)))))
_FORBIDDEN = frozenset({"aa", "ac", "cc", "ca"})   # §5.8.1.2
# UEB single-cell contractions spelled with two or three letters (§5.8.1.4).
_SINGLE_CELL_GROUPS = frozenset({
    "ch", "gh", "sh", "th", "wh", "ed", "er", "ou", "ow", "st", "ar", "en", "in",
    "ea", "bb", "cc", "dd", "ff", "gg", "be", "of",
    "and", "for", "the", "ing", "con", "dis", "com", "with",
})


def is_valid_abbreviation_key(key: str) -> bool:
    """True when ``key`` meets the §5.8.1 literary alphabetic-key rules."""
    return (LETTER_KEY_MIN_CELLS <= len(key) <= LETTER_KEY_MAX_CELLS
            and key.isascii() and key.isalpha() and key.islower()
            and any(c in _DOT_3_OR_6 for c in key)
            and key not in _FORBIDDEN
            and not any(group in key for group in _SINGLE_CELL_GROUPS))


def _candidates(name: str):
    letters = [c for c in name.lower() if c.isascii() and c.isalpha()]
    if not letters:
        return
    first, rest = letters[0], letters[1:]
    for size in range(LETTER_KEY_MIN_CELLS, LETTER_KEY_MAX_CELLS + 1):
        for picked in combinations(rest, size - 1):
            yield first + "".join(picked)


def abbreviation_keys(names: list[str] | tuple[str, ...]) -> list[str]:
    """One distinct abbreviation key per name, built from the name's letters.

    Raises ``ValueError`` when a name has no valid key (for example a name
    with no letter from k to z, or a number), so the caller can fall back to
    numeric keys.
    """
    keys: list[str] = []
    for name in names:
        key = next((k for k in _candidates(name)
                    if is_valid_abbreviation_key(k) and k not in keys), None)
        if key is None:
            raise ValueError(f"no BANA §5.8.1 abbreviation key for {name!r}")
        keys.append(key)
    return keys


def numeric_keys(count: int) -> list[str]:
    """Numeric keys 1..count; braille adds the numeric indicator (§5.8.1.6)."""
    return [str(n) for n in range(1, count + 1)]


def is_valid_numeric_key(key: str) -> bool:
    return key.isdigit()
