"""Answer labels: normalise to letters A-E and convert to/from positions (A2 rule 4)."""

from collections.abc import Sequence

LETTERS = "ABCDE"
MAX_OPTIONS = len(LETTERS)


def letter_from_position(position: int) -> str:
    """0-based position -> letter (0 -> "A")."""
    if not 0 <= position < MAX_OPTIONS:
        raise ValueError(f"position must be 0-{MAX_OPTIONS - 1}, got {position}")
    return LETTERS[position]


def position_from_letter(letter: str) -> int:
    """Letter -> 0-based position ("A" -> 0)."""
    if not isinstance(letter, str) or letter not in LETTERS or len(letter) != 1:
        raise ValueError(f"letter must be one of {', '.join(LETTERS)}, got {letter!r}")
    return LETTERS.index(letter)


def normalise_label(label: str | int) -> str:
    """Normalise a label to a letter A-E.

    Accepts letters in either case ("b") and the 1-based numeric labels some
    ARC items use ("1".."5" or 1..5). 0-based indices (HellaSwag, SWAG) are
    ambiguous here and must go through `letter_from_position` instead.
    """
    text = str(label).strip()
    if text.upper() in tuple(LETTERS):
        return text.upper()
    if text.isdigit() and 1 <= int(text) <= MAX_OPTIONS:
        return LETTERS[int(text) - 1]
    raise ValueError(f"cannot normalise answer label {label!r} to A-{LETTERS[-1]}")


def answer_position(choice_labels: Sequence[str | int], answer_key: str | int) -> int:
    """0-based position of `answer_key` among a question's own choice labels.

    Robust to any labelling scheme (A-D, 1-4, ...): the answer is located by
    matching it against the labels the question itself uses.
    """
    labels = [str(label).strip() for label in choice_labels]
    key = str(answer_key).strip()
    if key not in labels:
        raise ValueError(f"answer key {answer_key!r} not among choice labels {labels}")
    if labels.count(key) > 1:
        raise ValueError(f"answer key {answer_key!r} is ambiguous among choice labels {labels}")
    return labels.index(key)
