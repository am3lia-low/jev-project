"""Automatic checks for 04_validate.py and 06b_validate_swaps.py (A3.1, A3.3).

Every check returns a list of problems; an empty list means it passed. The
`check_paraphrase`, `check_swap` and `check_swap_set` functions combine the
individual checks and are what the scripts call.
"""

import difflib
import re
from collections.abc import Sequence

from .config import MAX_EDIT_SPAN, WORD_COUNT_TOLERANCE
from .labels import letter_from_position
from .schema import Question

# A word (keeping inner apostrophes and hyphens) or a single punctuation mark.
_TOKEN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]")
_WORD = re.compile(r"\w")


def normalise_text(text: str) -> str:
    """Casefolded text with whitespace collapsed, for comparing versions."""
    return " ".join(text.casefold().split())


def word_count(text: str) -> int:
    return sum(1 for token in _TOKEN.findall(text) if _WORD.match(token))


def option_key(text: str) -> str:
    """Whitespace-collapsed option text, case kept: "WW or Ww" and "WW or ww" differ (genotypes)."""
    return " ".join(text.split())


def duplicate_options(question: Question) -> list[str]:
    """Letters of options whose text repeats an earlier option's."""
    seen: set[str] = set()
    duplicates = []
    for i, option in enumerate(question.options):
        key = option_key(option)
        if key in seen:
            duplicates.append(letter_from_position(i))
        seen.add(key)
    return duplicates


def _fields(question: Question) -> list[tuple[str, str]]:
    """(label, text) for the question followed by each option, in original order."""
    options = [(f"option {letter_from_position(i)}", text) for i, text in enumerate(question.options)]
    return [("question", question.question), *options]


def _signature(question: Question) -> tuple[str, ...]:
    return tuple(normalise_text(text) for _, text in _fields(question))


def _paired_fields(original: Question, candidate: Question) -> list[tuple[str, str, str]]:
    """(label, original text, candidate text); needs equal option counts."""
    return [
        (label, before, after)
        for (label, before), (_, after) in zip(_fields(original), _fields(candidate), strict=True)
    ]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_answer_present(question: Question) -> list[str]:
    """The answer points at a non-empty option that no other option duplicates.

    A duplicated correct option would make the item ambiguous. Case counts
    (see `option_key`).
    """
    correct = option_key(question.options[question.answer_position])
    if not correct:
        return [f"correct option {question.answer} is empty"]
    duplicates = [
        letter_from_position(i)
        for i, option in enumerate(question.options)
        if i != question.answer_position and option_key(option) == correct
    ]
    if duplicates:
        return [f"correct option {question.answer} is duplicated by option(s) {', '.join(duplicates)}"]
    return []


def check_option_count(original: Question, candidate: Question) -> list[str]:
    if original.n_options != candidate.n_options:
        return [f"option count changed from {original.n_options} to {candidate.n_options}"]
    return []


def check_text_changed(original: Question, candidate: Question) -> list[str]:
    if _signature(original) == _signature(candidate):
        return ["text is unchanged (ignoring case and spacing)"]
    return []


def check_all_distinct(versions: Sequence[Question]) -> list[str]:
    """No two versions have the same question and options."""
    problems = []
    for i, first in enumerate(versions):
        for second in versions[i + 1 :]:
            if _signature(first) == _signature(second):
                problems.append(f"versions {first.version!r} and {second.version!r} are identical")
    return problems


def check_word_count(
    original: Question, candidate: Question, tolerance: int = WORD_COUNT_TOLERANCE
) -> list[str]:
    """The question and each option keep their word count to within +/- `tolerance`."""
    problems = []
    for label, before, after in _paired_fields(original, candidate):
        difference = word_count(after) - word_count(before)
        if abs(difference) > tolerance:
            problems.append(f"{label}: word count changed by {difference:+d} (limit +/-{tolerance})")
    return problems


def check_word_level_edits(
    original: Question, candidate: Question, max_span: int = MAX_EDIT_SPAN
) -> list[str]:
    """Only words were swapped: punctuation is untouched and each edit is short.

    Punctuation marks must appear in the same sequence (so sentence structure
    is intact), and no run of changed words may exceed `max_span` words on
    either side (so no clause was rewritten).
    """
    problems = []
    for label, before, after in _paired_fields(original, candidate):
        old_tokens, new_tokens = _TOKEN.findall(before), _TOKEN.findall(after)
        if _punctuation(old_tokens) != _punctuation(new_tokens):
            problems.append(f"{label}: punctuation changed")
        matcher = difflib.SequenceMatcher(
            a=[t.casefold() for t in old_tokens], b=[t.casefold() for t in new_tokens], autojunk=False
        )
        for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
            if tag != "equal" and max(a_end - a_start, b_end - b_start) > max_span:
                problems.append(
                    f"{label}: edit spans more than {max_span} words "
                    f"({' '.join(old_tokens[a_start:a_end])!r} -> {' '.join(new_tokens[b_start:b_end])!r})"
                )
    return problems


def _punctuation(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not _WORD.match(token)]


# ---------------------------------------------------------------------------
# What the scripts call
# ---------------------------------------------------------------------------
def check_paraphrase(original: Question, paraphrase: Question) -> list[str]:
    """04_validate.py: answer kept, option count unchanged, text actually changed (A3.1)."""
    problems = [*check_answer_present(paraphrase), *check_option_count(original, paraphrase)]
    return problems + check_text_changed(original, paraphrase)


def check_swap(original: Question, swap: Question) -> list[str]:
    """06b_validate_swaps.py: one swap against its original (A3.3)."""
    problems = [*check_answer_present(swap), *check_option_count(original, swap)]
    if original.n_options != swap.n_options:
        return problems  # the field-by-field checks below need matching option counts
    return [
        *problems,
        *check_text_changed(original, swap),
        *check_word_count(original, swap),
        *check_word_level_edits(original, swap),
    ]


def check_swap_set(original: Question, swaps: Sequence[Question]) -> list[str]:
    """06b_validate_swaps.py: the original plus all its swaps; each swap valid, all distinct."""
    problems = []
    for swap in swaps:
        problems += [f"{swap.version}: {problem}" for problem in check_swap(original, swap)]
    return problems + check_all_distinct([original, *swaps])
