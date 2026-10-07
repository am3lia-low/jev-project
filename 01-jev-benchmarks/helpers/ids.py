"""Stable question IDs such as "arcc-test-0417" (A2 rule 5).

An ID is `{dataset}-{source split}-{row index}`. The row index is the question's
position in the official HuggingFace split, so the ID never depends on sampling
and is shared by every version of the question (original, shuffled,
paraphrased, synonym-swapped).
"""

import re
from typing import NamedTuple

_SPLIT_ALIASES = {"validation": "val"}
_ID_PATTERN = re.compile(r"^(?P<dataset>[a-z0-9]+)-(?P<split>[a-z]+)-(?P<index>\d{4,})$")


class ParsedId(NamedTuple):
    dataset: str
    source_split: str
    index: int


def make_id(dataset: str, source_split: str, index: int) -> str:
    """Build an ID; "validation" is written as "val"."""
    if index < 0:
        raise ValueError(f"row index must be >= 0, got {index}")
    split = _SPLIT_ALIASES.get(source_split, source_split)
    question_id = f"{dataset}-{split}-{index:04d}"
    if not _ID_PATTERN.match(question_id):
        raise ValueError(f"invalid ID parts: dataset={dataset!r}, split={source_split!r}")
    return question_id


def parse_id(question_id: str) -> ParsedId:
    """Split an ID back into dataset, source split and row index."""
    match = _ID_PATTERN.match(question_id)
    if match is None:
        raise ValueError(f"not a valid question ID: {question_id!r}")
    return ParsedId(match["dataset"], match["split"], int(match["index"]))
