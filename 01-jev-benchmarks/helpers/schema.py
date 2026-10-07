"""The one question format shared by every script, plus JSONL read/write."""

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .labels import MAX_OPTIONS, letter_from_position, position_from_letter

ORIGINAL = "original"  # the human-written (or, for Level 3, drafted) wording
PARAPHRASE = "paraphrase"
VALID_SPLITS = ("train", "val", "test")

OrderName = Literal["original", "shuffled"]


@dataclass(frozen=True, slots=True)
class Question:
    """One multiple-choice question in one version.

    `options` and `answer` are always in the **original order**. The shuffled
    order is stored as `shuffled_order`: shuffled position `i` shows the option
    that sits at original position `shuffled_order[i]`.

    A paraphrase or synonym swap keeps the ID of its source question and sets
    `version`; its `options[i]` is the rewording of the source's `options[i]`,
    so `answer` is unchanged and each version carries its own `shuffled_order`.
    `meta` holds source details (e.g. video ID) that only some steps need.
    """

    id: str
    dataset: str
    split: str
    question: str
    options: tuple[str, ...]
    answer: str
    shuffled_order: tuple[int, ...] | None = None
    version: str = ORIGINAL
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept lists (e.g. straight from JSON) and freeze them.
        object.__setattr__(self, "options", tuple(self.options))
        if self.shuffled_order is not None:
            object.__setattr__(self, "shuffled_order", tuple(self.shuffled_order))

        if self.split not in VALID_SPLITS:
            raise ValueError(f"{self.id}: split must be one of {VALID_SPLITS}, got {self.split!r}")
        if not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"{self.id}: needs 2-{MAX_OPTIONS} options, got {len(self.options)}")
        if position_from_letter(self.answer) >= len(self.options):
            raise ValueError(f"{self.id}: answer {self.answer!r} is outside the options")
        if self.shuffled_order is not None and sorted(self.shuffled_order) != list(
            range(len(self.options))
        ):
            raise ValueError(
                f"{self.id}: shuffled_order {self.shuffled_order} is not a permutation "
                f"of 0-{len(self.options) - 1}"
            )

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def answer_position(self) -> int:
        """0-based position of the correct option in the original order."""
        return position_from_letter(self.answer)

    @property
    def shuffled_options(self) -> tuple[str, ...]:
        return self.view("shuffled")[0]

    @property
    def shuffled_answer(self) -> str:
        return self.view("shuffled")[1]

    def view(self, order: OrderName) -> tuple[tuple[str, ...], str]:
        """Return `(options, answer letter)` in the requested order."""
        if order == "original":
            return self.options, self.answer
        if order == "shuffled":
            if self.shuffled_order is None:
                raise ValueError(f"{self.id}: no shuffled order has been assigned yet")
            options = tuple(self.options[i] for i in self.shuffled_order)
            return options, letter_from_position(self.shuffled_order.index(self.answer_position))
        raise ValueError(f"order must be 'original' or 'shuffled', got {order!r}")

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["options"] = list(self.options)
        record["shuffled_order"] = None if self.shuffled_order is None else list(self.shuffled_order)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Question":
        return cls(**record)


def write_jsonl(path: str | Path, questions: Iterable[Question]) -> int:
    """Write questions as one JSON object per line; return how many were written.

    Writes to a temporary file first, so an interrupted run never leaves a
    half-written dataset behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    os.replace(temp_path, path)
    return count


def iter_jsonl(path: str | Path) -> Iterator[Question]:
    """Yield questions from a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield Question.from_dict(json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def read_jsonl(path: str | Path) -> list[Question]:
    return list(iter_jsonl(path))
