"""Level 0-3 views of the test set (A6).

| Level | Source                                   | Option order       |
|-------|------------------------------------------|--------------------|
| 0     | data/sampled/<dataset>/test.jsonl        | original           |
| 1     | data/sampled/<dataset>/test.jsonl        | balanced shuffle   |
| 2     | data/paraphrased/<dataset>/test.jsonl    | its own shuffle    |
| 3     | data/new/reviewed.jsonl (paper datasets) | balanced shuffle   |

Levels 0 and 1 are not separate files; they are the two stored orders of the
same questions. Every level yields `LevelItem`s, the exact options and answer
letter a model is shown, so the run scripts never handle orders themselves.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import DATASETS, NEW_REVIEWED_PATH, PARAPHRASED_DIR, SAMPLED_DIR
from .schema import OrderName, Question, read_jsonl

LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class LevelItem:
    level: int
    id: str
    dataset: str
    question: str
    options: tuple[str, ...]  # in the order shown to the model
    answer: str  # letter in that order
    version: str


def _item(level: int, question: Question, order: OrderName) -> LevelItem:
    options, answer = question.view(order)
    return LevelItem(level, question.id, question.dataset, question.question, options, answer, question.version)


def _test_file(root: Path, dataset: str) -> Path:
    path = root / dataset / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet; run the script that writes it first")
    return path


def level_items(level: int, datasets: Iterable[str] | None = None) -> list[LevelItem]:
    """All items of one level, dataset by dataset in `DATASETS` order.

    Level 3 only exists for the paper datasets; asking for SWAG yields nothing.
    """
    keys = list(DATASETS) if datasets is None else list(datasets)
    unknown = set(keys) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown dataset(s) {sorted(unknown)}")

    if level in (0, 1):
        order: OrderName = "original" if level == 0 else "shuffled"
        return [_item(level, q, order) for key in keys for q in read_jsonl(_test_file(SAMPLED_DIR, key))]
    if level == 2:
        return [
            _item(2, q, "shuffled") for key in keys for q in read_jsonl(_test_file(PARAPHRASED_DIR, key))
        ]
    if level == 3:
        if not NEW_REVIEWED_PATH.exists():
            raise FileNotFoundError(f"{NEW_REVIEWED_PATH} does not exist yet (Level 3 = reviewed 05 output)")
        wanted = set(keys)
        return [_item(3, q, "shuffled") for q in read_jsonl(NEW_REVIEWED_PATH) if q.dataset in wanted]
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
