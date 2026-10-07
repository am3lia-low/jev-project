"""Load the five datasets from HuggingFace into the shared question format (A1).

Each dataset has a pure `parse_*` function (one raw row -> `Question`) and
`load_questions` does the download. Run the download on the API laptop if
HuggingFace needs a login or is blocked on the work network.

Row fields below follow the HuggingFace dataset cards. If a dataset changes its
schema, `_require` fails with the missing field name rather than guessing.
"""

from collections.abc import Callable, Mapping
from typing import Any

from .config import DATASETS
from .ids import make_id
from .labels import answer_position, letter_from_position
from .schema import Question

Row = Mapping[str, Any]


def _require(row: Row, *fields: str) -> None:
    missing = [name for name in fields if name not in row]
    if missing:
        raise KeyError(f"row is missing field(s) {missing}; found {sorted(row)}")


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _video_id(source_id: str) -> str:
    """HellaSwag source IDs look like "activitynet~v_abc"; SWAG uses "v_abc"."""
    return source_id.split("~")[-1].strip()


def parse_hellaswag(row: Row, index: int, source_split: str, role: str) -> Question:
    _require(row, "ctx", "endings", "label", "source_id")
    if row["label"] in ("", None):
        raise ValueError(f"hs row {index} of {source_split} has no label (hidden test split?)")
    return Question(
        id=make_id("hs", source_split, index),
        dataset="hs",
        split=role,
        question=_clean(row["ctx"]),
        options=tuple(_clean(ending) for ending in row["endings"]),
        answer=letter_from_position(int(row["label"])),
        meta={
            "video_id": _video_id(row["source_id"]),
            "context": _clean(row["ctx"]),
            "source_id": row["source_id"],
            "activity_label": row.get("activity_label", ""),
            "source_split": source_split,
        },
    )


def _parse_choices_with_key(key: str, row: Row, index: int, source_split: str, role: str) -> Question:
    """ARC and CommonsenseQA share the `choices: {label, text}` + `answerKey` layout."""
    _require(row, "question", "choices", "answerKey")
    if row["answerKey"] in ("", None):
        raise ValueError(f"{key} row {index} of {source_split} has no answer (hidden test split?)")
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    return Question(
        id=make_id(key, source_split, index),
        dataset=key,
        split=role,
        question=_clean(row["question"]),
        options=tuple(_clean(text) for text in texts),
        answer=letter_from_position(answer_position(labels, row["answerKey"])),
        meta={"source_id": row.get("id", ""), "source_split": source_split},
    )


def parse_arc_easy(row: Row, index: int, source_split: str, role: str) -> Question:
    return _parse_choices_with_key("arce", row, index, source_split, role)


def parse_arc_challenge(row: Row, index: int, source_split: str, role: str) -> Question:
    return _parse_choices_with_key("arcc", row, index, source_split, role)


def parse_commonsenseqa(row: Row, index: int, source_split: str, role: str) -> Question:
    return _parse_choices_with_key("csqa", row, index, source_split, role)


def parse_swag(row: Row, index: int, source_split: str, role: str) -> Question:
    _require(row, "video-id", "startphrase", "label", "ending0", "ending1", "ending2", "ending3")
    return Question(
        id=make_id("swag", source_split, index),
        dataset="swag",
        split=role,
        question=_clean(row["startphrase"]),
        options=tuple(_clean(row[f"ending{i}"]) for i in range(4)),
        answer=letter_from_position(int(row["label"])),
        meta={
            "video_id": row["video-id"].strip(),
            "context": _clean(row["startphrase"]),
            "source_split": source_split,
        },
    )


_PARSERS: dict[str, Callable[[Row, int, str, str], Question]] = {
    "hs": parse_hellaswag,
    "arce": parse_arc_easy,
    "arcc": parse_arc_challenge,
    "csqa": parse_commonsenseqa,
    "swag": parse_swag,
}


def load_questions(dataset: str, source_split: str, role: str) -> list[Question]:
    """Download one official split and convert every row.

    `source_split` is the HuggingFace name ("train", "validation", "test");
    `role` is the project split ("train", "val", "test") the questions will
    be used as. Row order is preserved, so row index == the index in the ID.
    """
    from datasets import load_dataset  # imported here: not needed by most scripts

    spec = DATASETS[dataset]
    rows = load_dataset(spec.hf_path, spec.hf_config, split=source_split)
    parse = _PARSERS[dataset]
    return [parse(row, index, source_split, role) for index, row in enumerate(rows)]
