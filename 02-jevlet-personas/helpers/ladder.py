"""The difficulty ladder: benchmark questions, persona conditions and records (B2).

Five benchmarks from grade school to the expert frontier (LADDER_BENCHMARKS),
LADDER_N questions each, built by 10_build_ladder.py into
data/ladder/<benchmark>.jsonl:

    arce, arcc   Part A's sampled ARC test questions (originals), a seeded draw
    mmlu         cais/mmlu "all" test split, a seeded draw
    gpqa         Idavidrein/gpqa "gpqa_diamond" (198; gated: accept the terms on
                 HuggingFace and log in). The correct answer is always the first
                 column there, so options are shuffled (seeded).
    hle          cais/hle test split (gated), text-only multiple-choice questions:
                 answer_type "multipleChoice" and no image. The options are inside
                 the question text after "Answer Choices:"; they are split out
                 here, and questions whose options can't be parsed are dropped.

The HuggingFace field names for GPQA and HLE are as documented on their
dataset cards; unchecked here (HuggingFace needs a login for both). The
loaders fail loudly on a missing field; confirm on the API laptop.

A line: {"id", "benchmark", "question", "options": [...], "answer": letter, "meta": {...}}

Conditions: "none" (actual ability) and "<persona>-<placement>" for each
LADDER_PERSONAS entry, placement state or instructions, as in B1.

Records: one JSON object per question per line in
`results/raw/<model slug>/ladder/<benchmark>/<condition>.jsonl`:

    id, benchmark, condition, persona, placement, n_options, answer, pred,
    probabilities, correct (a failed call counts as wrong, as in Part A),
    cost_usd, cached, error
"""

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_common.parallel import run_parallel

from .bigfive import model_slug, read_records, write_records
from .choice import LETTERS, ChoiceAnswer, ChoiceAnswerer, choice_request, picked_letter
from .config import LADDER_DIR, LADDER_N, LADDER_PERSONAS, LADDER_SEED, PART_DIR, PLACEMENTS, RAW_RESULTS_DIR

PART_A_SAMPLED = PART_DIR.parent / "01-jev-benchmarks" / "data" / "sampled"
_CHOICE_LINE = re.compile(r"^\s*([A-Z])[.)]\s+(.*)$")


# ---------------------------------------------------------------------------
# Loading the benchmarks (10_build_ladder.py)
# ---------------------------------------------------------------------------
def _require(row: dict[str, Any], *fields: str) -> None:
    missing = [f for f in fields if f not in row]
    if missing:
        raise KeyError(f"fields {missing} not in the dataset row; it has {sorted(row)}")


def _draw(items: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    if len(items) <= LADDER_N:
        return items
    return sorted(random.Random(f"{LADDER_SEED}:{benchmark}").sample(items, LADDER_N), key=lambda i: i["id"])


def load_arc(benchmark: str) -> list[dict[str, Any]]:
    with (PART_A_SAMPLED / benchmark / "test.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    items = [{"id": r["id"], "benchmark": benchmark, "question": r["question"], "options": r["options"],
              "answer": r["answer"], "meta": r.get("meta", {})} for r in rows]
    return _draw(items, benchmark)


def load_mmlu() -> list[dict[str, Any]]:
    from datasets import load_dataset  # imported here: only 10_build_ladder.py needs it

    items = []
    for index, row in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        _require(row, "question", "choices", "answer", "subject")
        items.append({"id": f"mmlu-{index:05d}", "benchmark": "mmlu", "question": row["question"].strip(),
                      "options": [str(c).strip() for c in row["choices"]], "answer": LETTERS[int(row["answer"])],
                      "meta": {"subject": row["subject"]}})
    return _draw(items, "mmlu")


def load_gpqa() -> list[dict[str, Any]]:
    from datasets import load_dataset

    items = []
    for index, row in enumerate(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")):
        _require(row, "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
        correct = row["Correct Answer"].strip()
        options = [correct] + [row[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)]
        random.Random(f"{LADDER_SEED}:gpqa:{index}").shuffle(options)
        items.append({"id": f"gpqa-{index:03d}", "benchmark": "gpqa", "question": row["Question"].strip(),
                      "options": options, "answer": LETTERS[options.index(correct)],
                      "meta": {"domain": row.get("High-level domain"), "record_id": row.get("Record ID")}})
    return _draw(items, "gpqa")


def split_hle_choices(question: str) -> tuple[str, list[str]] | None:
    """(question stem, options) from an HLE question with an "Answer Choices:" block, or None."""
    head, sep, tail = question.rpartition("Answer Choices:")
    if not sep:
        return None
    options: list[str] = []
    for line in tail.strip().splitlines():
        match = _CHOICE_LINE.match(line)
        if match and match.group(1) == LETTERS[len(options)]:
            options.append(match.group(2).strip())
        elif options and line.strip():
            options[-1] += " " + line.strip()  # an option running over several lines
    if len(options) < 2:
        return None
    return head.strip(), options


def load_hle() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Text-only multiple-choice HLE questions, and counts of what was dropped and why."""
    from datasets import load_dataset

    items, counts = [], {"total": 0, "not multiple choice": 0, "has an image": 0, "options not parsed": 0}
    for row in load_dataset("cais/hle", split="test"):
        _require(row, "id", "question", "answer", "answer_type", "image")
        counts["total"] += 1
        if row["answer_type"] != "multipleChoice":
            counts["not multiple choice"] += 1
            continue
        if row["image"]:
            counts["has an image"] += 1
            continue
        parsed = split_hle_choices(row["question"])
        answer = str(row["answer"]).strip()
        if parsed is None or answer not in LETTERS[: len(parsed[1])]:
            counts["options not parsed"] += 1
            continue
        stem, options = parsed
        items.append({"id": f"hle-{row['id']}", "benchmark": "hle", "question": stem, "options": options,
                      "answer": answer, "meta": {"category": row.get("category"), "raw_subject": row.get("raw_subject")}})
    counts["kept"] = len(items)
    return _draw(items, "hle"), counts


def ladder_path(benchmark: str) -> Path:
    return LADDER_DIR / f"{benchmark}.jsonl"


def load_ladder(benchmark: str) -> list[dict[str, Any]]:
    path = ladder_path(benchmark)
    if not path.exists():
        raise SystemExit(f"{path} not found; run 10_build_ladder.py first")
    return read_records(path)


# ---------------------------------------------------------------------------
# Conditions, tasks and records
# ---------------------------------------------------------------------------
def conditions(personas: Sequence[str] = tuple(LADDER_PERSONAS),
               placements: Sequence[str] = PLACEMENTS) -> list[tuple[str, str | None, str | None]]:
    """(condition name, persona key, placement): "none" first, then each persona x placement."""
    return [("none", None, None), *((f"{p}-{pl}", p, pl) for p in personas for pl in placements)]


@dataclass(frozen=True, eq=False)
class LadderTask:
    item: dict[str, Any]
    condition: str  # "none" or "<persona>-<placement>"
    persona: str | None
    placement: str | None
    cache_tag: str | None = None

    @property
    def request(self) -> dict[str, Any]:
        text = LADDER_PERSONAS[self.persona] if self.persona else None
        return choice_request(self.item["question"], self.item["options"], text, self.placement)


def tasks_for(items: Sequence[dict[str, Any]], condition: tuple[str, str | None, str | None]) -> list[LadderTask]:
    name, persona, placement = condition
    return [LadderTask(item, name, persona, placement) for item in items]


def run_path(model: str, benchmark: str, condition: str) -> Path:
    return RAW_RESULTS_DIR / model_slug(model) / "ladder" / benchmark / f"{condition}.jsonl"


def _record(task: LadderTask, answer: ChoiceAnswer | None, error: str | None) -> dict[str, Any]:
    probabilities, result = answer if answer is not None else (None, None)
    letters = LETTERS[: len(task.item["options"])]
    pred = picked_letter(letters, probabilities, result)
    return {
        "id": task.item["id"], "benchmark": task.item["benchmark"], "condition": task.condition,
        "persona": task.persona, "placement": task.placement, "n_options": len(task.item["options"]),
        "answer": task.item["answer"], "pred": pred, "probabilities": probabilities,
        "correct": pred == task.item["answer"],
        "cost_usd": None if result is None else result.cost_usd,
        "cached": None if result is None else result.cached,
        "error": error,
    }


def run_tasks(tasks: list[LadderTask], answer: ChoiceAnswerer, model: str, workers: int) -> bool:
    """Answer one benchmark x condition and write its results file; True if the spending cap stopped it."""
    benchmark, condition = tasks[0].item["benchmark"], tasks[0].condition
    label = f"{benchmark} {condition}"
    done, failed, stopped = run_parallel(answer, tasks, workers, label)
    if stopped:
        print(f"  {label}: stopped at the spending cap after {len(done)}/{len(tasks)}; nothing written, re-run")
        return True
    records = {t.item["id"]: _record(t, result, None) for t, result in done}
    records |= {t.item["id"]: _record(t, None, error) for t, error in failed}
    path = run_path(model, benchmark, condition)
    write_records(path, [records[t.item["id"]] for t in tasks])
    accuracy = sum(r["correct"] for r in records.values()) / len(records)
    print(f"  {label}: {len(records)} questions, accuracy {accuracy:.1%}, {len(failed)} failed -> {path}")
    for task, error in failed[:10]:
        print(f"    {task.item['id']}: {error}")
    return False
