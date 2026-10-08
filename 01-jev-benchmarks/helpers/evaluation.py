"""Running a model on the test levels and the quiz, and the result records (A8, A9).

Shared by 08_run_jev.py and 09_run_llm.py; 10_analyse.py reads the records.
The Colab notebook (07b) writes the same record format for BERT without
importing this module.

A `Task` is one question as decision-index sends it to Jev: `{state,
questions: {q1: {type: "choice", instructions, criteria}}}`. The LLM
baselines get the same request rendered as a chat prompt (`chat_messages`),
so every model sees the same text.

Records: one JSON object per question per line in
`results/raw/<model slug>/<run>.jsonl`:

    id, dataset, run, quiz_set (quiz only, else null), n_options,
    answer         correct letter as shown (quiz: where the original sits)
    pred           letter with the highest probability; null if the call failed
    probabilities  {letter: p}, renormalised; null if the call failed
    correct        pred == answer; a failed call counts as wrong (A9)
    latency_s, cost_usd, cached, error

Run names: level0..level3, quiz, repeat_1..repeat_3 (A9 repeatability), and
level<n>_run<k> for extra runs of a level. BERT adds val.
"""

import json
import os
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_common.openrouter import CallResult

from .config import DATASETS, N_REPEATABILITY, QUIZ_DIR, RAW_RESULTS_DIR, SAMPLE_SEED
from .generation import run_parallel
from .levels import level_items
from .prompts import EVAL_QUESTION_KEY, jev_eval_request, load, quiz_request

QUIZ_PATH = QUIZ_DIR / "quiz.jsonl"

# (probabilities, call result) for one task; `cache_tag` makes a repeat a fresh request.
Answerer = Callable[["Task", str | None], tuple[dict[str, float], CallResult]]


@dataclass(frozen=True)
class Task:
    id: str
    dataset: str
    answer: str  # correct letter as shown
    n_options: int
    quiz_set: str | None
    request: dict[str, Any]  # {state, questions}

    @property
    def question(self) -> dict[str, Any]:
        return self.request["questions"][EVAL_QUESTION_KEY]

    @property
    def letters(self) -> tuple[str, ...]:
        return tuple(self.question["criteria"])


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def level_tasks(level: int, datasets: Iterable[str] | None = None) -> list[Task]:
    return [
        Task(item.id, item.dataset, item.answer, len(item.options), None, jev_eval_request(item))
        for item in level_items(level, datasets)
    ]


def quiz_tasks(datasets: Iterable[str] | None = None) -> list[Task]:
    if not QUIZ_PATH.exists():
        raise FileNotFoundError(f"{QUIZ_PATH} does not exist yet; run 06c_build_quiz.py first")
    wanted = set(DATASETS if datasets is None else datasets)
    with QUIZ_PATH.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [
        Task(r["id"], r["dataset"], r["original"], len(r["options"]), r["quiz_set"], quiz_request(r))
        for r in records
        if r["dataset"] in wanted
    ]


def repeat_tasks(datasets: Iterable[str] | None = None) -> list[Task]:
    """N_REPEATABILITY Level 0 questions, split as evenly as possible over the datasets (A9)."""
    keys = list(DATASETS if datasets is None else datasets)
    tasks = []
    for index, key in enumerate(keys):
        n = N_REPEATABILITY // len(keys) + (index < N_REPEATABILITY % len(keys))
        pool = level_tasks(0, [key])
        tasks += sorted(random.Random(f"{SAMPLE_SEED}:repeat:{key}").sample(pool, n), key=lambda t: t.id)
    return tasks


# ---------------------------------------------------------------------------
# LLM rendering (decision-index's `transformers` engine)
# ---------------------------------------------------------------------------
def render_prompt(task: Task) -> str:
    """The user message decision-index builds for a choice question.

    Copied from decision_index/engines/transformers_engine.py `render_prompt`
    (checked 2026-10-08). Our state is always empty and our instructions and
    option texts are plain strings.
    """
    state = task.request["state"]
    lines = ["State:", json.dumps(state, ensure_ascii=False) if state else "(empty)", "",
             "Question:", task.question["instructions"], "", "Options:"]
    lines += [f"{key}: {text}" for key, text in task.question["criteria"].items()]
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def chat_messages(task: Task) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load("llm_system").template},
        {"role": "user", "content": render_prompt(task)},
    ]


# ---------------------------------------------------------------------------
# Running and records
# ---------------------------------------------------------------------------
def model_slug(model: str) -> str:
    """Folder name for a model ID: "typesafe/jev-1.13" -> "typesafe__jev-1.13"."""
    return model.replace("/", "__")


def run_path(model: str, run: str) -> Path:
    return RAW_RESULTS_DIR / model_slug(model) / f"{run}.jsonl"


def _record(task: Task, run: str, probabilities: dict[str, float] | None,
            result: CallResult | None, error: str | None) -> dict[str, Any]:
    pred = max(task.letters, key=lambda k: probabilities.get(k, 0.0)) if probabilities else None
    return {
        "id": task.id,
        "dataset": task.dataset,
        "run": run,
        "quiz_set": task.quiz_set,
        "n_options": task.n_options,
        "answer": task.answer,
        "pred": pred,
        "probabilities": probabilities,
        "correct": pred == task.answer,
        "latency_s": None if result is None else round(result.latency_s, 4),
        "cost_usd": None if result is None else result.cost_usd,
        "cached": None if result is None else result.cached,
        "error": error,
    }


def run_tasks(
    tasks: list[Task], answer: Answerer, model: str, run: str, cache_tag: str | None, workers: int
) -> bool:
    """Answer every task and write `run_path(model, run)`; True if the spending cap stopped it.

    A stopped run writes nothing: re-run it (finished calls come from the
    cache) so a results file is always complete. Calls that fail after the
    client's retries are written as wrong answers with their error.
    """
    done, failed, stopped = run_parallel(lambda task: answer(task, cache_tag), tasks, workers, run)
    if stopped:
        print(f"  {run}: stopped at the spending cap after {len(done)}/{len(tasks)}; nothing written, re-run to finish")
        return True
    records = {task.id: _record(task, run, probabilities, result, None) for task, (probabilities, result) in done}
    records |= {task.id: _record(task, run, None, None, error) for task, error in failed}
    path = run_path(model, run)
    write_records(path, [records[task.id] for task in tasks])
    accuracy = sum(r["correct"] for r in records.values()) / len(records)
    print(f"  {run}: {len(records)} questions, accuracy {accuracy:.1%}, {len(failed)} failed -> {path}")
    for task, error in failed[:10]:
        print(f"    {task.id}: {error}")
    return False


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
