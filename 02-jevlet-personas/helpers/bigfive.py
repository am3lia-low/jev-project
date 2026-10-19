"""Running a model on the IPIP-NEO-120 items, and the result records (B1 steps 1-3).

A `Task` is one IPIP item under one condition, wording (`variant`) and run.
Jev gets it as a `score` question over the 5 scale labels; the persona goes
into the state as `{"persona": text}` or in front of the question
instructions. The LLM baselines get the same request rendered as a chat
prompt the way decision-index renders a choice question (as in Part A), with
the scale labels keyed "1".."5"; their answer is the probability-weighted mean
of the keys, read from the first token's top logprobs, so it is fractional
like Jev's.

Records: one JSON object per item per line in
`results/raw/<model slug>/bigfive/<condition>.jsonl` (`bigfive_paraphrased/`
for the paraphrased items, B1 step 4), sorted by (variant, run, item):

    item_id, domain, facet, keyed, condition,
    persona_domain, pole, placement   null for the baseline
    variant        persona wording 1..N (null for the baseline)
    run            1..N for repeated identical requests (Jev baseline), else 1
    answer         1-5, fractional; null if the call failed
    probabilities  LLMs: {"1".."5": p}, renormalised; Jev: as returned, if any
    raw            Jev's score answer as returned (null for LLMs)
    latency_s, cost_usd, cached, error
"""

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jev_common.cache import DiskCache
from jev_common.models import JEV_MODEL, llm_eval_params
from jev_common.openrouter import CallResult, OpenRouterClient, letter_probabilities
from jev_common.parallel import run_parallel

from .config import CACHE_DIR, LOG_DIR, PROMPTS_DIR, RAW_RESULTS_DIR
from .ipip import SCALE_MIN, Item, answer_from_jev, jev_score_question, scale_labels
from .personas import Condition, persona_text

QUESTION_KEY = "q1"
SCALE_KEYS = tuple(str(SCALE_MIN + i) for i in range(len(scale_labels())))  # "1".."5"



class Request(Protocol):
    """Anything that can be sent as one request: a B1 `Task` or a B4 `decay.Call`."""

    @property
    def request(self) -> dict[str, Any]: ...  # {state, questions}, with the question under QUESTION_KEY

    @property
    def cache_tag(self) -> str | None: ...


# (answer 1-5, probabilities or None, raw Jev answer or None, call result) for one request
Answer = tuple[float, dict[str, float] | None, dict[str, Any] | None, CallResult]
Answerer = Callable[[Request], Answer]


@dataclass(frozen=True)
class Task:
    item: Item
    condition: Condition
    variant: int | None
    run: int

    @property
    def persona(self) -> str | None:
        if self.condition.is_baseline:
            return None
        return persona_text(self.condition.domain, self.condition.pole, self.variant)

    @property
    def cache_tag(self) -> str | None:
        """Makes run k > 1 of an identical request a fresh call."""
        return None if self.run == 1 else f"run-{self.run}"

    @property
    def request(self) -> dict[str, Any]:
        """`{state, questions}` for one Jev request; pass to `OpenRouterClient.decide`."""
        question = jev_score_question(self.item)
        state: dict[str, Any] = {}
        if self.condition.placement == "state":
            state = {"persona": self.persona}
        elif self.condition.placement == "instructions":
            question = {**question, "instructions": f"{self.persona} {question['instructions']}"}
        return {"state": state, "questions": {QUESTION_KEY: question}}


def tasks_for(condition: Condition, items: Sequence[Item], variants: Iterable[int | None],
              runs: Iterable[int]) -> list[Task]:
    return [Task(item, condition, variant, run) for variant in variants for run in runs for item in items]


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------
def jev_answerer(client: OpenRouterClient, model: str) -> Answerer:
    def answer(task: Request) -> Answer:
        request = task.request
        result = client.decide(model, request["state"], request["questions"], cache_tag=task.cache_tag)
        raw = result.response["answers"][QUESTION_KEY]
        return answer_from_jev(raw), raw.get("probabilities"), raw, result

    return answer


def render_prompt(task: Request) -> str:
    """The user message, as decision-index renders a question (Part A's `render_prompt`)."""
    request = task.request
    question = request["questions"][QUESTION_KEY]
    state = request["state"]
    lines = ["State:", json.dumps(state, ensure_ascii=False) if state else "(empty)", "",
             "Question:", question["instructions"], "", "Options:"]
    lines += [f"{key}: {label}" for key, label in zip(SCALE_KEYS, question["criteria"], strict=True)]
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def chat_messages(task: Request) -> list[dict[str, str]]:
    system = (PROMPTS_DIR / "llm_system.txt").read_text(encoding="utf-8").strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": render_prompt(task)}]


def llm_answerer(client: OpenRouterClient, model: str, params: dict[str, Any]) -> Answerer:
    def answer(task: Request) -> Answer:
        result = client.chat(model, chat_messages(task), cache_tag=task.cache_tag, **params)
        probabilities = letter_probabilities(result.response, SCALE_KEYS)
        expected = sum(int(key) * p for key, p in probabilities.items())
        return expected, probabilities, None, result

    return answer


def model_answerer(client: OpenRouterClient, model: str) -> Answerer:
    """Jev's answerer for Jev, else the LLM answerer with that model's pinned settings."""
    if model == JEV_MODEL:
        return jev_answerer(client, model)
    return llm_answerer(client, model, llm_eval_params(model))


def make_client(namespace: str, max_cost_usd: float) -> OpenRouterClient:
    return OpenRouterClient(
        cache=DiskCache(namespace, CACHE_DIR), log_dir=LOG_DIR, max_cost_usd=max_cost_usd
    )


# ---------------------------------------------------------------------------
# Running and records
# ---------------------------------------------------------------------------
def model_slug(model: str) -> str:
    """Folder name for a model ID: "typesafe/jev-1.13" -> "typesafe__jev-1.13"."""
    return model.replace("/", "__")


def run_path(model: str, condition: Condition, folder: str = "bigfive") -> Path:
    return RAW_RESULTS_DIR / model_slug(model) / folder / f"{condition.name}.jsonl"


def _record(task: Task, answer: Answer | None, error: str | None) -> dict[str, Any]:
    value, probabilities, raw, result = answer if answer is not None else (None, None, None, None)
    return {
        "item_id": task.item.id,
        "domain": task.item.domain,
        "facet": task.item.facet,
        "keyed": task.item.keyed,
        "condition": task.condition.name,
        "persona_domain": task.condition.domain,
        "pole": task.condition.pole,
        "placement": task.condition.placement,
        "variant": task.variant,
        "run": task.run,
        "answer": None if value is None else round(value, 6),
        "probabilities": probabilities,
        "raw": raw,
        "latency_s": None if result is None else round(result.latency_s, 4),
        "cost_usd": None if result is None else result.cost_usd,
        "cached": None if result is None else result.cached,
        "error": error,
    }


def run_condition(tasks: list[Task], answer: Answerer, model: str, workers: int, folder: str = "bigfive") -> bool:
    """Answer every task of one condition and write its results file; True if the spending cap stopped it.

    A stopped run writes nothing: re-run it (finished calls come from the
    cache) so a results file is always complete. Calls that fail after the
    client's retries are written with `answer` null and their error.
    """
    condition = tasks[0].condition
    done, failed, stopped = run_parallel(answer, tasks, workers, condition.name)
    if stopped:
        print(f"  {condition.name}: stopped at the spending cap after {len(done)}/{len(tasks)}; "
              "nothing written, re-run to finish")
        return True
    records = [_record(task, result, None) for task, result in done]
    records += [_record(task, None, error) for task, error in failed]
    records.sort(key=lambda r: (r["variant"] or 0, r["run"], r["item_id"]))
    path = run_path(model, condition, folder)
    write_records(path, records)
    answered = [r["answer"] for r in records if r["answer"] is not None]
    mean = sum(answered) / len(answered) if answered else float("nan")
    print(f"  {condition.name}: {len(records)} answers, mean raw answer {mean:.2f}, {len(failed)} failed -> {path}")
    for task, error in failed[:10]:
        print(f"    {task.item.id} variant {task.variant} run {task.run}: {error}")
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
