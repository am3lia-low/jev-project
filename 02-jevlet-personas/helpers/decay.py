"""Persona decay over a growing history (B4).

No model keeps anything between requests, so our script builds each
conversation's history and sends it with every request. A request's state is

    {"persona": <persona text>, "history": [<turn>, ...]}

with the persona first (left out for the baseline and for the persona-in-
instructions control), then the history, then the probe question. At
checkpoint 0 there is no history key, so that request is exactly B1's.

A `Conversation` is one arm x condition x wording. Its history has
DECAY_TURNS turns:
    own    the model's own answers to IPIP items from DECAY_HISTORY_DOMAINS,
           built turn by turn (turn t is answered with turns 1..t-1 in the
           state, and our script writes the most likely answer label in)
    quiz   fixed ARC-Easy questions with their correct answers, the same for
           every condition and model; no calls are needed to build it
A turn is {"question": ..., "answer": ...}.

At each checkpoint k the probes are the persona trait's 24 IPIP items (all
DECAY_DOMAINS items for the baseline), one request each, sent with the first
k turns. Probe answers are never added to the history. The re-reminder
repeats the persona as a {"persona": ...} entry right after turn
DECAY_REMINDER_TURN (persona in the state only).

Results, per model slug and arm, under results/raw/<slug>/decay/<arm>/:
    <condition>.jsonl            one probe answer per line, sorted by
                                 (variant, reminder, checkpoint, item):
                                 the B1 record fields plus arm, checkpoint,
                                 reminder, history_chars
    histories/<conversation>.jsonl   own arm: one turn per line (turn,
                                 item_id, question, answer_label, answer,
                                 probabilities, cost_usd, cached)
"""

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from jev_common.parallel import run_parallel

from .bigfive import QUESTION_KEY, Answer, Answerer, model_slug, write_records
from .config import (
    DECAY_CHECKPOINTS,
    DECAY_HISTORY_DOMAINS,
    DECAY_REMINDER_CHECKPOINTS,
    DECAY_REMINDER_TURN,
    DECAY_SEED,
    DECAY_TURNS,
    QUIZ_SOURCE_PATH,
    RAW_RESULTS_DIR,
)
from .ipip import Item, jev_score_question, load_items, scale_labels
from .personas import BASELINE_CONDITION, Condition, persona_text

Turn = dict[str, str]


# ---------------------------------------------------------------------------
# Histories
# ---------------------------------------------------------------------------
def item_question(item: Item) -> str:
    return f'How accurately does this describe you: "{item.text}"?'


@cache
def history_items() -> tuple[Item, ...]:
    """The own arm's DECAY_TURNS history items: equal numbers from each history domain, seeded order."""
    rng = random.Random(f"{DECAY_SEED}:own")
    per_domain = DECAY_TURNS // len(DECAY_HISTORY_DOMAINS)
    chosen = []
    for domain in DECAY_HISTORY_DOMAINS:
        pool = [item for item in load_items() if item.domain == domain]
        chosen += rng.sample(pool, per_domain)
    rng.shuffle(chosen)
    return tuple(chosen)


@cache
def quiz_history(path: Path = QUIZ_SOURCE_PATH) -> tuple[Turn, ...]:
    """The quiz arm's DECAY_TURNS turns: seeded ARC-Easy questions with their correct answers."""
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = random.Random(f"{DECAY_SEED}:quiz").sample(rows, DECAY_TURNS)
    return tuple({"question": r["question"], "answer": r["options"]["ABCDEFGH".index(r["answer"])]} for r in rows)


def answer_label(value: float, probabilities: dict[str, float] | None, is_jev: bool) -> str:
    """The scale label a model would most likely say: the top probability, else the rounded answer.

    Jev's probabilities are keyed by position "0".."4"; the LLMs' by "1".."5".
    """
    labels = scale_labels()
    if probabilities:
        key = max(probabilities, key=probabilities.get)
        return labels[int(key) - (0 if is_jev else 1)]
    return labels[round(value) - 1]


# ---------------------------------------------------------------------------
# Conversations and requests
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Conversation:
    arm: str  # "own" or "quiz"
    condition: Condition
    variant: int | None  # persona wording; None for the baseline

    @property
    def name(self) -> str:
        return self.condition.name if self.variant is None else f"{self.condition.name}-v{self.variant}"

    @property
    def persona(self) -> str | None:
        if self.condition.is_baseline:
            return None
        return persona_text(self.condition.domain, self.condition.pole, self.variant)

    def probe_items(self, baseline_domains: Sequence[str]) -> list[Item]:
        domains = baseline_domains if self.condition.is_baseline else (self.condition.domain,)
        return [item for item in load_items() if item.domain in domains]

    def history_at(self, turns: Sequence[Turn], checkpoint: int, reminder: bool) -> tuple[Turn, ...]:
        """The first `checkpoint` turns, with the persona re-inserted after turn DECAY_REMINDER_TURN if asked."""
        history = list(turns[:checkpoint])
        if reminder:
            history.insert(DECAY_REMINDER_TURN, {"persona": self.persona})
        return tuple(history)


@dataclass(frozen=True, eq=False)
class Call:
    """One request: a history turn being answered (own arm) or a probe."""

    conversation: Conversation
    item: Item
    history: tuple[Turn, ...]
    checkpoint: int | None = None  # probes only
    reminder: bool = False

    @property
    def cache_tag(self) -> str | None:
        return None

    @property
    def request(self) -> dict[str, Any]:
        question = jev_score_question(self.item)
        state: dict[str, Any] = {}
        placement = self.conversation.condition.placement
        if placement == "state":
            state["persona"] = self.conversation.persona
        elif placement == "instructions":
            question = {**question, "instructions": f"{self.conversation.persona} {question['instructions']}"}
        if self.history:
            state["history"] = list(self.history)
        return {"state": state, "questions": {QUESTION_KEY: question}}


def conversations(arm: str, domains: Sequence[str], placements: Sequence[str], variants: Sequence[int],
                  poles: Sequence[str]) -> list[Conversation]:
    persona = [Conversation(arm, Condition(d, pole, p), v) for d in domains for pole in poles
               for p in placements for v in variants]
    return [Conversation(arm, BASELINE_CONDITION, None), *persona]


def probe_calls(conversation: Conversation, turns: Sequence[Turn], baseline_domains: Sequence[str]) -> list[Call]:
    items = conversation.probe_items(baseline_domains)
    reminders = [False]
    if conversation.condition.placement == "state":
        reminders.append(True)
    calls = []
    for reminder in reminders:
        for checkpoint in DECAY_REMINDER_CHECKPOINTS if reminder else DECAY_CHECKPOINTS:
            history = conversation.history_at(turns, checkpoint, reminder)
            calls += [Call(conversation, item, history, checkpoint, reminder) for item in items]
    return calls


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def build_own_history(conversation: Conversation, answer: Answerer, is_jev: bool) -> list[dict[str, Any]]:
    """Answer the history items one after another, each with the turns before it in the state."""
    turns: list[Turn] = []
    records = []
    for number, item in enumerate(history_items(), start=1):
        value, probabilities, _raw, result = answer(Call(conversation, item, tuple(turns)))
        label = answer_label(value, probabilities, is_jev)
        turns.append({"question": item_question(item), "answer": label})
        records.append({"turn": number, "item_id": item.id, "question": turns[-1]["question"],
                        "answer_label": label, "answer": round(value, 6), "probabilities": probabilities,
                        "cost_usd": result.cost_usd, "cached": result.cached})
    return records


def turns_from_records(records: Sequence[dict[str, Any]]) -> list[Turn]:
    return [{"question": r["question"], "answer": r["answer_label"]} for r in records]


def arm_dir(model: str, arm: str) -> Path:
    return RAW_RESULTS_DIR / model_slug(model) / "decay" / arm


def _probe_record(call: Call, answer: Answer | None, error: str | None) -> dict[str, Any]:
    value, probabilities, raw, result = answer if answer is not None else (None, None, None, None)
    condition = call.conversation.condition
    return {
        "arm": call.conversation.arm,
        "item_id": call.item.id, "domain": call.item.domain, "facet": call.item.facet, "keyed": call.item.keyed,
        "condition": condition.name, "persona_domain": condition.domain, "pole": condition.pole,
        "placement": condition.placement, "variant": call.conversation.variant,
        "checkpoint": call.checkpoint, "reminder": call.reminder,
        "history_chars": len(json.dumps(list(call.history), ensure_ascii=False)) if call.history else 0,
        "answer": None if value is None else round(value, 6),
        "probabilities": probabilities, "raw": raw,
        "cost_usd": None if result is None else result.cost_usd,
        "cached": None if result is None else result.cached,
        "error": error,
    }


def run_arm(arm: str, convs: list[Conversation], answer: Answerer, model: str, is_jev: bool,
            baseline_domains: Sequence[str], workers: int) -> bool:
    """Build the histories (own arm), then send every probe; True if the spending cap stopped it.

    A stopped run writes no probe results: re-run it (finished calls come from
    the cache). A conversation whose history could not be built is skipped and
    reported; its probes are not sent.
    """
    out = arm_dir(model, arm)
    turns_by_conv: dict[Conversation, list[Turn]] = {}
    if arm == "own":
        built, failed, stopped = run_parallel(lambda c: build_own_history(c, answer, is_jev), convs, workers,
                                              f"{arm} histories")
        if stopped:
            print(f"  {arm}: stopped at the spending cap while building histories; re-run to finish")
            return True
        for conversation, records in built:
            write_records(out / "histories" / f"{conversation.name}.jsonl", records)
            turns_by_conv[conversation] = turns_from_records(records)
        for conversation, error in failed:
            print(f"  {arm} {conversation.name}: history not built, its probes skipped: {error}")
    else:
        turns_by_conv = {conversation: list(quiz_history()) for conversation in convs}

    calls = [call for conv, turns in turns_by_conv.items() for call in probe_calls(conv, turns, baseline_domains)]
    done, failed_calls, stopped = run_parallel(answer, calls, workers, f"{arm} probes")
    if stopped:
        print(f"  {arm}: stopped at the spending cap after {len(done)}/{len(calls)} probes; "
              "nothing written, re-run to finish")
        return True
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call, result in done:
        by_condition[call.conversation.condition.name].append(_probe_record(call, result, None))
    for call, error in failed_calls:
        by_condition[call.conversation.condition.name].append(_probe_record(call, None, error))
    for name, records in by_condition.items():
        records.sort(key=lambda r: (r["variant"] or 0, r["reminder"], r["checkpoint"], r["item_id"]))
        write_records(out / f"{name}.jsonl", records)
    print(f"  {arm}: {len(done)} probe answers in {len(by_condition)} files, {len(failed_calls)} failed -> {out}")
    for call, error in failed_calls[:10]:
        print(f"    {call.conversation.name} checkpoint {call.checkpoint} {call.item.id}: {error}")
    return False
