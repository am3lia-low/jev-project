"""Cost estimates before a run, scaled from what each model actually cost in B1.

No API calls: the estimate fits cost = characters sent x rate + a fixed part
per call to the model's real B1 records, then applies it to the characters a
planned run would send. Used by the `estimate` steps of 03, 08 and 11.
"""

import json
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from jev_common.models import JEV_MODEL

from .bigfive import Request, chat_messages, model_slug, read_records, tasks_for
from .config import N_BASELINE_RUNS_JEV, N_BASELINE_RUNS_LLM, N_PERSONA_VARIANTS, RAW_RESULTS_DIR
from .ipip import load_items
from .personas import BASELINE_CONDITION, persona_conditions

# The fixed part per call: the price of the one output token an LLM baseline
# generates (OpenRouter list prices, 2026-10-18). Jev's output is free.
OUTPUT_TOKEN_USD = {JEV_MODEL: 0.0, "qwen/qwen3.8-27b": 3.00e-6, "google/gemma-4-26b-a4b-it": 0.32e-6}

Messages = Callable[[Any], list[dict[str, str]]]


def request_chars(model: str, call: Request, messages: Messages = chat_messages) -> int:
    """Size of what is sent: the Jev request JSON, or the LLM's chat messages."""
    if model == JEV_MODEL:
        return len(json.dumps(call.request, ensure_ascii=False))
    return sum(len(m["content"]) for m in messages(call))


@cache
def b1_cost_per_char(model: str) -> tuple[float, float]:
    """(USD per character sent, USD per call) from the model's actual B1 costs."""
    records = [r for path in (RAW_RESULTS_DIR / model_slug(model) / "bigfive").glob("*.jsonl")
               for r in read_records(path)]
    if not records:
        raise SystemExit(f"No B1 results for {model}; run 01_run_bigfive.py first (the estimate is scaled from them)")
    if model not in OUTPUT_TOKEN_USD:
        raise SystemExit(f"No output-token price for {model}; add it to OUTPUT_TOKEN_USD in helpers/costs.py")
    cost = sum(r["cost_usd"] or 0.0 for r in records)
    items = load_items()
    n_runs = N_BASELINE_RUNS_JEV if model == JEV_MODEL else N_BASELINE_RUNS_LLM
    tasks = tasks_for(BASELINE_CONDITION, items, [None], range(1, n_runs + 1))
    for condition in persona_conditions():
        tasks += tasks_for(condition, items, range(1, N_PERSONA_VARIANTS + 1), [1])
    chars = sum(request_chars(model, task) for task in tasks)
    per_call = OUTPUT_TOKEN_USD[model]
    return max(cost - per_call * len(records), 0.0) / chars, per_call


def estimate_usd(model: str, calls: Sequence[Request], messages: Messages = chat_messages) -> tuple[int, float]:
    """(characters sent, estimated USD) for `calls` on `model`."""
    rate, per_call = b1_cost_per_char(model)
    chars = sum(request_chars(model, call, messages) for call in calls)
    return chars, chars * rate + per_call * len(calls)
