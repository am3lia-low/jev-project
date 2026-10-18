"""Persona decay: does a persona fade as a history grows in front of the question? (B4)

Three steps, run in this order:
    estimate   LOCAL ONLY, no API calls. Counts the calls each model and arm
               would make and estimates the cost, scaled from what the same
               models actually cost per character in B1.
    check      A few API calls (well under $0.01): per model, the own arm's
               E-high-state wording 1 for 2 turns, then one probe; prints the
               state that was sent and the answers.
    run        The experiment: per model and arm, build the histories (own
               arm), then send every probe.

Design and file formats: helpers/decay.py. Every request is cached, so
re-running a step costs nothing for what is already done.

Run from this folder (check and run on the API laptop):
    python 03_run_decay.py estimate
    python 03_run_decay.py check
    python 03_run_decay.py run [--arms own quiz] [--models typesafe/jev-1.13] [--max-cost 5]
"""

import argparse
import json
from collections.abc import Sequence

from helpers.bigfive import Request, chat_messages, make_client, model_answerer, model_slug, read_records, tasks_for
from helpers.config import (
    DECAY_ARMS,
    DECAY_CHECKPOINTS,
    DECAY_DOMAINS,
    DECAY_TURNS,
    DECAY_VARIANTS,
    N_BASELINE_RUNS_JEV,
    N_BASELINE_RUNS_LLM,
    N_PERSONA_VARIANTS,
    PLACEMENTS,
    POLES,
    RAW_RESULTS_DIR,
)
from helpers.decay import (
    Call,
    Conversation,
    answer_label,
    conversations,
    history_items,
    item_question,
    probe_calls,
    quiz_history,
    run_arm,
)
from helpers.ipip import load_items
from helpers.personas import BASELINE_CONDITION, Condition, markers_verified, persona_conditions
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params

MODELS = (JEV_MODEL, *LLM_BASELINE_MODELS)


def all_calls(arm: str, domains: Sequence[str], placements: Sequence[str]) -> list[Call]:
    """Every call one model makes in one arm. Own-arm turn labels are a placeholder (sizes only)."""
    convs = conversations(arm, domains, placements, DECAY_VARIANTS, POLES)
    if arm == "quiz":
        return [call for conv in convs for call in probe_calls(conv, quiz_history(), domains)]
    placeholder = [{"question": item_question(item), "answer": "Moderately Accurate"} for item in history_items()]
    calls = []
    for conv in convs:
        calls += [Call(conv, item, tuple(placeholder[:t])) for t, item in enumerate(history_items())]
        calls += probe_calls(conv, placeholder, domains)
    return calls


def request_chars(model: str, call: Request) -> int:
    """Size of what is sent: the Jev request JSON, or the LLM's chat messages."""
    if model == JEV_MODEL:
        return len(json.dumps(call.request, ensure_ascii=False))
    return sum(len(m["content"]) for m in chat_messages(call))


def b1_cost_per_char(model: str) -> tuple[float, float]:
    """(USD per character sent, USD per call) from the model's actual B1 costs.

    Fits cost = chars x rate + fixed per call, with the fixed part taken as the
    output price of one token (LLMs answer with one token; Jev's output is free).
    """
    records = [r for path in (RAW_RESULTS_DIR / model_slug(model) / "bigfive").glob("*.jsonl")
               for r in read_records(path)]
    if not records:
        raise SystemExit(f"No B1 results for {model}; run 01_run_bigfive.py first (the estimate is scaled from them)")
    cost = sum(r["cost_usd"] or 0.0 for r in records)
    items = load_items()
    n_runs = N_BASELINE_RUNS_JEV if model == JEV_MODEL else N_BASELINE_RUNS_LLM
    tasks = tasks_for(BASELINE_CONDITION, items, [None], range(1, n_runs + 1))
    for condition in persona_conditions():
        tasks += tasks_for(condition, items, range(1, N_PERSONA_VARIANTS + 1), [1])
    chars = sum(request_chars(model, task) for task in tasks)
    per_call = {JEV_MODEL: 0.0, "qwen/qwen3.8-27b": 3.00e-6, "google/gemma-4-26b-a4b-it": 0.32e-6}[model]
    return max(cost - per_call * len(records), 0.0) / chars, per_call


def estimate(models: Sequence[str], arms: Sequence[str], domains: Sequence[str], placements: Sequence[str]) -> None:
    print("Estimate only: no API calls are made.\n")
    longest = Call(Conversation("own", Condition("E", "high", "state"), 1), load_items()[1],
                   tuple({"question": item_question(i), "answer": "Moderately Accurate"} for i in history_items()))
    print(f"History at {DECAY_TURNS} turns: own arm ~{len(json.dumps(list(longest.history)))} characters, "
          f"quiz arm {len(json.dumps(list(quiz_history())))} characters (about 4 characters per token)\n")
    total = 0.0
    print(f"{'model':28} {'arm':5} {'calls':>7} {'chars sent':>12} {'est. USD':>9}")
    for model in models:
        rate, per_call = b1_cost_per_char(model)
        for arm in arms:
            calls = all_calls(arm, domains, placements)
            chars = sum(request_chars(model, call) for call in calls)
            usd = chars * rate + per_call * len(calls)
            total += usd
            print(f"{model:28} {arm:5} {len(calls):7d} {chars:12,d} {usd:9.2f}")
    print(f"\nTotal estimate: ${total:.2f}. Checkpoints {DECAY_CHECKPOINTS}, traits {list(domains)}, "
          f"placements {list(placements)}, wordings {list(DECAY_VARIANTS)}.")


def check(models: Sequence[str]) -> None:
    conv = Conversation("own", Condition("E", "high", "state"), 1)
    probe = load_items()[1]  # ipip-002 "Make friends easily", E plus-keyed
    with make_client("decay", max_cost_usd=0.05) as client:
        for model in models:
            answer = model_answerer(client, model)
            turns = []
            for item in history_items()[:2]:
                value, probabilities, _raw, result = answer(Call(conv, item, tuple(turns)))
                label = answer_label(value, probabilities, model == JEV_MODEL)
                turns.append({"question": item_question(item), "answer": label})
                print(f"{model} | turn {len(turns)} {item.id} -> {value:.3f} ({label}) | ${result.cost_usd:.6f}")
            call = Call(conv, probe, tuple(turns), checkpoint=2)
            value, _probabilities, _raw, result = answer(call)
            print(f"{model} | probe {probe.id} at checkpoint 2 -> {value:.3f} | ${result.cost_usd:.6f}")
            print(f"  state sent: {json.dumps(call.request['state'], ensure_ascii=False)}")
        print(f"spent this run: ${client.spent_usd:.4f}")


def run(models: Sequence[str], arms: Sequence[str], domains: Sequence[str], placements: Sequence[str],
        workers: int, max_cost: float) -> None:
    with make_client("decay", max_cost) as client:
        for model in models:
            print(model)
            answer = model_answerer(client, model)
            for arm in arms:
                convs = conversations(arm, domains, placements, DECAY_VARIANTS, POLES)
                if run_arm(arm, convs, answer, model, model == JEV_MODEL, domains, workers):
                    print(f"spent this run: ${client.spent_usd:.4f}")
                    return
        print(f"spent this run: ${client.spent_usd:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("estimate", "check", "run"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--arms", nargs="+", choices=DECAY_ARMS, default=list(DECAY_ARMS))
    parser.add_argument("--domains", nargs="+", choices=DECAY_DOMAINS, default=list(DECAY_DOMAINS))
    parser.add_argument("--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=5.0, help="stop after spending this many USD")
    args = parser.parse_args()

    for model in args.models:
        if model != JEV_MODEL:
            llm_eval_params(model)  # fail before any call if a provider isn't pinned
    if not markers_verified():
        raise SystemExit("prompts/personas/markers.json is not marked verified")
    if args.step == "estimate":
        estimate(args.models, args.arms, args.domains, args.placements)
    elif args.step == "check":
        check(args.models)
    else:
        run(args.models, args.arms, args.domains, args.placements, args.workers, args.max_cost)


if __name__ == "__main__":
    main()
