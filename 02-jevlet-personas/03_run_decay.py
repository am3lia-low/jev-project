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

Arms (helpers/config.py, DECAY_ARM_SETTINGS): own and quiz are the main run
and the default. The follow-ups (2026-10-19) are opt-in with --arms:
    long       quiz Q&A up to 256 turns: does the state persona fade at last?
    appdata    unrelated app records in the state, no Q&A: does any non-empty
               state weaken a persona in the instructions?
Run `estimate` with the same arguments first; the long arm is the costly one.

Design and file formats: helpers/decay.py. Every request is cached, so
re-running a step costs nothing for what is already done.

Run from this folder (check and run on the API laptop):
    python 03_run_decay.py estimate
    python 03_run_decay.py check
    python 03_run_decay.py run [--arms own quiz] [--models typesafe/jev-1.13] [--max-cost 5]
    python 03_run_decay.py estimate --arms long appdata
    python 03_run_decay.py run --arms appdata
    python 03_run_decay.py run --arms long --placements state
"""

import argparse
import json
from collections.abc import Sequence

from helpers.bigfive import make_client, model_answerer
from helpers.config import (
    DECAY_ALL_ARMS,
    DECAY_ARM_SETTINGS,
    DECAY_ARMS,
    DECAY_DOMAINS,
    DECAY_VARIANTS,
    PLACEMENTS,
    POLES,
)
from helpers.costs import estimate_usd
from helpers.decay import (
    Call,
    Conversation,
    answer_label,
    conversations,
    fixed_turns,
    history_items,
    item_question,
    probe_calls,
    run_arm,
)
from helpers.ipip import load_items
from helpers.personas import Condition, markers_verified
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params

MODELS = (JEV_MODEL, *LLM_BASELINE_MODELS)


def all_calls(arm: str, domains: Sequence[str], placements: Sequence[str]) -> list[Call]:
    """Every call one model makes in one arm. Own-arm turn labels are a placeholder (sizes only)."""
    convs = conversations(arm, domains, placements, DECAY_VARIANTS, POLES)
    if arm != "own":
        return [call for conv in convs for call in probe_calls(conv, fixed_turns(arm), domains)]
    placeholder = [{"question": item_question(item), "answer": "Moderately Accurate"} for item in history_items()]
    calls = []
    for conv in convs:
        calls += [Call(conv, item, tuple(placeholder[:t])) for t, item in enumerate(history_items())]
        calls += probe_calls(conv, placeholder, domains)
    return calls



def estimate(models: Sequence[str], arms: Sequence[str], domains: Sequence[str], placements: Sequence[str]) -> None:
    print("Estimate only: no API calls are made.\n")
    for arm in arms:
        settings = DECAY_ARM_SETTINGS[arm]
        if arm == "own":
            turns = [{"question": item_question(i), "answer": "Moderately Accurate"} for i in history_items()]
        else:
            turns = list(fixed_turns(arm))
        print(f"{arm}: {settings['turns']} turns under \"{settings['key']}\", ~{len(json.dumps(turns))} characters "
              f"at full length (about 4 characters per token); checkpoints {settings['checkpoints']}")
    print("Requests already in the cache (checkpoint 0, the long arm's first 32 turns) cost nothing; "
          "the estimate counts them anyway.\n")
    total = 0.0
    print(f"{'model':28} {'arm':8} {'calls':>7} {'chars sent':>13} {'est. USD':>9}")
    for model in models:
        for arm in arms:
            calls = all_calls(arm, domains, placements)
            chars, usd = estimate_usd(model, calls)
            total += usd
            print(f"{model:28} {arm:8} {len(calls):7d} {chars:13,d} {usd:9.2f}")
    print(f"\nTotal estimate: ${total:.2f}. Traits {list(domains)}, placements {list(placements)}, "
          f"wordings {list(DECAY_VARIANTS)}.")


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
    parser.add_argument("--arms", nargs="+", choices=DECAY_ALL_ARMS, default=list(DECAY_ARMS))
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
