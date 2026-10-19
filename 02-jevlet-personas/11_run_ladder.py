"""Run Jev and the LLM baselines on the difficulty ladder (B2).

Each benchmark is asked with no persona (actual ability) and with each
"pretend to be [level]" persona in the state and in the instructions
(LADDER_PERSONAS; one level to start with).
    estimate   LOCAL ONLY, no API calls: call counts and estimated cost.
    check      One question per benchmark, every condition, every model;
               prints the answers (a few cents at most).
    run        Every question, benchmark, condition and model.

Needs data/ladder/ (10_build_ladder.py). Writes
results/raw/<model slug>/ladder/<benchmark>/<condition>.jsonl (format in
helpers/ladder.py). Every request is cached.

Run from this folder (check and run on the API laptop):
    python 11_run_ladder.py estimate
    python 11_run_ladder.py check
    python 11_run_ladder.py run [--benchmarks arce gpqa] [--models typesafe/jev-1.13] [--max-cost 5]
"""

import argparse
from collections.abc import Sequence

from helpers.bigfive import make_client
from helpers.choice import choice_chat_messages, model_choice_answerer
from helpers.config import LADDER_BENCHMARKS, LADDER_NAMES, PLACEMENTS
from helpers.costs import estimate_usd
from helpers.ladder import conditions, load_ladder, run_tasks, tasks_for
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params

MODELS = (JEV_MODEL, *LLM_BASELINE_MODELS)


def estimate(models: Sequence[str], benchmarks: Sequence[str], placements: Sequence[str]) -> None:
    print("Estimate only: no API calls are made.\n")
    total = 0.0
    print(f"{'model':28} {'benchmark':22} {'calls':>6} {'chars sent':>12} {'est. USD':>9}")
    for model in models:
        for benchmark in benchmarks:
            items = load_ladder(benchmark)
            calls = [t for c in conditions(placements=placements) for t in tasks_for(items, c)]
            chars, usd = estimate_usd(model, calls, choice_chat_messages)
            total += usd
            print(f"{model:28} {LADDER_NAMES[benchmark]:22} {len(calls):6d} {chars:12,d} {usd:9.2f}")
    print(f"\nTotal estimate: ${total:.2f}. Conditions: {[c[0] for c in conditions(placements=placements)]}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("estimate", "check", "run"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--benchmarks", nargs="+", choices=LADDER_BENCHMARKS, default=list(LADDER_BENCHMARKS))
    parser.add_argument("--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=5.0, help="stop after spending this many USD")
    args = parser.parse_args()

    for model in args.models:
        if model != JEV_MODEL:
            llm_eval_params(model)  # fail before any call if a provider isn't pinned
    if args.step == "estimate":
        estimate(args.models, args.benchmarks, args.placements)
        return
    with make_client("ladder", args.max_cost) as client:
        for model in args.models:
            print(model)
            answer = model_choice_answerer(client, model)
            for benchmark in args.benchmarks:
                items = load_ladder(benchmark)
                for condition in conditions(placements=args.placements):
                    tasks = tasks_for(items, condition)
                    if args.step == "check":
                        task = tasks[0]
                        probabilities, result = answer(task)
                        top = max(probabilities, key=probabilities.get)
                        print(f"  {benchmark} {condition[0]}: picked {top} (answer {task.item['answer']}), "
                              f"{probabilities} ${result.cost_usd:.6f}")
                    elif run_tasks(tasks, answer, model, args.workers):
                        print(f"spent this run: ${client.spent_usd:.4f}")
                        return
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
