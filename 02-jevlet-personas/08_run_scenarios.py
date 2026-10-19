"""Run Jev and the LLM baselines on the scenario questions (B1 step 5).

The same conditions as the Big Five runs (01_run_bigfive.py), so the behaviour
shift can be set next to the self-report shift:
    estimate   LOCAL ONLY, no API calls: call counts and estimated cost.
    check      One scenario, baseline and one persona in each placement, on
               every model; prints the answers (well under $0.01).
    baseline   No persona. Jev N_BASELINE_RUNS_JEV fresh repeats, each LLM
               N_BASELINE_RUNS_LLM.
    personas   High/low on one trait at a time x placement x
               N_PERSONA_VARIANTS wordings, one call each.

Needs data/scenarios/scenarios.json (07_make_scenarios.py --apply-review).
Writes results/raw/<model slug>/scenarios/<condition>.jsonl (format in
helpers/scenarios.py). Every request is cached.

Run from this folder (check, baseline and personas on the API laptop):
    python 08_run_scenarios.py estimate
    python 08_run_scenarios.py check
    python 08_run_scenarios.py baseline
    python 08_run_scenarios.py personas [--domains E A] [--placements state] [--models typesafe/jev-1.13]
"""

import argparse
from collections.abc import Sequence

from helpers.bigfive import make_client
from helpers.choice import choice_chat_messages, model_choice_answerer
from helpers.config import N_BASELINE_RUNS_JEV, N_BASELINE_RUNS_LLM, N_PERSONA_VARIANTS, PLACEMENTS
from helpers.costs import estimate_usd
from helpers.ipip import DOMAINS
from helpers.personas import BASELINE_CONDITION, Condition, markers_verified, persona_conditions
from helpers.scenarios import ScenarioTask, load_scenarios, run_condition, scenario_tasks
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params
from jev_common.openrouter import OpenRouterClient

MODELS = (JEV_MODEL, *LLM_BASELINE_MODELS)


def tasks_for(model: str, condition: Condition) -> list[ScenarioTask]:
    scenarios = load_scenarios()
    if condition.is_baseline:
        n_runs = N_BASELINE_RUNS_JEV if model == JEV_MODEL else N_BASELINE_RUNS_LLM
        return scenario_tasks(condition, scenarios, [None], range(1, n_runs + 1))
    return scenario_tasks(condition, scenarios, range(1, N_PERSONA_VARIANTS + 1), [1])


def estimate(models: Sequence[str], conditions: list[Condition]) -> None:
    print("Estimate only: no API calls are made.\n")
    total = 0.0
    print(f"{'model':28} {'calls':>7} {'chars sent':>12} {'est. USD':>9}")
    for model in models:
        calls = [task for condition in conditions for task in tasks_for(model, condition)]
        chars, usd = estimate_usd(model, calls, choice_chat_messages)
        total += usd
        print(f"{model:28} {len(calls):7d} {chars:12,d} {usd:9.2f}")
    print(f"\nTotal estimate: ${total:.2f} for {len(load_scenarios())} scenarios, baseline + "
          f"{len(conditions) - 1} persona conditions.")


def check(client: OpenRouterClient, models: list[str]) -> None:
    scenario = next(s for s in load_scenarios() if s.domain == "E")
    print(f"{scenario.id}: {scenario.situation}\n  A: {scenario.options[0]}\n  B: {scenario.options[1]}"
          f"\n  high action: {scenario.high_letter}")
    conditions = [BASELINE_CONDITION, *(Condition("E", "high", p) for p in PLACEMENTS),
                  *(Condition("E", "low", p) for p in PLACEMENTS)]
    for model in models:
        answer = model_choice_answerer(client, model)
        for condition in conditions:
            task = ScenarioTask(scenario, condition, None if condition.is_baseline else 1, 1)
            probabilities, result = answer(task)
            print(f"{model} | {condition.name} | P(high) {probabilities.get(scenario.high_letter, 0):.3f} | "
                  f"{probabilities} | ${result.cost_usd:.6f}")


def run(client: OpenRouterClient, models: list[str], conditions: list[Condition], workers: int) -> bool:
    for model in models:
        print(model)
        answer = model_choice_answerer(client, model)
        for condition in conditions:
            if run_condition(tasks_for(model, condition), answer, model, workers):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("estimate", "check", "baseline", "personas"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--domains", nargs="+", choices=tuple(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=3.0, help="stop after spending this many USD")
    args = parser.parse_args()

    for model in args.models:
        if model != JEV_MODEL:
            llm_eval_params(model)  # fail before any call if a provider isn't pinned
    if not markers_verified():
        raise SystemExit("prompts/personas/markers.json is not marked verified")
    personas = persona_conditions(tuple(args.domains), tuple(args.placements))
    if args.step == "estimate":
        estimate(args.models, [BASELINE_CONDITION, *personas])
        return
    with make_client("scenarios_run", args.max_cost) as client:
        if args.step == "check":
            check(client, args.models)
        elif args.step == "baseline":
            run(client, args.models, [BASELINE_CONDITION], args.workers)
        else:
            run(client, args.models, personas, args.workers)
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
