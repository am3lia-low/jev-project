"""Run Jev and the LLM baselines on the IPIP-NEO-120 items (B1 steps 1-3).

Three steps, run in this order:
    check      One item, baseline and one persona in each placement, on every
               model; prints the raw answers. Confirms Jev's `score` answer
               format (position 0-4, see helpers/ipip.py) before spending more.
    baseline   No persona. Jev N_BASELINE_RUNS_JEV fresh repeats, each LLM
               N_BASELINE_RUNS_LLM (temperature 0, so repeats are identical).
    personas   High/low on one trait at a time x placement (state or
               instructions) x N_PERSONA_VARIANTS wordings, one call each.

Writes results/raw/<model slug>/bigfive/<condition>.jsonl (format in
helpers/bigfive.py). Every request is cached, so re-running a step costs
nothing for what is already done.

--items paraphrased runs the reworded items from 05_paraphrase_ipip.py instead
(B1 step 4, contamination check) and writes to bigfive_paraphrased/.

Run from this folder on the API laptop:
    python 01_run_bigfive.py check
    python 01_run_bigfive.py baseline
    python 01_run_bigfive.py personas [--domains E A] [--placements state] [--models typesafe/jev-1.13]
    python 01_run_bigfive.py baseline --items paraphrased
    python 01_run_bigfive.py personas --items paraphrased
"""

import argparse
import json

from helpers.bigfive import Task, make_client, model_answerer, run_condition, tasks_for
from helpers.config import (
    IPIP_PARAPHRASED_PATH,
    ITEM_SETS,
    N_BASELINE_RUNS_JEV,
    N_BASELINE_RUNS_LLM,
    N_PERSONA_VARIANTS,
    PLACEMENTS,
)
from helpers.ipip import DOMAINS, IPIP_PATH, load_items
from helpers.personas import BASELINE_CONDITION, Condition, markers_verified, persona_conditions
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params
from jev_common.openrouter import OpenRouterClient

MODELS = (JEV_MODEL, *LLM_BASELINE_MODELS)


def check(client: OpenRouterClient, models: list[str]) -> None:
    item = load_items()[1]  # ipip-002 "Make friends easily", E plus-keyed
    conditions = [BASELINE_CONDITION, *(Condition("E", "high", p) for p in PLACEMENTS)]
    for model in models:
        answer = model_answerer(client, model)
        for condition in conditions:
            task = Task(item, condition, None if condition.is_baseline else 1, 1)
            value, probabilities, raw, result = answer(task)
            print(f"{model} | {condition.name} | answer {value:.3f} | ${result.cost_usd:.6f}")
            print(f"  raw: {json.dumps(raw if raw is not None else probabilities)}")


def paraphrases_verified() -> bool:
    if not IPIP_PARAPHRASED_PATH.exists():
        return False
    return bool(json.loads(IPIP_PARAPHRASED_PATH.read_text(encoding="utf-8")).get("verified"))


def run(client: OpenRouterClient, models: list[str], conditions: list[Condition], workers: int,
        item_set: str = "original") -> bool:
    items = load_items(IPIP_PARAPHRASED_PATH if item_set == "paraphrased" else IPIP_PATH)
    for model in models:
        print(model)
        answer = model_answerer(client, model)
        for condition in conditions:
            if condition.is_baseline:
                n_runs = N_BASELINE_RUNS_JEV if model == JEV_MODEL else N_BASELINE_RUNS_LLM
                tasks = tasks_for(condition, items, [None], range(1, n_runs + 1))
            else:
                tasks = tasks_for(condition, items, range(1, N_PERSONA_VARIANTS + 1), [1])
            if run_condition(tasks, answer, model, workers, ITEM_SETS[item_set]):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("check", "baseline", "personas"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--domains", nargs="+", choices=tuple(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--items", choices=tuple(ITEM_SETS), default="original",
                        help="the original IPIP wording, or the reviewed paraphrases (B1 step 4)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=3.0, help="stop after spending this many USD")
    args = parser.parse_args()

    for model in args.models:
        if model != JEV_MODEL:
            llm_eval_params(model)  # fail before any call if a provider isn't pinned
    if args.step == "personas" and not markers_verified():
        raise SystemExit("prompts/personas/markers.json is not marked verified; check the adjectives "
                         "against Goldberg (1992) and set \"verified\": true first.")
    if args.items == "paraphrased":
        if args.step == "check":
            raise SystemExit("check uses the original item; run baseline or personas with --items paraphrased")
        if not paraphrases_verified():
            raise SystemExit(f"{IPIP_PARAPHRASED_PATH} is missing or not verified; run 05_paraphrase_ipip.py "
                             "and --apply-review first")

    with make_client("bigfive", args.max_cost) as client:
        if args.step == "check":
            check(client, args.models)
        elif args.step == "baseline":
            run(client, args.models, [BASELINE_CONDITION], args.workers, args.items)
        else:
            run(client, args.models, persona_conditions(tuple(args.domains), tuple(args.placements)), args.workers,
                args.items)
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
