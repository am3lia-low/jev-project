"""Run Jev on the test levels and the quiz (A9).

Four steps, run in this order:
    sanity   8a: Level 0; the paper datasets' accuracy must agree with the
             paper (its number inside our bootstrap 95% CI). Debug the client
             or templates before going on if it doesn't.
    repeat   8b: N_REPEATABILITY Level 0 questions, 3 runs each; reports how
             often all 3 runs agree. At or above REPEAT_AGREEMENT_SINGLE_RUN,
             one run per level is enough; else use `levels --runs 3`.
    levels   8c: Levels 0-3 (Level 0 comes from the sanity run's cache).
    quiz     8d: the memorisation quiz (data/quiz/quiz.jsonl).

Writes results/raw/typesafe__jev-1.13/<run>.jsonl (format in
helpers/evaluation.py). Every request is cached, so re-running a step costs
nothing for what is already done.

Run from this folder on the API laptop:
    python 08_run_jev.py sanity
    python 08_run_jev.py repeat
    python 08_run_jev.py levels [--levels 1 2 3] [--runs 3]
    python 08_run_jev.py quiz
"""

import argparse
from collections import defaultdict

from helpers import config
from helpers.evaluation import Task, level_tasks, quiz_tasks, read_records, repeat_tasks, run_path, run_tasks
from helpers.generation import make_client
from jev_common.models import JEV_MODEL
from jev_common.openrouter import OpenRouterClient, choice_probabilities
from jev_common.stats import bootstrap_ci


def jev_answerer(client: OpenRouterClient):
    def answer(task: Task, cache_tag: str | None):
        result = client.decide(JEV_MODEL, task.request["state"], task.request["questions"], cache_tag=cache_tag)
        question_key = next(iter(task.request["questions"]))
        return choice_probabilities(result.response, question_key), result

    return answer


def sanity(client: OpenRouterClient, workers: int) -> None:
    if run_tasks(level_tasks(0), jev_answerer(client), JEV_MODEL, "level0", None, workers):
        return
    by_dataset = defaultdict(list)
    for record in read_records(run_path(JEV_MODEL, "level0")):
        by_dataset[record["dataset"]].append(record["correct"])
    print("Level 0 vs the paper (A9 sanity check):")
    for key, correct in by_dataset.items():
        ci = bootstrap_ci(correct)
        paper = config.PAPER_JEV_ACCURACY.get(key)
        verdict = "no paper number" if paper is None else ("OK" if ci.low <= paper <= ci.high else "CHECK")
        paper_text = "" if paper is None else f", paper {paper:.1%}"
        print(f"  {key}: {ci.estimate:.1%} [{ci.low:.1%}, {ci.high:.1%}]{paper_text} -> {verdict}")


def repeat(client: OpenRouterClient, workers: int) -> None:
    tasks = repeat_tasks()
    runs = [f"repeat_{k}" for k in range(1, config.N_REPEATABILITY_RUNS + 1)]
    for run in runs:
        if run_tasks(tasks, jev_answerer(client), JEV_MODEL, run, run.replace("_", "-"), workers):
            return
    preds = defaultdict(list)
    for run in runs:
        for record in read_records(run_path(JEV_MODEL, run)):
            preds[record["id"]].append(record["pred"])
    agreement = sum(len(set(p)) == 1 for p in preds.values()) / len(preds)
    advice = "single runs are enough" if agreement >= config.REPEAT_AGREEMENT_SINGLE_RUN else "use `levels --runs 3` (majority vote)"
    print(f"All {len(runs)} runs agree on {agreement:.1%} of {len(preds)} questions: {advice}")


def levels(client: OpenRouterClient, workers: int, chosen: list[int], n_runs: int) -> None:
    for level in chosen:
        try:
            tasks = level_tasks(level)
        except FileNotFoundError as error:
            print(f"  level{level}: skipped, {error}")
            continue
        for k in range(1, n_runs + 1):
            run = f"level{level}" if k == 1 else f"level{level}_run{k}"
            if run_tasks(tasks, jev_answerer(client), JEV_MODEL, run, None if k == 1 else f"run-{k}", workers):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("sanity", "repeat", "levels", "quiz"))
    parser.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1, 2, 3])
    parser.add_argument("--runs", type=int, default=1, help="runs per level (levels step only)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=0.50, help="stop after spending this many USD")
    args = parser.parse_args()

    with make_client("jev", args.max_cost) as client:
        if args.step == "sanity":
            sanity(client, args.workers)
        elif args.step == "repeat":
            repeat(client, args.workers)
        elif args.step == "levels":
            levels(client, args.workers, args.levels, args.runs)
        else:
            run_tasks(quiz_tasks(), jev_answerer(client), JEV_MODEL, "quiz", None, args.workers)
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
