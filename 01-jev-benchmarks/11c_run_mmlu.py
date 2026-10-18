"""Run Jev and the LLM baselines on both versions of the MMLU number-swap items (A11).

The same request format as the levels (A9): Jev gets one `choice` question
whose instructions are the bare question (`prompts/eval/mmlu.txt`, as for
ARC), options A-D; the LLMs get it rendered as a chat prompt, pinned to their
providers. One run per version (Jev's repeatability is 99.5%, A9).

Writes results/mmlu/raw/<model slug>/{original,swapped}.jsonl (record format
in helpers/evaluation.py; `answer` is the correct letter of that version).
Kept out of results/raw/ so 10_analyse.py never reads it.

Run from this folder on the API laptop, after 11b_mmlu_build.py:
    python 11c_run_mmlu.py [--models typesafe/jev-1.13 qwen/qwen3.8-27b]
"""

import argparse
import json

from helpers import config
from helpers.evaluation import Task, chat_messages, run_tasks
from helpers.generation import make_client
from helpers.labels import letter_from_position
from helpers.prompts import EVAL_QUESTION_KEY, fill
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS, llm_eval_params
from jev_common.openrouter import OpenRouterClient, choice_probabilities, letter_probabilities

ITEMS_PATH = config.MMLU_DIR / "items.jsonl"
RAW_DIR = config.MMLU_RESULTS_DIR / "raw"
VERSIONS = ("original", "swapped")


def mmlu_tasks(version: str) -> list[Task]:
    if not ITEMS_PATH.exists():
        raise SystemExit(f"{ITEMS_PATH} does not exist yet; run 11b_mmlu_build.py first")
    with ITEMS_PATH.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    tasks = []
    for record in records:
        item = record[version]
        request = {"state": {}, "questions": {EVAL_QUESTION_KEY: {
            "type": "choice",
            "instructions": fill("eval/mmlu", question=item["question"]),
            "criteria": {letter_from_position(i): text for i, text in enumerate(item["options"])},
        }}}
        tasks.append(Task(record["id"], record["subject"], item["answer"], len(item["options"]), None, request))
    return tasks


def answerer(client: OpenRouterClient, model: str):
    """As 08_run_jev.py's (Jev) and 09_run_llm.py's (LLMs) answerers."""
    if model == JEV_MODEL:
        def answer(task: Task, cache_tag: str | None):
            result = client.decide(model, task.request["state"], task.request["questions"], cache_tag=cache_tag)
            return choice_probabilities(result.response, EVAL_QUESTION_KEY), result
        return answer

    params = llm_eval_params(model)

    def answer(task: Task, cache_tag: str | None):
        result = client.chat(model, chat_messages(task), cache_tag=cache_tag, **params)
        return letter_probabilities(result.response, task.letters), result
    return answer


def main() -> None:
    models = (JEV_MODEL, *LLM_BASELINE_MODELS)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", choices=models, default=list(models))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=0.50, help="stop after spending this many USD")
    args = parser.parse_args()
    for model in args.models:
        if model != JEV_MODEL:
            llm_eval_params(model)  # fail before any call if a provider isn't pinned

    tasks = {version: mmlu_tasks(version) for version in VERSIONS}
    with make_client("mmlu_eval", args.max_cost) as client:
        stopped = False
        for model in args.models:
            print(model)
            for version in VERSIONS:
                if stopped := run_tasks(tasks[version], answerer(client, model), model, version, None,
                                        args.workers, root=RAW_DIR):
                    break
            if stopped:
                break
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
