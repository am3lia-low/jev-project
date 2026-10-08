"""Run the LLM baselines on the test levels and the quiz (A4, A9).

As in the paper: the same request Jev gets, rendered as a chat prompt the way
decision-index's `transformers` engine does (system line + "State / Question /
Options" + "Reply with exactly one option key."), thinking off, one generated
token. The option letters' probabilities are read from the first token's top
logprobs and renormalised. decision-index instead scores every option key in
one forward pass; over OpenRouter only the top 20 tokens are visible, so a
letter outside them gets 0.

Each baseline is pinned to one provider (`LLM_PROVIDERS` in
src/jev_common/models.py) with no fallback, so all of a model's answers come
from the same deployment.

Writes results/raw/<model slug>/<run>.jsonl (format in helpers/evaluation.py).

Run from this folder on the API laptop, after 08_run_jev.py:
    python 09_run_llm.py levels [--models qwen/qwen3.8-27b] [--levels 0 1]
    python 09_run_llm.py quiz
"""

import argparse

from helpers.evaluation import Task, chat_messages, level_tasks, quiz_tasks, run_tasks
from helpers.generation import make_client
from jev_common.models import LLM_BASELINE_MODELS, llm_eval_params
from jev_common.openrouter import OpenRouterClient, letter_probabilities


def llm_answerer(client: OpenRouterClient, model: str):
    params = llm_eval_params(model)

    def answer(task: Task, cache_tag: str | None):
        result = client.chat(model, chat_messages(task), cache_tag=cache_tag, **params)
        return letter_probabilities(result.response, task.letters), result

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("levels", "quiz"))
    parser.add_argument("--models", nargs="+", choices=LLM_BASELINE_MODELS, default=list(LLM_BASELINE_MODELS))
    parser.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=2.0, help="stop after spending this many USD")
    args = parser.parse_args()

    for model in args.models:
        llm_eval_params(model)  # fail before any call if a provider isn't pinned

    with make_client("llm", args.max_cost) as client:
        for model in args.models:
            print(model)
            answer = llm_answerer(client, model)
            if args.step == "quiz":
                stopped = run_tasks(quiz_tasks(), answer, model, "quiz", None, args.workers)
            else:
                stopped = False
                for level in args.levels:
                    try:
                        tasks = level_tasks(level)
                    except FileNotFoundError as error:
                        print(f"  level{level}: skipped, {error}")
                        continue
                    if stopped := run_tasks(tasks, answer, model, f"level{level}", None, args.workers):
                        break
            if stopped:
                break
        print(f"spent this run: ${client.spent_usd:.4f}")


if __name__ == "__main__":
    main()
