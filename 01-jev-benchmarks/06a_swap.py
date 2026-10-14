"""Pick the quiz items and make 3 synonym-swap versions of each (A3.3, A7 steps 1-2).

Items: N_QUIZ_BENCHMARK_PER_DATASET test questions per paper dataset (original
wording and order) and N_QUIZ_CONTROL_PER_DATASET Level 3 questions per paper
dataset, written to `data/swaps/items.jsonl`. Level 3 must be reviewed first
(`05_generate_new.py --apply-review`); `--benchmark-only` swaps the benchmark
items before that.

Swaps: PARAPHRASE_MODEL writes 3 versions per item in one request, written to
`data/swaps/raw.jsonl` as versions "swap1".."swap3" of the item's ID.
`06b_validate_swaps.py` checks them; `--retry-failed` regenerates rejected
items (up to MAX_ATTEMPTS).

Run from this folder on the API laptop:
    python 06a_swap.py
    python 06a_swap.py --retry-failed      # after 06b_validate_swaps.py
"""

import argparse
import json
import random
from dataclasses import replace

from helpers import config
from helpers.generation import chat_json, make_client, question_json, run_parallel
from helpers.prompts import fill
from helpers.schema import Question, read_jsonl, write_jsonl
from jev_common.models import require_paraphrase_model
from jev_common.openrouter import OpenRouterClient

ITEMS_PATH = config.SWAPS_DIR / "items.jsonl"
RAW_PATH = config.SWAPS_DIR / "raw.jsonl"
FAILED_PATH = config.SWAPS_DIR / "failed.json"  # written by 06b_validate_swaps.py
SWAP_VERSIONS = tuple(f"swap{i}" for i in range(1, config.N_SWAPS_PER_ITEM + 1))


def select_items(benchmark_only: bool) -> list[Question]:
    """Quiz items, tagged `meta["quiz_set"]` = "benchmark" or "control"."""
    items: list[Question] = []
    for key in config.PAPER_DATASETS:
        test = read_jsonl(config.SAMPLED_DIR / key / "test.jsonl")
        chosen = random.Random(f"{config.QUIZ_SEED}:items:{key}").sample(test, config.N_QUIZ_BENCHMARK_PER_DATASET)
        items += [replace(q, shuffled_order=None, meta={"quiz_set": "benchmark"}) for q in chosen]
    if benchmark_only:
        return items
    if not config.NEW_REVIEWED_PATH.exists():
        raise SystemExit(f"{config.NEW_REVIEWED_PATH} not found: review Level 3 first, or use --benchmark-only")
    level3 = read_jsonl(config.NEW_REVIEWED_PATH)
    for key in config.PAPER_DATASETS:
        pool = sorted((q for q in level3 if q.dataset == key), key=lambda q: q.id)
        if len(pool) < config.N_QUIZ_CONTROL_PER_DATASET:
            raise SystemExit(f"{key}: only {len(pool)} Level 3 questions, need {config.N_QUIZ_CONTROL_PER_DATASET}")
        chosen = random.Random(f"{config.QUIZ_SEED}:control:{key}").sample(pool, config.N_QUIZ_CONTROL_PER_DATASET)
        items += [replace(q, shuffled_order=None, meta={"quiz_set": "control"}) for q in chosen]
    return items


def swap(client: OpenRouterClient, model: str, item: Question, first_attempt: int) -> list[Question]:
    """3 swap versions in one request; malformed replies are retried up to MAX_ATTEMPTS in total."""
    prompt = fill("swap", n_options=item.n_options, item_json=question_json(item))
    error: Exception | None = None
    for attempt in range(first_attempt, config.MAX_ATTEMPTS):
        try:
            data, result = chat_json(client, model, prompt, attempt)
            versions = data.get("versions")
            if not isinstance(versions, list) or len(versions) != len(SWAP_VERSIONS):
                raise ValueError(f"expected {len(SWAP_VERSIONS)} versions, got {versions!r:.200}")
            swaps = []
            for name, version in zip(SWAP_VERSIONS, versions, strict=True):
                options = version.get("options") if isinstance(version, dict) else None
                if not isinstance(options, list) or len(options) != item.n_options:
                    raise ValueError(f"{name}: expected {item.n_options} options")
                swaps.append(replace(
                    item,
                    question=" ".join(str(version.get("question", "")).split()),
                    options=tuple(" ".join(str(o).split()) for o in options),
                    version=name,
                    meta={**item.meta, "model": model, "attempt": attempt, "cost_usd": result.cost_usd / 3},
                ))
            return swaps
        except ValueError as exc:
            error = exc
    raise ValueError(f"{item.id}: no usable swaps after {config.MAX_ATTEMPTS} attempts: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark-only", action="store_true", help="skip control items (Level 3 not ready)")
    parser.add_argument("--retry-failed", action="store_true", help="regenerate what 06b rejected")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=1.0, help="stop after spending this many USD")
    args = parser.parse_args()
    model = require_paraphrase_model()

    items = select_items(args.benchmark_only)
    write_jsonl(ITEMS_PATH, items)
    previous: dict[str, list[Question]] = {}
    if RAW_PATH.exists():
        for q in read_jsonl(RAW_PATH):
            previous.setdefault(q.id, []).append(q)

    failed: dict[str, int] = {}
    if args.retry_failed:
        if not FAILED_PATH.exists():
            raise SystemExit(f"{FAILED_PATH} not found; run 06b_validate_swaps.py first")
        failed = {qid: info.get("attempt", -1) for qid, info in json.loads(FAILED_PATH.read_text(encoding="utf-8")).items()}
    todo = []
    for item in items:
        if item.id in failed:
            if failed[item.id] + 1 < config.MAX_ATTEMPTS:
                todo.append((item, failed[item.id] + 1))
        elif item.id not in previous:
            todo.append((item, 0))

    with make_client("swap", args.max_cost) as client:
        done, errors, stopped = run_parallel(
            lambda pair: swap(client, model, pair[0], pair[1]), todo, args.workers, "swaps"
        )
        print(f"spent this run: ${client.spent_usd:.4f}")
    merged = {**previous, **{item.id: swaps for (item, _), swaps in done}}
    written = write_jsonl(RAW_PATH, [s for item in items for s in merged.get(item.id, [])])
    print(f"Wrote {RAW_PATH}: {written // len(SWAP_VERSIONS)}/{len(items)} items swapped, {len(errors)} failed now")
    for (item, _), error in errors:
        print(f"  {item.id}: {error}")
    if stopped:
        print(f"Stopped at the ${args.max_cost:.2f} cap; re-run to continue (cached calls are free)")


if __name__ == "__main__":
    main()
