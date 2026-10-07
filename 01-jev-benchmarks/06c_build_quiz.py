"""Assemble the memorisation quiz (A7 step 4).

For every item that passed `06b_validate_swaps.py`: options A-D are the
original and its 3 swaps (each a whole question with its own options), option
E is "None of the provided options". The original's position is balanced
across A-D within each (quiz set, dataset) group; the swaps fill the other
slots in a seeded random order. Writes `data/quiz/quiz.jsonl`, one record per
item, which `08_run_jev.py` and `09_run_llm.py` turn into requests with
`helpers.prompts.quiz_request`.

Run from this folder:
    python 06c_build_quiz.py
"""

import json
import random
from collections import Counter, defaultdict

from helpers import config
from helpers.generation import instance_text
from helpers.labels import letter_from_position
from helpers.schema import Question, read_jsonl
from helpers.shuffle import balanced_positions

ITEMS_PATH = config.SWAPS_DIR / "items.jsonl"
SWAPS_PATH = config.SWAPS_DIR / "swaps.jsonl"
QUIZ_PATH = config.QUIZ_DIR / "quiz.jsonl"
N_VERSIONS = 1 + config.N_SWAPS_PER_ITEM  # A-D
NONE_OPTION = "None of the provided options."


def main() -> None:
    items = {q.id: q for q in read_jsonl(ITEMS_PATH)}
    swaps: dict[str, list[Question]] = defaultdict(list)
    for swap in read_jsonl(SWAPS_PATH):
        swaps[swap.id].append(swap)

    groups: dict[tuple[str, str], list[Question]] = defaultdict(list)
    for item_id in sorted(swaps):
        item = items[item_id]
        groups[(item.meta["quiz_set"], item.dataset)].append(item)

    records = []
    for (quiz_set, dataset), group in sorted(groups.items()):
        rng = random.Random(f"{config.QUIZ_SEED}:quiz:{quiz_set}:{dataset}")
        positions = balanced_positions(len(group), N_VERSIONS, rng)
        for item, position in zip(group, positions, strict=True):
            others = sorted(swaps[item.id], key=lambda q: q.version)
            random.Random(f"{config.QUIZ_SEED}:{item.id}").shuffle(others)
            versions = others[:position] + [item] + others[position:]
            records.append({
                "id": item.id,
                "dataset": dataset,
                "quiz_set": quiz_set,
                "options": {
                    **{letter_from_position(i): instance_text(v) for i, v in enumerate(versions)},
                    letter_from_position(N_VERSIONS): NONE_OPTION,
                },
                "versions": {letter_from_position(i): v.version for i, v in enumerate(versions)},
                "original": letter_from_position(position),
            })

    QUIZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUIZ_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sets = Counter(r["quiz_set"] for r in records)
    print(f"Wrote {QUIZ_PATH}: {len(records)} items {dict(sets)}")
    for quiz_set in sorted(sets):
        positions = Counter(r["original"] for r in records if r["quiz_set"] == quiz_set)
        print(f"  {quiz_set}: original at {dict(sorted(positions.items()))}")
    expected = len(config.PAPER_DATASETS) * (config.N_QUIZ_BENCHMARK_PER_DATASET + config.N_QUIZ_CONTROL_PER_DATASET)
    if len(records) < expected:
        print(f"  {expected - len(records)} short of {expected}: items whose swaps never passed 06b are left out")


if __name__ == "__main__":
    main()
