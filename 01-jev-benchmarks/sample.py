"""Download the 5 datasets, dedupe SWAG vs HellaSwag, sample, split and save (A2).

For each dataset this writes `data/<dataset>/{train,val,test}.jsonl`, with
labels as letters A-E, stable IDs, and the balanced-shuffle order stored next
to the original order. Splitting happens here, by original question, before
any LLM step. A summary goes to `data/sampling_report.json`.

Run from this folder (needs HuggingFace access, e.g. on the API laptop):
    python sample.py
    python sample.py --datasets arce arcc
"""

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from helpers import config
from helpers.dedupe import dedupe_swag
from helpers.loaders import load_questions
from helpers.schema import Question, write_jsonl
from helpers.shuffle import balanced_shuffle, position_counts

# SWAG is deduped against the sampled HellaSwag test questions, so HellaSwag goes first.
ORDER = ("hs", "arce", "arcc", "csqa", "swag")


def draw(pool: list[Question], n: int, dataset: str, label: str) -> list[Question]:
    """Draw `n` questions from `pool`, reproducibly (independent of pool order)."""
    if len(pool) < n:
        raise ValueError(f"{dataset}: need {n} {label} questions but the pool has only {len(pool)}")
    rng = random.Random(f"{config.SAMPLE_SEED}:{dataset}:{label}")
    return rng.sample(sorted(pool, key=lambda q: q.id), n)


def as_split(questions: list[Question], split: str) -> list[Question]:
    return [replace(q, split=split) for q in questions]


def sample_dataset(
    key: str, hellaswag_test: list[Question] | None
) -> tuple[dict[str, list[Question]], dict]:
    """Return `{split: questions}` (shuffled orders set) and a report for one dataset."""
    spec = config.DATASETS[key]
    report: dict = {"name": spec.name}

    test_pool = load_questions(key, spec.test_source, "test")
    train_pool = load_questions(key, config.TRAIN_SOURCE, "train")
    report["pool_sizes"] = {"test": len(test_pool), "train": len(train_pool)}

    if key == "swag":
        if hellaswag_test is None:
            raise ValueError("swag needs the sampled HellaSwag test questions; sample 'hs' first")
        result = dedupe_swag(train_pool, hellaswag_test)
        train_pool = result.kept
        report["dedupe"] = {
            "dropped_by_video": len(result.dropped_by_video),
            "dropped_by_context": len(result.dropped_by_context),
            "kept": len(result.kept),
        }

    test = draw(test_pool, config.N_TEST, key, "test")

    if spec.val_source == config.TRAIN_SOURCE:
        # One pool, two disjoint draws: sample both at once, then split.
        drawn = draw(train_pool, config.N_TRAIN + config.N_VAL, key, "train+val")
        train, val = drawn[: config.N_TRAIN], as_split(drawn[config.N_TRAIN :], "val")
    else:
        val_pool = load_questions(key, spec.val_source, "val")
        report["pool_sizes"]["val"] = len(val_pool)
        train, val = draw(train_pool, config.N_TRAIN, key, "train"), draw(val_pool, config.N_VAL, key, "val")

    splits = {"train": train, "val": val, "test": test}
    shuffled = {
        split: balanced_shuffle(questions, config.SHUFFLE_SEED) for split, questions in splits.items()
    }
    report["sizes"] = {split: len(questions) for split, questions in shuffled.items()}
    report["correct_position_counts"] = {
        split: {str(n): dict(sorted(counts.items())) for n, counts in position_counts(questions).items()}
        for split, questions in shuffled.items()
    }
    return shuffled, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--out", type=Path, default=config.DATA_DIR, help="output folder")
    args = parser.parse_args()

    report: dict = {}
    hellaswag_test: list[Question] | None = None
    for key in sorted(args.datasets, key=ORDER.index):
        print(f"Sampling {config.DATASETS[key].name} ...")
        if key == "swag" and hellaswag_test is None:
            raise SystemExit("SWAG is deduped against HellaSwag: include 'hs' in --datasets")
        splits, report[key] = sample_dataset(key, hellaswag_test)
        for split, questions in splits.items():
            write_jsonl(args.out / key / f"{split}.jsonl", questions)
        if key == "hs":
            hellaswag_test = splits["test"]
        print(f"  {report[key]['sizes']}")

    report_path = args.out / "sampling_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
