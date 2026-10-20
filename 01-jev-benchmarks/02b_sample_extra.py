"""Draw extra train questions for the bigger-pool BERT sensitivity analysis (A8, decided 2026-10-20).

For each dataset in N_TRAIN_EXTRA, draws that many more train questions (None =
all that are left) from the same train pool as 02_sample.py, after the same
filters (duplicate options, SWAG video dedupe against the sampled HellaSwag
test questions, text overlap with the sampled test questions), leaving out
every question already in train, val or test. The existing splits are read,
never rewritten, so every model is still tested on the same questions.

Writes `data/sampled/<dataset>/train_extra.jsonl` (split "train", with a
balanced-shuffle order) and adds a "train_extra" entry per dataset to
`data/sampling_report.json`. 03_paraphrase.py and 04_validate.py then treat
it like any other split; `07a_build_bert_data.py --extra-train` adds it to
the training data.

Run from this folder (needs HuggingFace access):
    python 02b_sample_extra.py
"""

import argparse
import json
import random

from helpers import config
from helpers.dedupe import dedupe_swag, drop_test_overlap
from helpers.loaders import load_questions
from helpers.schema import read_jsonl, write_jsonl
from helpers.shuffle import balanced_shuffle, position_counts
from helpers.text_checks import duplicate_options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=list(config.N_TRAIN_EXTRA),
                        default=list(config.N_TRAIN_EXTRA))
    args = parser.parse_args()

    report_path = config.DATA_DIR / "sampling_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in args.datasets:
        out = config.SAMPLED_DIR / key / f"{config.EXTRA_TRAIN_SPLIT}.jsonl"
        if out.exists():
            raise SystemExit(f"{out} already exists; delete it first to draw again")
        existing = {split: read_jsonl(config.SAMPLED_DIR / key / f"{split}.jsonl") for split in config.SPLITS}
        used = {q.id for questions in existing.values() for q in questions}

        pool = [q for q in load_questions(key, config.TRAIN_SOURCE, "train") if not duplicate_options(q)]
        if key == "swag":
            pool = dedupe_swag(pool, read_jsonl(config.SAMPLED_DIR / "hs" / "test.jsonl")).kept
        pool, _ = drop_test_overlap(pool, existing["test"])
        pool = [q for q in pool if q.id not in used]

        wanted = config.N_TRAIN_EXTRA[key]
        n = len(pool) if wanted is None else wanted
        if n > len(pool):
            raise SystemExit(f"{key}: want {n} extra train questions but only {len(pool)} are left")
        rng = random.Random(f"{config.SAMPLE_SEED}:{key}:{config.EXTRA_TRAIN_SPLIT}")
        extra = balanced_shuffle(rng.sample(sorted(pool, key=lambda q: q.id), n), config.SHUFFLE_SEED)
        write_jsonl(out, extra)

        counts = position_counts(extra)
        report[key][config.EXTRA_TRAIN_SPLIT] = {
            "left_in_pool": len(pool), "drawn": len(extra),
            "correct_position_counts": {str(k): dict(sorted(c.items())) for k, c in counts.items()},
        }
        print(f"{config.DATASETS[key].name}: {len(extra)} extra train questions (of {len(pool)} left) -> {out}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated {report_path}. Next (API laptop): python 03_paraphrase.py, then python 04_validate.py")


if __name__ == "__main__":
    main()
