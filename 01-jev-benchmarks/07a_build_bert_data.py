"""Build the flat data files the Colab BERT notebook reads (A5, A6).

Writes `data/bert/`:
    train.jsonl, val.jsonl         originals + paraphrases of the train/val
                                   questions (A5: one model trained on both
                                   versions), each in its balanced-shuffle order
    level0.jsonl .. level3.jsonl   the test levels exactly as Jev sees them
                                   (helpers/levels.py)

One JSON object per line: id, dataset, split, version ("original" /
"paraphrase"), question, options (in the order shown), label (0-based
position of the correct option), answer (its letter). The notebook pads to 5
options with masked dummies (A2 rule 6), so options are not padded here.

Copy data/bert/ to Google Drive (MyDrive/jev-bert/data/) and run
notebooks/07b_bert_colab.ipynb.

Run from this folder, after 04_validate.py (and 05 --apply-review for Level 3):
    python 07a_build_bert_data.py
    python 07a_build_bert_data.py --originals-only   # before the paraphrases exist
"""

import argparse
import json
from collections import Counter

from helpers import config
from helpers.labels import position_from_letter
from helpers.levels import LEVELS, LevelItem, level_items
from helpers.schema import Question, read_jsonl
from helpers.evaluation import write_records


def flat(question: Question) -> dict:
    options, answer = question.view("shuffled")
    return {
        "id": question.id,
        "dataset": question.dataset,
        "split": question.split,
        "version": question.version,
        "question": question.question,
        "options": list(options),
        "label": position_from_letter(answer),
        "answer": answer,
    }


def flat_item(item: LevelItem) -> dict:
    return {
        "id": item.id,
        "dataset": item.dataset,
        "split": "test",
        "version": item.version,
        "question": item.question,
        "options": list(item.options),
        "label": position_from_letter(item.answer),
        "answer": item.answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--originals-only", action="store_true",
                        help="build train/val without paraphrases (testing, or the original-only ablation)")
    args = parser.parse_args()

    for split in ("train", "val"):
        records, counts = [], Counter()
        for key in config.DATASETS:
            originals = read_jsonl(config.SAMPLED_DIR / key / f"{split}.jsonl")
            records += [flat(q) for q in originals]
            counts[(key, "original")] += len(originals)
            if args.originals_only:
                continue
            path = config.PARAPHRASED_DIR / key / f"{split}.jsonl"
            if not path.exists():
                raise SystemExit(f"{path} not found: run 03_paraphrase.py and 04_validate.py first "
                                 "(or pass --originals-only)")
            paraphrases = read_jsonl(path)
            records += [flat(q) for q in paraphrases]
            counts[(key, "paraphrase")] += len(paraphrases)
        write_records(config.BERT_DATA_DIR / f"{split}.jsonl", records)
        detail = ", ".join(f"{key} {n}" + ("" if version == "original" else " para")
                           for (key, version), n in counts.items())
        print(f"{split}: {len(records)} ({detail})")

    for level in LEVELS:
        path = config.BERT_DATA_DIR / f"level{level}.jsonl"
        try:
            items = level_items(level)
        except FileNotFoundError as error:
            print(f"level{level}: skipped, {error}")
            path.unlink(missing_ok=True)  # never leave a stale level behind
            continue
        write_records(path, [flat_item(item) for item in items])
        print(f"level{level}: {len(items)}")

    manifest = {p.name: sum(1 for _ in p.open(encoding="utf-8")) for p in sorted(config.BERT_DATA_DIR.glob("*.jsonl"))}
    (config.BERT_DATA_DIR / "manifest.json").write_text(
        json.dumps({"originals_only": args.originals_only, "rows": manifest}, indent=1), encoding="utf-8"
    )
    print(f"Wrote {config.BERT_DATA_DIR}; copy it to MyDrive/jev-bert/data/ for the notebook")


if __name__ == "__main__":
    main()
