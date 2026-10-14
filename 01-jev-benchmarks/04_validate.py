"""Check the paraphrases and keep the ones that pass (A3.1).

Automatic checks on every paraphrase in `data/paraphrased/raw/` (answer still
present and unambiguous, option count unchanged, text actually changed).
Passing paraphrases get a balanced shuffle and are written to
`data/paraphrased/<dataset>/<split>.jsonl` (Level 2 and BERT training data).
Rejected and missing ones go to `data/paraphrased/failed.json`, which
`03_paraphrase.py --retry-failed` regenerates.

Also writes the manual-check sheet `data/reviews/paraphrase_50.csv` (random
test paraphrases; never overwritten) and, once you have filled it in,
summarises it.

Run from this folder:
    python 04_validate.py
"""

import argparse
import json
import random
from collections import Counter

from helpers import config
from helpers.generation import is_yes, numbered_options, read_review_sheet, write_review_sheet
from helpers.schema import read_jsonl, write_jsonl
from helpers.shuffle import balanced_shuffle
from helpers.text_checks import check_paraphrase

SHEET = config.REVIEW_DIR / "paraphrase_50.csv"
FAILED_PATH = config.PARAPHRASED_DIR / "failed.json"
REPORT_PATH = config.PARAPHRASED_DIR / "validation_report.json"


def summarise_sheet() -> None:
    rows = [row for row in read_review_sheet(SHEET) if row["meaning_preserved"]]
    if not rows:
        print(f"{SHEET}: not filled in yet")
        return
    kept = sum(is_yes(row["meaning_preserved"]) for row in rows)
    difficulty = Counter(row["difficulty"].casefold() or "blank" for row in rows)
    print(f"{SHEET}: {len(rows)} rated; meaning kept {kept}/{len(rows)} ({kept / len(rows):.0%}); "
          f"difficulty {dict(difficulty)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    args = parser.parse_args()

    # Keep other datasets' entries when only some are validated (--datasets).
    failed: dict[str, dict] = {}
    report: dict[str, dict] = {}
    if FAILED_PATH.exists():
        failed = {qid: info for qid, info in json.loads(FAILED_PATH.read_text(encoding="utf-8")).items()
                  if info["dataset"] not in args.datasets}
    if REPORT_PATH.exists():
        report = {key: value for key, value in json.loads(REPORT_PATH.read_text(encoding="utf-8"))["datasets"].items()
                  if key not in args.datasets}
    passed_test = []
    for key in args.datasets:
        report[key] = {}
        for split in config.SPLITS:
            originals = read_jsonl(config.SAMPLED_DIR / key / f"{split}.jsonl")
            raw_file = config.PARAPHRASED_RAW_DIR / key / f"{split}.jsonl"
            raw = {q.id: q for q in read_jsonl(raw_file)} if raw_file.exists() else {}
            passed = []
            problem_counts: Counter[str] = Counter()
            for original in originals:
                candidate = raw.get(original.id)
                if candidate is None:
                    failed[original.id] = {"dataset": key, "split": split, "attempt": -1, "problems": ["missing"]}
                    problem_counts["missing"] += 1
                    continue
                problems = check_paraphrase(original, candidate)
                if problems:
                    failed[original.id] = {
                        "dataset": key, "split": split,
                        "attempt": candidate.meta.get("attempt", 0), "problems": problems,
                    }
                    problem_counts.update(" ".join(p.split()[:3]) for p in problems)
                else:
                    passed.append(candidate)
            shuffled = balanced_shuffle(passed, config.SHUFFLE_SEED)
            write_jsonl(config.PARAPHRASED_DIR / key / f"{split}.jsonl", shuffled)
            if split == "test":
                by_id = {q.id: q for q in originals}
                passed_test += [(by_id[p.id], p) for p in shuffled]
            report[key][split] = {"total": len(originals), "passed": len(passed), "problems": dict(problem_counts)}
            print(f"{key}/{split}: {len(passed)}/{len(originals)} passed {dict(problem_counts) or ''}")

    out_of_attempts = [qid for qid, info in failed.items() if info["attempt"] >= config.MAX_ATTEMPTS - 1]
    FAILED_PATH.write_text(json.dumps(failed, indent=1), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps({"datasets": report, "out_of_attempts": out_of_attempts}, indent=1), encoding="utf-8")
    print(f"{len(failed)} failed ({len(out_of_attempts)} out of attempts, dropped); see {FAILED_PATH}")
    if len(failed) > len(out_of_attempts):
        print("Next: python 03_paraphrase.py --retry-failed, then run this again")

    if not SHEET.exists():
        if len(passed_test) < config.N_MANUAL_PARAPHRASE:
            print(f"Only {len(passed_test)} test paraphrases so far; the review sheet waits for {config.N_MANUAL_PARAPHRASE}")
            return
        sample = random.Random(f"{config.SAMPLE_SEED}:manual-paraphrase").sample(
            sorted(passed_test, key=lambda pair: pair[0].id), config.N_MANUAL_PARAPHRASE
        )
        rows = [{
            "id": original.id,
            "correct_answer": original.answer,
            "original": f"{original.question}\n{numbered_options(original.options)}",
            "paraphrase": f"{p.question}\n{numbered_options(p.options)}",
            "meaning_preserved": "",
            "difficulty": "",
            "notes": "",
        } for original, p in sample]
        write_review_sheet(SHEET, list(rows[0]), rows)
        print(f"Wrote {SHEET}: fill in meaning_preserved (yes/no) and difficulty (easier/same/harder)")
    else:
        summarise_sheet()


if __name__ == "__main__":
    main()
