"""Check the synonym swaps and keep the items whose swaps all pass (A3.3, A7 step 3).

Automatic checks per item (answer present, option count unchanged, text
changed, word count within +/-WORD_COUNT_TOLERANCE, word-level edits only, all
4 versions distinct). Items that pass go to `data/swaps/swaps.jsonl`;
rejected and missing ones to `data/swaps/failed.json`, which
`06a_swap.py --retry-failed` regenerates.

Also writes the manual-check sheet `data/reviews/swaps_30.csv` (random passing
items; never overwritten) and, once filled in, summarises it.

Run from this folder:
    python 06b_validate_swaps.py
"""

import json
import random
from collections import Counter, defaultdict

from helpers import config
from helpers.generation import instance_text, is_yes, read_review_sheet, write_review_sheet
from helpers.schema import Question, read_jsonl, write_jsonl
from helpers.text_checks import check_swap_set

ITEMS_PATH = config.SWAPS_DIR / "items.jsonl"
RAW_PATH = config.SWAPS_DIR / "raw.jsonl"
SWAPS_PATH = config.SWAPS_DIR / "swaps.jsonl"
FAILED_PATH = config.SWAPS_DIR / "failed.json"
SHEET = config.REVIEW_DIR / "swaps_30.csv"


def main() -> None:
    items = read_jsonl(ITEMS_PATH)
    raw: dict[str, list[Question]] = defaultdict(list)
    for swap in read_jsonl(RAW_PATH) if RAW_PATH.exists() else []:
        raw[swap.id].append(swap)

    passed: list[tuple[Question, list[Question]]] = []
    failed: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for item in items:
        swaps = sorted(raw.get(item.id, []), key=lambda q: q.version)
        if len(swaps) != config.N_SWAPS_PER_ITEM:
            failed[item.id] = {"attempt": -1, "problems": ["missing"]}
            counts["missing"] += 1
            continue
        problems = check_swap_set(item, swaps)
        if problems:
            failed[item.id] = {"attempt": swaps[0].meta.get("attempt", 0), "problems": problems}
            counts.update(p.split(": ", 1)[-1].split(" (")[0][:40] for p in problems)
        else:
            passed.append((item, swaps))

    write_jsonl(SWAPS_PATH, [s for _, swaps in passed for s in swaps])
    FAILED_PATH.write_text(json.dumps(failed, indent=1), encoding="utf-8")
    by_set = Counter(item.meta["quiz_set"] for item, _ in passed)
    out_of_attempts = sum(info["attempt"] >= config.MAX_ATTEMPTS - 1 for info in failed.values())
    print(f"{len(passed)}/{len(items)} items passed {dict(by_set)}; {len(failed)} failed "
          f"({out_of_attempts} out of attempts)")
    for problem, count in counts.most_common(10):
        print(f"  {count:4d}  {problem}")
    if len(failed) > out_of_attempts:
        print("Next: python 06a_swap.py --retry-failed, then run this again")

    if SHEET.exists():
        rows = [r for r in read_review_sheet(SHEET) if r["meaning_kept_all"]]
        if rows:
            kept = sum(is_yes(r["meaning_kept_all"]) for r in rows)
            natural = sum(is_yes(r["none_sounds_off"]) for r in rows)
            print(f"{SHEET}: {len(rows)} rated; meaning kept {kept}/{len(rows)}, none sounds off {natural}/{len(rows)}")
        else:
            print(f"{SHEET}: not filled in yet")
    elif len(passed) >= config.N_MANUAL_SWAPS:
        sample = random.Random(f"{config.QUIZ_SEED}:manual-swaps").sample(passed, config.N_MANUAL_SWAPS)
        rows = [{
            "id": item.id,
            "quiz_set": item.meta["quiz_set"],
            "original": instance_text(item),
            **{s.version: instance_text(s) for s in swaps},
            "meaning_kept_all": "",
            "none_sounds_off": "",
            "notes": "",
        } for item, swaps in sample]
        write_review_sheet(SHEET, list(rows[0]), rows)
        print(f"Wrote {SHEET}: fill in meaning_kept_all and none_sounds_off (yes/no)")


if __name__ == "__main__":
    main()
