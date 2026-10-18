"""Pick N_MMLU verified swaps and build both versions of each item (A11).

Selection: a seeded order of each subject's verified items; each subject
gets a share of N_MMLU in proportion to its verified count (largest
remainder), filled from the front of its order, skipping items flagged in
the manual check.

Versions, written to `data/mmlu/items.jsonl`:
    original  MMLU's question and 4 options, in MMLU's order
    swapped   new question; options = new answer, old answer (the trap) and
              the 2 wrong answers. The new answer's position is balanced over
              A-D, and the trap's over the 3 slots left (seeded).

Also writes the manual-check sheet `data/reviews/mmlu_swaps_30.csv` (from the
first selection; never overwritten). Once filled in, items rated "no" in any
column are dropped and replaced by the next verified item of their subject.

Run from this folder:
    python 11b_mmlu_build.py
"""

import json
import random
from collections import Counter, defaultdict
from typing import Any

from helpers import config
from helpers.generation import is_yes, numbered_options, read_review_sheet, write_review_sheet
from helpers.labels import letter_from_position
from helpers.shuffle import balanced_positions

VERIFIED_PATH = config.MMLU_DIR / "verified.jsonl"
ITEMS_PATH = config.MMLU_DIR / "items.jsonl"
SHEET = config.REVIEW_DIR / "mmlu_swaps_30.csv"
CHECKS = ("only_numbers_changed", "new_answer_right", "wrong_as_plausible")


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Shares of `total` in proportion to `counts`, largest remainder first; never above a count."""
    available = sum(counts.values())
    if available <= total:
        return dict(counts)
    exact = {k: total * n / available for k, n in counts.items()}
    shares = {k: int(v) for k, v in exact.items()}
    for key in sorted(exact, key=lambda k: (shares[k] - exact[k], k))[: total - sum(shares.values())]:
        shares[key] += 1
    return shares


def select(verified: list[dict[str, Any]], flagged: set[str]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(verified, key=lambda r: r["id"]):
        by_subject[item["subject"]].append(item)
    for subject, items in by_subject.items():
        random.Random(f"{config.MMLU_SEED}:order:{subject}").shuffle(items)
    usable = {s: [r for r in items if r["id"] not in flagged] for s, items in by_subject.items()}
    shares = quotas({s: len(items) for s, items in usable.items()}, config.N_MMLU)
    return sorted((r for s, items in usable.items() for r in items[: shares[s]]), key=lambda r: r["id"])


def build(chosen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(f"{config.MMLU_SEED}:positions")
    answer_slots = balanced_positions(len(chosen), 4, rng)
    trap_slots = balanced_positions(len(chosen), 3, rng)
    records = []
    for item, answer_slot, trap_index in zip(chosen, answer_slots, trap_slots, strict=True):
        old_answer = item["options"][ord(item["answer"]) - ord("A")]
        wrong = list(item["swap"]["wrong_answers"])
        random.Random(f"{config.MMLU_SEED}:{item['id']}").shuffle(wrong)
        free = [slot for slot in range(4) if slot != answer_slot]
        trap_slot = free[trap_index]
        options: list[str] = [""] * 4
        options[answer_slot], options[trap_slot] = item["swap"]["answer"], old_answer
        for slot, text in zip((s for s in free if s != trap_slot), wrong, strict=True):
            options[slot] = text
        records.append({
            "id": item["id"],
            "subject": item["subject"],
            "original": {"question": item["question"], "options": item["options"], "answer": item["answer"]},
            "swapped": {"question": item["swap"]["question"], "options": options,
                        "answer": letter_from_position(answer_slot), "trap": letter_from_position(trap_slot)},
        })
    return records


def main() -> None:
    with VERIFIED_PATH.open(encoding="utf-8") as handle:
        verified = [json.loads(line) for line in handle if line.strip()]
    rated = [r for r in read_review_sheet(SHEET) if any(r[c] for c in CHECKS)] if SHEET.exists() else []
    flagged = {r["id"] for r in rated if not all(is_yes(r[c]) for c in CHECKS)}

    chosen = select(verified, flagged)
    records = build(chosen)
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {ITEMS_PATH}: {len(records)} items {dict(Counter(r['subject'] for r in records))}")
    print(f"  new answer at {dict(sorted(Counter(r['swapped']['answer'] for r in records).items()))}, "
          f"trap at {dict(sorted(Counter(r['swapped']['trap'] for r in records).items()))}")
    if len(records) < config.N_MMLU:
        print(f"  {config.N_MMLU - len(records)} short of N_MMLU = {config.N_MMLU}: not enough verified items")

    if SHEET.exists():
        if rated:
            print(f"{SHEET}: {len(rated)} rated, {len(flagged)} flagged and replaced: {', '.join(sorted(flagged)) or 'none'}")
        else:
            print(f"{SHEET}: not filled in yet")
    else:
        by_id = {r["id"]: r for r in verified}
        sample = random.Random(f"{config.MMLU_SEED}:manual").sample(records, min(config.N_MANUAL_MMLU, len(records)))
        rows = [{
            "id": r["id"],
            "original_question": r["original"]["question"],
            "original_options": numbered_options(r["original"]["options"]),
            "old_answer": r["original"]["answer"],
            "swapped_question": r["swapped"]["question"],
            "swapped_options": numbered_options(r["swapped"]["options"]),
            "new_answer": r["swapped"]["answer"],
            "trap": r["swapped"]["trap"],
            "solve": by_id[r["id"]]["swap"]["solve"],
            **dict.fromkeys(CHECKS, ""),
            "notes": "",
        } for r in sorted(sample, key=lambda r: r["id"])]
        write_review_sheet(SHEET, list(rows[0]), rows)
        print(f"Wrote {SHEET}: fill in {', '.join(CHECKS)} (yes/no), then run this again")


if __name__ == "__main__":
    main()
