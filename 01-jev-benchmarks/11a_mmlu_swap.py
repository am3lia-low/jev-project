"""Make a number-swapped version of each MMLU maths candidate and check it by code (A11).

Candidates: MMLU maths test questions whose 4 options are plain numbers and
whose question contains a number (`helpers.mmlu.load_candidates`), written to
`data/mmlu/candidates.jsonl`.

Swaps: NEW_QUESTION_MODEL rewrites the numbers, writes a `solve()` function
for the answer, the new answer and 2 wrong answers (`prompts/mmlu_swap.txt`),
or says the item is not swappable. Replies go to `data/mmlu/raw.jsonl`.

Checks (`helpers.mmlu.check_swap`): only numbers changed, solve(original)
gives the original answer, solve(new) the stated new answer, all answers
distinct and the old answer not among the wrong ones. Passing swaps go to
`data/mmlu/verified.jsonl`, the rest to `data/mmlu/failed.json`;
`--retry-failed` regenerates failed ones (up to MAX_ATTEMPTS), never the
ones the model called not swappable.

Run from this folder on the API laptop:
    python 11a_mmlu_swap.py
    python 11a_mmlu_swap.py --retry-failed
"""

import argparse
import json
from collections import Counter
from typing import Any

from helpers import config
from helpers.generation import chat_json, make_client, numbered_options, run_parallel
from helpers.mmlu import check_swap, load_candidates
from helpers.prompts import fill
from jev_common.models import NEW_QUESTION_MODEL
from jev_common.openrouter import OpenRouterClient

CANDIDATES_PATH = config.MMLU_DIR / "candidates.jsonl"
RAW_PATH = config.MMLU_DIR / "raw.jsonl"
VERIFIED_PATH = config.MMLU_DIR / "verified.jsonl"
FAILED_PATH = config.MMLU_DIR / "failed.json"
TEMPERATURE = 0.2  # arithmetic, not style: lower than NEW_QUESTION_MODEL's default


def read_records(path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_records(path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temp_path.replace(path)
    return count


def generate(client: OpenRouterClient, item: dict[str, Any], attempt: int) -> dict[str, Any]:
    """One reply; a reply that is not JSON is retried as a fresh request up to MAX_ATTEMPTS."""
    prompt = fill("mmlu_swap", question=item["question"], options=numbered_options(item["options"]),
                  answer=item["options"][ord(item["answer"]) - ord("A")])
    error: Exception | None = None
    for next_attempt in range(attempt, config.MAX_ATTEMPTS):
        try:
            data, result = chat_json(client, NEW_QUESTION_MODEL, prompt, next_attempt, temperature=TEMPERATURE)
            return {"id": item["id"], "reply": data, "model": NEW_QUESTION_MODEL,
                    "attempt": next_attempt, "cost_usd": result.cost_usd}
        except ValueError as exc:
            error = exc
    raise ValueError(f"{item['id']}: no JSON reply after {config.MAX_ATTEMPTS} attempts: {error}")


def verify(candidates: list[dict[str, Any]], raw: dict[str, dict[str, Any]]) -> None:
    verified, failed, counts = [], {}, Counter()
    for item in candidates:
        record = raw.get(item["id"])
        if record is None:
            failed[item["id"]] = {"attempt": -1, "problems": ["missing"]}
            counts["missing"] += 1
            continue
        reply = record["reply"]
        if reply.get("swappable") is False:
            failed[item["id"]] = {"attempt": config.MAX_ATTEMPTS, "problems": [f"not swappable: {reply.get('reason', '')}"]}
            counts["not swappable"] += 1
            continue
        problems = check_swap(item, reply)
        if problems:
            failed[item["id"]] = {"attempt": record["attempt"], "problems": problems}
            counts.update(p.split(":")[0].split(" = ")[0][:40] for p in problems)
            continue
        verified.append({**item, "swap": {
            "question": " ".join(reply["new_question"].split()),
            "answer": " ".join(str(reply["new_answer"]).split()),
            "wrong_answers": [" ".join(str(w).split()) for w in reply["wrong_answers"]],
            "solve": reply["solve"],
            "original_args": reply["original_args"],
            "new_args": reply["new_args"],
            "model": record["model"],
            "attempt": record["attempt"],
        }})

    write_records(VERIFIED_PATH, verified)
    FAILED_PATH.write_text(json.dumps(failed, indent=1), encoding="utf-8")
    by_subject = Counter(r["subject"] for r in verified)
    retryable = sum(0 <= info["attempt"] < config.MAX_ATTEMPTS - 1 for info in failed.values())
    print(f"{len(verified)}/{len(candidates)} verified {dict(by_subject)}; {len(failed)} failed ({retryable} can be retried)")
    for problem, count in counts.most_common(12):
        print(f"  {count:4d}  {problem}")
    if retryable:
        print("Next: python 11a_mmlu_swap.py --retry-failed")
    elif len(verified) >= config.N_MMLU:
        print("Next: python 11b_mmlu_build.py")
    else:
        print(f"Only {len(verified)} verified, fewer than N_MMLU = {config.N_MMLU}; 11b keeps all of them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--retry-failed", action="store_true", help="regenerate swaps that failed the checks")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=5.0, help="stop after spending this many USD")
    args = parser.parse_args()

    if CANDIDATES_PATH.exists():
        candidates = read_records(CANDIDATES_PATH)
    else:
        candidates = load_candidates()
        write_records(CANDIDATES_PATH, candidates)
    print(f"{len(candidates)} candidates: {dict(Counter(c['subject'] for c in candidates))}")
    raw = {r["id"]: r for r in read_records(RAW_PATH)} if RAW_PATH.exists() else {}

    todo = [(item, 0) for item in candidates if item["id"] not in raw]
    if args.retry_failed:
        if not FAILED_PATH.exists():
            raise SystemExit(f"{FAILED_PATH} not found; run 11a once without --retry-failed first")
        failed = json.loads(FAILED_PATH.read_text(encoding="utf-8"))
        todo += [(item, failed[item["id"]]["attempt"] + 1) for item in candidates
                 if item["id"] in raw and item["id"] in failed
                 and 0 <= failed[item["id"]]["attempt"] < config.MAX_ATTEMPTS - 1]

    if todo:
        with make_client("mmlu_swap", args.max_cost) as client:
            done, errors, stopped = run_parallel(lambda pair: generate(client, *pair), todo, args.workers, "swaps")
            print(f"spent this run: ${client.spent_usd:.4f}")
        raw |= {record["id"]: record for _, record in done}
        write_records(RAW_PATH, [raw[c["id"]] for c in candidates if c["id"] in raw])
        for (item, _), error in errors:
            print(f"  {item['id']}: {error}")
        if stopped:
            print(f"Stopped at the ${args.max_cost:.2f} cap; re-run to continue (cached calls are free)")
    verify(candidates, raw)


if __name__ == "__main__":
    main()
