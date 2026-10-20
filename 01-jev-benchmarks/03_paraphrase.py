"""Paraphrase the sampled questions with an OpenRouter LLM (A3.1).

Pilot (run first): the same N_PILOT train questions through both candidate
models, written to `data/paraphrased/pilot/` with a blinded review sheet
`data/reviews/pilot_30.csv` (variants X/Y; the key is in the pilot folder).
Fill in the sheet, then `--pilot-results` shows which model kept meaning best.
Set `PARAPHRASE_MODEL` in src/jev_common/models.py to the winner.

Full run: every sampled question (train, val, test) through `PARAPHRASE_MODEL`,
written to `data/paraphrased/raw/<dataset>/<split>.jsonl`. `04_validate.py`
checks them; `--retry-failed` regenerates what it rejected (up to
MAX_ATTEMPTS per question). Responses are cached, so re-runs only pay for
what is missing.

Run from this folder on the API laptop:
    python 03_paraphrase.py --pilot
    python 03_paraphrase.py --pilot-results
    python 03_paraphrase.py                     # full run
    python 03_paraphrase.py --retry-failed      # after 04_validate.py
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from helpers import config
from helpers.generation import (
    chat_json,
    is_yes,
    make_client,
    numbered_options,
    question_json,
    read_review_sheet,
    run_parallel,
    write_review_sheet,
)
from helpers.prompts import fill
from helpers.schema import PARAPHRASE, Question, read_jsonl, write_jsonl
from helpers.text_checks import check_paraphrase
from jev_common.models import PARAPHRASE_CANDIDATES, require_paraphrase_model
from jev_common.openrouter import OpenRouterClient

PILOT_SHEET = config.REVIEW_DIR / "pilot_30.csv"
PILOT_KEY = config.PILOT_DIR / "blind_key.json"
FAILED_PATH = config.PARAPHRASED_DIR / "failed.json"  # written by 04_validate.py


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def paraphrase(client: OpenRouterClient, model: str, question: Question, first_attempt: int) -> Question:
    """One paraphrase; malformed replies are retried up to MAX_ATTEMPTS in total."""
    prompt = fill("paraphrase", n_options=question.n_options, item_json=question_json(question))
    error: Exception | None = None
    for attempt in range(first_attempt, config.MAX_ATTEMPTS):
        try:
            data, result = chat_json(client, model, prompt, attempt)
            options = data.get("options")
            if not isinstance(data.get("question"), str) or not isinstance(options, list):
                raise ValueError(f"reply lacks question/options: {data}")
            if len(options) != question.n_options:
                raise ValueError(f"reply has {len(options)} options, expected {question.n_options}")
            return replace(
                question,
                question=" ".join(data["question"].split()),
                options=tuple(" ".join(str(option).split()) for option in options),
                shuffled_order=None,  # 04_validate.py assigns the balanced shuffle
                version=PARAPHRASE,
                meta={
                    "source_split": question.meta.get("source_split"),
                    "model": model,
                    "attempt": attempt,
                    "cost_usd": result.cost_usd,
                },
            )
        except ValueError as exc:
            error = exc
    raise ValueError(f"{question.id}: no usable paraphrase after {config.MAX_ATTEMPTS} attempts: {error}")


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------
def pilot_questions() -> list[Question]:
    """N_PILOT train questions, spread evenly over the datasets."""
    keys = list(config.DATASETS)
    per_dataset = config.N_PILOT // len(keys)
    chosen = []
    for key in keys:
        pool = read_jsonl(config.SAMPLED_DIR / key / "train.jsonl")
        chosen += random.Random(f"{config.SAMPLE_SEED}:pilot:{key}").sample(pool, per_dataset)
    return chosen


def run_pilot(workers: int, max_cost: float) -> None:
    questions = pilot_questions()
    outputs: dict[str, dict[str, Question]] = {}
    with make_client("paraphrase_pilot", max_cost) as client:
        for model in PARAPHRASE_CANDIDATES:
            done, failed, stopped = run_parallel(
                lambda q, m=model: paraphrase(client, m, q, 0), questions, workers, model
            )
            outputs[model] = {q.id: p for q, p in done}
            write_jsonl(config.PILOT_DIR / f"{model_slug(model)}.jsonl", [p for _, p in done])
            cost = sum(p.meta["cost_usd"] for p in outputs[model].values())
            passed = sum(not check_paraphrase(q, outputs[model][q.id]) for q in questions if q.id in outputs[model])
            print(f"{model}: {len(done)} done, {len(failed)} failed, auto-checks passed {passed}, "
                  f"${cost:.4f} (~${cost / max(len(done), 1) * 8500:.2f} for all 8,500)")
            for question, error in failed:
                print(f"  {question.id}: {error}")
            if stopped:
                raise SystemExit("stopped at the spending cap")
        print(f"spent this run: ${client.spent_usd:.4f}")

    rng = random.Random(f"{config.SAMPLE_SEED}:pilot:blind")
    key: dict[str, dict[str, str]] = {}
    rows = []
    for question in questions:
        models = list(PARAPHRASE_CANDIDATES)
        rng.shuffle(models)
        key[question.id] = dict(zip(("X", "Y"), models, strict=True))
        for variant, model in key[question.id].items():
            p = outputs[model].get(question.id)
            rows.append({
                "id": question.id,
                "variant": variant,
                "correct_answer": question.answer,
                "original": f"{question.question}\n{numbered_options(question.options)}",
                "paraphrase": "" if p is None else f"{p.question}\n{numbered_options(p.options)}",
                "auto_check_problems": "generation failed" if p is None else "; ".join(check_paraphrase(question, p)),
                "meaning_preserved": "",
                "difficulty": "",
                "notes": "",
            })
    PILOT_KEY.parent.mkdir(parents=True, exist_ok=True)
    PILOT_KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    if write_review_sheet(PILOT_SHEET, list(rows[0]), rows):
        print(f"Wrote {PILOT_SHEET}: fill in meaning_preserved (yes/no) and difficulty (easier/same/harder)")
    else:
        print(f"{PILOT_SHEET} already exists; left untouched")


def pilot_results() -> None:
    key = json.loads(PILOT_KEY.read_text(encoding="utf-8"))
    rows = read_review_sheet(PILOT_SHEET)
    tally: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        model = key[row["id"]][row["variant"]]
        if not row["meaning_preserved"]:
            tally[model]["unrated"] += 1
            continue
        tally[model]["rated"] += 1
        tally[model]["meaning_kept"] += is_yes(row["meaning_preserved"])
        tally[model][f"difficulty_{row['difficulty'].casefold() or 'blank'}"] += 1
        tally[model]["auto_fail"] += bool(row["auto_check_problems"])
    for model, counts in tally.items():
        rated = counts["rated"] or 1
        print(f"{model}: meaning kept {counts['meaning_kept']}/{counts['rated']} "
              f"({counts['meaning_kept'] / rated:.0%}), {dict(counts)}")


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def raw_path(dataset: str, split: str) -> Path:
    return config.PARAPHRASED_RAW_DIR / dataset / f"{split}.jsonl"


def paraphrase_split(
    client: OpenRouterClient, model: str, key: str, split: str, failed_ids: dict[str, int], workers: int
) -> bool:
    """Paraphrase what is missing (or rejected) in one dataset split; True if the cap stopped it."""
    originals = read_jsonl(config.SAMPLED_DIR / key / f"{split}.jsonl")
    path = raw_path(key, split)
    previous = {q.id: q for q in read_jsonl(path)} if path.exists() else {}
    todo = []
    for q in originals:
        if q.id in failed_ids:
            next_attempt = failed_ids[q.id] + 1
            if next_attempt < config.MAX_ATTEMPTS:  # else out of attempts; 04 reports it dropped
                todo.append((q, next_attempt))
        elif q.id not in previous:
            todo.append((q, 0))
    done, failed, stopped = run_parallel(
        lambda item: paraphrase(client, model, item[0], item[1]), todo, workers, f"{key}/{split}"
    )
    merged = {**previous, **{q.id: p for (q, _), p in done}}
    written = write_jsonl(path, [merged[q.id] for q in originals if q.id in merged])
    print(f"  {key}/{split}: {written}/{len(originals)} paraphrased, {len(failed)} failed now")
    for (question, _), error in failed:
        print(f"    {question.id}: {error}")
    return stopped


def run_full(datasets: list[str], workers: int, max_cost: float, retry_failed: bool) -> None:
    model = require_paraphrase_model()
    failed_ids: dict[str, int] = {}
    if retry_failed:
        if not FAILED_PATH.exists():
            raise SystemExit(f"{FAILED_PATH} not found; run 04_validate.py first")
        failed_ids = {qid: info.get("attempt", -1) for qid, info in json.loads(FAILED_PATH.read_text(encoding="utf-8")).items()}

    with make_client("paraphrase", max_cost) as client:
        stopped = any(
            paraphrase_split(client, model, key, split, failed_ids, workers)
            for key in datasets
            for split in config.sampled_splits(key)
        )
        if stopped:
            print(f"Stopped at the ${max_cost:.2f} cap; re-run to continue (cached calls are free)")
        print(f"spent this run: ${client.spent_usd:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="pilot both candidate models")
    mode.add_argument("--pilot-results", action="store_true", help="summarise the filled pilot sheet")
    mode.add_argument("--retry-failed", action="store_true", help="regenerate what 04_validate.py rejected")
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=3.0, help="stop after spending this many USD")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(args.workers, min(args.max_cost, 0.25))
    elif args.pilot_results:
        pilot_results()
    else:
        run_full(args.datasets, args.workers, args.max_cost, args.retry_failed)


if __name__ == "__main__":
    main()
