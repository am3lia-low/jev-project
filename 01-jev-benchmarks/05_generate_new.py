"""Draft brand-new questions in each paper dataset's style, then apply your review (A3.2).

Drafting: NEW_QUESTION_MODEL writes N_NEW_PER_DATASET questions per paper
dataset, N_NEW_PER_CALL at a time, shown N_STYLE_EXAMPLES sampled *train*
questions for style and the topics already drafted (to avoid repeats). Each
draft is compared with every question in the dataset's official splits: drafts
at NEAR_COPY_REJECT or above are discarded and replaced (the model tends to
reproduce textbook questions that the benchmark already contains); for the
rest, the most similar question and its similarity are recorded so possible
near-copies stand out in the review.
Writes `data/new/drafts.jsonl` and the review sheet
`data/reviews/new_questions_400.csv` (never overwritten).

Review: in the sheet, fix wording, options or answer in place where needed and
fill in answer_correct, one_correct_option, not_near_copy and keep (yes/no).
`--apply-review` then writes the kept questions, with a balanced shuffle, to
`data/new/reviewed.jsonl` (Level 3).

Run from this folder on the API laptop:
    python 05_generate_new.py
    python 05_generate_new.py --apply-review
"""

import argparse
import difflib
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from helpers import config
from helpers.dedupe import normalise_context
from helpers.generation import (
    chat_json,
    is_yes,
    make_client,
    numbered_options,
    read_review_sheet,
    write_review_sheet,
)
from helpers.ids import make_id
from helpers.labels import LETTERS, letter_from_position, normalise_label
from helpers.loaders import load_questions
from helpers.prompts import fill
from helpers.schema import Question, read_jsonl, write_jsonl
from helpers.shuffle import balanced_shuffle
from jev_common.models import NEW_QUESTION_MODEL
from jev_common.openrouter import BudgetExceeded, OpenRouterClient

DRAFTS_PATH = config.NEW_DIR / "drafts.jsonl"
SHEET = config.REVIEW_DIR / "new_questions_400.csv"
N_OPTIONS = {"hs": 4, "arce": 4, "arcc": 4, "csqa": 5}  # ARC is mostly 4; drafts use 4
MAX_EXTRA_CALLS = 10  # per dataset, to replace malformed, repeated or near-copy drafts
SOURCE_SPLITS = ("train", "validation", "test")
_WORD = re.compile(r"[a-z0-9]+")


class BenchmarkIndex:
    """Every labelled question in one dataset's official splits, for near-copy search."""

    def __init__(self, key: str) -> None:
        self.pool: list[Question] = []
        for split in SOURCE_SPLITS:
            try:
                self.pool += load_questions(key, split, "test")
            except (ValueError, KeyError):  # hidden-label test splits
                continue
            except Exception as error:  # noqa: BLE001 - e.g. a split that does not exist
                print(f"  {key} {split}: skipped ({type(error).__name__})")
        self.texts = [_text(q) for q in self.pool]
        self.document_frequency: Counter[str] = Counter()
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(self.texts):
            for word in set(_WORD.findall(text)):
                self.document_frequency[word] += 1
                self.index[word].append(i)

    def nearest(self, question: Question) -> tuple[Question | None, float]:
        """Most similar benchmark question (question + correct option) and its similarity 0-1."""
        text = _text(question)
        words = sorted(set(_WORD.findall(text)), key=lambda w: self.document_frequency.get(w, 0))
        candidates = {i for word in words[:4] for i in self.index.get(word, [])}
        best, best_score = None, 0.0
        for i in candidates:
            score = difflib.SequenceMatcher(None, text, self.texts[i], autojunk=False).ratio()
            if score > best_score:
                best, best_score = i, score
        return (self.pool[best] if best is not None else None), best_score


def draft_dataset(client: OpenRouterClient, key: str) -> tuple[list[Question], int]:
    """Sequential calls for one dataset, so each call sees the topics already used.

    Returns the accepted drafts and how many near-copies were discarded.
    """
    benchmark = BenchmarkIndex(key)
    spec = config.DATASETS[key]
    n_options = N_OPTIONS[key]
    train = [q for q in read_jsonl(config.SAMPLED_DIR / key / "train.jsonl") if q.n_options == n_options]
    drafts: list[Question] = []
    rejected: list[str] = []
    seen: set[str] = set()
    n_calls = config.N_NEW_PER_DATASET // config.N_NEW_PER_CALL + MAX_EXTRA_CALLS
    for call in range(n_calls):
        if len(drafts) >= config.N_NEW_PER_DATASET:
            break
        rng = random.Random(f"{config.SAMPLE_SEED}:new:{key}:{call}")
        examples = rng.sample(train, config.N_STYLE_EXAMPLES)
        prompt = fill(
            "generate_new/_common",
            n=config.N_NEW_PER_CALL,
            dataset_name=spec.name,
            style=fill(f"generate_new/{key}"),
            n_examples=len(examples),
            examples="\n\n".join(
                f"{q.question}\n{numbered_options(q.options)}\nAnswer: {q.answer}" for q in examples
            ),
            n_options=n_options,
            avoid="; ".join([q.question[:60] for q in drafts] + rejected) or "none yet",
        )
        data, result = chat_json(client, NEW_QUESTION_MODEL, prompt)
        for raw in data.get("questions") or []:
            try:
                question = " ".join(str(raw["question"]).split())
                options = tuple(" ".join(str(o).split()) for o in raw["options"])
                answer = normalise_label(raw["answer"])
            except (KeyError, TypeError, ValueError):
                continue
            norm = normalise_context(question)
            if len(options) != n_options or LETTERS.index(answer) >= n_options or norm in seen:
                continue
            seen.add(norm)
            draft = Question(
                id=make_id(key, "new", len(drafts)),
                dataset=key,
                split="test",
                question=question,
                options=options,
                answer=answer,
                meta={"model": NEW_QUESTION_MODEL, "call": call, "cost_usd": result.cost_usd / config.N_NEW_PER_CALL},
            )
            nearest, similarity = benchmark.nearest(draft)
            if similarity >= config.NEAR_COPY_REJECT:
                rejected.append(question[:60])
                continue
            draft.meta.update(
                nearest_id=nearest.id if nearest else "",
                nearest_text=f"{nearest.question} / {nearest.options[nearest.answer_position]}" if nearest else "",
                similarity=round(similarity, 3),
            )
            drafts.append(draft)
            if len(drafts) >= config.N_NEW_PER_DATASET:
                break
    return drafts, len(rejected)


def _text(question: Question) -> str:
    return normalise_context(f"{question.question} {question.options[question.answer_position]}")


def generate(datasets: list[str], max_cost: float) -> None:
    with make_client("generate_new", max_cost) as client, ThreadPoolExecutor(len(datasets)) as pool:
        futures = {key: pool.submit(draft_dataset, client, key) for key in datasets}
        drafts: list[Question] = []
        for key, future in futures.items():
            try:
                found, n_rejected = future.result()
            except BudgetExceeded:
                raise SystemExit(f"stopped at the ${max_cost:.2f} cap; re-run to continue (cached calls are free)")
            note = "" if len(found) >= config.N_NEW_PER_DATASET else " (short: raise MAX_EXTRA_CALLS and re-run)"
            print(f"{config.DATASETS[key].name}: {len(found)} drafts, {n_rejected} near-copies discarded{note}")
            drafts += found
        print(f"spent this run: ${client.spent_usd:.4f}")
    write_jsonl(DRAFTS_PATH, drafts)
    flagged = sum(d.meta["similarity"] >= config.NEAR_COPY_SIMILARITY for d in drafts)
    print(f"Wrote {DRAFTS_PATH}: {len(drafts)} drafts, {flagged} flagged as possible near-copies")

    rows = [{
        "id": d.id,
        "dataset": d.dataset,
        "question": d.question,
        **{f"option_{letter_from_position(i)}": (d.options[i] if i < d.n_options else "") for i in range(len(LETTERS))},
        "answer": d.answer,
        "nearest_benchmark": d.meta["nearest_text"],
        "similarity": d.meta["similarity"],
        "possible_near_copy": "YES" if d.meta["similarity"] >= config.NEAR_COPY_SIMILARITY else "",
        "answer_correct": "",
        "one_correct_option": "",
        "not_near_copy": "",
        "keep": "",
        "notes": "",
    } for d in drafts]
    if write_review_sheet(SHEET, list(rows[0]), rows):
        print(f"Wrote {SHEET}: check every row, edit text in place, fill in the yes/no columns")
    else:
        print(f"{SHEET} already exists; left untouched")


def apply_review() -> None:
    drafts = {d.id: d for d in read_jsonl(DRAFTS_PATH)}
    kept: list[Question] = []
    problems = []
    for row in read_review_sheet(SHEET):
        if not is_yes(row["keep"]):
            continue
        unchecked = [c for c in ("answer_correct", "one_correct_option", "not_near_copy") if not is_yes(row[c])]
        if unchecked:
            problems.append(f"{row['id']}: keep=yes but not yes in {unchecked}")
            continue
        options = tuple(row[f"option_{letter}"] for letter in LETTERS if row[f"option_{letter}"])
        try:
            kept.append(Question(
                id=row["id"],
                dataset=row["dataset"],
                split="test",
                question=" ".join(row["question"].split()),
                options=options,
                answer=normalise_label(row["answer"]),
                meta={**drafts[row["id"]].meta, "edited": row["question"] != drafts[row["id"]].question
                      or options != drafts[row["id"]].options},
            ))
        except ValueError as error:
            problems.append(f"{row['id']}: {error}")
    if problems:
        raise SystemExit("Fix these rows in the sheet first:\n  " + "\n  ".join(problems))

    by_dataset: dict[str, list[Question]] = defaultdict(list)
    for question in kept:
        by_dataset[question.dataset].append(question)
    reviewed = [q for key in config.PAPER_DATASETS for q in balanced_shuffle(by_dataset[key], config.SHUFFLE_SEED)]
    write_jsonl(config.NEW_REVIEWED_PATH, reviewed)
    for key in config.PAPER_DATASETS:
        count = len(by_dataset[key])
        note = "" if count >= config.N_NEW_PER_DATASET else f" (short of {config.N_NEW_PER_DATASET}; draft more)"
        print(f"{config.DATASETS[key].name}: {count} kept{note}")
    print(f"Wrote {config.NEW_REVIEWED_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply-review", action="store_true", help="write reviewed.jsonl from the filled sheet")
    parser.add_argument("--datasets", nargs="+", choices=config.PAPER_DATASETS, default=list(config.PAPER_DATASETS))
    parser.add_argument("--max-cost", type=float, default=2.0, help="stop after spending this many USD")
    args = parser.parse_args()
    if args.apply_review:
        apply_review()
    else:
        generate(args.datasets, args.max_cost)


if __name__ == "__main__":
    main()
