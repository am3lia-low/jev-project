"""Paraphrase the 120 IPIP-NEO-120 items for the contamination check (B1 step 4).

If a model has seen the IPIP items (they are public and widely copied), it may
answer the original wording from memory. A paraphrase keeps each item's
meaning and keying but changes its words; 01_run_bigfive.py --items
paraphrased then runs it, and 06_analyse_paraphrase.py compares the two.

Two steps:
    generate        PARAPHRASE_MODEL rewords each item (prompts/ipip_paraphrase.txt),
                    automatic checks flag suspect ones, and the review sheet
                    data/reviews/ipip_paraphrases.csv is written (all 120 rows).
                    About 120 short calls, around $0.01 (API laptop).
    --apply-review  After you fill in the sheet: writes
                    data/ipip/ipip_neo_120_paraphrased.json, marked verified.
                    01_run_bigfive.py --items paraphrased refuses to run before this.

The review sheet, per row: `same_meaning` y/n (same meaning, strength and
direction as the original); if n, write a replacement in `edit` (used as is).
A sheet is never overwritten once written; delete it to start again.

Run from this folder:
    python 05_paraphrase_ipip.py                  # API laptop
    python 05_paraphrase_ipip.py --apply-review   # any laptop
"""

import argparse
import json
import re
from string import Template
from typing import Any

from helpers.bigfive import make_client, read_records, write_records
from helpers.config import (
    IPIP_PARAPHRASE_DRAFTS_PATH,
    IPIP_PARAPHRASED_PATH,
    IPIP_REVIEW_PATH,
    PROMPTS_DIR,
)
from helpers.ipip import DOMAINS, IPIP_PATH, Item, load_items
from jev_common.generation import chat_json, is_yes, read_review_sheet, write_review_sheet
from jev_common.models import require_paraphrase_model
from jev_common.parallel import run_parallel

SHEET_FIELDS = ["item_id", "domain", "facet_name", "keyed", "original", "paraphrase", "flags",
                "same_meaning", "edit", "notes"]
STOPWORDS = {"about", "always", "with", "that", "than", "them", "they", "things", "what", "when", "from",
             "have", "into", "other", "others", "people", "some", "very", "your", "just"}
NEGATION = re.compile(r"\b(?:not|never|no)\b|n't\b", re.IGNORECASE)
MAX_ATTEMPTS = 3


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.casefold()) if len(w) > 3 and w not in STOPWORDS}


def checks(item: Item, paraphrase: str) -> list[str]:
    """Problems that make a reply unusable (retried) and flags for the reviewer (kept)."""
    problems = []
    if not paraphrase.strip():
        problems.append("empty")
    elif paraphrase.strip().casefold().rstrip(".") == item.text.casefold().rstrip("."):
        problems.append("same as the original")
    elif re.match(r"^(i|you)\b", paraphrase.strip(), re.IGNORECASE) or paraphrase.strip().endswith("?"):
        problems.append("not in the item form (starts with I/you or is a question)")
    return problems


def flags(item: Item, paraphrase: str) -> list[str]:
    found = []
    original = content_words(item.text)
    if original and len(original & content_words(paraphrase)) / len(original) >= 0.5:
        found.append("keeps half or more of the key words")
    if bool(NEGATION.search(paraphrase)) and not NEGATION.search(item.text):
        found.append("adds a negation")
    if len(paraphrase.split()) > 2 * len(item.text.split()) + 3:
        found.append("much longer")
    return found


def generate_one(client, model: str, item: Item) -> dict[str, Any]:
    prompt = Template((PROMPTS_DIR / "ipip_paraphrase.txt").read_text(encoding="utf-8")).substitute(
        domain_name=DOMAINS[item.domain], facet_name=item.facet_name, text=item.text)
    problems: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        reply, result = chat_json(client, model, prompt, attempt)
        paraphrase = " ".join(str(reply.get("paraphrase", "")).split())
        problems = checks(item, paraphrase)
        if not problems:
            return {"item_id": item.id, "original": item.text, "paraphrase": paraphrase,
                    "flags": flags(item, paraphrase), "model": model, "attempt": attempt}
    raise ValueError(f"{item.id}: no usable paraphrase after {MAX_ATTEMPTS} attempts: {problems}")


def generate(max_cost: float, workers: int) -> None:
    model = require_paraphrase_model()
    items = load_items()
    with make_client("ipip_paraphrase", max_cost) as client:
        done, failed, stopped = run_parallel(lambda item: generate_one(client, model, item), items, workers,
                                             "paraphrases")
        print(f"spent this run: ${client.spent_usd:.4f}")
    if stopped or failed:
        for item, error in failed:
            print(f"  {item.id}: {error}")
        raise SystemExit("Not every item has a paraphrase; re-run (finished calls come from the cache)")
    drafts = sorted((record for _, record in done), key=lambda r: r["item_id"])
    write_records(IPIP_PARAPHRASE_DRAFTS_PATH, drafts)
    n_flagged = sum(bool(d["flags"]) for d in drafts)
    print(f"{len(drafts)} paraphrases -> {IPIP_PARAPHRASE_DRAFTS_PATH} ({n_flagged} flagged for a closer look)")
    by_id = {item.id: item for item in items}
    rows = [{"item_id": d["item_id"], "domain": by_id[d["item_id"]].domain,
             "facet_name": by_id[d["item_id"]].facet_name, "keyed": by_id[d["item_id"]].keyed,
             "original": d["original"], "paraphrase": d["paraphrase"], "flags": "; ".join(d["flags"]),
             "same_meaning": "", "edit": "", "notes": ""} for d in drafts]
    if write_review_sheet(IPIP_REVIEW_PATH, SHEET_FIELDS, rows):
        print(f"Review sheet: {IPIP_REVIEW_PATH}. Fill in same_meaning (and edit where n), then --apply-review.")
    else:
        print(f"{IPIP_REVIEW_PATH} already exists and was left alone; delete it to write a fresh one.")


def apply_review() -> None:
    if not IPIP_REVIEW_PATH.exists():
        raise SystemExit(f"{IPIP_REVIEW_PATH} not found; run the generate step first")
    rows = {row["item_id"]: row for row in read_review_sheet(IPIP_REVIEW_PATH)}
    items = load_items()
    missing = [item.id for item in items if item.id not in rows or not rows[item.id]["same_meaning"]]
    no_edit = [i for i, row in rows.items() if not is_yes(row["same_meaning"]) and row["same_meaning"]
               and not row["edit"]]
    if missing or no_edit:
        raise SystemExit(f"Review not finished: same_meaning empty for {len(missing)} items {missing[:8]}; "
                         f"n without an edit for {no_edit[:8]}")
    instrument = json.loads(IPIP_PATH.read_text(encoding="utf-8"))
    paraphrased, edited = [], 0
    for item in instrument["items"]:
        row = rows[item["id"]]
        text = row["paraphrase"] if is_yes(row["same_meaning"]) else row["edit"]
        edited += not is_yes(row["same_meaning"])
        paraphrased.append({**item, "text": " ".join(text.split())})
    out = {**instrument, "instrument": f"{instrument['instrument']}, items paraphrased (B1 step 4)",
           "paraphrased_by": sorted({r["model"] for r in read_records(IPIP_PARAPHRASE_DRAFTS_PATH)}),
           "reviewed": "data/reviews/ipip_paraphrases.csv", "verified": True,
           "originals": {item["id"]: item["text"] for item in instrument["items"]}, "items": paraphrased}
    IPIP_PARAPHRASED_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(paraphrased)} items ({edited} replaced by your edit) -> {IPIP_PARAPHRASED_PATH}")
    print("Next (API laptop): python 01_run_bigfive.py baseline --items paraphrased, then personas --items paraphrased")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply-review", action="store_true", help="write the paraphrased instrument from the sheet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=0.5, help="stop after spending this many USD")
    args = parser.parse_args()
    if args.apply_review:
        apply_review()
    else:
        generate(args.max_cost, args.workers)


if __name__ == "__main__":
    main()
