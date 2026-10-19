"""Write the scenario questions for the behaviour check (B1 step 5).

Self-reports often fail to predict what a model does (The Personality
Illusion), so B1 step 5 asks whether a persona also moves behaviour. Each
scenario is an everyday situation with two acceptable actions: one a person
high in an IPIP facet would take, one a person low in it would take.

Two steps:
    generate        NEW_QUESTION_MODEL writes N_SCENARIO_DRAFTS_PER_FACET drafts
                    per facet (prompts/scenarios.txt; 30 calls, about $0.30, API
                    laptop). Automatic checks flag problems; the review sheet
                    data/reviews/scenarios.csv lists every draft.
    --apply-review  After you fill in the sheet: keeps the first
                    N_SCENARIOS_PER_FACET kept drafts per facet, balances which
                    action is shown as A (seeded, half per trait) and writes
                    data/scenarios/scenarios.json, marked verified.

The review sheet, per row: `keep` y/n. Keep a draft if `high` is what a person
high in the facet would do, `low` what a person low in it would do, and
neither action is plainly better or worse. You may fix the wording in the
situation/high/low cells; the edited text is used. A sheet is never
overwritten once written; delete it to start again.

Run from this folder:
    python 07_make_scenarios.py                  # API laptop
    python 07_make_scenarios.py --apply-review   # any laptop
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from string import Template
from typing import Any

from helpers.bigfive import make_client, write_records
from helpers.config import (
    N_SCENARIO_DRAFTS_PER_FACET,
    N_SCENARIOS_PER_FACET,
    PERSONAS_DIR,
    PROMPTS_DIR,
    SCENARIO_DRAFTS_PATH,
    SCENARIO_MAX_LENGTH_RATIO,
    SCENARIO_REVIEW_PATH,
    SCENARIO_SEED,
    SCENARIOS_PATH,
)
from helpers.ipip import DOMAINS, load_items
from jev_common.generation import chat_json, is_yes, read_review_sheet, write_review_sheet
from jev_common.models import NEW_QUESTION_MODEL
from jev_common.parallel import run_parallel

DOMAIN_DESCRIPTIONS = {
    "N": "how easily a person feels negative emotions such as worry, anger and sadness",
    "E": "how much a person seeks company, activity and attention",
    "O": "how much a person seeks new ideas, experiences and imagination",
    "A": "how much a person puts others' needs and harmony first",
    "C": "how organised, careful and driven a person is",
}
SHEET_FIELDS = ["scenario_id", "domain", "facet_name", "situation", "high", "low", "flags", "keep", "notes"]
TRAIT_WORDS = {"extravert", "extraverted", "introvert", "introverted", "outgoing", "shy", "anxious", "nervous",
               "calm", "organised", "organized", "careless", "kind", "curious", "agreeable", "neurotic",
               "conscientious", "open-minded"}


def facets() -> list[tuple[str, int, str, list[str]]]:
    """(domain, facet, facet name, item texts) for the 30 IPIP facets, in instrument order."""
    by_facet: dict[tuple[str, int], list] = defaultdict(list)
    for item in load_items():
        by_facet[(item.domain, item.facet)].append(item)
    return [(d, f, items[0].facet_name, [i.text for i in items]) for (d, f), items in sorted(
        by_facet.items(), key=lambda kv: ("NEOAC".index(kv[0][0]), kv[0][1]))]


def flags(draft: dict[str, str]) -> list[str]:
    found = []
    high, low = draft["high"], draft["low"]
    if high.casefold() == low.casefold():
        found.append("the two actions are the same")
    if min(len(high), len(low)) and max(len(high), len(low)) / min(len(high), len(low)) > SCENARIO_MAX_LENGTH_RATIO:
        found.append("actions differ a lot in length")
    markers = json.loads((PERSONAS_DIR / "markers.json").read_text(encoding="utf-8"))["markers"]
    words = TRAIT_WORDS | {m.casefold() for poles in markers.values() for pool in poles.values() for m in pool}
    used = sorted(w for w in words if re.search(rf"\b{re.escape(w)}\b", " ".join(draft.values()).casefold()))
    if used:
        found.append(f"trait words: {', '.join(used)}")
    if not draft["situation"].lower().startswith("you") and " you" not in draft["situation"].lower():
        found.append("situation not in the second person")
    return found


def generate_facet(client, domain: str, facet: int, facet_name: str, items: list[str]) -> list[dict[str, Any]]:
    prompt = Template((PROMPTS_DIR / "scenarios.txt").read_text(encoding="utf-8")).substitute(
        domain_name=DOMAINS[domain], domain_description=DOMAIN_DESCRIPTIONS[domain], facet_name=facet_name,
        facet_items="\n".join(f"- {text}" for text in items), n=N_SCENARIO_DRAFTS_PER_FACET)
    reply, _result = chat_json(client, NEW_QUESTION_MODEL, prompt)
    drafts = []
    for number, s in enumerate(reply.get("scenarios", []), start=1):
        draft = {k: " ".join(str(s.get(k, "")).split()) for k in ("situation", "high", "low")}
        if not all(draft.values()):
            continue
        drafts.append({"scenario_id": f"scn-{domain}{facet}-{number}", "domain": domain, "facet": facet,
                       "facet_name": facet_name, **draft, "flags": flags(draft), "model": NEW_QUESTION_MODEL})
    if len(drafts) < N_SCENARIOS_PER_FACET:
        raise ValueError(f"{domain}{facet}: only {len(drafts)} usable drafts")
    return drafts


def generate(max_cost: float, workers: int) -> None:
    todo = facets()
    with make_client("scenarios", max_cost) as client:
        done, failed, stopped = run_parallel(lambda f: generate_facet(client, *f), todo, workers, "facets")
        print(f"spent this run: ${client.spent_usd:.4f}")
    if stopped or failed:
        for f, error in failed:
            print(f"  {f[0]}{f[1]}: {error}")
        raise SystemExit("Not every facet has drafts; re-run (finished calls come from the cache)")
    order = {(d, f): i for i, (d, f, _, _) in enumerate(todo)}
    drafts = [d for _, batch in sorted(done, key=lambda pair: order[pair[0][:2]]) for d in batch]
    write_records(SCENARIO_DRAFTS_PATH, drafts)
    print(f"{len(drafts)} drafts -> {SCENARIO_DRAFTS_PATH} ({sum(bool(d['flags']) for d in drafts)} flagged)")
    rows = [{**{k: d[k] for k in ("scenario_id", "domain", "facet_name", "situation", "high", "low")},
             "flags": "; ".join(d["flags"]), "keep": "", "notes": ""} for d in drafts]
    if write_review_sheet(SCENARIO_REVIEW_PATH, SHEET_FIELDS, rows):
        print(f"Review sheet: {SCENARIO_REVIEW_PATH}. Fill in keep (y/n), then --apply-review.")
    else:
        print(f"{SCENARIO_REVIEW_PATH} already exists and was left alone; delete it to write a fresh one.")


def apply_review() -> None:
    if not SCENARIO_REVIEW_PATH.exists():
        raise SystemExit(f"{SCENARIO_REVIEW_PATH} not found; run the generate step first")
    rows = read_review_sheet(SCENARIO_REVIEW_PATH)
    empty = [r["scenario_id"] for r in rows if not r["keep"]]
    if empty:
        raise SystemExit(f"Review not finished: keep is empty for {len(empty)} drafts, e.g. {empty[:8]}")
    facet_of = {(d, f): name for d, f, name, _ in facets()}
    kept: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        domain, facet = row["scenario_id"].split("-")[1][0], int(row["scenario_id"].split("-")[1][1:])
        if is_yes(row["keep"]) and len(kept[(domain, facet)]) < N_SCENARIOS_PER_FACET:
            kept[(domain, facet)].append({**row, "domain": domain, "facet": facet})
    short = {f"{d}{f}": len(kept[(d, f)]) for d, f in facet_of if len(kept[(d, f)]) < N_SCENARIOS_PER_FACET}
    if short:
        print(f"Warning: fewer than {N_SCENARIOS_PER_FACET} kept for facets {short}; they are used as they are")
    scenarios = []
    for domain in DOMAINS:
        chosen = [r for (d, f), batch in sorted(kept.items()) if d == domain for r in batch]
        high_first = [i < len(chosen) // 2 + len(chosen) % 2 for i in range(len(chosen))]
        random.Random(f"{SCENARIO_SEED}:{domain}").shuffle(high_first)
        for row, first in zip(chosen, high_first):
            options = (row["high"], row["low"]) if first else (row["low"], row["high"])
            scenarios.append({"id": row["scenario_id"], "domain": domain, "facet": row["facet"],
                              "facet_name": facet_of[(domain, row["facet"])], "situation": row["situation"],
                              "options": list(options), "high_letter": "A" if first else "B"})
    out = {"description": "Scenario questions for the behaviour check (PLAN B1 step 5)",
           "generated_by": NEW_QUESTION_MODEL, "reviewed": "data/reviews/scenarios.csv", "verified": True,
           "scenarios": scenarios}
    SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENARIOS_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    per_domain = Counter(s["domain"] for s in scenarios)
    high_a = Counter(s["domain"] for s in scenarios if s["high_letter"] == "A")
    print(f"{len(scenarios)} scenarios -> {SCENARIOS_PATH}; per trait (high action shown as A): "
          + ", ".join(f"{d} {per_domain[d]} ({high_a[d]})" for d in DOMAINS))
    print("Next (API laptop): python 08_run_scenarios.py estimate, then check, baseline, personas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply-review", action="store_true", help="write scenarios.json from the review sheet")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-cost", type=float, default=1.0, help="stop after spending this many USD")
    args = parser.parse_args()
    if args.apply_review:
        apply_review()
    else:
        generate(args.max_cost, args.workers)


if __name__ == "__main__":
    main()
