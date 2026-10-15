"""IPIP-NEO-120 items, scoring key and reverse-keyed items (B1).

Items live in data/ipip/ipip_neo_120.json: 120 items, 24 per Big Five domain
(N, E, O, A, C), 4 per facet, each keyed "plus" or "minus". Answers are on the
5-point scale "Very Inaccurate" (1) to "Very Accurate" (5); a minus-keyed item
scores 6 - answer.

Jev asks each item as a `score` question over the 5 scale labels. Its answer
`score` is a probability-weighted position 0-4, so the 1-5 answer is
`score + 1` (fractional answers are kept, not rounded). Checked live
2026-10-15: the answer also has `probabilities` per position ("0".."4"), a
`legend` (position -> label) and `confidence`.
"""

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

PART_DIR = Path(__file__).resolve().parent.parent
IPIP_PATH = PART_DIR / "data" / "ipip" / "ipip_neo_120.json"
DOMAINS = {"N": "Neuroticism", "E": "Extraversion", "O": "Openness", "A": "Agreeableness", "C": "Conscientiousness"}
SCALE_MIN, SCALE_MAX = 1, 5


@dataclass(frozen=True)
class Item:
    id: str  # "ipip-001" ... "ipip-120", in the standard administration order
    text: str
    domain: str  # N, E, O, A or C
    facet: int  # 1-6 within the domain
    facet_name: str
    keyed: str  # "plus" or "minus"

    @property
    def reverse_keyed(self) -> bool:
        return self.keyed == "minus"


@cache
def _instrument(path: Path = IPIP_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path = IPIP_PATH) -> tuple[Item, ...]:
    return tuple(Item(**item) for item in _instrument(path)["items"])


def scale_labels(path: Path = IPIP_PATH) -> tuple[str, ...]:
    return tuple(_instrument(path)["scale"])


def instructions(path: Path = IPIP_PATH) -> str:
    return _instrument(path)["instructions"]


def shuffled_items(items: Sequence[Item], run: int, seed: int) -> list[Item]:
    """Item order for one run (B1 step 1: order shuffled each run), reproducible."""
    order = list(items)
    random.Random(f"{seed}:run:{run}").shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def keyed_score(item: Item, answer: float) -> float:
    """1-5 answer -> trait-direction score (minus-keyed items reversed)."""
    if not SCALE_MIN <= answer <= SCALE_MAX:
        raise ValueError(f"{item.id}: answer must be {SCALE_MIN}-{SCALE_MAX}, got {answer}")
    return SCALE_MIN + SCALE_MAX - answer if item.reverse_keyed else answer


def domain_scores(items: Sequence[Item], answers: Mapping[str, float]) -> dict[str, float]:
    """Mean keyed score per domain (1-5) over the items answered.

    `answers` maps item ID to the 1-5 answer; missing items are skipped, so
    report how many were answered alongside.
    """
    totals: dict[str, list[float]] = defaultdict(list)
    for item in items:
        if item.id in answers:
            totals[item.domain].append(keyed_score(item, answers[item.id]))
    return {domain: sum(values) / len(values) for domain, values in totals.items()}


def facet_scores(items: Sequence[Item], answers: Mapping[str, float]) -> dict[tuple[str, int], float]:
    """Mean keyed score per (domain, facet)."""
    totals: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in items:
        if item.id in answers:
            totals[(item.domain, item.facet)].append(keyed_score(item, answers[item.id]))
    return {key: sum(values) / len(values) for key, values in totals.items()}


# ---------------------------------------------------------------------------
# Jev score questions
# ---------------------------------------------------------------------------
def jev_score_question(item: Item, path: Path = IPIP_PATH) -> dict[str, Any]:
    """One Jev `score` question for an item, over the 5 scale labels (low to high)."""
    return {
        "type": "score",
        "instructions": f'{instructions(path)} How accurately does this describe you: "{item.text}"?',
        "criteria": list(scale_labels(path)),
    }


def answer_from_jev(score_answer: Mapping[str, Any]) -> float:
    """Jev score answer (position 0-4) -> 1-5 answer."""
    if score_answer.get("type") != "score":
        raise ValueError(f"expected a score answer, got {score_answer.get('type')!r}")
    return float(score_answer["score"]) + SCALE_MIN
