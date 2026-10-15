"""Big Five persona conditions and their wordings (B1 step 3).

A condition is the baseline (no persona) or one trait pushed high or low,
with the persona text placed in the Jev state or in the question
instructions. Each persona condition has N_PERSONA_VARIANTS wordings: a seeded
draw of N_PERSONA_ADJECTIVES Goldberg markers for that pole (markers.json),
filled into persona.txt, e.g. "You are a person who is very talkative, very
bold, ...". The draw depends only on (trait, pole, variant), so both
placements and every model see the same wordings.
"""

import json
import random
from dataclasses import dataclass
from functools import cache
from typing import Any

from .config import (
    BASELINE,
    N_PERSONA_ADJECTIVES,
    N_PERSONA_VARIANTS,
    PERSONA_QUALIFIER,
    PERSONA_SEED,
    PERSONAS_DIR,
    PLACEMENTS,
    POLES,
)
from .ipip import DOMAINS


@dataclass(frozen=True)
class Condition:
    domain: str | None  # N, E, O, A or C; None for the baseline
    pole: str | None  # "high" or "low"
    placement: str | None  # "state" or "instructions"

    @property
    def name(self) -> str:
        """File name of the condition's results: "baseline" or e.g. "E-high-state"."""
        return BASELINE if self.domain is None else f"{self.domain}-{self.pole}-{self.placement}"

    @property
    def is_baseline(self) -> bool:
        return self.domain is None


BASELINE_CONDITION = Condition(None, None, None)


def persona_conditions(
    domains: tuple[str, ...] = tuple(DOMAINS), placements: tuple[str, ...] = PLACEMENTS
) -> list[Condition]:
    return [Condition(d, pole, p) for d in domains for pole in POLES for p in placements]


@cache
def _markers() -> dict[str, Any]:
    return json.loads((PERSONAS_DIR / "markers.json").read_text(encoding="utf-8"))


def markers_verified() -> bool:
    return bool(_markers().get("verified"))


def persona_text(domain: str, pole: str, variant: int) -> str:
    """Wording `variant` (1..N_PERSONA_VARIANTS) of the persona for `domain` pushed to `pole`."""
    if not 1 <= variant <= N_PERSONA_VARIANTS:
        raise ValueError(f"variant must be 1-{N_PERSONA_VARIANTS}, got {variant}")
    pool = _markers()["markers"][domain][pole]
    if len(pool) < N_PERSONA_ADJECTIVES:
        raise ValueError(f"{domain}-{pole}: {len(pool)} markers, need {N_PERSONA_ADJECTIVES}")
    adjectives = random.Random(f"{PERSONA_SEED}:{domain}:{pole}:{variant}").sample(pool, N_PERSONA_ADJECTIVES)
    words = [f"{PERSONA_QUALIFIER} {a}" for a in adjectives]
    listed = ", ".join(words[:-1]) + f" and {words[-1]}"
    template = (PERSONAS_DIR / "persona.txt").read_text(encoding="utf-8").strip()
    return template.format(adjectives=listed)
