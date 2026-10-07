"""Reliability checks for the Big Five measurements (B1 step 2).

- Cronbach's alpha per trait: rows are response sets, columns are that
  trait's items, values are keyed scores. Only traits with alpha >= 0.70 are
  interpreted. Alpha measures whether items move together *across
  respondents whose trait differs*, so the rows must vary in the trait: pool
  every condition (baseline and high/low personas) x run, as Serapio-Garcia
  et al. did across persona prompts. Repeat runs of a single condition differ
  only by noise and give alpha near 0 even for a perfectly consistent model.
- Acquiescence: the mean raw (unreversed) answer across plus- and
  minus-keyed items. A model that agrees with everything scores high on both,
  which reverse keying would otherwise hide as a neutral trait score.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .ipip import Item, keyed_score

ALPHA_THRESHOLD = 0.70
SCALE_MIDPOINT = 3.0


def cronbach_alpha(scores: ArrayLike) -> float:
    """Alpha for a (response sets x items) matrix.

    Returns nan when total-score variance is 0 (e.g. identical answers in
    every run), where alpha is undefined.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(f"need at least 2 runs x 2 items, got shape {matrix.shape}")
    if np.isnan(matrix).any():
        raise ValueError("scores contain missing values; drop incomplete runs first")
    k = matrix.shape[1]
    item_variances = matrix.var(axis=0, ddof=1).sum()
    total_variance = matrix.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_variances / total_variance))


def alpha_per_domain(items: Sequence[Item], runs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Cronbach's alpha per domain; `runs` is one {item ID: 1-5 answer} per response set.

    Pass response sets from all conditions together (see the module docstring).
    """
    alphas = {}
    for domain in sorted({item.domain for item in items}):
        domain_items = [item for item in items if item.domain == domain]
        matrix = [[keyed_score(item, run[item.id]) for item in domain_items] for run in runs]
        alphas[domain] = cronbach_alpha(matrix)
    return alphas


def interpretable(alphas: Mapping[str, float], threshold: float = ALPHA_THRESHOLD) -> dict[str, bool]:
    return {domain: bool(alpha >= threshold) for domain, alpha in alphas.items()}


def acquiescence(items: Sequence[Item], answers: Mapping[str, float]) -> dict[str, float]:
    """Mean raw answer on plus- and minus-keyed items, and their average.

    `bias` above the scale midpoint (3) means a lean towards agreeing
    regardless of content; near 3 means none.
    """
    plus = [answers[i.id] for i in items if i.id in answers and not i.reverse_keyed]
    minus = [answers[i.id] for i in items if i.id in answers and i.reverse_keyed]
    if not plus or not minus:
        raise ValueError("need answers to both plus- and minus-keyed items")
    plus_mean, minus_mean = float(np.mean(plus)), float(np.mean(minus))
    return {"plus_mean": plus_mean, "minus_mean": minus_mean, "bias": (plus_mean + minus_mean) / 2}
