"""Seeded balanced shuffle (A2 rule 3).

The correct answer's position is balanced within each option-count group
(3, 4 and 5 options), so a model cannot gain from a position preference. Also
used to balance where the original sits among options A-D in the quiz (A7).
"""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from .schema import Question


def balanced_positions(count: int, n_options: int, rng: random.Random) -> list[int]:
    """`count` target positions in `range(n_options)`, as even as possible, in random order.

    Every position appears `count // n_options` times, and `count % n_options`
    randomly chosen positions appear once more.
    """
    if count < 0 or n_options < 1:
        raise ValueError(f"need count >= 0 and n_options >= 1, got {count}, {n_options}")
    positions = [i % n_options for i in range(count)]
    # Cycling fills positions 0..r-1 with the remainder; pick which ones randomly.
    remainder = count % n_options
    if remainder:
        extra = rng.sample(range(n_options), remainder)
        positions = positions[: count - remainder] + extra
    rng.shuffle(positions)
    return positions


def shuffled_order_for(question: Question, target_position: int, seed: int) -> tuple[int, ...]:
    """A permutation that puts the correct option at `target_position`.

    The other options fill the remaining slots in an order that depends only
    on `seed` and the question ID, so it is stable if other questions change.
    """
    if not 0 <= target_position < question.n_options:
        raise ValueError(f"{question.id}: target position {target_position} is out of range")
    rng = random.Random(f"{seed}:{question.id}:{question.version}")
    others = [i for i in range(question.n_options) if i != question.answer_position]
    rng.shuffle(others)
    others_iter = iter(others)
    return tuple(
        question.answer_position if slot == target_position else next(others_iter)
        for slot in range(question.n_options)
    )


def balanced_shuffle(questions: Sequence[Question], seed: int) -> list[Question]:
    """Return copies with `shuffled_order` set, in the same order as the input.

    Balance is computed within each option-count group among the questions
    passed in, so call it once per dataset and split. Deterministic for a
    given seed and set of questions, whatever order they arrive in.
    """
    by_count: dict[int, list[Question]] = defaultdict(list)
    for question in questions:
        by_count[question.n_options].append(question)

    orders: dict[tuple[str, str], tuple[int, ...]] = {}
    for n_options in sorted(by_count):
        group = sorted(by_count[n_options], key=lambda q: (q.id, q.version))
        rng = random.Random(f"{seed}:group:{n_options}")
        targets = balanced_positions(len(group), n_options, rng)
        for question, target in zip(group, targets, strict=True):
            orders[(question.id, question.version)] = shuffled_order_for(question, target, seed)

    return [replace(q, shuffled_order=orders[(q.id, q.version)]) for q in questions]


def position_counts(questions: Sequence[Question]) -> dict[int, Counter[int]]:
    """Correct-answer position counts in the shuffled order, per option count."""
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for question in questions:
        counts[question.n_options][question.shuffled_order.index(question.answer_position)] += 1
    return dict(counts)
