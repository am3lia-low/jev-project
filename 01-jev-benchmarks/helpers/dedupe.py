"""Keep test questions out of training data (A1, A2 rule 2).

- `drop_test_overlap`: within one dataset, drop train/val candidates whose
  question text matches a test question. ARC's official splits share a few
  question stems (sometimes with the same options), so splitting by row is
  not enough.
- `dedupe_swag`: HellaSwag and SWAG both draw on ActivityNet Captions, so a
  SWAG training question can share its source video, or even its context
  text, with a HellaSwag test question. Such questions must not reach BERT's
  training set.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import Question

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_context(text: str) -> str:
    """Case, punctuation and spacing-insensitive form of a context for matching."""
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


def drop_test_overlap(
    pool: Sequence[Question], test: Sequence[Question]
) -> tuple[list[Question], list[Question]]:
    """Split `pool` into (kept, dropped): dropped questions share their text with a test question.

    Text is compared after `normalise_context`, and options are ignored: the
    same question stem with different options still teaches the test item.
    """
    test_texts = {normalise_context(q.question) for q in test}
    kept: list[Question] = []
    dropped: list[Question] = []
    for question in pool:
        (dropped if normalise_context(question.question) in test_texts else kept).append(question)
    return kept, dropped


@dataclass(frozen=True)
class DedupeResult:
    kept: list[Question]
    dropped_by_video: list[Question]
    dropped_by_context: list[Question]  # context matched, video ID did not

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_by_video) + len(self.dropped_by_context)


def dedupe_swag(swag: Sequence[Question], hellaswag_test: Sequence[Question]) -> DedupeResult:
    """Split `swag` into questions to keep and questions that overlap `hellaswag_test`.

    Needs the `video_id` and `context` fields that `loaders` stores in `meta`.
    `loaders` puts both datasets' video IDs in one form (bare YouTube ID,
    casefolded, since SWAG stores them lowercased); contexts are compared
    after `normalise_context`.
    """
    video_ids = {q.meta["video_id"] for q in hellaswag_test if q.meta.get("video_id")}
    contexts = {normalise_context(q.meta["context"]) for q in hellaswag_test if q.meta.get("context")}
    contexts.discard("")

    kept: list[Question] = []
    by_video: list[Question] = []
    by_context: list[Question] = []
    for question in swag:
        if question.meta.get("video_id") and question.meta["video_id"] in video_ids:
            by_video.append(question)
        elif normalise_context(question.meta.get("context", "")) in contexts:
            by_context.append(question)
        else:
            kept.append(question)
    return DedupeResult(kept, by_video, by_context)
