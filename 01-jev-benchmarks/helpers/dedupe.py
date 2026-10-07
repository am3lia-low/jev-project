"""Drop SWAG training questions that overlap HellaSwag test questions (A1).

HellaSwag and SWAG both draw on ActivityNet Captions, so a SWAG training
question can share its source video, or even its context text, with a
HellaSwag test question. Such questions must not reach BERT's training set.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import Question

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_context(text: str) -> str:
    """Case, punctuation and spacing-insensitive form of a context for matching."""
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


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
    Video IDs are case-sensitive (YouTube IDs); contexts are compared after
    `normalise_context`.
    """
    video_ids = {q.meta["video_id"] for q in hellaswag_test if q.meta.get("video_id")}
    contexts = {normalise_context(q.meta["context"]) for q in hellaswag_test if q.meta.get("context")}
    contexts.discard("")

    kept: list[Question] = []
    by_video: list[Question] = []
    by_context: list[Question] = []
    for question in swag:
        if question.meta.get("video_id") in video_ids:
            by_video.append(question)
        elif normalise_context(question.meta.get("context", "")) in contexts:
            by_context.append(question)
        else:
            kept.append(question)
    return DedupeResult(kept, by_video, by_context)
