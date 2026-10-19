"""Shared plumbing for the LLM generation scripts (03, 05, 06a, 11a) and review sheets.

- `make_client`: OpenRouter client with this part's cache, call log and a cap.
- `chat_json` (re-exported from jev_common.generation): one chat request that
  must answer with a JSON object.
- `run_parallel` (re-exported from jev_common.parallel): run a function over
  many items in threads, keep going past single failures, stop on the spending cap.
- Review sheets (re-exported from jev_common.generation): the CSVs in
  data/reviews/ that you fill in by hand. They are never overwritten once
  written, so a filled-in sheet cannot be lost.
"""

import json
from collections.abc import Sequence

from jev_common.cache import DiskCache
from jev_common.generation import (  # noqa: F401 - re-exported for the scripts
    chat_json,
    is_yes,
    parse_json_reply,
    read_review_sheet,
    write_review_sheet,
)
from jev_common.openrouter import OpenRouterClient
from jev_common.parallel import run_parallel  # noqa: F401 - re-exported for the scripts

from .config import CACHE_DIR, LOG_DIR
from .labels import letter_from_position
from .schema import Question


def make_client(namespace: str, max_cost_usd: float) -> OpenRouterClient:
    return OpenRouterClient(
        cache=DiskCache(namespace, CACHE_DIR), log_dir=LOG_DIR, max_cost_usd=max_cost_usd
    )


# ---------------------------------------------------------------------------
# Rendering and review sheets
# ---------------------------------------------------------------------------
def numbered_options(options: Sequence[str]) -> str:
    """Options as "A. ..." lines, for prompts and review sheets."""
    return "\n".join(f"{letter_from_position(i)}. {text}" for i, text in enumerate(options))


def instance_text(question: Question) -> str:
    """One quiz option: a whole question with its options as a), b), ... (A7).

    Lower-case letters, so they don't clash with the quiz's own options A-E.
    """
    options = "\n".join(f"{chr(ord('a') + i)}) {text}" for i, text in enumerate(question.options))
    return f"{question.question}\n{options}"


def question_json(question: Question) -> str:
    """{"question": ..., "options": [...]} in original order, for prompts."""
    return json.dumps({"question": question.question, "options": list(question.options)}, ensure_ascii=False)
