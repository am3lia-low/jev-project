"""Shared plumbing for the LLM generation scripts (03, 05, 06a) and review sheets.

- `make_client`: OpenRouter client with this part's cache, call log and a cap.
- `chat_json`: one chat request that must answer with a JSON object.
- `run_parallel` (re-exported from jev_common.parallel): run a function over
  many items in threads, keep going past single failures, stop on the spending cap.
- Review sheets: the CSVs in data/reviews/ that you fill in by hand. They are
  never overwritten once written, so a filled-in sheet cannot be lost.
"""

import csv
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from jev_common.cache import DiskCache
from jev_common.models import GENERATION_PARAMS
from jev_common.openrouter import CallResult, OpenRouterClient
from jev_common.parallel import run_parallel  # noqa: F401 - re-exported for the scripts

from .config import CACHE_DIR, LOG_DIR
from .labels import letter_from_position
from .schema import Question

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def make_client(namespace: str, max_cost_usd: float) -> OpenRouterClient:
    return OpenRouterClient(
        cache=DiskCache(namespace, CACHE_DIR), log_dir=LOG_DIR, max_cost_usd=max_cost_usd
    )


def parse_json_reply(text: str | None) -> dict[str, Any]:
    """The JSON object in a model reply, tolerating code fences and stray text around it."""
    if not text:
        raise ValueError("empty reply")
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in reply: {text[:200]!r}") from None
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"reply is JSON but not an object: {text[:200]!r}")
    return value


def chat_json(
    client: OpenRouterClient, model: str, prompt: str, attempt: int = 0
) -> tuple[dict[str, Any], CallResult]:
    """Send `prompt` as one user message and parse the JSON object it answers with.

    `attempt` > 0 sends the same prompt again as a fresh request (new cache
    entry), for regenerating items that failed validation.
    """
    params = {**GENERATION_PARAMS.get(model, {}), "response_format": {"type": "json_object"}}
    result = client.chat(
        model,
        [{"role": "user", "content": prompt}],
        cache_tag=f"attempt-{attempt}" if attempt else None,
        **params,
    )
    message = result.response["choices"][0]["message"]
    return parse_json_reply(message.get("content")), result


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


def write_review_sheet(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> bool:
    """Write a CSV for manual review, unless it already exists. Returns True if written.

    UTF-8 with BOM so Excel opens non-ASCII text correctly.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return True


def read_review_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def is_yes(value: str) -> bool:
    return value.strip().casefold() in {"y", "yes", "1", "true", "x"}
