"""Generator-model plumbing and review sheets, shared by both parts.

- `chat_json`: one chat request that must answer with a JSON object.
- Review sheets: CSVs you fill in by hand. They are never overwritten once
  written, so a filled-in sheet cannot be lost.

Part A re-exports these from helpers/generation.py; Part B imports them here.
"""

import csv
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .models import GENERATION_PARAMS
from .openrouter import CallResult, OpenRouterClient

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


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
    client: OpenRouterClient, model: str, prompt: str, attempt: int = 0, **overrides: Any
) -> tuple[dict[str, Any], CallResult]:
    """Send `prompt` as one user message and parse the JSON object it answers with.

    `attempt` > 0 sends the same prompt again as a fresh request (new cache
    entry), for regenerating items that failed validation. `overrides`
    replace the model's GENERATION_PARAMS (e.g. a lower temperature).
    """
    params = {**GENERATION_PARAMS.get(model, {}), **overrides, "response_format": {"type": "json_object"}}
    result = client.chat(
        model,
        [{"role": "user", "content": prompt}],
        cache_tag=f"attempt-{attempt}" if attempt else None,
        **params,
    )
    message = result.response["choices"][0]["message"]
    return parse_json_reply(message.get("content")), result


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
