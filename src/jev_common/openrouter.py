"""OpenRouter client: Jev decisions and LLM chat completions (A9).

Two endpoints, checked against the OpenRouter docs on 2026-10-08:

- Jev runs on the Decisions API (`POST /api/alpha/decisions`), not chat
  completions. A request is `{model, state, questions}`; each question has a
  `type` ("choice", "score" or "noul"), `instructions` and `criteria`. A choice
  answer comes back as `{choice, confidence, probabilities: {option: p}}`.
- LLMs (paraphraser, question writer, baselines) run on
  `POST /api/v1/chat/completions`.

Every call is retried with backoff on rate limits, timeouts and server errors,
logged to `<log_dir>/openrouter.jsonl` (no API key, no request text), and,
when a `DiskCache` is given, cached so a re-run only pays for what is missing.
`max_cost_usd` stops a client once the spend it has seen reaches the cap.
"""

import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .cache import DiskCache

BASE_URL = "https://openrouter.ai/api"
DECISIONS_PATH = "/alpha/decisions"
CHAT_PATH = "/v1/chat/completions"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Worth retrying: rate limit, timeouts, provider and server errors. 400 (bad
# request), 401 (key), 402 (out of credits) and 403 are not.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
MAX_ATTEMPTS = 6


class OpenRouterError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"OpenRouter returned {status}: {message}")
        self.status = status


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CallResult:
    response: dict[str, Any]
    latency_s: float  # of the original call, also when served from the cache
    cached: bool

    @property
    def cost_usd(self) -> float:
        return float((self.response.get("usage") or {}).get("cost") or 0.0)


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, OpenRouterError):
        return error.status in RETRY_STATUSES
    return isinstance(error, httpx.TransportError)


class OpenRouterClient:
    """One client per script run; safe to share between threads.

    With several threads the cap can be overshot by up to one call per thread.
    """

    def __init__(
        self,
        cache: DiskCache | None = None,
        log_dir: str | Path | None = None,
        max_cost_usd: float | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        load_dotenv(REPO_ROOT / ".env")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set; add it to .env (API laptop only)")
        self.cache = cache
        self.log_path = Path(log_dir) / "openrouter.jsonl" if log_dir is not None else None
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0  # new (uncached) calls made by this client
        self._lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -----------------------------------------------------------------------
    # Public calls
    # -----------------------------------------------------------------------
    def decide(
        self,
        model: str,
        state: Any,
        questions: Mapping[str, Any],
        cache_tag: str | None = None,
        **extra: Any,
    ) -> CallResult:
        """Jev: one Decisions API request (all questions see the same state).

        `cache_tag` (e.g. "repeat-2") makes an otherwise identical request a
        separate cache entry, so it is really sent again; it is not sent.
        """
        body = {"model": model, "state": state, "questions": dict(questions), **extra}
        return self._call(DECISIONS_PATH, body, cache_tag)

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        cache_tag: str | None = None,
        **params: Any,
    ) -> CallResult:
        """LLM: one chat completion. `params` go into the body (temperature, logprobs, ...)."""
        body = {"model": model, "messages": [dict(m) for m in messages], **params}
        return self._call(CHAT_PATH, body, cache_tag)

    def account(self) -> dict[str, Any]:
        """Usage and limit of this API key (free; not cached or counted)."""
        response = self._http.get("/v1/key")
        response.raise_for_status()
        return response.json()["data"]

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _call(self, path: str, body: dict[str, Any], cache_tag: str | None = None) -> CallResult:
        key: dict[str, Any] = {"path": path, "body": body}
        if cache_tag is not None:
            key["tag"] = cache_tag
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return CallResult(hit["response"], hit["latency_s"], cached=True)

        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of the ${self.max_cost_usd:.2f} cap for this run"
            )

        start = time.perf_counter()
        response = self._post_with_retry(path, body)
        result = CallResult(response, time.perf_counter() - start, cached=False)
        with self._lock:
            self.spent_usd += result.cost_usd
        if self.cache is not None:
            self.cache.put(key, {"response": response, "latency_s": result.latency_s})
        return result

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        reraise=True,
    )
    def _post_with_retry(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        status, payload, error = 0, None, ""
        try:
            response = self._http.post(path, json=body)
            status = response.status_code
            payload = response.json() if response.content else {}
            # Errors can also arrive with a 200 status, inside the body.
            if status != 200 or "error" in payload:
                info = payload.get("error") or {}
                error = str(info.get("message") or response.text[:500])
                raise OpenRouterError(int(info.get("code") or status), error)
            return payload
        except (httpx.TransportError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, json.JSONDecodeError):
                raise OpenRouterError(status or 502, f"unparseable response: {exc}") from exc
            raise
        finally:
            self._log(path, body, status, time.perf_counter() - start, payload, error)

    def _log(
        self,
        path: str,
        body: dict[str, Any],
        status: int,
        latency_s: float,
        payload: dict[str, Any] | None,
        error: str,
    ) -> None:
        if self.log_path is None:
            return
        usage = (payload or {}).get("usage") or {}
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "path": path,
            "model": body.get("model"),
            "status": status,
            "latency_s": round(latency_s, 3),
            "id": (payload or {}).get("id"),
            "served_by": (payload or {}).get("model"),
            "cost_usd": usage.get("cost"),
            "error": error or None,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Reading answers
# ---------------------------------------------------------------------------
def choice_probabilities(response: Mapping[str, Any], question_key: str) -> dict[str, float]:
    """Probabilities of a Jev choice answer, renormalised to sum to 1."""
    answer = response["answers"][question_key]
    if answer.get("type") != "choice":
        raise ValueError(f"answer {question_key!r} is {answer.get('type')!r}, not 'choice'")
    probabilities = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if not probabilities:  # documented as optional: fall back to the pick alone
        return {answer["choice"]: 1.0}
    return _renormalise(probabilities)


def letter_probabilities(response: Mapping[str, Any], letters: Sequence[str]) -> dict[str, float]:
    """Option-letter probabilities from the first generated token (A4, as in the paper).

    Reads `top_logprobs` of the first token, keeps tokens that are one of
    `letters` once stripped (" B" and "B" both count), and renormalises.
    Needs a request with `logprobs=True`, `top_logprobs` and `max_tokens=1`.
    Letters absent from the top list get 0; all zero raises.
    """
    content = ((response["choices"][0].get("logprobs") or {}).get("content")) or []
    if not content:
        raise ValueError("response has no logprobs; did the provider ignore logprobs=True?")
    totals = dict.fromkeys(letters, 0.0)
    for entry in content[0].get("top_logprobs") or []:
        token = entry["token"].strip()
        if token in totals:
            totals[token] += math.exp(entry["logprob"])
    if not any(totals.values()):
        tokens = [e["token"] for e in content[0].get("top_logprobs") or []]
        raise ValueError(f"none of {list(letters)} among the top tokens {tokens}")
    return _renormalise(totals)


def _renormalise(probabilities: Mapping[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError(f"probabilities sum to {total}")
    return {k: v / total for k, v in probabilities.items()}
