"""On-disk cache keyed by a hash of each request, so API scripts can resume (A3).

One JSON file per request under `<root>/<namespace>/<hash prefix>/<hash>.json`,
where `root` is the calling part's `cache/` folder (gitignored).
The key hashes the request payload only, so API keys and headers never reach
the cache. Delete a file (or a namespace folder) to force a re-run.
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def request_key(request: Any) -> str:
    """SHA-256 of the request in canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """Cache of responses for one script or purpose (`namespace`)."""

    def __init__(self, namespace: str, root: str | Path) -> None:
        self.directory = Path(root) / namespace

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: Any) -> Any | None:
        """Return the cached response, or None on a miss or an unreadable file."""
        path = self._path(request_key(request))
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)["response"]
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, OSError) as error:
            logger.warning("Ignoring unreadable cache file %s: %s", path, error)
            return None

    def put(self, request: Any, response: Any) -> None:
        """Store a response next to its request (kept for auditing)."""
        path = self._path(request_key(request))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump({"request": request, "response": response}, handle, ensure_ascii=False)
        os.replace(temp_path, path)

    def get_or_call(self, request: Any, call: Callable[[Any], Any]) -> Any:
        """Return the cached response, else `call(request)`, cache and return it.

        Failed calls raise and are not cached, so a re-run retries them.
        """
        cached = self.get(request)
        if cached is not None:
            return cached
        response = call(request)
        self.put(request, response)
        return response

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.json"))
