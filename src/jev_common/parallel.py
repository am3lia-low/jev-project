"""Run one API call per item in threads, keep going past single failures, stop on the spending cap.

Used by both parts' run scripts (Part A via helpers/generation.py).
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from tqdm import tqdm

from .openrouter import BudgetExceeded

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    function: Callable[[T], R], items: Sequence[T], workers: int, description: str
) -> tuple[list[tuple[T, R]], list[tuple[T, str]], bool]:
    """Apply `function` to every item in `workers` threads.

    Returns (successes, failures as (item, error message), stopped_by_budget).
    Successes keep the input order. After the spending cap is hit, the
    remaining items are skipped, not counted as failures.
    """
    results: dict[int, R] = {}
    failures: list[tuple[T, str]] = []
    stopped = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, item): index for index, item in enumerate(items)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            index = futures[future]
            try:
                results[index] = future.result()
            except BudgetExceeded:
                if not stopped:
                    stopped = True
                    for other in futures:
                        other.cancel()
            except Exception as error:  # noqa: BLE001 - one bad item must not stop the run
                failures.append((items[index], f"{type(error).__name__}: {error}"))
    successes = [(items[i], results[i]) for i in sorted(results)]
    return successes, failures, stopped
