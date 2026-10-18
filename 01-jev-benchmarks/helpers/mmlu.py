"""MMLU maths items, number parsing and the solve() code check (A11).

Records are plain dicts, not `Question`s: a swap needs fields the shared
format has no place for (the generated `solve` code and its arguments).

- `load_candidates`: MMLU maths test questions whose 4 options are all plain
  numbers and whose question contains a number.
- `parse_number` / `same_number`: "$1,200.50", "12.5%", "2/69" -> Fraction,
  compared to the precision the text is written with.
- `run_solve`: execute one generated `solve(**args)` in a separate Python
  process with a timeout, so a bad or endless function can't hang the run.
- `check_swap`: every automatic check on one generated swap.
"""

import json
import re
import subprocess
import sys
from fractions import Fraction
from typing import Any

from .config import MMLU_HF_PATH, MMLU_SUBJECTS, SOLVE_TIMEOUT_S
from .labels import letter_from_position

_NUMBER = re.compile(r"^-?\$?-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?$")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def parse_number(text: str) -> Fraction | None:
    """The value of a plain-number option, or None if it isn't one.

    "12.5%" is 12.5 (the percent sign is kept as formatting, not divided out),
    so a solve() for a percentage answer returns the number before the sign.
    """
    compact = str(text).replace(" ", "")
    if not _NUMBER.match(compact):
        return None
    body = compact.replace("$", "").replace(",", "").rstrip("%")
    try:
        return Fraction(body)
    except (ValueError, ZeroDivisionError):
        return None


def tolerance(text: str) -> Fraction:
    """Half a unit in the last written decimal place; exact for integers and fractions."""
    body = str(text).replace(" ", "").replace(",", "").rstrip("%")
    if "." in body and "/" not in body:
        return Fraction(1, 2 * 10 ** len(body.split(".")[1]))
    return Fraction(1, 10**9)


def same_number(value: float | Fraction, text: str) -> bool:
    """Whether a computed value is the number written as `text`."""
    target = parse_number(text)
    if target is None:
        return False
    return abs(Fraction(value).limit_denominator(10**9) - target) <= tolerance(text)


def numbers_in(text: str) -> list[str]:
    return _DIGITS.findall(text)


def without_numbers(text: str) -> str:
    """The text with every number replaced by "#", to check only numbers changed."""
    return _DIGITS.sub("#", _clean(text))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_candidates() -> list[dict[str, Any]]:
    """Candidate items from the MMLU maths test splits (needs HuggingFace)."""
    from datasets import load_dataset

    candidates = []
    for subject, short in MMLU_SUBJECTS.items():
        for index, row in enumerate(load_dataset(MMLU_HF_PATH, subject, split="test")):
            options = [_clean(c) for c in row["choices"]]
            question = _clean(row["question"])
            if len(options) != 4 or not numbers_in(question):
                continue
            if any(parse_number(o) is None for o in options):
                continue
            if len({parse_number(o) for o in options}) != 4:
                continue  # two options with the same value
            candidates.append({
                "id": f"mmlu-{short}-{index:04d}",
                "subject": subject,
                "question": question,
                "options": options,
                "answer": letter_from_position(int(row["answer"])),
            })
    return candidates


# ---------------------------------------------------------------------------
# solve() check
# ---------------------------------------------------------------------------
_RUNNER = """
import json, math, sys
from fractions import Fraction
payload = json.loads(sys.stdin.read())
scope = {"math": math, "Fraction": Fraction}
exec(payload["code"], scope)
results = [scope["solve"](**args) for args in payload["calls"]]
print(json.dumps([str(Fraction(r).limit_denominator(10**12)) if isinstance(r, Fraction) else repr(float(r)) for r in results]))
"""


def run_solve(code: str, *calls: dict[str, Any]) -> list[Fraction]:
    """`solve(**args)` for each `args`, run in an isolated Python process.

    Raises ValueError if the code fails, times out or returns a non-number.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=json.dumps({"code": code, "calls": list(calls)}),
            capture_output=True, text=True, timeout=SOLVE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"solve() took longer than {SOLVE_TIMEOUT_S}s") from None
    if done.returncode != 0:
        raise ValueError(f"solve() failed: {done.stderr.strip().splitlines()[-1] if done.stderr.strip() else 'no output'}")
    try:
        return [Fraction(v) for v in json.loads(done.stdout)]
    except (ValueError, TypeError, json.JSONDecodeError):
        raise ValueError(f"solve() returned something that isn't a number: {done.stdout.strip()[:100]}") from None


def check_swap(item: dict[str, Any], swap: dict[str, Any]) -> list[str]:
    """Problems with one generated swap of `item` (empty list = passes).

    `swap` is the model's JSON reply: new_question, solve, original_args,
    new_args, new_answer, wrong_answers (2).
    """
    problems = []
    for key in ("new_question", "solve", "original_args", "new_args", "new_answer", "wrong_answers"):
        if key not in swap:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    old_answer = item["options"][ord(item["answer"]) - ord("A")]
    new_question, new_answer = _clean(swap["new_question"]), _clean(str(swap["new_answer"]))
    wrong = [_clean(str(w)) for w in swap["wrong_answers"]] if isinstance(swap["wrong_answers"], list) else []

    if without_numbers(new_question) != without_numbers(item["question"]):
        problems.append("wording changed (not only numbers)")
    if numbers_in(new_question) == numbers_in(item["question"]):
        problems.append("no number changed")
    if len(wrong) != 2:
        problems.append(f"expected 2 wrong answers, got {len(wrong)}")
    for label, text in (("new answer", new_answer), *((f"wrong answer {i + 1}", w) for i, w in enumerate(wrong))):
        if parse_number(text) is None:
            problems.append(f"{label} {text!r} is not a plain number")
    if problems:
        return problems

    values = [parse_number(t) for t in (new_answer, *wrong)]
    if len(set(values)) != 3:
        problems.append("new answer and wrong answers are not all different")
    if parse_number(new_answer) in {parse_number(o) for o in item["options"]}:
        problems.append("new answer equals one of the original options")
    if parse_number(old_answer) in values[1:]:
        problems.append("a wrong answer equals the old answer")
    if not isinstance(swap["original_args"], dict) or not isinstance(swap["new_args"], dict):
        return [*problems, "original_args / new_args are not objects"]
    try:
        original_value, new_value = run_solve(swap["solve"], swap["original_args"], swap["new_args"])
    except ValueError as error:
        return [*problems, str(error)]
    if not same_number(original_value, old_answer):
        problems.append(f"solve(original) = {float(original_value):g}, not the original answer {old_answer}")
    if not same_number(new_value, new_answer):
        problems.append(f"solve(new) = {float(new_value):g}, not the stated new answer {new_answer}")
    return problems
