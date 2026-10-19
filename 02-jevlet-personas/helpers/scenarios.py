"""Scenario questions for the behaviour check, and running a model on them (B1 step 5).

A scenario is an everyday situation with two actions: one a person high in an
IPIP facet would typically take, one a person low in it would take. Built by
07_make_scenarios.py and reviewed by hand; data/scenarios/scenarios.json holds
the kept ones, N_SCENARIOS_PER_FACET per facet, with the high action shown as A
in half of each trait's scenarios.

A `ScenarioTask` is one scenario under one B1 condition, wording and run. Jev
gets it as a `choice` question ("<situation> What do you do?", options A/B);
the persona is placed as in B1. The behaviour score is the probability of the
high action, 0-1.

Records: one JSON object per scenario per line in
`results/raw/<model slug>/scenarios/<condition>.jsonl`, sorted by
(variant, run, scenario):

    scenario_id, domain, facet, condition, persona_domain, pole, placement,
    variant, run,
    high_letter    where the high action is shown
    p_high         probability of the high action (null if the call failed)
    pred           letter picked (ties: Jev's own choice)
    probabilities, cost_usd, cached, error
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from jev_common.parallel import run_parallel

from .bigfive import model_slug, write_records
from .choice import ChoiceAnswer, ChoiceAnswerer, choice_request, picked_letter
from .config import RAW_RESULTS_DIR, SCENARIOS_PATH
from .personas import Condition, persona_text


@dataclass(frozen=True)
class Scenario:
    id: str  # "scn-E1-1": trait, facet, number
    domain: str
    facet: int
    facet_name: str
    situation: str
    options: tuple[str, str]  # as shown, A then B
    high_letter: str  # "A" or "B"


@cache
def load_scenarios(path: Path = SCENARIOS_PATH) -> tuple[Scenario, ...]:
    if not path.exists():
        raise SystemExit(f"{path} not found; run 07_make_scenarios.py and --apply-review first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("verified"):
        raise SystemExit(f"{path} is not marked verified; finish the review (07_make_scenarios.py --apply-review)")
    return tuple(Scenario(**{**s, "options": tuple(s["options"])}) for s in data["scenarios"])


@dataclass(frozen=True)
class ScenarioTask:
    scenario: Scenario
    condition: Condition
    variant: int | None
    run: int

    @property
    def cache_tag(self) -> str | None:
        """Makes run k > 1 of an identical request a fresh call."""
        return None if self.run == 1 else f"run-{self.run}"

    @property
    def request(self) -> dict[str, Any]:
        persona = None
        if not self.condition.is_baseline:
            persona = persona_text(self.condition.domain, self.condition.pole, self.variant)
        return choice_request(f"{self.scenario.situation} What do you do?", self.scenario.options, persona,
                              self.condition.placement)


def scenario_tasks(condition: Condition, scenarios: Sequence[Scenario], variants: Iterable[int | None],
                   runs: Iterable[int]) -> list[ScenarioTask]:
    return [ScenarioTask(s, condition, v, r) for v in variants for r in runs for s in scenarios]


def run_path(model: str, condition: Condition) -> Path:
    return RAW_RESULTS_DIR / model_slug(model) / "scenarios" / f"{condition.name}.jsonl"


def _record(task: ScenarioTask, answer: ChoiceAnswer | None, error: str | None) -> dict[str, Any]:
    probabilities, result = answer if answer is not None else (None, None)
    s = task.scenario
    return {
        "scenario_id": s.id, "domain": s.domain, "facet": s.facet,
        "condition": task.condition.name, "persona_domain": task.condition.domain,
        "pole": task.condition.pole, "placement": task.condition.placement,
        "variant": task.variant, "run": task.run, "high_letter": s.high_letter,
        "p_high": None if probabilities is None else round(probabilities.get(s.high_letter, 0.0), 6),
        "pred": picked_letter(("A", "B"), probabilities, result),
        "probabilities": probabilities,
        "cost_usd": None if result is None else result.cost_usd,
        "cached": None if result is None else result.cached,
        "error": error,
    }


def run_condition(tasks: list[ScenarioTask], answer: ChoiceAnswerer, model: str, workers: int) -> bool:
    """Answer every task of one condition and write its results file; True if the spending cap stopped it.

    A stopped run writes nothing: re-run it (finished calls come from the cache).
    """
    condition = tasks[0].condition
    done, failed, stopped = run_parallel(answer, tasks, workers, condition.name)
    if stopped:
        print(f"  {condition.name}: stopped at the spending cap after {len(done)}/{len(tasks)}; "
              "nothing written, re-run to finish")
        return True
    records = [_record(task, result, None) for task, result in done]
    records += [_record(task, None, error) for task, error in failed]
    records.sort(key=lambda r: (r["variant"] or 0, r["run"], r["scenario_id"]))
    path = run_path(model, condition)
    write_records(path, records)
    answered = [r["p_high"] for r in records if r["p_high"] is not None]
    mean = sum(answered) / len(answered) if answered else float("nan")
    print(f"  {condition.name}: {len(records)} answers, mean P(high action) {mean:.2f}, {len(failed)} failed -> {path}")
    for task, error in failed[:10]:
        print(f"    {task.scenario.id} variant {task.variant} run {task.run}: {error}")
    return False
