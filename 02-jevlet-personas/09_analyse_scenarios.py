"""Scenario (behaviour) results next to the Big Five (self-report) results (B1 step 5). Local only.

Reads results/raw/<slug>/scenarios/ (08_run_scenarios.py) and, for the
comparison, results/raw/<slug>/bigfive/ (01_run_bigfive.py). Skips what is
missing, so it can be run at any stage.

Two measures of the same persona, both scaled to 0-1 so they can sit side by side:
    behaviour     probability of the high action, averaged over a trait's scenarios
    self-report   the trait's IPIP score, (score - 1) / 4
Shift = high persona minus low persona on the persona's own trait; 0 means
no pull, 1 the most. A persona "moves" a measure when the bootstrap 95% CI of
its shift lies above 0.

Writes results/tables/scenarios_*.csv and results/scenarios_summary.md
    scenarios_shifts         per model x persona trait x placement: behaviour
                             and self-report shift with CIs, and which moved
    scenarios_shift_matrix   behaviour high - low on every trait, per persona
                             trait x placement (the diagonal is the target)
    scenarios_baseline       no persona: behaviour and self-report score per trait
    scenarios_noise          Jev baseline: mean per-scenario SD over its repeats

Run from this folder:
    python 09_analyse_scenarios.py
"""

import numpy as np
import pandas as pd

from helpers.bigfive import read_records
from helpers.config import RAW_RESULTS_DIR, SCENARIO_SUMMARY_PATH, TABLES_DIR
from helpers.ipip import DOMAINS, SCALE_MAX, SCALE_MIN, keyed_score, load_items
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
MODEL_ORDER = [JEV_MODEL, *LLM_BASELINE_MODELS]
ITEM_BY_ID = {item.id: item for item in load_items()}


def _load(folder: str) -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_RESULTS_DIR.glob(f"*/{folder}/*.jsonl")):
        model = path.parent.parent.name.replace("__", "/", 1)
        rows += [{**r, "model": model} for r in read_records(path)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["variant"] = df["variant"].fillna(0).astype(int)
    return df


def load_behaviour() -> pd.DataFrame:
    df = _load("scenarios")
    if not df.empty:
        df = df.rename(columns={"scenario_id": "unit", "p_high": "value"})
    return df


def load_self_report() -> pd.DataFrame:
    df = _load("bigfive")
    if df.empty:
        return df
    df["value"] = [np.nan if pd.isna(a) else (keyed_score(ITEM_BY_ID[i], a) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)
                   for i, a in zip(df["item_id"], df["answer"])]
    return df.rename(columns={"item_id": "unit"})


def _sort(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return table.sort_values(keys, key=lambda c: c.map(order) if c.name == "model" else c).reset_index(drop=True)


def _shift(rows: pd.DataFrame) -> dict[str, float]:
    """High - low persona on the rows' units, each unit averaged over wordings; CI over units."""
    hi = rows[rows["pole"] == "high"].groupby("unit")["value"].mean()
    lo = rows[rows["pole"] == "low"].groupby("unit")["value"].mean()
    paired = pd.concat([hi, lo], axis=1, keys=["high", "low"]).dropna()
    if paired.empty:
        return {"shift": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    diff = stats.bootstrap_statistic([paired["high"].to_numpy(), paired["low"].to_numpy()],
                                     lambda a, b: float(a.mean() - b.mean()))
    return {"shift": diff.estimate, "ci_low": diff.low, "ci_high": diff.high, "n": len(paired)}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def shifts_table(behaviour: pd.DataFrame, self_report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    persona = behaviour[(behaviour["condition"] != "baseline") & (behaviour["domain"] == behaviour["persona_domain"])]
    sr = self_report[(self_report["condition"] != "baseline") & (self_report["domain"] == self_report["persona_domain"])] \
        if not self_report.empty else self_report
    for (model, domain, placement), group in persona.groupby(["model", "persona_domain", "placement"]):
        b = _shift(group)
        s = _shift(sr[(sr["model"] == model) & (sr["persona_domain"] == domain) & (sr["placement"] == placement)]) \
            if not sr.empty else _shift(sr)
        rows.append({"model": model, "persona_domain": domain, "placement": placement,
                     "behaviour_shift": b["shift"], "behaviour_ci_low": b["ci_low"], "behaviour_ci_high": b["ci_high"],
                     "n_scenarios": b["n"], "self_report_shift": s["shift"], "self_report_ci_low": s["ci_low"],
                     "self_report_ci_high": s["ci_high"], "moves_behaviour": bool(b["ci_low"] > 0),
                     "moves_self_report": bool(s["ci_low"] > 0)})
    table = pd.DataFrame(rows)
    return table if table.empty else _sort(table, ["model", "persona_domain", "placement"])


def shift_matrix(behaviour: pd.DataFrame) -> pd.DataFrame:
    persona = behaviour[behaviour["condition"] != "baseline"]
    rows = []
    for (model, persona_domain, placement), group in persona.groupby(["model", "persona_domain", "placement"]):
        row = {"model": model, "persona_domain": persona_domain, "placement": placement}
        for domain in DOMAINS:
            row[domain] = _shift(group[group["domain"] == domain])["shift"]
        rows.append(row)
    table = pd.DataFrame(rows)
    return table if table.empty else _sort(table, ["model", "persona_domain", "placement"])


def baseline_table(behaviour: pd.DataFrame, self_report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in (("behaviour", behaviour), ("self_report", self_report)):
        if df.empty:
            continue
        base = df[df["condition"] == "baseline"]
        for model, group in base.groupby("model"):
            rows.append({"model": model, "measure": name, **group.groupby("domain")["value"].mean().to_dict()})
    table = pd.DataFrame(rows)
    return table if table.empty else _sort(table, ["model", "measure"])


def noise_table(behaviour: pd.DataFrame) -> pd.DataFrame:
    base = behaviour[behaviour["condition"] == "baseline"]
    rows = []
    for model, group in base.groupby("model"):
        if group["run"].nunique() > 1:
            rows.append({"model": model, "n_runs": group["run"].nunique(),
                         "mean_scenario_sd": group.groupby("unit")["value"].std(ddof=1).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _cell(estimate: float, low: float, high: float) -> str:
    return "–" if pd.isna(estimate) else f"{estimate:.2f} [{low:.2f}, {high:.2f}]"


def summary(shifts: pd.DataFrame, matrix: pd.DataFrame, baseline: pd.DataFrame) -> str:
    lines = ["# Scenarios (behaviour) vs self-report: automatic summary (B1 step 5)", "",
             "Written by `09_analyse_scenarios.py`; do not edit by hand. Both measures on 0-1: behaviour = "
             "probability of the high action; self-report = (IPIP trait score - 1) / 4. Shift = high persona minus "
             "low persona on the persona's own trait.", ""]
    if shifts.empty:
        return "\n".join([*lines, "No persona runs yet."]) + "\n"
    lines += ["## Does the persona move self-report, behaviour, or both?", "",
              "| Model | Trait | Placement | Self-report shift | Behaviour shift | Moves |", "|---|---|---|---|---|---|"]
    for _, r in shifts.iterrows():
        moves = {(True, True): "both", (True, False): "self-report only", (False, True): "behaviour only",
                 (False, False): "neither"}[(r["moves_self_report"], r["moves_behaviour"])]
        lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['persona_domain']} | {r['placement']} | "
                     f"{_cell(r['self_report_shift'], r['self_report_ci_low'], r['self_report_ci_high'])} | "
                     f"{_cell(r['behaviour_shift'], r['behaviour_ci_low'], r['behaviour_ci_high'])} | {moves} |")
    if not matrix.empty:
        lines += ["", "## Behaviour shift on every trait (diagonal = target)", "",
                  "| Model | Persona | Placement | " + " | ".join(DOMAINS) + " |",
                  "|---|---|---|" + "---|" * len(DOMAINS)]
        for _, r in matrix.iterrows():
            lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['persona_domain']} | {r['placement']} | "
                         + " | ".join("–" if pd.isna(r[d]) else f"{r[d]:+.2f}" for d in DOMAINS) + " |")
    if not baseline.empty:
        lines += ["", "## No persona: trait scores (0-1)", "",
                  "| Model | Measure | " + " | ".join(DOMAINS) + " |", "|---|---|" + "---|" * len(DOMAINS)]
        for _, r in baseline.iterrows():
            lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['measure']} | "
                         + " | ".join("–" if pd.isna(r.get(d)) else f"{r[d]:.2f}" for d in DOMAINS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    behaviour = load_behaviour()
    if behaviour.empty:
        raise SystemExit(f"No scenario results in {RAW_RESULTS_DIR} yet; run 08_run_scenarios.py first")
    self_report = load_self_report()
    print(f"Loaded {len(behaviour)} scenario answers ({int(behaviour['value'].isna().sum())} failed, left out) and "
          f"{len(self_report)} Big Five answers")
    tables = {"scenarios_shifts": shifts_table(behaviour, self_report), "scenarios_shift_matrix": shift_matrix(behaviour),
              "scenarios_baseline": baseline_table(behaviour, self_report), "scenarios_noise": noise_table(behaviour)}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.empty:
            print(f"  {name}: no data yet")
            continue
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
        print(f"  {name}: {len(table)} rows")
    SCENARIO_SUMMARY_PATH.write_text(summary(tables["scenarios_shifts"], tables["scenarios_shift_matrix"],
                                             tables["scenarios_baseline"]), encoding="utf-8")
    print(f"Wrote {TABLES_DIR} and {SCENARIO_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
