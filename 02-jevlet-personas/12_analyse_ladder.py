"""Difficulty-ladder tables, figure and summary (B2). Local only, no API calls.

Reads results/raw/<slug>/ladder/<benchmark>/<condition>.jsonl (11_run_ladder.py)
and skips what is missing, so it can be run at any stage.

Two questions (PLAN B2):
    No persona   Where does each model fall to chance? Accuracy per benchmark
                 with a bootstrap CI, against chance (the mean of 1 / options;
                 binomial test, one-sided) and published human accuracy.
    Persona      Does "pretend to be [level]" lower accuracy towards that
                 level? Drop = no-persona accuracy - persona accuracy on the
                 same questions, with a bootstrap CI and an exact McNemar test.

Writes results/tables/ladder_*.csv, results/figures/ladder_accuracy.png and
results/ladder_summary.md
    ladder_accuracy   per model x benchmark x condition: n, accuracy, CI,
                      chance, p vs chance, human reference
    ladder_drops      per model x benchmark x persona condition: drop with CI, McNemar p

Run from this folder:
    python 12_analyse_ladder.py
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.bigfive import read_records
from helpers.config import (
    FIGURES_DIR,
    HUMAN_ACCURACY,
    LADDER_BENCHMARKS,
    LADDER_DIR,
    LADDER_NAMES,
    LADDER_SUMMARY_PATH,
    RAW_RESULTS_DIR,
    TABLES_DIR,
)
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
MODEL_ORDER = [JEV_MODEL, *LLM_BASELINE_MODELS]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Categorical slots of the validated default palette: no persona, then each persona condition.
CONDITION_COLOURS = ["#0b0b0b", "#2a78d6", "#eb6834", "#1f9e6e", "#9b59b6"]


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_RESULTS_DIR.glob("*/ladder/*/*.jsonl")):
        model = path.parents[2].name.replace("__", "/", 1)
        rows += [{**r, "model": model} for r in read_records(path)]
    return pd.DataFrame(rows)


def _sort(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    bench = {b: i for i, b in enumerate(LADDER_BENCHMARKS)}

    def key(column: pd.Series) -> pd.Series:
        if column.name == "model":
            return column.map(order)
        if column.name == "benchmark":
            return column.map(bench)
        if column.name == "condition":
            return column.map(lambda c: (c != "none", c))
        return column

    return table.sort_values(keys, key=key).reset_index(drop=True)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, benchmark, condition), group in df.groupby(["model", "benchmark", "condition"]):
        correct = group["correct"].astype(float).to_numpy()
        ci = stats.bootstrap_ci(correct)
        chance = float((1 / group["n_options"]).mean())
        human = HUMAN_ACCURACY.get(benchmark) or {}
        rows.append({"model": model, "benchmark": benchmark, "condition": condition, "n": len(group),
                     "accuracy": ci.estimate, "ci_low": ci.low, "ci_high": ci.high, "chance": chance,
                     "p_above_chance": stats.binomial_test_vs_chance(int(correct.sum()), len(correct), chance),
                     "human_non_expert": human.get("non-expert"), "human_expert": human.get("expert"),
                     "n_failed": int(group["error"].notna().sum())})
    table = pd.DataFrame(rows)
    return table if table.empty else _sort(table, ["model", "benchmark", "condition"])


def drops_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, benchmark), group in df.groupby(["model", "benchmark"]):
        none = group[group["condition"] == "none"].set_index("id")["correct"]
        for condition, persona in group[group["condition"] != "none"].groupby("condition"):
            paired = pd.concat([none, persona.set_index("id")["correct"]], axis=1, keys=["none", "persona"]).dropna()
            if paired.empty:
                continue
            a, b = paired["none"].astype(float).to_numpy(), paired["persona"].astype(float).to_numpy()
            drop = stats.bootstrap_statistic([a, b], lambda x, y: float(x.mean() - y.mean()))
            test = stats.mcnemar_exact(a, b)
            rows.append({"model": model, "benchmark": benchmark, "condition": condition, "n": len(paired),
                         "accuracy_none": a.mean(), "accuracy_persona": b.mean(), "drop": drop.estimate,
                         "ci_low": drop.low, "ci_high": drop.high, "mcnemar_p": test.p_value})
    table = pd.DataFrame(rows)
    return table if table.empty else _sort(table, ["model", "benchmark", "condition"])


def figure(accuracy: pd.DataFrame, path) -> None:
    models = [m for m in MODEL_ORDER if m in set(accuracy["model"])]
    benchmarks = [b for b in LADDER_BENCHMARKS if b in set(accuracy["benchmark"])]
    conds = sorted(set(accuracy["condition"]), key=lambda c: (c != "none", c))
    x = np.arange(len(benchmarks))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False, sharey=True,
                             constrained_layout=True)
    for c, model in enumerate(models):
        ax = axes[0][c]
        sub = accuracy[accuracy["model"] == model].set_index(["benchmark", "condition"])
        chance = [accuracy[accuracy["benchmark"] == b]["chance"].mean() for b in benchmarks]
        ax.plot(x, chance, color=MUTED, linestyle=":", linewidth=1.5, label="chance")
        for i, cond in enumerate(conds):
            ys = [sub["accuracy"].get((b, cond), np.nan) for b in benchmarks]
            lo = [sub["ci_low"].get((b, cond), np.nan) for b in benchmarks]
            hi = [sub["ci_high"].get((b, cond), np.nan) for b in benchmarks]
            colour = CONDITION_COLOURS[i % len(CONDITION_COLOURS)]
            ax.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
            ax.plot(x, ys, color=colour, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                    label="no persona" if cond == "none" else cond)
        labelled = set()
        for j, b in enumerate(benchmarks):
            human = HUMAN_ACCURACY.get(b) or {}
            for kind, marker in (("non-expert", "v"), ("expert", "^")):
                if kind in human:
                    ax.plot(j, human[kind], marker=marker, color=MUTED, markersize=6, linestyle="none",
                            label=None if kind in labelled else f"human {kind}")
                    labelled.add(kind)
        ax.set_xticks(x, [LADDER_NAMES[b].split(" (")[0] for b in benchmarks], fontsize=8, color=MUTED, rotation=20)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.set_title(NAMES.get(model, model), fontsize=11, color=INK)
        if c == 0:
            ax.set_ylabel("accuracy", fontsize=9, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Accuracy up the difficulty ladder, with and without a 'pretend' persona", fontsize=12, color=INK)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def summary(accuracy: pd.DataFrame, drops: pd.DataFrame) -> str:
    lines = ["# Difficulty ladder: automatic summary (B2)", "",
             "Written by `12_analyse_ladder.py`; do not edit by hand. Accuracy with a bootstrap 95% CI. "
             "\"At chance\" means the binomial test against chance is not significant (p >= 0.05).", ""]
    report_path = LADDER_DIR / "build_report.json"
    if report_path.exists():
        hle = json.loads(report_path.read_text(encoding="utf-8")).get("hle")
        if hle:
            lines += [f"HLE: {hle.get('kept')} text-only multiple-choice questions of {hle.get('total')}"
                      + (" (exploratory: fewer than planned)" if hle.get("exploratory") else "") + ".", ""]
    if accuracy.empty:
        return "\n".join([*lines, "No results yet."]) + "\n"
    none = accuracy[accuracy["condition"] == "none"]
    lines += ["## No persona: actual ability", "", "| Model | " + " | ".join(
        LADDER_NAMES[b] for b in LADDER_BENCHMARKS if b in set(none["benchmark"])) + " |",
        "|---|" + "---|" * none["benchmark"].nunique()]
    for model, group in none.groupby("model", sort=False):
        cells = []
        for _, r in group.iterrows():
            flag = " (at chance)" if r["p_above_chance"] >= 0.05 else ""
            cells.append(f"{r['accuracy']:.1%} [{r['ci_low']:.1%}, {r['ci_high']:.1%}]{flag}")
        lines.append(f"| {NAMES.get(model, model)} | " + " | ".join(cells) + " |")
    chance_row = none.groupby("benchmark", sort=False)["chance"].mean()
    lines.append("| Chance | " + " | ".join(f"{v:.1%}" for v in chance_row) + " |")
    humans = [HUMAN_ACCURACY.get(b) for b in chance_row.index]
    lines.append("| Humans (non-expert / expert) | " + " | ".join(
        "–" if not h else f"{h['non-expert']:.0%} / {h['expert']:.0%} ({h['source']})" for h in humans) + " |")
    if not drops.empty:
        lines += ["", "## Persona: drop in accuracy (no persona - persona)", "",
                  "| Model | Benchmark | Condition | No persona | Persona | Drop | McNemar p |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in drops.iterrows():
            lines.append(f"| {NAMES.get(r['model'], r['model'])} | {LADDER_NAMES[r['benchmark']]} | {r['condition']} | "
                         f"{r['accuracy_none']:.1%} | {r['accuracy_persona']:.1%} | {r['drop']:+.1%} "
                         f"[{r['ci_low']:+.1%}, {r['ci_high']:+.1%}] | {r['mcnemar_p']:.3g} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    if df.empty:
        raise SystemExit(f"No ladder results in {RAW_RESULTS_DIR} yet; run 11_run_ladder.py first")
    print(f"Loaded {len(df)} answers ({int(df['error'].notna().sum())} failed calls, counted as wrong)")
    accuracy, drops = accuracy_table(df), drops_table(df)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("ladder_accuracy", accuracy), ("ladder_drops", drops)):
        if not table.empty:
            table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
            print(f"  {name}: {len(table)} rows")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure(accuracy, FIGURES_DIR / "ladder_accuracy.png")
    LADDER_SUMMARY_PATH.write_text(summary(accuracy, drops), encoding="utf-8")
    print(f"Wrote {TABLES_DIR}, {FIGURES_DIR / 'ladder_accuracy.png'}, {LADDER_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
