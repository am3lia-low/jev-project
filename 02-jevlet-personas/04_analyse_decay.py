"""Persona decay tables, figure and summary (B4). Local only, no API calls.

Reads every results file under results/raw/<model slug>/decay/<arm>/ and
skips what is missing, so it can be run at any stage.

The measure is the persona's pull at each checkpoint: the high-persona score
minus the low-persona score on the persona's own trait (1-5 scale, so 0 is
no pull and 4 the most). Anything the history does to both poles alike
cancels out. Retention is that shift divided by the shift at checkpoint 0
(no history) for the same model, arm, placement and trait: 1 means no decay,
0 means the persona is gone.

Writes results/tables/decay_*.csv and results/decay_summary.md
    decay_shifts     per model x arm x placement x persona trait x checkpoint
                     (x reminder): high, low, high - low with a bootstrap CI
                     over the trait's 24 items, and retention
    decay_pooled     the same pooled over the persona traits (72 items)
    decay_baseline   no-persona trait scores per checkpoint: what the history
                     does on its own
    decay_turns      own arm: mean answer to the history items per condition,
                     to check the history itself is in character

and results/figures/decay_retention.png (pooled shift per checkpoint).

Run from this folder:
    python 04_analyse_decay.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.bigfive import read_records
from helpers.config import (
    DECAY_ALL_ARMS,
    DECAY_ARM_SETTINGS,
    DECAY_REMINDER_TURN,
    DECAY_SUMMARY_PATH,
    FIGURES_DIR,
    PLACEMENTS,
    RAW_RESULTS_DIR,
    TABLES_DIR,
)
from helpers.ipip import SCALE_MAX, SCALE_MIN, keyed_score, load_items
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
MODEL_ORDER = [JEV_MODEL, *LLM_BASELINE_MODELS]
ARM_NAMES = {"own": "Own answers in the history", "quiz": "Quiz Q&A in the history",
             "long": "Long quiz history", "appdata": "App records in the state"}
ITEM_BY_ID = {item.id: item for item in load_items()}
SCALE = SCALE_MAX - SCALE_MIN
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Categorical slots 1-2 of the validated default palette.
PLACEMENT_COLOURS = {"state": "#2a78d6", "instructions": "#eb6834"}
PLACEMENT_NAMES = {"state": "persona in state", "instructions": "persona in instructions"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _model(path) -> str:
    return path.parents[2].name.replace("__", "/", 1)


def load() -> pd.DataFrame:
    rows = []
    if RAW_RESULTS_DIR.exists():
        for path in sorted(RAW_RESULTS_DIR.glob("*/decay/*/*.jsonl")):
            rows += [{**r, "model": _model(path)} for r in read_records(path)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["variant"] = df["variant"].fillna(0).astype(int)  # 0 = baseline (no wording)
    df["keyed_score"] = [
        np.nan if pd.isna(a) else keyed_score(ITEM_BY_ID[i], a) for i, a in zip(df["item_id"], df["answer"])
    ]
    return df


def load_turns() -> pd.DataFrame:
    rows = []
    if RAW_RESULTS_DIR.exists():
        for path in sorted(RAW_RESULTS_DIR.glob("*/decay/own/histories/*.jsonl")):
            model = path.parents[3].name.replace("__", "/", 1)
            rows += [{**r, "model": model, "conversation": path.stem} for r in read_records(path)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _display_order(column: pd.Series) -> pd.Series:
    """Sort key: models, arms and placements in their configured order, anything else as is."""
    for order in (MODEL_ORDER, DECAY_ALL_ARMS, PLACEMENTS):
        if column.isin(order).all():
            return column.map({value: i for i, value in enumerate(order)})
    return column


def _item_means(rows: pd.DataFrame) -> pd.Series:
    """Per-item keyed score, averaged over wordings."""
    return rows.groupby("item_id")["keyed_score"].mean()


def _shift(hi: pd.Series, lo: pd.Series) -> dict[str, float]:
    paired = pd.concat([hi, lo], axis=1, keys=["high", "low"]).dropna()
    if paired.empty:
        return {}
    diff = stats.bootstrap_statistic(
        [(paired["high"].to_numpy() - SCALE_MIN) / SCALE, (paired["low"].to_numpy() - SCALE_MIN) / SCALE],
        lambda a, b: float(a.mean() - b.mean()),
    )
    return {"n_items": len(paired), "high": paired["high"].mean(), "low": paired["low"].mean(),
            "shift": diff.estimate * SCALE, "ci_low": diff.low * SCALE, "ci_high": diff.high * SCALE}


def shifts_table(df: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    persona = df[df["condition"] != "baseline"]
    persona = persona[persona["domain"] == persona["persona_domain"]]  # the persona's own trait
    keys = ["model", "arm", "placement", "checkpoint", "reminder"] + ([] if pooled else ["persona_domain"])
    rows = []
    for key, group in persona.groupby(keys):
        result = _shift(_item_means(group[group["pole"] == "high"]), _item_means(group[group["pole"] == "low"]))
        if result:
            rows.append({**dict(zip(keys, key)), **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    start_keys = [k for k in keys if k not in ("checkpoint", "reminder")]
    start = table[(table["checkpoint"] == 0) & ~table["reminder"]].set_index(start_keys)["shift"]
    table["retention"] = [
        row["shift"] / start.get(tuple(row[k] for k in start_keys), np.nan) for _, row in table.iterrows()
    ]
    table = table.sort_values(keys, key=_display_order)
    front = keys + ["n_items", "high", "low", "shift", "ci_low", "ci_high", "retention"]
    return table[front].reset_index(drop=True)


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "baseline"]
    if base.empty:
        return pd.DataFrame()
    table = base.groupby(["model", "arm", "checkpoint", "domain"])["keyed_score"].mean().unstack("domain")
    return table.reset_index().sort_values(["model", "arm", "checkpoint"], key=_display_order).reset_index(drop=True)


def turns_table(turns: pd.DataFrame) -> pd.DataFrame:
    """Own arm: the history items' mean raw answer per conversation (a sanity check, not a result)."""
    if turns.empty:
        return pd.DataFrame()
    return (turns.groupby(["model", "conversation"])
            .agg(turns=("turn", "count"), mean_answer=("answer", "mean"),
                 cost_usd=("cost_usd", "sum")).reset_index())


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure_retention(pooled: pd.DataFrame, path) -> None:
    """Shift (high - low) per checkpoint: columns = models, rows = arms, one line per placement."""
    models = [m for m in MODEL_ORDER if m in set(pooled["model"])]
    arms = [a for a in DECAY_ALL_ARMS if a in set(pooled["arm"])]
    fig, axes = plt.subplots(len(arms), len(models), figsize=(3.6 * len(models) + 0.6, 3.0 * len(arms) + 1.0),
                             squeeze=False, sharey=True, constrained_layout=True)
    for r, arm in enumerate(arms):
        for c, model in enumerate(models):
            ax = axes[r][c]
            checkpoints = DECAY_ARM_SETTINGS[arm]["checkpoints"]
            x_of = {k: i for i, k in enumerate(checkpoints)}
            sub = pooled[(pooled["model"] == model) & (pooled["arm"] == arm)]
            for placement in PLACEMENTS:
                colour = PLACEMENT_COLOURS[placement]
                line = sub[(sub["placement"] == placement) & ~sub["reminder"]].sort_values("checkpoint")
                if line.empty:
                    continue
                x = line["checkpoint"].map(x_of).to_numpy()
                ax.fill_between(x, line["ci_low"], line["ci_high"], color=colour, alpha=0.15, linewidth=0)
                ax.plot(x, line["shift"], color=colour, linewidth=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=PLACEMENT_NAMES[placement])
                remind = sub[(sub["placement"] == placement) & sub["reminder"]].sort_values("checkpoint")
                if not remind.empty:
                    xr = remind["checkpoint"].map(x_of).to_numpy()
                    ax.errorbar(xr, remind["shift"], yerr=[remind["shift"] - remind["ci_low"],
                                                          remind["ci_high"] - remind["shift"]],
                                fmt="D", color=colour, markersize=7, markerfacecolor=SURFACE,
                                markeredgewidth=2, elinewidth=1.2, capsize=0, linestyle="none",
                                label=f"{PLACEMENT_NAMES[placement]}, re-reminded at turn {DECAY_REMINDER_TURN}")
            ax.set_xticks(range(len(checkpoints)), [str(k) for k in checkpoints], fontsize=9, color=MUTED)
            ax.set_ylim(0, SCALE)
            ax.set_yticks(range(SCALE + 1))
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if r == 0:
                ax.set_title(NAMES.get(model, model), fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ARM_NAMES[arm]}\nshift, high - low (1-5 scale)", fontsize=9, color=MUTED)
            if r == len(arms) - 1:
                ax.set_xlabel("turns of history (records, for app data) before the question", fontsize=9,
                              color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Persona pull as the history grows (0 = no pull, 4 = maximum)", fontsize=12, color=INK)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary(pooled: pd.DataFrame, baseline: pd.DataFrame) -> str:
    lines = ["# Persona decay: automatic summary (B4)", "",
             "Written by `04_analyse_decay.py`; do not edit by hand. Shift = high-persona score minus "
             "low-persona score on the persona's own trait (1-5 scale; 0 = no pull). Retention = shift "
             "divided by the shift with no history. Pooled over the persona traits; per-trait numbers are "
             "in `tables/decay_shifts.csv`.", ""]
    if pooled.empty:
        return "\n".join([*lines, "No results yet."]) + "\n"
    lines += ["## Shift with no history and at the longest history", "",
              "| Model | History | Placement | Shift at 0 | Longest (turns) | Shift there | Retention there |"
              " Re-reminded, there |", "|---|---|---|---|---|---|---|---|"]
    for (model, arm, placement), group in pooled.groupby(["model", "arm", "placement"], sort=False):
        last = max(DECAY_ARM_SETTINGS[arm]["checkpoints"])
        plain = group[~group["reminder"]].set_index("checkpoint")
        remind = group[group["reminder"]].set_index("checkpoint")

        def cell(table: pd.DataFrame, k: int) -> str:
            if k not in table.index:
                return "–"
            row = table.loc[k]
            return f"{row['shift']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"

        retention = plain["retention"].get(last, np.nan)
        lines.append(f"| {NAMES.get(model, model)} | {arm} | {placement} | {cell(plain, 0)} | {last} | {cell(plain, last)} | "
                     f"{'–' if pd.isna(retention) else f'{retention:.2f}'} | {cell(remind, last)} |")
    if not baseline.empty:
        domains = [d for d in ("N", "E", "O", "A", "C") if d in baseline.columns]
        lines += ["", "## No-persona scores per checkpoint", "",
                  "What the history does on its own. Trait scores, 1-5.", "",
                  "| Model | History | Turns | " + " | ".join(domains) + " |",
                  "|---|---|---|" + "---|" * len(domains)]
        for _, row in baseline.iterrows():
            values = " | ".join(f"{row[d]:.2f}" for d in domains)
            lines.append(f"| {NAMES.get(row['model'], row['model'])} | {row['arm']} | {row['checkpoint']} | {values} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    if df.empty:
        raise SystemExit(f"No decay results in {RAW_RESULTS_DIR} yet; run 03_run_decay.py first")
    failed = int(df["answer"].isna().sum())
    print(f"Loaded {len(df)} probe answers ({failed} failed calls, left out)")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    shifts = shifts_table(df, pooled=False)
    pooled = shifts_table(df, pooled=True)
    baseline = baseline_table(df)
    turns = turns_table(load_turns())
    for name, table in (("decay_shifts", shifts), ("decay_pooled", pooled), ("decay_baseline", baseline),
                        ("decay_turns", turns)):
        if table.empty:
            print(f"  {name}: no data yet")
            continue
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
        print(f"  {name}: {len(table)} rows")
    if not pooled.empty:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figure_retention(pooled, FIGURES_DIR / "decay_retention.png")
    DECAY_SUMMARY_PATH.write_text(summary(pooled, baseline), encoding="utf-8")
    print(f"Wrote {TABLES_DIR}, {FIGURES_DIR / 'decay_retention.png'}, {DECAY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
