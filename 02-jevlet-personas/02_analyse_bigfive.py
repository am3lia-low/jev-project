"""Big Five tables and summary (B1 steps 1-3).

Reads every results file under results/raw/<model slug>/bigfive/ and skips
what is missing, so it can be run at any stage. A response set is one
(condition, variant, run): 120 answers. Sets with a failed call are left out
of the scores and alpha, and counted.

Writes results/tables/bigfive_*.csv, results/bigfive_summary.md
    bigfive_scores        domain scores (1-5) per response set
    bigfive_reliability   alpha per domain, pooled over every complete set of
                          a model (all conditions; see helpers/reliability.py),
                          and the baseline's acquiescence
    bigfive_noise         Jev baseline: SD of domain scores over its repeats,
                          and the mean per-item SD
    bigfive_shifts        per persona trait x placement: target-domain score
                          for high, low and the baseline, high - low with a
                          bootstrap CI over the trait's 24 items, and the
                          largest off-target high - low shift
    bigfive_shift_matrix  high - low on every measured domain, per persona
                          trait x placement (the diagonal is the target)

and results/figures/bigfive_shift_matrix.png (the shift matrix as heatmaps).

Run from this folder:
    python 02_analyse_bigfive.py
"""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from helpers.bigfive import read_records
from helpers.config import BIGFIVE_SUMMARY_PATH, FIGURES_DIR, PLACEMENTS, RAW_RESULTS_DIR, TABLES_DIR
from helpers.ipip import DOMAINS, SCALE_MAX, SCALE_MIN, domain_scores, keyed_score, load_items
from helpers.reliability import ALPHA_THRESHOLD, acquiescence, alpha_per_domain
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
ITEMS = load_items()
ITEM_BY_ID = {item.id: item for item in ITEMS}
SET_KEY = ["model", "condition", "variant", "run"]
MODEL_ORDER = [JEV_MODEL, *LLM_BASELINE_MODELS]
INK, MUTED = "#0b0b0b", "#52514e"
# Diverging blue <-> red with a neutral grey midpoint (validated default palette).
SHIFT_CMAP = LinearSegmentedColormap.from_list("shift", ["#184f95", "#6da7ec", "#f0efec", "#ee8a89", "#b8302f"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    rows = []
    if RAW_RESULTS_DIR.exists():
        for path in sorted(RAW_RESULTS_DIR.glob("*/bigfive/*.jsonl")):
            model = path.parent.parent.name.replace("__", "/", 1)
            rows += [{**r, "model": model} for r in read_records(path)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["variant"] = df["variant"].fillna(0).astype(int)  # 0 = baseline (no wording)
    df["keyed_score"] = [
        np.nan if pd.isna(a) else keyed_score(ITEM_BY_ID[i], a) for i, a in zip(df["item_id"], df["answer"])
    ]
    return df


def complete_sets(df: pd.DataFrame) -> tuple[list[tuple[tuple, dict[str, float]]], Counter]:
    """[(set key, {item ID: answer})] for sets with all 120 answers; Counter of incomplete sets per model."""
    sets, incomplete = [], Counter()
    for key, group in df.groupby(SET_KEY, sort=True):
        answers = dict(zip(group["item_id"], group["answer"]))
        if len(answers) == len(ITEMS) and not group["answer"].isna().any():
            sets.append((key, answers))
        else:
            incomplete[key[0]] += 1
    return sets, incomplete


def _model_rank(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def scores_table(sets: list[tuple[tuple, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for (model, condition, variant, run), answers in sets:
        rows.append({"model": model, "condition": condition, "variant": variant or None, "run": run,
                     **domain_scores(ITEMS, answers)})
    return pd.DataFrame(rows)


def reliability_table(sets: list[tuple[tuple, dict[str, float]]], incomplete: Counter) -> pd.DataFrame:
    rows = []
    for model in sorted({key[0] for key, _ in sets}, key=_model_rank):
        model_sets = [answers for key, answers in sets if key[0] == model]
        baseline = [answers for key, answers in sets if key[0] == model and key[1] == "baseline"]
        n_conditions = len({key[1] for key, _ in sets if key[0] == model})
        alphas = alpha_per_domain(ITEMS, model_sets) if len(model_sets) >= 2 else {}
        bias = [acquiescence(ITEMS, answers)["bias"] for answers in baseline]
        for domain in DOMAINS:
            alpha = alphas.get(domain, np.nan)
            rows.append({"model": model, "domain": domain, "alpha": alpha,
                         "interpretable": bool(alpha >= ALPHA_THRESHOLD),
                         "n_sets": len(model_sets), "n_conditions": n_conditions,
                         "n_incomplete_sets": incomplete[model],
                         "baseline_acquiescence": np.mean(bias) if bias else np.nan})
    return pd.DataFrame(rows)


def noise_table(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run noise of the baseline, for models with repeated baseline runs (Jev)."""
    rows = []
    for model in sorted(scores["model"].unique()):
        base = scores[(scores["model"] == model) & (scores["condition"] == "baseline")]
        if len(base) < 2:
            continue
        answers = df[(df["model"] == model) & (df["condition"] == "baseline")]
        item_sd = answers.groupby("item_id")["answer"].std(ddof=1).mean()
        rows.append({"model": model, "n_runs": len(base), "mean_item_sd": item_sd,
                     **{f"{d}_sd": base[d].std(ddof=1) for d in DOMAINS}})
    return pd.DataFrame(rows)


def _to_unit(values: np.ndarray) -> np.ndarray:
    return (values - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)


def _item_means(df: pd.DataFrame, model: str, condition: str, domain: str) -> pd.Series:
    """Per-item keyed score for `domain`'s items under one condition, averaged over wordings and runs."""
    rows = df[(df["model"] == model) & (df["condition"] == condition) & (df["domain"] == domain)]
    return rows.groupby("item_id")["keyed_score"].mean()


def shifts_table(df: pd.DataFrame) -> pd.DataFrame:
    """B1 step 3: does the persona move its target trait, and only that trait?"""
    rows = []
    for model in sorted(df["model"].unique()):
        for domain in DOMAINS:
            for placement in PLACEMENTS:
                high, low = f"{domain}-high-{placement}", f"{domain}-low-{placement}"
                hi, lo = _item_means(df, model, high, domain), _item_means(df, model, low, domain)
                if hi.empty or lo.empty:
                    continue
                paired = pd.concat([hi, lo], axis=1, keys=["high", "low"]).dropna()
                diff = stats.bootstrap_statistic(
                    [_to_unit(paired["high"].to_numpy()), _to_unit(paired["low"].to_numpy())],
                    lambda a, b: float(a.mean() - b.mean()),
                )
                scale = SCALE_MAX - SCALE_MIN
                base = _item_means(df, model, "baseline", domain)
                off = {other: _item_means(df, model, high, other).mean() - _item_means(df, model, low, other).mean()
                       for other in DOMAINS if other != domain}
                worst = max(off, key=lambda k: abs(off[k]) if not np.isnan(off[k]) else -1)
                rows.append({
                    "model": model, "persona_domain": domain, "placement": placement,
                    "n_items": len(paired),
                    "high": paired["high"].mean(), "baseline": base.mean() if not base.empty else np.nan,
                    "low": paired["low"].mean(),
                    "high_minus_low": diff.estimate * scale, "ci_low": diff.low * scale, "ci_high": diff.high * scale,
                    "largest_off_target_domain": worst, "largest_off_target_shift": off[worst],
                })
    return pd.DataFrame(rows)


def shift_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    """High - low persona shift on every measured domain (keyed scores, 1-5 scale)."""
    rows = []
    for model in sorted(df["model"].unique(), key=_model_rank):
        for placement in PLACEMENTS:
            for persona in DOMAINS:
                high, low = f"{persona}-high-{placement}", f"{persona}-low-{placement}"
                for measured in DOMAINS:
                    hi, lo = _item_means(df, model, high, measured), _item_means(df, model, low, measured)
                    if hi.empty or lo.empty:
                        continue
                    rows.append({"model": model, "placement": placement, "persona_domain": persona,
                                 "measured_domain": measured, "shift": hi.mean() - lo.mean()})
    return pd.DataFrame(rows)


def figure_shift_matrix(matrix: pd.DataFrame, path) -> None:
    """One heatmap per model x placement: rows = persona trait, columns = measured trait."""
    models = [m for m in MODEL_ORDER if m in set(matrix["model"])]
    placements = [p for p in PLACEMENTS if p in set(matrix["placement"])]
    domains = list(DOMAINS)
    fig, axes = plt.subplots(len(placements), len(models), figsize=(3.4 * len(models) + 0.8, 3.3 * len(placements) + 0.9),
                             squeeze=False, constrained_layout=True)
    limit = SCALE_MAX - SCALE_MIN
    image = None
    for r, placement in enumerate(placements):
        for c, model in enumerate(models):
            ax = axes[r][c]
            sub = matrix[(matrix["model"] == model) & (matrix["placement"] == placement)]
            grid = sub.pivot(index="persona_domain", columns="measured_domain", values="shift").reindex(
                index=domains, columns=domains)
            image = ax.imshow(grid.to_numpy(), cmap=SHIFT_CMAP, vmin=-limit, vmax=limit)
            for i in range(len(domains)):
                for j in range(len(domains)):
                    value = grid.iat[i, j]
                    if np.isnan(value):
                        continue
                    ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=9,
                            color="#ffffff" if abs(value) >= 2.5 else INK,
                            fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(len(domains)), domains, fontsize=9, color=MUTED)
            ax.set_yticks(range(len(domains)), domains, fontsize=9, color=MUTED)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(domains)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(domains)), minor=True)
            ax.grid(which="minor", color="#fcfcfb", linewidth=2)
            ax.tick_params(which="minor", length=0)
            if r == 0:
                ax.set_title(NAMES.get(model, model), fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f"Persona in {placement}\npersona trait (high - low)", fontsize=9, color=MUTED)
            if r == len(placements) - 1:
                ax.set_xlabel("measured trait", fontsize=9, color=MUTED)
    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
    bar.set_label("high - low persona shift (1-5 scale)", fontsize=9, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle("Big Five persona shifts: diagonal = target trait, off-diagonal = side effects",
                 fontsize=12, color=INK)
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary(reliability: pd.DataFrame, noise: pd.DataFrame, shifts: pd.DataFrame) -> str:
    lines = ["# Big Five personas (draft)", "",
             "Written by `02_analyse_bigfive.py`. Numbers only; the reading is done by a person.", ""]
    if reliability.empty:
        return "\n".join(lines + ["No complete response sets yet."]) + "\n"
    for model in reliability["model"].unique():
        name = NAMES.get(model, model)
        rel = reliability[reliability["model"] == model]
        lines += [f"## {name}", ""]
        alphas = ", ".join(f"{r.domain} {r.alpha:.2f}" for r in rel.itertuples())
        n_cond = rel["n_conditions"].iloc[0]
        lines.append(f"- **Alpha** over {rel['n_sets'].iloc[0]} response sets from {n_cond} condition(s): {alphas}"
                     f" (interpret only >= {ALPHA_THRESHOLD:.2f}).")
        if n_cond == 1:
            lines.append("  - Baseline only: alpha needs sets that differ in the trait, so it is not meaningful yet.")
        if not np.isnan(rel["baseline_acquiescence"].iloc[0]):
            lines.append(f"- **Acquiescence** (baseline mean raw answer, plus- and minus-keyed averaged; 3 = none): "
                         f"{rel['baseline_acquiescence'].iloc[0]:.2f}.")
        model_noise = noise[noise["model"] == model] if not noise.empty else noise
        if not model_noise.empty:
            r = model_noise.iloc[0]
            lines.append(f"- **Baseline noise** over {int(r['n_runs'])} repeats: mean per-item SD {r['mean_item_sd']:.3f}.")
        model_shifts = shifts[shifts["model"] == model] if not shifts.empty else shifts
        if model_shifts.empty:
            lines.append("- **Persona shifts:** no persona results yet.")
        else:
            lines.append("- **Persona shifts** (target trait, high - low, 1-5 scale, 95% CI over items):")
            for r in model_shifts.itertuples():
                lines.append(f"  - {DOMAINS[r.persona_domain]}, {r.placement}: {r.high_minus_low:+.2f} "
                             f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}] (high {r.high:.2f}, baseline {r.baseline:.2f}, "
                             f"low {r.low:.2f}); largest off-target {r.largest_off_target_domain} "
                             f"{r.largest_off_target_shift:+.2f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load()
    if df.empty:
        raise SystemExit(f"No Big Five results in {RAW_RESULTS_DIR} yet")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def save(name: str, table: pd.DataFrame) -> None:
        if table.empty:
            print(f"  {name}: no data yet")
            return
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
        print(f"  {name}: {len(table)} rows")

    sets, incomplete = complete_sets(df)
    print(f"Loaded {len(df)} answers, {len(sets)} complete response sets"
          f"{f', incomplete: {dict(incomplete)}' if incomplete else ''}")
    scores = scores_table(sets)
    reliability = reliability_table(sets, incomplete)
    noise = noise_table(df, scores) if not scores.empty else pd.DataFrame()
    shifts = shifts_table(df)
    matrix = shift_matrix_table(df)
    for name, table in (("bigfive_scores", scores), ("bigfive_reliability", reliability),
                        ("bigfive_noise", noise), ("bigfive_shifts", shifts), ("bigfive_shift_matrix", matrix)):
        save(name, table)
    if not matrix.empty:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figure_shift_matrix(matrix, FIGURES_DIR / "bigfive_shift_matrix.png")
    BIGFIVE_SUMMARY_PATH.write_text(summary(reliability, noise, shifts), encoding="utf-8")
    print(f"Wrote {TABLES_DIR}, {FIGURES_DIR}, {BIGFIVE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
