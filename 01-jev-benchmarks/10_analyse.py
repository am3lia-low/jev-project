"""Tables, figures and the memorisation verdict (A5, A7, A8).

Reads every result file there is and skips what is missing, so it can be run
at any stage:
    results/raw/<model slug>/<run>.jsonl         08_run_jev.py, 09_run_llm.py
    results/bert/<untrained | seed_n>/<run>.jsonl  07b notebook (downloaded from Colab)

Writes results/tables/*.csv, results/figures/*.png and results/verdict.md.

How models are combined:
- A level run more than once (level<n>_run<k>) is scored by majority vote.
- Fine-tuned BERT: accuracy is the mean over seeds, with a bootstrap CI over
  questions and the SD over seeds. The log-odds comparison uses each
  question's seed-averaged correctness; the paired tests (McNemar, which need
  0/1) and the Level 3 test use the majority over seeds.
- The above-chance rule (A5) uses BERT's val predictions on the original
  wording only (200 per dataset), so the two versions of one question are not
  counted as independent.

--sensitivity <BERT results folder> (A8, bigger-pool sensitivity analysis):
re-runs only the above-chance rule and the ladder rule with that folder's BERT
in place of results/bert/, and writes results/tables/sensitivity_<folder>_*.csv
and results/sensitivity_<folder>.md next to the main results. The main tables
and verdict.md are not touched: the verdict stays on the original BERT run.

Run from this folder:
    python 10_analyse.py
    python 10_analyse.py --sensitivity results/bert_bigpool
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import config
from helpers.evaluation import read_records
from helpers.levels import LEVELS
from jev_common import stats
from jev_common.models import BERT_MODEL, JEV_MODEL, LLM_BASELINE_MODELS

BERT = f"{BERT_MODEL} (fine-tuned)"
BERT_UNTRAINED = f"{BERT_MODEL} (untrained)"
NAMES = {JEV_MODEL: "Jev", BERT: "BERT-base", BERT_UNTRAINED: "BERT-base untrained",
         LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
# Fixed colour per model (validated categorical palette, slots 1-4), so a model
# keeps its colour whichever others are present.
COLOURS = {JEV_MODEL: "#2a78d6", BERT: "#eb6834", LLM_BASELINE_MODELS[0]: "#1baf7a",
           LLM_BASELINE_MODELS[1]: "#eda100", BERT_UNTRAINED: "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LADDER_DATASETS = ("swag", "arce", "arcc", "csqa")  # pooled memorisation test; HellaSwag apart (A8)
ALL = "all"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(bert_dir: Path = config.BERT_RESULTS_DIR) -> pd.DataFrame:
    rows = []
    if config.RAW_RESULTS_DIR.exists():
        for folder in sorted(p for p in config.RAW_RESULTS_DIR.iterdir() if p.is_dir()):
            model = folder.name.replace("__", "/")
            for path in sorted(folder.glob("*.jsonl")):
                rows += [{**r, "model": model, "seed": -1} for r in read_records(path)]
    if bert_dir.exists():
        for folder in sorted(p for p in bert_dir.iterdir() if p.is_dir()):
            untrained = folder.name == "untrained"
            seed = -1 if untrained else int(folder.name.removeprefix("seed_"))
            for path in sorted(folder.glob("*.jsonl")):
                rows += [{**r, "model": BERT_UNTRAINED if untrained else BERT, "seed": seed}
                         for r in read_records(path)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    split = df["run"].str.extract(r"^(?P<base>.+?)(?:_run(?P<k>\d+))?$")
    df["base_run"], df["run_k"] = split["base"], split["k"].fillna(1).astype(int)
    df["correct"] = df["correct"].astype(float)
    df["confidence"] = [max(p.values()) if isinstance(p, dict) else 0.0 for p in df["probabilities"]]
    df["failed"] = df["error"].notna()
    if "version" not in df:
        df["version"] = None
    return df


def per_question(df: pd.DataFrame, model: str, run: str) -> pd.DataFrame:
    """One row per question: dataset, correct (BERT: fraction of seeds), binary (0/1).

    Several runs of one API level are combined by majority vote on `pred`.
    """
    sub = df[(df["model"] == model) & (df["base_run"] == run)]
    if sub.empty:
        return pd.DataFrame(columns=["dataset", "correct", "binary"])
    if model in (BERT, BERT_UNTRAINED):
        grouped = sub.groupby("id").agg(dataset=("dataset", "first"), correct=("correct", "mean"))
        grouped["binary"] = (grouped["correct"] > 0.5).astype(float)
        return grouped

    def vote(group: pd.DataFrame) -> pd.Series:
        preds = Counter(group.sort_values("run_k")["pred"].dropna())
        pred = preds.most_common(1)[0][0] if preds else None
        correct = float(pred == group["answer"].iloc[0])
        return pd.Series({"dataset": group["dataset"].iloc[0], "correct": correct, "binary": correct})

    if sub["run_k"].max() == 1:
        return sub.set_index("id")[["dataset", "correct"]].assign(binary=lambda g: g["correct"])
    return sub.groupby("id")[["dataset", "pred", "answer", "run_k"]].apply(vote)


def with_pooled(frame: pd.DataFrame, datasets) -> list[tuple[str, pd.DataFrame]]:
    """[(dataset, rows of that dataset), ..., ("all", all of those rows)]."""
    parts = [(key, frame[frame["dataset"] == key]) for key in datasets]
    parts = [(key, part) for key, part in parts if not part.empty]
    if len(parts) > 1:
        parts.append((ALL, pd.concat(part for _, part in parts)))
    return parts


def models_in(df: pd.DataFrame) -> list[str]:
    order = [JEV_MODEL, *LLM_BASELINE_MODELS, BERT, BERT_UNTRAINED]
    present = set(df["model"])
    return [m for m in order if m in present] + sorted(present - set(order))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in models_in(df):
        for level in LEVELS:
            run = f"level{level}"
            questions = per_question(df, model, run)
            raw = df[(df["model"] == model) & (df["base_run"] == run)]
            for key, part in with_pooled(questions, config.DATASETS):
                records = raw if key == ALL else raw[raw["dataset"] == key]
                ci = stats.bootstrap_ci(part["correct"])
                per_seed = records.groupby("seed")["correct"].mean()
                eces = [stats.expected_calibration_error(g["confidence"], g["correct"])
                        for _, g in records.groupby(["seed", "run_k"])]
                latency = records["latency_s"].dropna()
                rows.append({
                    "model": NAMES.get(model, model), "level": level, "dataset": key, "n": len(part),
                    "accuracy": ci.estimate, "ci_low": ci.low, "ci_high": ci.high,
                    "sd_over_seeds": per_seed.std() if model == BERT and len(per_seed) > 1 else np.nan,
                    "ece": float(np.mean(eces)), "failed": int(records["failed"].sum()),
                    "latency_p50_s": latency.median() if len(latency) else np.nan,
                    "latency_p95_s": latency.quantile(0.95) if len(latency) else np.nan,
                    "cost_per_1k_usd": records["cost_usd"].fillna(0).mean() * 1000,
                })
    return pd.DataFrame(rows)


def above_chance(df: pd.DataFrame) -> pd.DataFrame:
    """A5: BERT is a valid difficulty control on a dataset if its val CI low >= chance + margin."""
    val = df[(df["model"] == BERT) & (df["base_run"] == "val") & (df["version"] == "original")]
    rows = []
    for key in config.DATASETS:
        part = val[val["dataset"] == key]
        if part.empty:
            continue
        correct = part.groupby("id")["correct"].mean()
        chance = float((1 / part.groupby("id")["n_options"].first()).mean())
        ci = stats.bootstrap_ci(correct)
        rows.append({"dataset": key, "n": len(correct), "accuracy": ci.estimate, "ci_low": ci.low,
                     "chance": chance, "threshold": chance + config.ABOVE_CHANCE_MARGIN,
                     "valid": ci.low >= chance + config.ABOVE_CHANCE_MARGIN})
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in models_in(df):
        for x, y in ((0, 1), (1, 2), (0, 2)):
            qa, qb = paired(per_question(df, model, f"level{x}"), per_question(df, model, f"level{y}"))
            for key, part in with_pooled(qa, config.DATASETS):
                result = stats.mcnemar_exact(part["binary"], qb.loc[part.index, "binary"])
                rows.append({"model": NAMES.get(model, model), "levels": f"{x} vs {y}", "dataset": key,
                             "n": len(part), "right_only_first": result.a_only, "right_only_second": result.b_only,
                             "p_value": result.p_value})
    return pd.DataFrame(rows)


def level3_table(df: pd.DataFrame) -> pd.DataFrame:
    """Level 3 (new questions) vs Level 0 on the paper datasets; different questions, so unpaired."""
    rows = []
    for model in models_in(df):
        l0, l3 = per_question(df, model, "level0"), per_question(df, model, "level3")
        if l3.empty or l0.empty:
            continue
        l0 = l0[l0["dataset"].isin(config.PAPER_DATASETS)]
        zero, three = dict(with_pooled(l0, config.PAPER_DATASETS)), dict(with_pooled(l3, config.PAPER_DATASETS))
        for key in zero.keys() & three.keys():
            a, b = zero[key]["binary"], three[key]["binary"]
            test = stats.two_proportion_test(int(a.sum()), len(a), int(b.sum()), len(b))
            rows.append({"model": NAMES.get(model, model), "dataset": key, "level0": a.mean(), "level3": b.mean(),
                         "n_level0": len(a), "n_level3": len(b), "z": test.z, "p_value": test.p_value})
    return pd.DataFrame(rows).sort_values(["model", "dataset"]) if rows else pd.DataFrame()


def memorisation_table(df: pd.DataFrame, valid: dict[str, bool]) -> pd.DataFrame:
    """A8: each API model's Level 0 -> 2 log-odds drop vs BERT's, on the same questions."""
    ref0, ref2 = per_question(df, BERT, "level0"), per_question(df, BERT, "level2")
    if ref0.empty or ref2.empty:
        return pd.DataFrame()
    rows = []
    for model in models_in(df):
        if model in (BERT, BERT_UNTRAINED):
            continue
        sub0, sub2 = per_question(df, model, "level0"), per_question(df, model, "level2")
        common = sub0.index.intersection(sub2.index).intersection(ref0.index).intersection(ref2.index)
        if common.empty:
            continue
        frame = sub0.loc[common, ["dataset"]]
        pooled_keys = [k for k in LADDER_DATASETS if valid.get(k)]
        groups = [(k, frame[frame["dataset"] == k].index, valid.get(k, False)) for k in config.DATASETS]
        groups.append((f"pooled ({', '.join(pooled_keys) or 'none valid'})",
                       frame[frame["dataset"].isin(pooled_keys)].index, True))
        for key, ids, bert_valid in groups:
            if ids.empty:
                continue
            result = stats.compare_drops(sub0.loc[ids, "binary"], sub2.loc[ids, "binary"],
                                         ref0.loc[ids, "correct"], ref2.loc[ids, "correct"],
                                         config.MEMORISATION_DROP_RATIO)
            rows.append({"model": NAMES.get(model, model), "dataset": key, "n": len(ids),
                         "bert_control_valid": bert_valid, "subject_drop": result.subject_drop,
                         "bert_drop": result.reference_drop, "difference": result.difference.estimate,
                         "ci_low": result.difference.low, "ci_high": result.difference.high,
                         "supported": result.supported})
    return pd.DataFrame(rows)


def quiz_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quiz = df[df["base_run"] == "quiz"]
    rates, compare, positions = [], [], []
    for model in models_in(quiz):
        sub = quiz[quiz["model"] == model]
        for quiz_set in ("benchmark", "control"):
            part = sub[sub["quiz_set"] == quiz_set]
            for key, group in with_pooled(part, config.PAPER_DATASETS):
                hits = int(group["correct"].sum())
                ci = stats.bootstrap_ci(group["correct"])
                rates.append({"model": NAMES.get(model, model), "quiz_set": quiz_set, "dataset": key,
                              "n": len(group), "picked_original": ci.estimate, "ci_low": ci.low, "ci_high": ci.high,
                              "p_vs_chance": stats.binomial_test_vs_chance(hits, len(group), config.QUIZ_CHANCE),
                              "picked_none_E": float((group["pred"] == "E").mean()),
                              "failed": int(group["failed"].sum())})
        bench, control = dict(with_pooled(sub[sub["quiz_set"] == "benchmark"], config.PAPER_DATASETS)), \
            dict(with_pooled(sub[sub["quiz_set"] == "control"], config.PAPER_DATASETS))
        for key in [k for k in (*config.PAPER_DATASETS, ALL) if k in bench and k in control]:
            b, c = bench[key]["correct"], control[key]["correct"]
            test = stats.two_proportion_test(int(b.sum()), len(b), int(c.sum()), len(c))
            compare.append({"model": NAMES.get(model, model), "dataset": key, "benchmark": b.mean(),
                            "control": c.mean(), "z": test.z, "p_value": test.p_value})
        for letter in "ABCDE":
            at = sub[sub["answer"] == letter]
            positions.append({"model": NAMES.get(model, model), "letter": letter,
                              "picked_rate": float((sub["pred"] == letter).mean()),
                              "original_here": len(at),
                              "picked_original_when_here": at["correct"].mean() if len(at) else np.nan})
    return pd.DataFrame(rates), pd.DataFrame(compare), pd.DataFrame(positions)


def repeatability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df[df["base_run"].str.startswith("repeat_")].groupby("model"):
        preds = sub.pivot_table(index="id", columns="base_run", values="pred", aggfunc="first")
        agree = preds.nunique(axis=1, dropna=False).eq(1)
        rows.append({"model": NAMES.get(model, model), "runs": preds.shape[1], "n": len(preds),
                     "all_runs_agree": agree.mean(),
                     "single_runs_enough": agree.mean() >= config.REPEAT_AGREEMENT_SINGLE_RUN})
    return pd.DataFrame(rows)


def bert_training_table() -> pd.DataFrame:
    rows = []
    for path in sorted(config.BERT_RESULTS_DIR.glob("seed_*/train_log.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed": log["seed"], "best_epoch": log.get("best_epoch"),
                     "train_seconds": log.get("train_seconds"), "gpu": log.get("gpu"),
                     "train_rows": log.get("train_rows"),
                     "best_val_accuracy": max(e["val_accuracy"] for e in log["epochs"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def figure_levels(accuracy: pd.DataFrame, path: Path) -> None:
    """Accuracy by level, one panel per dataset, one line per model, 95% CI bars."""
    keys = [k for k in config.DATASETS if k in set(accuracy["dataset"])]
    fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 3.4), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        for model, name in NAMES.items():
            rows = accuracy[(accuracy["model"] == name) & (accuracy["dataset"] == key)].sort_values("level")
            if rows.empty:
                continue
            untrained = model == BERT_UNTRAINED
            ax.errorbar(rows["level"], rows["accuracy"],
                        yerr=[rows["accuracy"] - rows["ci_low"], rows["ci_high"] - rows["accuracy"]],
                        color=COLOURS[model], linewidth=1 if untrained else 2, linestyle="--" if untrained else "-",
                        marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.5,
                        elinewidth=1, capsize=0, label=name)
        ax.set_title(config.DATASETS[key].name, color=INK, fontsize=10)
        ax.set_xticks(LEVELS)
        ax.set_xlabel("Level", color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0][0].set_ylabel("Accuracy", color=MUTED, fontsize=9)
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_quiz(rates: pd.DataFrame, path: Path) -> None:
    """Pooled % of quiz items where the original was picked, benchmark vs control, vs 25% chance."""
    pooled = rates[rates["dataset"] == ALL]
    models = list(dict.fromkeys(pooled["model"]))
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(models), 3.4))
    width = 0.36
    for offset, (quiz_set, colour) in zip((-width / 2, width / 2), (("benchmark", "#4a3aa7"), ("control", "#8a8984"))):
        rows = pooled[pooled["quiz_set"] == quiz_set].set_index("model").reindex(models)
        x = np.arange(len(models)) + offset
        ax.bar(x, rows["picked_original"], width - 0.04, color=colour, label=f"{quiz_set} items")
        ax.errorbar(x, rows["picked_original"], yerr=[rows["picked_original"] - rows["ci_low"],
                    rows["ci_high"] - rows["picked_original"]], fmt="none", ecolor=INK, elinewidth=1)
    ax.axhline(config.QUIZ_CHANCE, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(models) - 0.5, config.QUIZ_CHANCE, " chance", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel("Original picked", color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict(memorisation: pd.DataFrame, quiz_rates: pd.DataFrame, quiz_compare: pd.DataFrame,
            level3: pd.DataFrame, valid: dict[str, bool]) -> str:
    """The three signals of A8 per API model, stated with their numbers. The final call is yours."""
    alpha = config.SIGNIFICANCE
    lines = ["# Memorisation verdict (draft)", "",
             "Written by `10_analyse.py`. It states the three signals of PLAN A8 with their numbers; "
             "the final judgement is made by a person, not by this script.", ""]
    if valid:
        lines.append("BERT difficulty control valid on: "
                     + (", ".join(k for k, v in valid.items() if v) or "none")
                     + "; uninformative on: " + (", ".join(k for k, v in valid.items() if not v) or "none") + ".")
        lines.append("")
    subjects = [n for n in dict.fromkeys([*memorisation.get("model", []), *quiz_rates.get("model", [])])]
    if not subjects:
        lines.append("No Jev or LLM results yet.")
    for name in subjects:
        lines += [f"## {name}", ""]
        signals = 0
        mem = memorisation[(memorisation["model"] == name) & memorisation["dataset"].str.startswith("pooled")] \
            if not memorisation.empty else memorisation
        if mem.empty:
            lines.append("- **Ladder:** no Level 0 + Level 2 results for both this model and BERT yet.")
        else:
            r = mem.iloc[0]
            signals += bool(r["supported"])
            lines.append(f"- **Ladder** ({r['dataset']}, n={r['n']}): log-odds drop {r['subject_drop']:.2f} vs "
                         f"BERT {r['bert_drop']:.2f}; difference {r['difference']:.2f} "
                         f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] -> "
                         f"{'supports' if r['supported'] else 'does not support'} memorisation "
                         f"(needs CI > 0 and drop >= {config.MEMORISATION_DROP_RATIO:g}x BERT's).")
        rates = quiz_rates[(quiz_rates["model"] == name) & (quiz_rates["dataset"] == ALL)] \
            if not quiz_rates.empty else quiz_rates
        if len(rates) < 2:
            lines.append("- **Quiz:** no results for both item sets yet.")
        else:
            bench = rates[rates["quiz_set"] == "benchmark"].iloc[0]
            control = rates[rates["quiz_set"] == "control"].iloc[0]
            cmp = quiz_compare[(quiz_compare["model"] == name) & (quiz_compare["dataset"] == ALL)].iloc[0]
            bench_above, control_above = bench["p_vs_chance"] < alpha, control["p_vs_chance"] < alpha
            gap = cmp["p_value"] < alpha and bench["picked_original"] > control["picked_original"]
            evidence = bench_above and (gap if control_above else True)
            signals += evidence
            lines.append(f"- **Quiz:** original picked on {bench['picked_original']:.1%} of benchmark items "
                         f"(p={bench['p_vs_chance']:.3g} vs 25%) and {control['picked_original']:.1%} of control "
                         f"items (p={control['p_vs_chance']:.3g}); gap p={cmp['p_value']:.3g} -> "
                         f"{'evidence of' if evidence else 'no evidence of'} memorised wording.")
        l3 = level3[(level3["model"] == name) & (level3["dataset"] == ALL)] if not level3.empty else level3
        if l3.empty:
            lines.append("- **Level 3:** no Level 3 results yet.")
        else:
            r = l3.iloc[0]
            drop = r["p_value"] < alpha and r["level0"] > r["level3"]
            signals += drop
            lines.append(f"- **Level 3:** {r['level3']:.1%} on new questions vs {r['level0']:.1%} at Level 0 "
                         f"(p={r['p_value']:.3g}) -> {'a significant drop' if drop else 'no significant drop'}.")
        lines += ["", f"Signals pointing to memorisation: {signals} of 3.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def sensitivity(bert_dir: Path) -> None:
    """The ladder rule with another BERT run as the reference, next to the main run's (A8)."""
    name = bert_dir.name
    runs = {"main": load(), name: load(bert_dir)}
    if runs[name][runs[name]["model"] == BERT].empty:
        raise SystemExit(f"No fine-tuned BERT results in {bert_dir} yet")
    chance, ladder = {}, {}
    for label, df in runs.items():
        chance[label] = above_chance(df)
        valid = dict(zip(chance[label]["dataset"], chance[label]["valid"])) if not chance[label].empty else {}
        ladder[label] = memorisation_table(df, valid)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    chance[name].to_csv(config.TABLES_DIR / f"sensitivity_{name}_above_chance.csv", index=False, float_format="%.4f")
    ladder[name].to_csv(config.TABLES_DIR / f"sensitivity_{name}_ladder.csv", index=False, float_format="%.4f")

    def bert_accuracy(df: pd.DataFrame) -> dict[str, float]:
        return {run: per_question(df, BERT, run)["correct"].mean() for run in ("level0", "level2")}

    lines = [f"# Sensitivity: BERT trained on a bigger pool (`{name}`)", "",
             f"Written by `10_analyse.py --sensitivity`; do not edit by hand. Pre-registered in PLAN A8 "
             "(2026-10-20): the verdict stays on the main BERT run; this checks whether it holds when BERT is "
             "trained on more data. Same test questions, same Jev and LLM results, same rules.", "",
             "## BERT above-chance rule (val originals)", "", "| Dataset | Main: accuracy, CI low | Valid | "
             f"{name}: accuracy, CI low | Valid |", "|---|---|---|---|---|"]
    merged = chance["main"].merge(chance[name], on="dataset", suffixes=("_main", "_new"))
    for _, r in merged.iterrows():
        lines.append(f"| {r['dataset']} | {r['accuracy_main']:.1%}, {r['ci_low_main']:.1%} | "
                     f"{'yes' if r['valid_main'] else 'no'} | {r['accuracy_new']:.1%}, {r['ci_low_new']:.1%} | "
                     f"{'yes' if r['valid_new'] else 'no'} |")
    accuracy = {label: bert_accuracy(df) for label, df in runs.items()}
    lines += ["", "BERT accuracy, all datasets: " + "; ".join(
        f"{label} Level 0 {a['level0']:.1%}, Level 2 {a['level2']:.1%}" for label, a in accuracy.items()), "",
              "## Ladder rule, pooled over the datasets where each BERT run is valid", "",
              "| Model | Run | Pool | Model drop | BERT drop | Ratio | Difference [95% CI] | Supported |",
              "|---|---|---|---|---|---|---|---|"]
    for label, table in ladder.items():
        pooled = table[table["dataset"].str.startswith("pooled")]
        for _, r in pooled.iterrows():
            ratio = r["subject_drop"] / r["bert_drop"] if r["bert_drop"] > 0 else float("nan")
            lines.append(f"| {r['model']} | {label} | {r['dataset'].removeprefix('pooled ')} | {r['subject_drop']:.2f} | "
                         f"{r['bert_drop']:.2f} | {ratio:.2f}x | {r['difference']:.2f} [{r['ci_low']:.2f}, "
                         f"{r['ci_high']:.2f}] | {'yes' if r['supported'] else 'no'} |")
    out = config.RESULTS_DIR / f"sensitivity_{name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} and results/tables/sensitivity_{name}_*.csv (main tables and verdict.md untouched)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sensitivity", type=Path, metavar="BERT_RESULTS_DIR",
                        help="re-run the ladder rule with another BERT run (A8 sensitivity analysis)")
    args = parser.parse_args()
    if args.sensitivity is not None:
        sensitivity(args.sensitivity)
        return
    df = load()
    if df.empty:
        raise SystemExit(f"No results in {config.RAW_RESULTS_DIR} or {config.BERT_RESULTS_DIR} yet")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def save(name: str, table: pd.DataFrame) -> None:
        if table.empty:
            print(f"  {name}: no data yet")
            return
        table.to_csv(config.TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
        print(f"  {name}: {len(table)} rows")

    print(f"Loaded {len(df)} records: {dict(Counter(zip(df['model'].map(lambda m: NAMES.get(m, m)), df['base_run'])))}")
    accuracy = accuracy_table(df)
    chance = above_chance(df)
    valid = dict(zip(chance["dataset"], chance["valid"])) if not chance.empty else {}
    memorisation = memorisation_table(df, valid)
    level3 = level3_table(df)
    quiz_rates, quiz_compare, quiz_positions = quiz_tables(df)
    for name, table in (("accuracy", accuracy), ("bert_above_chance", chance), ("mcnemar", mcnemar_table(df)),
                        ("level3_vs_level0", level3), ("memorisation_ladder", memorisation),
                        ("quiz", quiz_rates), ("quiz_benchmark_vs_control", quiz_compare),
                        ("quiz_positions", quiz_positions), ("repeatability", repeatability_table(df)),
                        ("bert_training", bert_training_table())):
        save(name, table)
    if not accuracy.empty:
        figure_levels(accuracy, config.FIGURES_DIR / "accuracy_by_level.png")
    if not quiz_rates.empty:
        figure_quiz(quiz_rates, config.FIGURES_DIR / "quiz_original_picked.png")
    config.VERDICT_PATH.write_text(verdict(memorisation, quiz_rates, quiz_compare, level3, valid), encoding="utf-8")
    print(f"Wrote {config.TABLES_DIR}, {config.FIGURES_DIR}, {config.VERDICT_PATH}")


if __name__ == "__main__":
    main()
