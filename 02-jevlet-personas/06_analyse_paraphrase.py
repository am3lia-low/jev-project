"""Original vs paraphrased IPIP items: the contamination check (B1 step 4). Local only, no API calls.

A model that answers the original items partly from memory should change its
answers more when the wording changes than a model that reads the meaning.
Compares, per model, its results in results/raw/<slug>/bigfive/ (original)
with results/raw/<slug>/bigfive_paraphrased/ (from 01_run_bigfive.py --items
paraphrased). Only conditions present in both are compared.

Writes results/tables/paraphrase_*.csv and results/paraphrase_summary.md
    paraphrase_items     baseline: per item, mean answer on each wording
                         (over runs) and the keyed difference
    paraphrase_agreement baseline, per model: item-level correlation and mean
                         absolute difference between wordings; Jev's
                         run-to-run noise on the original as the reference
    paraphrase_domains   baseline, per model x domain: trait score on each
                         wording, difference with a bootstrap CI over items
    paraphrase_shifts    per model x persona trait x placement: high - low on
                         the target trait on each wording, with CIs, and the ratio
    paraphrase_alpha     alpha per domain on the paraphrased items

There is no pre-registered threshold: this is a descriptive check, read
against the LLMs and Jev's own noise.

Run from this folder:
    python 06_analyse_paraphrase.py
"""

import numpy as np
import pandas as pd

from helpers.bigfive import read_records
from helpers.config import (
    IPIP_PARAPHRASED_PATH,
    ITEM_SETS,
    PARAPHRASE_SUMMARY_PATH,
    RAW_RESULTS_DIR,
    TABLES_DIR,
)
from helpers.ipip import DOMAINS, SCALE_MAX, SCALE_MIN, keyed_score, load_items
from helpers.reliability import ALPHA_THRESHOLD, alpha_per_domain
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}
MODEL_ORDER = [JEV_MODEL, *LLM_BASELINE_MODELS]
ITEMS = load_items()
ITEM_BY_ID = {item.id: item for item in ITEMS}


def load() -> pd.DataFrame:
    rows = []
    for item_set, folder in ITEM_SETS.items():
        for path in sorted(RAW_RESULTS_DIR.glob(f"*/{folder}/*.jsonl")):
            model = path.parent.parent.name.replace("__", "/", 1)
            rows += [{**r, "model": model, "item_set": item_set} for r in read_records(path)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["variant"] = df["variant"].fillna(0).astype(int)
    df["keyed_score"] = [
        np.nan if pd.isna(a) else keyed_score(ITEM_BY_ID[i], a) for i, a in zip(df["item_id"], df["answer"])
    ]
    # Keep only (model, condition) pairs run on both wordings.
    both = df.groupby(["model", "condition"])["item_set"].nunique()
    keep = set(both[both == len(ITEM_SETS)].index)
    return df[[pair in keep for pair in zip(df["model"], df["condition"])]]


def _models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _paired_diff(a: np.ndarray, b: np.ndarray) -> stats.ConfidenceInterval:
    """Mean of a - b on the 1-5 scale, bootstrapped over items (the bootstrap takes 0-1 values)."""
    scale = SCALE_MAX - SCALE_MIN
    diff = stats.bootstrap_statistic([(np.asarray(a) - SCALE_MIN) / scale, (np.asarray(b) - SCALE_MIN) / scale],
                                     lambda x, y: float(x.mean() - y.mean()))
    return stats.ConfidenceInterval(diff.estimate * scale, diff.low * scale, diff.high * scale)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def items_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "baseline"]
    means = base.groupby(["model", "item_id", "item_set"])[["answer", "keyed_score"]].mean().unstack("item_set")
    table = pd.DataFrame({
        "original": means[("answer", "original")], "paraphrased": means[("answer", "paraphrased")],
        "keyed_difference": means[("keyed_score", "paraphrased")] - means[("keyed_score", "original")],
    }).reset_index()
    paraphrased = {i.id: i.text for i in load_items(IPIP_PARAPHRASED_PATH)} if IPIP_PARAPHRASED_PATH.exists() else {}
    table["domain"] = table["item_id"].map(lambda i: ITEM_BY_ID[i].domain)
    table["text"] = table["item_id"].map(lambda i: ITEM_BY_ID[i].text)
    table["paraphrase"] = table["item_id"].map(paraphrased)
    return table


def agreement_table(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[(df["condition"] == "baseline") & (df["item_set"] == "original")]
    for model in _models(items):
        sub = items[items["model"] == model].dropna(subset=["original", "paraphrased"])
        runs = base[base["model"] == model].groupby("item_id")["answer"]
        noise = runs.std(ddof=1).mean() if runs.size().max() > 1 else np.nan
        rows.append({"model": model, "n_items": len(sub),
                     "correlation": np.corrcoef(sub["original"], sub["paraphrased"])[0, 1],
                     "mean_abs_difference": (sub["paraphrased"] - sub["original"]).abs().mean(),
                     "original_run_noise_sd": noise})
    return pd.DataFrame(rows)


def domains_table(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in _models(items):
        for domain in DOMAINS:
            sub = items[(items["model"] == model) & (items["domain"] == domain)]
            orig = np.array([keyed_score(ITEM_BY_ID[i], a) for i, a in zip(sub["item_id"], sub["original"])])
            para = orig + sub["keyed_difference"].to_numpy()
            diff = _paired_diff(para, orig)
            rows.append({"model": model, "domain": domain, "original": orig.mean(), "paraphrased": para.mean(),
                         "difference": diff.estimate, "ci_low": diff.low, "ci_high": diff.high})
    return pd.DataFrame(rows)


def _shift(rows: pd.DataFrame) -> tuple[float, float, float]:
    hi = rows[rows["pole"] == "high"].groupby("item_id")["keyed_score"].mean()
    lo = rows[rows["pole"] == "low"].groupby("item_id")["keyed_score"].mean()
    paired = pd.concat([hi, lo], axis=1, keys=["high", "low"]).dropna()
    if paired.empty:
        return np.nan, np.nan, np.nan
    diff = _paired_diff(paired["high"].to_numpy(), paired["low"].to_numpy())
    return diff.estimate, diff.low, diff.high


def shifts_table(df: pd.DataFrame) -> pd.DataFrame:
    persona = df[(df["condition"] != "baseline") & (df["domain"] == df["persona_domain"])]
    rows = []
    for (model, domain, placement), group in persona.groupby(["model", "persona_domain", "placement"]):
        o = _shift(group[group["item_set"] == "original"])
        p = _shift(group[group["item_set"] == "paraphrased"])
        rows.append({"model": model, "persona_domain": domain, "placement": placement,
                     "shift_original": o[0], "ci_low_original": o[1], "ci_high_original": o[2],
                     "shift_paraphrased": p[0], "ci_low_paraphrased": p[1], "ci_high_paraphrased": p[2],
                     "ratio": p[0] / o[0] if o[0] else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return table.sort_values(["model", "persona_domain", "placement"],
                             key=lambda c: c.map(order) if c.name == "model" else c).reset_index(drop=True)


def alpha_table(df: pd.DataFrame) -> pd.DataFrame:
    para = df[df["item_set"] == "paraphrased"]
    items = load_items(IPIP_PARAPHRASED_PATH) if IPIP_PARAPHRASED_PATH.exists() else ITEMS
    rows = []
    for model in _models(para):
        sets = []
        for _, group in para[para["model"] == model].groupby(["condition", "variant", "run"]):
            answers = dict(zip(group["item_id"], group["answer"]))
            if len(answers) == len(items) and not group["answer"].isna().any():
                sets.append(answers)
        alphas = alpha_per_domain(items, sets) if len(sets) >= 2 else {}
        rows += [{"model": model, "domain": d, "alpha": alphas.get(d, np.nan), "n_sets": len(sets),
                  "interpretable": bool(alphas.get(d, 0) >= ALPHA_THRESHOLD)} for d in DOMAINS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary(agreement: pd.DataFrame, domains: pd.DataFrame, shifts: pd.DataFrame, items: pd.DataFrame) -> str:
    lines = ["# IPIP paraphrases: automatic summary (B1 step 4)", "",
             "Written by `06_analyse_paraphrase.py`; do not edit by hand. Compares each model's answers to the "
             "original IPIP wording with its answers to the reviewed paraphrases. A model answering from memory "
             "should change more than the others, and more than its own run-to-run noise.", ""]
    if agreement.empty:
        return "\n".join([*lines, "No baseline run on both wordings yet."]) + "\n"
    lines += ["## Baseline: item-level agreement", "",
              "| Model | Items | Correlation | Mean abs. difference (1-5) | Run-to-run noise (SD) |",
              "|---|---|---|---|---|"]
    for _, r in agreement.iterrows():
        noise = "–" if pd.isna(r["original_run_noise_sd"]) else f"{r['original_run_noise_sd']:.2f}"
        lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['n_items']} | {r['correlation']:.2f} | "
                     f"{r['mean_abs_difference']:.2f} | {noise} |")
    lines += ["", "## Baseline: trait scores (1-5), paraphrased - original", "",
              "| Model | " + " | ".join(DOMAINS) + " |", "|---|" + "---|" * len(DOMAINS)]
    for model, group in domains.groupby("model", sort=False):
        cells = [f"{r['difference']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]" for _, r in group.iterrows()]
        lines.append(f"| {NAMES.get(model, model)} | " + " | ".join(cells) + " |")
    if not shifts.empty:
        lines += ["", "## Persona pull (high - low on the target trait)", "",
                  "| Model | Trait | Placement | Original | Paraphrased | Ratio |", "|---|---|---|---|---|---|"]
        for _, r in shifts.iterrows():
            lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['persona_domain']} | {r['placement']} | "
                         f"{r['shift_original']:.2f} [{r['ci_low_original']:.2f}, {r['ci_high_original']:.2f}] | "
                         f"{r['shift_paraphrased']:.2f} [{r['ci_low_paraphrased']:.2f}, "
                         f"{r['ci_high_paraphrased']:.2f}] | {r['ratio']:.2f} |")
    top = items.reindex(items["keyed_difference"].abs().sort_values(ascending=False).index).head(10)
    lines += ["", "## Items that moved most (baseline, keyed difference)", "",
              "Check these first: a large move can also mean the paraphrase changed the meaning.", "",
              "| Model | Item | Original | Paraphrase | Keyed difference |", "|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(f"| {NAMES.get(r['model'], r['model'])} | {r['item_id']} | {r['text']} | "
                     f"{r['paraphrase'] or '–'} | {r['keyed_difference']:+.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    if df.empty or "paraphrased" not in set(df["item_set"]):
        raise SystemExit("No condition run on both wordings yet; run 01_run_bigfive.py --items paraphrased first")
    print(f"Loaded {len(df)} answers ({int(df['answer'].isna().sum())} failed calls, left out)")
    items = items_table(df)
    tables = {"paraphrase_items": items, "paraphrase_agreement": agreement_table(df, items),
              "paraphrase_domains": domains_table(items), "paraphrase_shifts": shifts_table(df),
              "paraphrase_alpha": alpha_table(df)}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.empty:
            print(f"  {name}: no data yet")
            continue
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
        print(f"  {name}: {len(table)} rows")
    PARAPHRASE_SUMMARY_PATH.write_text(summary(tables["paraphrase_agreement"], tables["paraphrase_domains"],
                                               tables["paraphrase_shifts"], items), encoding="utf-8")
    print(f"Wrote {TABLES_DIR} and {PARAPHRASE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
