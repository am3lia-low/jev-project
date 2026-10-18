"""Tables and the draft summary for the MMLU number-swap sub-experiment (A11).

Per model, over the same items in both versions:
- accuracy on original and swapped (bootstrap 95% CI), paired McNemar;
- log-odds drop original -> swapped with a bootstrap CI;
- trap rate: share of swapped items where the old answer is picked, and its
  share of the model's wrong answers (chance MMLU_TRAP_CHANCE = 1/3), with a
  one-sided binomial test. Failed calls count as wrong for accuracy (A9) but
  are left out of the wrong answers, as they picked no option;
- memorisation supported if the drop's CI is entirely above 0 AND the trap's
  share of wrong answers is above 1/3 (p < SIGNIFICANCE). Fixed 2026-10-18,
  before any run.
Jev vs each LLM: bootstrap CI of (Jev drop - LLM drop), a comparison, not a
control (the LLMs may also have seen MMLU).

Writes results/mmlu/tables/*.csv and results/mmlu/summary.md (overwritten on
every run; the reading is written by hand elsewhere).

Run from this folder:
    python 11d_analyse_mmlu.py
"""

import json

import numpy as np
import pandas as pd

from helpers import config
from helpers.evaluation import model_slug, read_records
from jev_common import stats
from jev_common.models import JEV_MODEL, LLM_BASELINE_MODELS

ITEMS_PATH = config.MMLU_DIR / "items.jsonl"
RAW_DIR = config.MMLU_RESULTS_DIR / "raw"
TABLES_DIR = config.MMLU_RESULTS_DIR / "tables"
SUMMARY_PATH = config.MMLU_RESULTS_DIR / "summary.md"
NAMES = {JEV_MODEL: "Jev", LLM_BASELINE_MODELS[0]: "Qwen3.8-27B", LLM_BASELINE_MODELS[1]: "Gemma-4-26B-A4B"}


def load(model: str, traps: dict[str, str]) -> pd.DataFrame | None:
    """One row per item: subject, correct_original, correct_swapped, pred_swapped, trap."""
    paths = {v: RAW_DIR / model_slug(model) / f"{v}.jsonl" for v in ("original", "swapped")}
    if not all(p.exists() for p in paths.values()):
        return None
    original = {r["id"]: r for r in read_records(paths["original"])}
    swapped = {r["id"]: r for r in read_records(paths["swapped"])}
    ids = [i for i in traps if i in original and i in swapped]
    return pd.DataFrame({
        "id": ids,
        "subject": [original[i]["dataset"] for i in ids],
        "correct_original": [float(original[i]["correct"]) for i in ids],
        "correct_swapped": [float(swapped[i]["correct"]) for i in ids],
        "pred_swapped": [swapped[i]["pred"] for i in ids],
        "trap": [traps[i] for i in ids],
    }).set_index("id")


def drop(original, swapped) -> float:
    return stats.log_odds_drop(original, swapped)


def model_row(name: str, frame: pd.DataFrame) -> dict:
    o, s = frame["correct_original"].to_numpy(), frame["correct_swapped"].to_numpy()
    acc_o, acc_s = stats.bootstrap_ci(o), stats.bootstrap_ci(s)
    drop_ci = stats.bootstrap_statistic([o, s], drop)
    mcnemar = stats.mcnemar_exact(o, s)
    answered_wrong = frame[frame["pred_swapped"].notna() & (frame["correct_swapped"] == 0)]
    n_trap = int((answered_wrong["pred_swapped"] == answered_wrong["trap"]).sum())
    n_wrong = len(answered_wrong)
    p_trap = stats.binomial_test_vs_chance(n_trap, n_wrong, config.MMLU_TRAP_CHANCE) if n_wrong else 1.0
    trap_above = n_wrong > 0 and p_trap < config.SIGNIFICANCE
    return {
        "model": name, "n": len(frame),
        "acc_original": acc_o.estimate, "acc_original_low": acc_o.low, "acc_original_high": acc_o.high,
        "acc_swapped": acc_s.estimate, "acc_swapped_low": acc_s.low, "acc_swapped_high": acc_s.high,
        "mcnemar_original_only": mcnemar.a_only, "mcnemar_swapped_only": mcnemar.b_only,
        "mcnemar_p": mcnemar.p_value,
        "drop": drop_ci.estimate, "drop_low": drop_ci.low, "drop_high": drop_ci.high,
        "trap_rate": n_trap / len(frame), "wrong_answers": n_wrong, "trap_picks": n_trap,
        "trap_share_of_wrong": n_trap / n_wrong if n_wrong else np.nan, "trap_p": p_trap,
        "supported": bool(drop_ci.low > 0 and trap_above),
    }


def summary(rows: pd.DataFrame, versus: pd.DataFrame, n_items: int) -> str:
    lines = ["# MMLU number swap (draft)", "",
             f"Written by `11d_analyse_mmlu.py` over {n_items} items (PLAN A11). It states the pass rule's numbers; "
             "the reading is done by hand.", "",
             "Rule (fixed 2026-10-18, before any run): log-odds drop CI entirely above 0 **and** the old answer's "
             f"share of wrong answers above {config.MMLU_TRAP_CHANCE:.0%} (one-sided binomial, p < {config.SIGNIFICANCE:g}).", ""]
    for r in rows.itertuples():
        share = "no wrong answers" if not r.wrong_answers else f"{r.trap_share_of_wrong:.0%} of {r.wrong_answers} wrong answers (p={r.trap_p:.3g})"
        lines += [f"## {r.model}", "",
                  f"- Accuracy: original {r.acc_original:.1%} [{r.acc_original_low:.1%}, {r.acc_original_high:.1%}], "
                  f"swapped {r.acc_swapped:.1%} [{r.acc_swapped_low:.1%}, {r.acc_swapped_high:.1%}]; McNemar p={r.mcnemar_p:.3g}.",
                  f"- Log-odds drop {r.drop:.2f} [{r.drop_low:.2f}, {r.drop_high:.2f}].",
                  f"- Old answer picked on {r.trap_rate:.1%} of swapped items; {share}.",
                  f"- -> {'supports' if r.supported else 'does not support'} memorisation.", ""]
    if not versus.empty:
        lines += ["## Jev vs the LLMs", ""]
        lines += [f"- vs {r.baseline}: drop difference {r.difference:.2f} [{r.low:.2f}, {r.high:.2f}] -> "
                  f"{'Jev relies on recall more' if r.low > 0 else 'no clear difference'}." for r in versus.itertuples()]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not ITEMS_PATH.exists():
        raise SystemExit(f"{ITEMS_PATH} does not exist yet; run 11b_mmlu_build.py first")
    with ITEMS_PATH.open(encoding="utf-8") as handle:
        traps = {r["id"]: r["swapped"]["trap"] for r in map(json.loads, filter(str.strip, handle))}
    frames = {m: f for m in (JEV_MODEL, *LLM_BASELINE_MODELS) if (f := load(m, traps)) is not None}
    if not frames:
        raise SystemExit(f"No results in {RAW_DIR} yet; run 11c_run_mmlu.py first")

    rows = pd.DataFrame([model_row(NAMES[m], f) for m, f in frames.items()])
    by_subject = pd.DataFrame([
        {"model": NAMES[m], "subject": subject, "n": len(part),
         "acc_original": part["correct_original"].mean(), "acc_swapped": part["correct_swapped"].mean()}
        for m, f in frames.items() for subject, part in f.groupby("subject")
    ])
    versus = []
    if JEV_MODEL in frames:
        jev = frames[JEV_MODEL]
        for baseline in LLM_BASELINE_MODELS:
            if baseline not in frames:
                continue
            both = jev.join(frames[baseline], rsuffix="_b", how="inner")
            ci = stats.bootstrap_statistic(
                [both["correct_original"], both["correct_swapped"], both["correct_original_b"], both["correct_swapped_b"]],
                lambda jo, js, bo, bs: drop(jo, js) - drop(bo, bs))
            versus.append({"baseline": NAMES[baseline], "n": len(both),
                           "difference": ci.estimate, "low": ci.low, "high": ci.high})
    versus = pd.DataFrame(versus)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (("memorisation", rows), ("accuracy_by_subject", by_subject), ("jev_vs_llm", versus)):
        if not table.empty:
            table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.4f")
    SUMMARY_PATH.write_text(summary(rows, versus, len(traps)), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
