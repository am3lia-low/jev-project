# Memorisation: findings (Part A, Levels 0–3, quiz and MMLU number swap)

All runs finished 2026-10-18 on the API laptop; written up 2026-10-18. The design is in [PLAN A6–A8](../../docs/PLAN.md#a6-test-sets-and-the-memorisation-ladder). This file is written by hand. The numbers come from `10_analyse.py`; its automatic draft is [`verdict.md`](verdict.md) (overwritten on every run) and the full tables are in [`tables/`](tables/). The BERT side is in [`bert_summary.md`](bert_summary.md).

## What was run

| | Jev (`typesafe/jev-1.13`) | Qwen3.8-27B | Gemma-4-26B-A4B | BERT-base |
|---|---|---|---|---|
| Levels 0–2 | 2,000 each | 2,000 each | 2,000 each | 2,000 each × 5 seeds |
| Level 3 (new questions) | 200 | 200 | 200 | 200 × 5 seeds |
| Quiz | 494 | 494 | 494 | — |
| Failed calls | 0 | 0 | 0 | — |

- Level 2 and quiz runs: Jev $0.05, LLMs $0.08. LLMs stayed on their pinned providers (DekaLLM for Qwen, Makora for Gemma).
- Jev repeatability: all 3 runs agree on 99.5% of 200 questions, so one run per level.

## Accuracy by level (all datasets pooled)

| Level | Jev | Qwen | Gemma | BERT |
|---|---|---|---|---|
| 0 original | 92.9% | 93.2% | 88.4% | 47.6% |
| 1 shuffled options | 93.1% | 92.4% | 87.5% | 47.6% |
| 2 paraphrased + shuffled | 89.7% | 88.5% | 83.5% | 45.2% |
| 3 new questions (paper datasets, vs Level 0 on the same 4) | 100.0% (vs 94.8%) | 99.5% (vs 94.1%) | 100.0% (vs 91.3%) | 54.4% (vs 42.8%) |

## Findings

### 1. Option order doesn't matter (Level 0 → 1)

No model changes significantly when the options are shuffled (McNemar p > 0.15 for every model and dataset). Whatever Jev may have memorised, it isn't answer positions.

### 2. Ladder (Level 0 → 2): Jev passes the pre-registered rule, narrowly

The rule (PLAN A8, fixed before any Level 2 run): the model's log-odds drop must be ≥ 2× BERT's and the bootstrap 95% CI of the difference must lie above 0, pooled over the datasets where BERT passes the above-chance rule (SWAG, ARC-Easy, CommonsenseQA).

| | Pooled drop (log-odds) | × BERT's 0.20 | Difference, 95% CI | Rule |
|---|---|---|---|---|
| Jev | 0.41 | 2.05× | 0.21 [0.03, 0.40] | passes |
| Qwen | 0.60 | 3.0× | 0.40 [0.21, 0.61] | passes |
| Gemma | 0.34 | 1.7× | 0.14 [−0.03, 0.31] | fails |

Why this is weak evidence for Jev:
- **It passes only when pooled.** Jev passes on none of the three datasets on its own, and ARC-Easy went *up* (99.25% → 99.5%).
- **It's marginal:** 2.05× against a 2× threshold, with a CI lower bound of 0.03.
- **Qwen drops more than Jev.** Paraphrased wording costs ordinary LLMs more than it costs BERT, so the ladder may be picking up an LLM-wide paraphrase effect, not Jev-specific recall.
- **BERT was trained on paraphrases in the same style** (PLAN A3.1), so Level 2 wording is familiar to it and new to the LLMs. Some of the extra drop may come from this.
- HellaSwag (reported separately, BERT invalid there) "passes" for all three LLMs, but only because BERT's HellaSwag drop is negative.

### 3. Quiz: uninformative about memorised wording

| Original picked | Benchmark items (n=396) | Control items (n=98) |
|---|---|---|
| Jev | 62.6% | 81.6% |
| Qwen | 0.3% | 1.0% |
| Gemma | 70.5% | 80.6% |

- Every model picks the original **more** often on control items, which it cannot have seen (Level 3 questions written for this project), than on benchmark items. The quiz seems to reward the most natural-sounding wording over the synonym swaps. It doesn't detect prior exposure.
- Jev's benchmark-vs-control gap comes mostly from HellaSwag: original picked on 35.7% of benchmark items vs 91.3% of controls, and option E ("none") picked on 63% of HellaSwag benchmark items.
- **Qwen's quiz result is set aside:** it picks E on 492 of 494 items (the prompt tells it to when it doesn't recognise the dataset), so its result is uninformative, not a negative.

### 4. Level 3: no drop on unseen questions

All three LLMs score 99.5–100% on the 200 new questions, *higher* than their Level 0 accuracy on the same four datasets. BERT also scores higher (54.4% vs 42.8%), so the new questions are easier than the benchmark ones. Part of the reason is an option-length cue: the correct option is the longest in 45% of Level 3 questions vs ~26% at Level 0 ([`bert_summary.md`](bert_summary.md#level-3-has-an-option-length-cue)). That limits how much a missing drop can show, but it's the opposite of what a model relying on recall would produce.

### 5. Sensitivity check: BERT on a bigger training pool (pre-registered 2026-10-20)

BERT was re-trained with more training questions on the three ladder datasets: ARC-Easy +950 (all that was left), CommonsenseQA and SWAG +1,100 each (17,300 training rows instead of 11,000; 5 seeds; Kaggle, finished 2026-10-21). Same test questions, same Jev and LLM results, same rules. As fixed in [PLAN A8](../../docs/PLAN.md#a8-metrics-and-analysis) before the run, **the verdict stays on the main BERT run**; this only asks whether the result holds. Numbers: [`sensitivity_bert_bigpool.md`](sensitivity_bert_bigpool.md).

**BERT got better, but its paraphrase drop barely moved.**

| | Main BERT | Bigger-pool BERT |
|---|---|---|
| Val accuracy (originals): ARC-Easy / CSQA / SWAG | 48.8 / 44.4 / 63.0% | 53.4 / 47.6 / 65.5% |
| Datasets passing the above-chance rule | ARC-Easy, CSQA, SWAG | the same (ARC-Challenge 37.5%, still below) |
| Pooled log-odds drop, Level 0 → 2 | 0.20 | 0.21 |

**The ladder rule with the bigger-pool BERT:**

| | Ratio to BERT's drop | Difference [95% CI] | Rule |
|---|---|---|---|
| Jev | 1.98x (was 2.06x) | 0.20 [0.03, 0.39] | **fails** (was: passes) |
| Qwen | 2.90x (was 3.02x) | 0.39 [0.20, 0.60] | passes (as before) |
| Gemma | 1.63x (was 1.70x) | 0.13 [−0.04, 0.30] | fails (as before) |

**Why the flip is not caused by the bigger pool, and what it does show.** The two BERT runs differ by 0.01 in their drop. The five seeds *within* one run differ by up to 0.08:

| | BERT drop, mean of 5 seeds | Range over seeds | Jev's ratio, seed by seed |
|---|---|---|---|
| Main BERT | 0.198 (SD 0.027) | 0.154 – 0.221 | 1.86x – 2.66x |
| Bigger-pool BERT | 0.207 (SD 0.030) | 0.156 – 0.234 | 1.75x – 2.63x |

So Jev sits on the 2x line: against single seeds of the *main* run, its drop clears 2x on 3 of 5 and misses on 2. The two halves of the rule behave differently:
- **"Jev drops more than BERT" (CI above 0) is robust:** it holds with both BERT runs, CI low 0.03 each time.
- **"Jev drops at least 2x as much" is a coin flip:** it depends on which BERT seeds happen to be averaged, not on Jev.

Qwen passes both halves with both runs, by a clear margin. This adds to the reading below: Jev's ladder signal is not one that survives a reasonable change to the reference. Per dataset, CommonsenseQA now passes on its own (drop 0.45 vs 0.16), SWAG and ARC-Easy still fail.

### 6. MMLU number swap: Jev drops sharply, but doesn't pick the old answer (PLAN A11, run 2026-10-27)

A separate sub-experiment with its own pass rule (fixed 2026-10-18), reported next to the three signals above, not as a fourth. Full write-up: [`mmlu_findings.md`](mmlu_findings.md).

| Model | Original | Numbers swapped | Drop [95% CI] | Old answer's share of wrong answers | Rule |
|---|---|---|---|---|---|
| Jev | 96.0% | 75.0% | 2.03 [1.46, 2.89] | 40% of 50 (p = 0.20) | not met |
| Qwen | 68.0% | 61.5% | 0.28 [−0.07, 0.65] | 32% of 77 (p = 0.61) | not met |
| Gemma | 75.0% | 70.5% | 0.23 [−0.12, 0.58] | 29% of 59 (p = 0.81) | not met |

- **No model meets the rule.** Jev passes the drop half by a wide margin and fails the trap half.
- **Jev's drop is Jev-specific:** larger than both LLMs' (CI of the difference above 0 for each), while the LLMs' own drops are not significant. So the swapped items aren't simply harder.
- **Jev scores 20-28 points above the LLMs on the original maths items**, despite arithmetic being a listed weakness, and falls to about their level once the numbers change.
- **The trap test had little power** (only 50 wrong answers), and a post-hoc probability check also finds only a slight, non-significant lean towards the old answer.

## Verdict (drafted 2026-10-27; to be confirmed by hand, TODO 9)

**Pre-registered results, as fixed in PLAN A8 and A11:**

| Test | Jev | Qwen | Gemma |
|---|---|---|---|
| Ladder (main BERT run) | passes, narrowly (2.05x) | passes (3.0x) | fails |
| Quiz | no evidence | set aside | no evidence |
| Level 3 | no drop | no drop | no drop |
| MMLU number swap | not met (drop yes, trap no) | not met | not met |

**Reading.** On the paper benchmarks (ARC, CommonsenseQA, HellaSwag, SWAG), the evidence that Jev memorised them is **weak and inconclusive**. Jev's only pass is narrow, pooled, flips with an equally valid BERT (finding 5), and Qwen drops more. Rewording the questions costs Jev about what it costs an ordinary LLM.

**MMLU maths is different.** Changing only the numbers costs Jev 21 points while the LLMs barely move, and Jev's original-item accuracy is far above what its maths ability suggests. This is the strongest sign in Part A that Jev's score depends on exposure to the test items. The pre-registered rule is not met, because Jev does not reliably pick the remembered old answer, so it is reported as "not supported" with this caveat alongside it.

**In one line:** no pre-registered test shows Jev memorising the paper benchmarks beyond what a standard LLM shows; its MMLU maths accuracy, however, looks inflated by exposure to those items, which fits Deußer et al.'s concern about MMLU.

ARC-Challenge and HellaSwag stay out of the ladder pool: BERT is near chance on them, and the RACE fallback was not used (decided 2026-10-20).
