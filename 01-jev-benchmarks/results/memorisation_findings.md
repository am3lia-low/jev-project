# Memorisation: findings (Part A, Levels 0–3 + quiz)

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

## Reading (draft; the final verdict is by hand, TODO 9)

Jev meets the pre-registered ladder criterion, but narrowly, only when pooled, and by less than the Qwen baseline. The quiz and Level 3 don't support memorisation. Overall this is **weak, inconclusive evidence** that Jev has memorised the benchmarks, no stronger than for a standard LLM. The PLAN fixes the rule, so it is reported as passing, with the caveats above alongside it.

The [MMLU number-swap sub-experiment](../../docs/PLAN.md#a11-mmlu-number-swap-sub-experiment) (PLAN A11) is designed to settle it: changing a calculation's numbers turns a recalled answer into a wrong one, which none of the tests above can do.

Still open: ARC-Challenge stays out of the ladder pool. The BERT learning curve (2026-10-18) shows BERT still improving on it but below the threshold; it can only rejoin through the RACE fallback, decided on BERT's evidence alone (PLAN A8).
