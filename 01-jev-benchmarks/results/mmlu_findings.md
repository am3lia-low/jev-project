# MMLU number swap: findings (PLAN A11)

Run 2026-10-27 on the API laptop; written up 2026-10-27. The design is in [PLAN A11](../../docs/PLAN.md#a11-mmlu-number-swap-sub-experiment). This file is written by hand. The numbers come from `11d_analyse_mmlu.py`; its automatic draft is [`mmlu/summary.md`](mmlu/summary.md) and the full tables are in [`mmlu/tables/`](mmlu/tables/).

## What was run

- **Items:** 200 MMLU maths test questions (elementary 92, high school 96, college 12). Sonnet changed only the numbers and wrote a `solve()` function; the new answer was checked by running it. 30 items checked by hand (2026-10-24): all 30 passed all three checks.
- **Two versions per item:** the original (MMLU's question and options), and the swapped one (new numbers; options = new answer, **the old answer as a trap**, and 2 typical mistakes on the new numbers).
- **Models:** Jev, Qwen and Gemma, one run each. 1,200 calls, 0 failed, **$0.013**.

## How to read the numbers

- **Drop** = log-odds of accuracy on the original minus on the swapped version. 0 means the new numbers change nothing.
- **Trap share** = of a model's wrong answers on the swapped version, the share that picked the old answer. With 3 wrong options, chance is 1/3.
- **The rule** (fixed 2026-10-18, before any run): memorisation is supported if the drop's 95% CI is above 0 **and** the trap share is above 1/3 (binomial, p < 0.05).

## Findings

### 1. No model meets the pre-registered rule

| Model | Original | Swapped | Drop [95% CI] | Trap share of wrong answers | Rule |
|---|---|---|---|---|---|
| Jev | 96.0% | 75.0% | 2.03 [1.46, 2.89] | 40% of 50 (p = 0.20) | not met |
| Qwen | 68.0% | 61.5% | 0.28 [−0.07, 0.65] | 32% of 77 (p = 0.61) | not met |
| Gemma | 75.0% | 70.5% | 0.23 [−0.12, 0.58] | 29% of 59 (p = 0.81) | not met |

Jev passes the drop half by a wide margin and fails the trap half. Qwen and Gemma fail both.

### 2. Jev drops sharply when only the numbers change; the LLMs don't

| Subject (items) | Jev | Qwen | Gemma |
|---|---|---|---|
| Elementary (92) | 98.9 → 92.4% | 79.3 → 81.5% | 92.4 → 94.6% |
| High school (96) | 92.7 → 62.5% | 59.4 → 45.8% | 61.5 → 50.0% |
| College (12) | 100 → 41.7% | 50.0 → 33.3% | 50.0 → 50.0% |

- Jev's drop is larger than both LLMs': the difference is 1.74 [1.10, 2.66] vs Qwen and 1.80 [1.15, 2.71] vs Gemma. Both CIs are above 0, so by A11's comparison **Jev relies on recall more than the LLMs**.
- Jev got 44 items right in the original and wrong when swapped; it got 2 the other way round.
- **On the original items, Jev beats both LLMs by 20-28 points**, most of all on the harder subjects (college maths 12/12 vs 6/12). This is a "System One" model whose vendor docs list arithmetic and multi-step reasoning as weaknesses. After the swap, Jev lands close to the LLMs (75% vs 61.5-70.5%).
- **The swapped items aren't simply harder.** If they were, the LLMs would drop too; their drops are small and not significant. (The LLMs may also have seen MMLU, so they are a comparison, not a clean control.)

### 3. Jev doesn't specifically fall back on the old answer

- When Jev is wrong on a swapped item, it picks the old answer 40% of the time. That is above the 1/3 chance level, but with only 50 wrong answers it is not significant (p = 0.20). Of the 44 items that flipped from right to wrong, 18 went to the trap.
- **Extra check (post-hoc, not part of the rule):** across all 200 swapped items, Jev puts on average 0.105 probability on the trap vs 0.088 on each other wrong option. The difference, 0.017 [−0.009, 0.045], includes 0. For Qwen (−0.017) and Gemma (−0.030) the trap gets slightly *less* than the other wrong options.
- So Jev leans slightly towards the old answer, but not clearly. The test had little power: Jev is right so often that it produced few wrong answers to count.

## Reading

By the pre-registered rule, the MMLU test **does not support memorisation** for any model, and it is reported that way.

The drop half is still the clearest signal in Part A. Jev scores far above what its maths ability suggests on the original MMLU items, and falls to about the LLMs' level once the numbers change. The LLMs don't drop. The likeliest reading is that **Jev's MMLU maths accuracy is inflated by exposure to these items**. The evidence does not show *how*: Jev does not reliably pick the remembered answer value (finding 3). It may recognise an item's form or answer pattern without holding the exact number. That last point is a guess, not a result.

## Caveats

- **Few college items (12)**, so the per-subject numbers there are rough.
- **The trap test is underpowered for a model this accurate.** 50 wrong answers can't show a modest trap preference. The rule was fixed before the run and stays as it is.
- **One run per model**, as in A9 (Jev's repeatability is 99.5%).
- **Swapped numbers can be less "round"** than MMLU's. The LLMs not dropping argues against this explaining Jev's drop, but it can't be ruled out for Jev alone.
- **The extra probability check was chosen after seeing the results.** Read it as description only.
