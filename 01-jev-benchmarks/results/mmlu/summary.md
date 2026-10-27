# MMLU number swap (draft)

Written by `11d_analyse_mmlu.py` over 200 items (PLAN A11). It states the pass rule's numbers; the reading is done by hand.

Rule (fixed 2026-10-18, before any run): log-odds drop CI entirely above 0 **and** the old answer's share of wrong answers above 33% (one-sided binomial, p < 0.05).

## Jev

- Accuracy: original 96.0% [93.0%, 98.5%], swapped 75.0% [69.0%, 80.5%]; McNemar p=3.08e-11.
- Log-odds drop 2.03 [1.46, 2.89].
- Old answer picked on 10.0% of swapped items; 40% of 50 wrong answers (p=0.196).
- -> does not support memorisation.

## Qwen3.8-27B

- Accuracy: original 68.0% [61.5%, 74.5%], swapped 61.5% [54.5%, 68.0%]; McNemar p=0.148.
- Log-odds drop 0.28 [-0.07, 0.65].
- Old answer picked on 12.5% of swapped items; 32% of 77 wrong answers (p=0.606).
- -> does not support memorisation.

## Gemma-4-26B-A4B

- Accuracy: original 75.0% [69.0%, 81.0%], swapped 70.5% [64.0%, 76.5%]; McNemar p=0.253.
- Log-odds drop 0.23 [-0.12, 0.58].
- Old answer picked on 8.5% of swapped items; 29% of 59 wrong answers (p=0.808).
- -> does not support memorisation.

## Jev vs the LLMs

- vs Qwen3.8-27B: drop difference 1.74 [1.10, 2.66] -> Jev relies on recall more.
- vs Gemma-4-26B-A4B: drop difference 1.80 [1.15, 2.71] -> Jev relies on recall more.

