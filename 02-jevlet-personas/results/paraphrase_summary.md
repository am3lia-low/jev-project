# IPIP paraphrases: automatic summary (B1 step 4)

Written by `06_analyse_paraphrase.py`; do not edit by hand. Compares each model's answers to the original IPIP wording with its answers to the reviewed paraphrases. A model answering from memory should change more than the others, and more than its own run-to-run noise.

## Baseline: item-level agreement

| Model | Items | Correlation | Mean abs. difference (1-5) | Run-to-run noise (SD) |
|---|---|---|---|---|
| Jev | 120 | 0.95 | 0.18 | 0.04 |
| Qwen3.8-27B | 120 | 0.87 | 0.17 | – |
| Gemma-4-26B-A4B | 120 | 0.78 | 0.41 | – |

## Baseline: trait scores (1-5), paraphrased - original

| Model | N | E | O | A | C |
|---|---|---|---|---|---|
| Jev | -0.01 [-0.08, +0.06] | -0.04 [-0.12, +0.04] | +0.03 [-0.05, +0.11] | +0.03 [-0.08, +0.14] | -0.03 [-0.14, +0.07] |
| Qwen3.8-27B | +0.10 [+0.00, +0.22] | -0.02 [-0.10, +0.04] | +0.06 [-0.01, +0.14] | +0.02 [-0.15, +0.20] | -0.07 [-0.18, +0.04] |
| Gemma-4-26B-A4B | -0.14 [-0.48, +0.18] | -0.01 [-0.22, +0.20] | +0.17 [-0.20, +0.55] | -0.14 [-0.50, +0.22] | +0.04 [-0.27, +0.34] |

## Persona pull (high - low on the target trait)

| Model | Trait | Placement | Original | Paraphrased | Ratio |
|---|---|---|---|---|---|
| Jev | A | instructions | 3.03 [2.68, 3.31] | 3.01 [2.68, 3.29] | 0.99 |
| Jev | A | state | 3.41 [3.04, 3.69] | 3.34 [2.99, 3.61] | 0.98 |
| Jev | C | instructions | 3.22 [3.02, 3.35] | 3.19 [2.93, 3.36] | 0.99 |
| Jev | C | state | 3.69 [3.46, 3.84] | 3.63 [3.34, 3.81] | 0.99 |
| Jev | E | instructions | 3.29 [3.14, 3.42] | 3.23 [3.02, 3.40] | 0.98 |
| Jev | E | state | 3.63 [3.46, 3.76] | 3.55 [3.32, 3.73] | 0.98 |
| Jev | N | instructions | 2.81 [2.51, 3.09] | 2.95 [2.62, 3.25] | 1.05 |
| Jev | N | state | 3.18 [2.84, 3.50] | 3.23 [2.85, 3.57] | 1.01 |
| Jev | O | instructions | 2.67 [2.31, 2.99] | 2.62 [2.24, 2.97] | 0.98 |
| Jev | O | state | 3.06 [2.64, 3.43] | 3.00 [2.56, 3.40] | 0.98 |
| Qwen3.8-27B | A | instructions | 3.07 [2.69, 3.42] | 2.82 [2.39, 3.21] | 0.92 |
| Qwen3.8-27B | A | state | 3.56 [3.27, 3.80] | 3.34 [2.99, 3.64] | 0.94 |
| Qwen3.8-27B | C | instructions | 3.31 [3.12, 3.49] | 3.27 [3.08, 3.44] | 0.99 |
| Qwen3.8-27B | C | state | 3.92 [3.82, 3.98] | 3.93 [3.83, 3.98] | 1.00 |
| Qwen3.8-27B | E | instructions | 3.68 [3.49, 3.83] | 3.68 [3.51, 3.83] | 1.00 |
| Qwen3.8-27B | E | state | 3.93 [3.85, 3.99] | 3.94 [3.87, 3.99] | 1.00 |
| Qwen3.8-27B | N | instructions | 3.28 [2.95, 3.57] | 3.32 [3.02, 3.59] | 1.01 |
| Qwen3.8-27B | N | state | 3.69 [3.49, 3.86] | 3.72 [3.54, 3.87] | 1.01 |
| Qwen3.8-27B | O | instructions | 2.73 [2.34, 3.09] | 2.66 [2.23, 3.06] | 0.98 |
| Qwen3.8-27B | O | state | 3.44 [3.14, 3.70] | 3.31 [2.93, 3.64] | 0.96 |
| Gemma-4-26B-A4B | A | instructions | 3.67 [3.25, 4.00] | 3.64 [3.22, 3.98] | 0.99 |
| Gemma-4-26B-A4B | A | state | 3.65 [3.23, 3.99] | 3.58 [3.09, 3.98] | 0.98 |
| Gemma-4-26B-A4B | C | instructions | 3.67 [3.35, 3.93] | 3.44 [2.92, 3.85] | 0.94 |
| Gemma-4-26B-A4B | C | state | 3.93 [3.78, 4.00] | 3.87 [3.64, 4.00] | 0.99 |
| Gemma-4-26B-A4B | E | instructions | 3.87 [3.68, 4.00] | 3.60 [3.21, 3.90] | 0.93 |
| Gemma-4-26B-A4B | E | state | 3.93 [3.85, 3.99] | 3.68 [3.41, 3.90] | 0.94 |
| Gemma-4-26B-A4B | N | instructions | 3.41 [3.02, 3.74] | 3.62 [3.32, 3.88] | 1.06 |
| Gemma-4-26B-A4B | N | state | 3.69 [3.42, 3.89] | 3.77 [3.48, 3.97] | 1.02 |
| Gemma-4-26B-A4B | O | instructions | 3.24 [2.72, 3.68] | 2.92 [2.32, 3.46] | 0.90 |
| Gemma-4-26B-A4B | O | state | 3.49 [3.00, 3.86] | 3.46 [3.01, 3.82] | 0.99 |

## Items that moved most (baseline, keyed difference)

Check these first: a large move can also mean the paraphrase changed the meaning.

| Model | Item | Original | Paraphrase | Keyed difference |
|---|---|---|---|---|
| Gemma-4-26B-A4B | ipip-094 | Distrust people | Find people hard to trust | -2.00 |
| Gemma-4-26B-A4B | ipip-096 | Am not easily annoyed | Am not irritable | -2.00 |
| Gemma-4-26B-A4B | ipip-065 | Handle tasks smoothly | Manage assignments with ease | -2.00 |
| Gemma-4-26B-A4B | ipip-068 | Do not like poetry | Do not enjoy poems | +1.99 |
| Gemma-4-26B-A4B | ipip-001 | Worry about things | Feel concerned about things | +1.98 |
| Gemma-4-26B-A4B | ipip-043 | Feel others' emotions | Sense other people's feelings | +1.98 |
| Gemma-4-26B-A4B | ipip-044 | Am concerned about others | Care about other people | +1.97 |
| Gemma-4-26B-A4B | ipip-119 | Try not to think about the needy | Try to avoid thinking about people who are struggling | -1.97 |
| Gemma-4-26B-A4B | ipip-099 | Obstruct others' plans | Get in the way of others' plans | -1.86 |
| Gemma-4-26B-A4B | ipip-033 | Enjoy wild flights of fantasy | Delight in extravagant flights of fancy | -1.86 |
