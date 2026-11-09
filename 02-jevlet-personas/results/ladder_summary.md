# Difficulty ladder: automatic summary (B2)

Written by `12_analyse_ladder.py`; do not edit by hand. Accuracy with a bootstrap 95% CI. "At chance" means the binomial test against chance is not significant (p >= 0.05).

HLE: 513 text-only multiple-choice questions of 2500.

## No persona: actual ability

| Model | ARC-Easy | ARC-Challenge | MMLU | GPQA Diamond | HLE (text-only MCQ) |
|---|---|---|---|---|---|
| Jev | 99.0% [97.5%, 100.0%] | 98.5% [96.5%, 100.0%] | 92.0% [88.0%, 95.5%] | 73.7% [67.7%, 79.8%] | 22.5% [17.0%, 28.5%] |
| Qwen3.8-27B | 98.5% [96.5%, 100.0%] | 98.5% [96.5%, 100.0%] | 79.5% [74.0%, 85.0%] | 48.0% [40.9%, 55.1%] | 14.0% [9.5%, 19.0%] (at chance) |
| Gemma-4-26B-A4B | 99.5% [98.5%, 100.0%] | 98.5% [96.5%, 100.0%] | 78.5% [73.0%, 84.0%] | 41.4% [34.3%, 48.5%] | 10.5% [6.5%, 15.0%] (at chance) |
| Chance | 25.0% | 25.0% | 25.0% | 25.0% | 16.5% |
| Humans (non-expert / expert) | – | – | 34% / 90% (Hendrycks et al. 2021) | 34% / 65% (Rein et al. 2023 (main set)) | – |

## Persona: drop in accuracy (no persona - persona)

| Model | Benchmark | Condition | No persona | Persona | Drop | McNemar p |
|---|---|---|---|---|---|---|
| Jev | ARC-Easy | secondary-instructions | 99.0% | 99.0% | +0.0% [+0.0%, +0.0%] | 1 |
| Jev | ARC-Easy | secondary-state | 99.0% | 99.0% | +0.0% [+0.0%, +0.0%] | 1 |
| Jev | ARC-Challenge | secondary-instructions | 98.5% | 98.5% | +0.0% [+0.0%, +0.0%] | 1 |
| Jev | ARC-Challenge | secondary-state | 98.5% | 98.5% | +0.0% [+0.0%, +0.0%] | 1 |
| Jev | MMLU | secondary-instructions | 92.0% | 92.0% | +0.0% [-2.0%, +2.0%] | 1 |
| Jev | MMLU | secondary-state | 92.0% | 91.0% | +1.0% [-1.0%, +3.0%] | 0.625 |
| Jev | GPQA Diamond | secondary-instructions | 73.7% | 77.3% | -3.5% [-7.1%, -0.5%] | 0.0654 |
| Jev | GPQA Diamond | secondary-state | 73.7% | 73.2% | +0.5% [-3.5%, +4.5%] | 1 |
| Jev | HLE (text-only MCQ) | secondary-instructions | 22.5% | 20.0% | +2.5% [-0.5%, +6.0%] | 0.227 |
| Jev | HLE (text-only MCQ) | secondary-state | 22.5% | 21.0% | +1.5% [-2.0%, +5.0%] | 0.581 |
| Qwen3.8-27B | ARC-Easy | secondary-instructions | 98.5% | 98.5% | +0.0% [-2.0%, +2.0%] | 1 |
| Qwen3.8-27B | ARC-Easy | secondary-state | 98.5% | 99.5% | -1.0% [-2.5%, +0.0%] | 0.5 |
| Qwen3.8-27B | ARC-Challenge | secondary-instructions | 98.5% | 98.5% | +0.0% [-1.5%, +1.5%] | 1 |
| Qwen3.8-27B | ARC-Challenge | secondary-state | 98.5% | 98.0% | +0.5% [-1.0%, +2.5%] | 1 |
| Qwen3.8-27B | MMLU | secondary-instructions | 79.5% | 78.5% | +1.0% [-3.0%, +5.0%] | 0.804 |
| Qwen3.8-27B | MMLU | secondary-state | 79.5% | 80.5% | -1.0% [-5.0%, +3.0%] | 0.815 |
| Qwen3.8-27B | GPQA Diamond | secondary-instructions | 48.0% | 44.9% | +3.0% [-2.0%, +8.1%] | 0.345 |
| Qwen3.8-27B | GPQA Diamond | secondary-state | 48.0% | 47.5% | +0.5% [-5.1%, +6.1%] | 1 |
| Qwen3.8-27B | HLE (text-only MCQ) | secondary-instructions | 14.0% | 12.5% | +1.5% [-2.0%, +5.0%] | 0.581 |
| Qwen3.8-27B | HLE (text-only MCQ) | secondary-state | 14.0% | 15.0% | -1.0% [-4.5%, +2.5%] | 0.774 |
| Gemma-4-26B-A4B | ARC-Easy | secondary-instructions | 99.5% | 94.0% | +5.5% [+2.5%, +9.0%] | 0.000977 |
| Gemma-4-26B-A4B | ARC-Easy | secondary-state | 99.5% | 93.0% | +6.5% [+3.5%, +10.0%] | 0.000244 |
| Gemma-4-26B-A4B | ARC-Challenge | secondary-instructions | 98.5% | 93.5% | +5.0% [+2.0%, +8.0%] | 0.00195 |
| Gemma-4-26B-A4B | ARC-Challenge | secondary-state | 98.5% | 95.5% | +3.0% [+1.0%, +5.5%] | 0.0312 |
| Gemma-4-26B-A4B | MMLU | secondary-instructions | 78.5% | 74.0% | +4.5% [-1.0%, +10.0%] | 0.136 |
| Gemma-4-26B-A4B | MMLU | secondary-state | 78.5% | 77.5% | +1.0% [-4.5%, +6.5%] | 0.856 |
| Gemma-4-26B-A4B | GPQA Diamond | secondary-instructions | 41.4% | 40.4% | +1.0% [-4.0%, +6.1%] | 0.851 |
| Gemma-4-26B-A4B | GPQA Diamond | secondary-state | 41.4% | 41.9% | -0.5% [-6.1%, +5.1%] | 1 |
| Gemma-4-26B-A4B | HLE (text-only MCQ) | secondary-instructions | 10.5% | 11.5% | -1.0% [-5.5%, +3.5%] | 0.824 |
| Gemma-4-26B-A4B | HLE (text-only MCQ) | secondary-state | 10.5% | 10.5% | +0.0% [-4.0%, +4.0%] | 1 |
