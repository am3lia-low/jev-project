# Scenarios (behaviour) vs self-report: automatic summary (B1 step 5)

Written by `09_analyse_scenarios.py`; do not edit by hand. Both measures on 0-1: behaviour = probability of the high action; self-report = (IPIP trait score - 1) / 4. Shift = high persona minus low persona on the persona's own trait.

## Does the persona move self-report, behaviour, or both?

| Model | Trait | Placement | Self-report shift | Behaviour shift | Moves |
|---|---|---|---|---|---|
| Jev | A | instructions | 0.76 [0.67, 0.83] | 0.85 [0.73, 0.94] | both |
| Jev | A | state | 0.85 [0.76, 0.92] | 0.86 [0.74, 0.95] | both |
| Jev | C | instructions | 0.80 [0.76, 0.84] | 0.69 [0.47, 0.88] | both |
| Jev | C | state | 0.92 [0.87, 0.96] | 0.71 [0.47, 0.90] | both |
| Jev | E | instructions | 0.82 [0.78, 0.86] | 0.97 [0.92, 1.00] | both |
| Jev | E | state | 0.91 [0.86, 0.94] | 0.97 [0.92, 1.00] | both |
| Jev | N | instructions | 0.70 [0.63, 0.77] | 0.64 [0.48, 0.78] | both |
| Jev | N | state | 0.80 [0.71, 0.87] | 0.64 [0.47, 0.79] | both |
| Jev | O | instructions | 0.67 [0.58, 0.75] | 0.97 [0.94, 0.99] | both |
| Jev | O | state | 0.76 [0.66, 0.86] | 0.98 [0.96, 0.99] | both |
| Qwen3.8-27B | A | instructions | 0.77 [0.67, 0.86] | 0.81 [0.68, 0.92] | both |
| Qwen3.8-27B | A | state | 0.89 [0.82, 0.95] | 0.83 [0.71, 0.93] | both |
| Qwen3.8-27B | C | instructions | 0.83 [0.78, 0.87] | 0.71 [0.50, 0.88] | both |
| Qwen3.8-27B | C | state | 0.98 [0.96, 1.00] | 0.71 [0.48, 0.90] | both |
| Qwen3.8-27B | E | instructions | 0.92 [0.87, 0.96] | 0.96 [0.93, 0.99] | both |
| Qwen3.8-27B | E | state | 0.98 [0.96, 1.00] | 0.98 [0.95, 0.99] | both |
| Qwen3.8-27B | N | instructions | 0.82 [0.74, 0.89] | 0.64 [0.46, 0.79] | both |
| Qwen3.8-27B | N | state | 0.92 [0.87, 0.96] | 0.65 [0.47, 0.80] | both |
| Qwen3.8-27B | O | instructions | 0.68 [0.58, 0.77] | 0.93 [0.88, 0.97] | both |
| Qwen3.8-27B | O | state | 0.86 [0.79, 0.93] | 0.96 [0.92, 0.98] | both |
| Gemma-4-26B-A4B | A | instructions | 0.92 [0.81, 1.00] | 0.89 [0.78, 0.98] | both |
| Gemma-4-26B-A4B | A | state | 0.91 [0.81, 1.00] | 0.86 [0.74, 0.95] | both |
| Gemma-4-26B-A4B | C | instructions | 0.92 [0.84, 0.98] | 0.68 [0.41, 0.90] | both |
| Gemma-4-26B-A4B | C | state | 0.98 [0.95, 1.00] | 0.69 [0.41, 0.92] | both |
| Gemma-4-26B-A4B | E | instructions | 0.97 [0.92, 1.00] | 0.98 [0.95, 1.00] | both |
| Gemma-4-26B-A4B | E | state | 0.98 [0.96, 1.00] | 0.98 [0.94, 1.00] | both |
| Gemma-4-26B-A4B | N | instructions | 0.85 [0.76, 0.94] | 0.79 [0.63, 0.92] | both |
| Gemma-4-26B-A4B | N | state | 0.92 [0.85, 0.97] | 0.71 [0.53, 0.86] | both |
| Gemma-4-26B-A4B | O | instructions | 0.81 [0.68, 0.92] | 0.95 [0.87, 1.00] | both |
| Gemma-4-26B-A4B | O | state | 0.87 [0.75, 0.97] | 0.97 [0.91, 1.00] | both |

## Behaviour shift on every trait (diagonal = target)

| Model | Persona | Placement | N | E | O | A | C |
|---|---|---|---|---|---|---|---|
| Jev | A | instructions | -0.48 | +0.46 | +0.55 | +0.85 | +0.29 |
| Jev | A | state | -0.45 | +0.44 | +0.54 | +0.86 | +0.28 |
| Jev | C | instructions | -0.38 | +0.10 | -0.32 | +0.23 | +0.69 |
| Jev | C | state | -0.35 | +0.09 | -0.35 | +0.20 | +0.71 |
| Jev | E | instructions | -0.14 | +0.97 | +0.60 | -0.00 | -0.11 |
| Jev | E | state | -0.15 | +0.97 | +0.60 | +0.01 | -0.09 |
| Jev | N | instructions | +0.64 | +0.09 | +0.07 | -0.41 | +0.05 |
| Jev | N | state | +0.64 | +0.06 | +0.06 | -0.41 | +0.05 |
| Jev | O | instructions | -0.33 | +0.49 | +0.97 | +0.36 | +0.50 |
| Jev | O | state | -0.29 | +0.45 | +0.98 | +0.37 | +0.48 |
| Qwen3.8-27B | A | instructions | -0.51 | +0.43 | +0.66 | +0.81 | +0.33 |
| Qwen3.8-27B | A | state | -0.56 | +0.46 | +0.68 | +0.83 | +0.36 |
| Qwen3.8-27B | C | instructions | -0.37 | +0.18 | -0.24 | +0.25 | +0.71 |
| Qwen3.8-27B | C | state | -0.45 | +0.19 | -0.22 | +0.27 | +0.71 |
| Qwen3.8-27B | E | instructions | -0.20 | +0.96 | +0.52 | +0.08 | -0.14 |
| Qwen3.8-27B | E | state | -0.23 | +0.98 | +0.57 | +0.11 | -0.11 |
| Qwen3.8-27B | N | instructions | +0.64 | +0.17 | +0.23 | -0.32 | -0.03 |
| Qwen3.8-27B | N | state | +0.65 | +0.16 | +0.21 | -0.35 | -0.01 |
| Qwen3.8-27B | O | instructions | -0.27 | +0.43 | +0.93 | +0.37 | +0.36 |
| Qwen3.8-27B | O | state | -0.30 | +0.43 | +0.96 | +0.40 | +0.41 |
| Gemma-4-26B-A4B | A | instructions | -0.51 | +0.39 | +0.70 | +0.89 | +0.44 |
| Gemma-4-26B-A4B | A | state | -0.46 | +0.41 | +0.72 | +0.86 | +0.38 |
| Gemma-4-26B-A4B | C | instructions | -0.37 | -0.20 | -0.35 | +0.22 | +0.68 |
| Gemma-4-26B-A4B | C | state | -0.36 | -0.16 | -0.40 | +0.19 | +0.69 |
| Gemma-4-26B-A4B | E | instructions | -0.23 | +0.98 | +0.57 | +0.04 | -0.18 |
| Gemma-4-26B-A4B | E | state | -0.23 | +0.98 | +0.59 | +0.07 | -0.10 |
| Gemma-4-26B-A4B | N | instructions | +0.79 | +0.04 | +0.10 | -0.48 | -0.17 |
| Gemma-4-26B-A4B | N | state | +0.71 | +0.12 | +0.11 | -0.47 | -0.05 |
| Gemma-4-26B-A4B | O | instructions | -0.39 | +0.16 | +0.95 | +0.33 | +0.48 |
| Gemma-4-26B-A4B | O | state | -0.36 | +0.28 | +0.97 | +0.33 | +0.48 |

## No persona: trait scores (0-1)

| Model | Measure | N | E | O | A | C |
|---|---|---|---|---|---|---|
| Jev | behaviour | 0.15 | 0.56 | 0.71 | 0.79 | 0.75 |
| Jev | self_report | 0.37 | 0.44 | 0.55 | 0.74 | 0.69 |
| Qwen3.8-27B | behaviour | 0.13 | 0.53 | 0.82 | 0.81 | 0.72 |
| Qwen3.8-27B | self_report | 0.41 | 0.48 | 0.52 | 0.69 | 0.61 |
| Gemma-4-26B-A4B | behaviour | 0.17 | 0.47 | 0.71 | 0.79 | 0.73 |
| Gemma-4-26B-A4B | self_report | 0.22 | 0.45 | 0.56 | 0.80 | 0.80 |
