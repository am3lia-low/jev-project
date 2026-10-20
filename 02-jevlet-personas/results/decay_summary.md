# Persona decay: automatic summary (B4)

Written by `04_analyse_decay.py`; do not edit by hand. Shift = high-persona score minus low-persona score on the persona's own trait (1-5 scale; 0 = no pull). Retention = shift divided by the shift with no history. Pooled over the persona traits; per-trait numbers are in `tables/decay_shifts.csv`.

## Shift with no history and at the longest history

| Model | History | Placement | Shift at 0 | Longest (turns) | Shift there | Retention there | Re-reminded, there |
|---|---|---|---|---|---|---|---|
| Jev | own | state | 3.51 [3.35, 3.65] | 32 | 2.91 [2.67, 3.14] | 0.83 | 2.90 [2.65, 3.13] |
| Jev | own | instructions | 3.12 [2.98, 3.25] | 32 | 1.22 [1.05, 1.39] | 0.39 | – |
| Jev | quiz | state | 3.51 [3.35, 3.65] | 32 | 3.33 [3.16, 3.49] | 0.95 | 3.30 [3.13, 3.45] |
| Jev | quiz | instructions | 3.12 [2.98, 3.25] | 32 | 2.45 [2.30, 2.60] | 0.79 | – |
| Jev | long | state | 3.51 [3.35, 3.65] | 256 | 3.27 [3.10, 3.42] | 0.93 | – |
| Jev | appdata | state | 3.51 [3.35, 3.65] | 32 | 3.28 [3.11, 3.44] | 0.94 | – |
| Jev | appdata | instructions | 3.12 [2.98, 3.25] | 32 | 2.27 [2.14, 2.40] | 0.73 | – |
| Qwen3.8-27B | own | state | 3.83 [3.75, 3.91] | 32 | 3.22 [2.98, 3.45] | 0.84 | 3.30 [3.06, 3.51] |
| Qwen3.8-27B | own | instructions | 3.47 [3.32, 3.60] | 32 | 2.79 [2.58, 2.99] | 0.80 | – |
| Qwen3.8-27B | quiz | state | 3.83 [3.75, 3.91] | 32 | 3.60 [3.40, 3.77] | 0.94 | 3.67 [3.50, 3.81] |
| Qwen3.8-27B | quiz | instructions | 3.47 [3.32, 3.60] | 32 | 3.60 [3.45, 3.72] | 1.04 | – |
| Qwen3.8-27B | long | state | 3.83 [3.75, 3.91] | 256 | 3.63 [3.46, 3.79] | 0.95 | – |
| Qwen3.8-27B | appdata | state | 3.83 [3.75, 3.91] | 32 | 3.60 [3.43, 3.74] | 0.94 | – |
| Qwen3.8-27B | appdata | instructions | 3.47 [3.32, 3.60] | 32 | 3.52 [3.37, 3.66] | 1.02 | – |
| Gemma-4-26B-A4B | own | state | 3.87 [3.76, 3.95] | 32 | 3.57 [3.35, 3.76] | 0.92 | 3.56 [3.34, 3.75] |
| Gemma-4-26B-A4B | own | instructions | 3.57 [3.34, 3.76] | 32 | 2.47 [2.22, 2.72] | 0.69 | – |
| Gemma-4-26B-A4B | quiz | state | 3.87 [3.76, 3.95] | 32 | 3.77 [3.58, 3.92] | 0.98 | 3.79 [3.60, 3.93] |
| Gemma-4-26B-A4B | quiz | instructions | 3.57 [3.34, 3.76] | 32 | 3.34 [3.10, 3.57] | 0.94 | – |
| Gemma-4-26B-A4B | long | state | 3.87 [3.76, 3.95] | 256 | 3.80 [3.64, 3.92] | 0.98 | – |
| Gemma-4-26B-A4B | appdata | state | 3.87 [3.76, 3.95] | 32 | 3.63 [3.40, 3.82] | 0.94 | – |
| Gemma-4-26B-A4B | appdata | instructions | 3.57 [3.34, 3.76] | 32 | 3.71 [3.51, 3.89] | 1.04 | – |

## No-persona scores per checkpoint

What the history does on its own. Trait scores, 1-5.

| Model | History | Turns | N | E | C |
|---|---|---|---|---|---|
| Jev | own | 0 | 2.46 | 2.75 | 3.73 |
| Jev | own | 2 | 2.98 | 2.98 | 3.04 |
| Jev | own | 4 | 2.99 | 2.99 | 3.01 |
| Jev | own | 8 | 3.00 | 3.00 | 3.00 |
| Jev | own | 16 | 3.00 | 3.00 | 3.00 |
| Jev | own | 32 | 3.00 | 3.00 | 3.00 |
| Jev | quiz | 0 | 2.46 | 2.75 | 3.73 |
| Jev | quiz | 2 | 2.47 | 2.70 | 3.60 |
| Jev | quiz | 4 | 2.52 | 2.74 | 3.66 |
| Jev | quiz | 8 | 2.33 | 2.67 | 3.85 |
| Jev | quiz | 16 | 2.32 | 2.67 | 3.90 |
| Jev | quiz | 32 | 2.32 | 2.70 | 3.89 |
| Jev | long | 0 | 2.46 | 2.75 | 3.73 |
| Jev | long | 32 | 2.32 | 2.70 | 3.89 |
| Jev | long | 64 | 2.24 | 2.68 | 4.00 |
| Jev | long | 128 | 2.29 | 2.71 | 3.96 |
| Jev | long | 256 | 2.30 | 2.73 | 3.96 |
| Jev | appdata | 0 | 2.46 | 2.75 | 3.73 |
| Jev | appdata | 1 | 2.76 | 2.76 | 3.32 |
| Jev | appdata | 2 | 2.70 | 2.73 | 3.40 |
| Jev | appdata | 8 | 2.68 | 2.74 | 3.45 |
| Jev | appdata | 32 | 2.72 | 2.74 | 3.40 |
| Qwen3.8-27B | own | 0 | 2.65 | 2.91 | 3.44 |
| Qwen3.8-27B | own | 2 | 2.94 | 3.01 | 3.13 |
| Qwen3.8-27B | own | 4 | 2.99 | 3.00 | 3.05 |
| Qwen3.8-27B | own | 8 | 3.00 | 3.00 | 3.02 |
| Qwen3.8-27B | own | 16 | 3.00 | 3.00 | 3.01 |
| Qwen3.8-27B | own | 32 | 3.00 | 3.00 | 3.01 |
| Qwen3.8-27B | quiz | 0 | 2.65 | 2.91 | 3.44 |
| Qwen3.8-27B | quiz | 2 | 2.68 | 2.94 | 3.52 |
| Qwen3.8-27B | quiz | 4 | 2.48 | 2.92 | 3.70 |
| Qwen3.8-27B | quiz | 8 | 2.27 | 2.93 | 3.88 |
| Qwen3.8-27B | quiz | 16 | 2.15 | 2.91 | 3.96 |
| Qwen3.8-27B | quiz | 32 | 2.14 | 2.91 | 3.93 |
| Qwen3.8-27B | long | 0 | 2.65 | 2.91 | 3.44 |
| Qwen3.8-27B | long | 32 | 2.14 | 2.91 | 3.93 |
| Qwen3.8-27B | long | 64 | 2.10 | 2.94 | 4.01 |
| Qwen3.8-27B | long | 128 | 2.02 | 2.94 | 4.03 |
| Qwen3.8-27B | long | 256 | 2.07 | 3.00 | 4.05 |
| Qwen3.8-27B | appdata | 0 | 2.65 | 2.91 | 3.44 |
| Qwen3.8-27B | appdata | 1 | 2.24 | 2.87 | 3.97 |
| Qwen3.8-27B | appdata | 2 | 2.15 | 2.87 | 4.03 |
| Qwen3.8-27B | appdata | 8 | 2.12 | 2.87 | 4.05 |
| Qwen3.8-27B | appdata | 32 | 2.09 | 2.81 | 4.00 |
| Gemma-4-26B-A4B | own | 0 | 1.93 | 2.79 | 4.28 |
| Gemma-4-26B-A4B | own | 2 | 2.52 | 2.75 | 3.42 |
| Gemma-4-26B-A4B | own | 4 | 2.68 | 2.86 | 3.42 |
| Gemma-4-26B-A4B | own | 8 | 2.64 | 2.88 | 3.27 |
| Gemma-4-26B-A4B | own | 16 | 2.76 | 2.93 | 3.02 |
| Gemma-4-26B-A4B | own | 32 | 2.64 | 2.87 | 3.07 |
| Gemma-4-26B-A4B | quiz | 0 | 1.93 | 2.79 | 4.28 |
| Gemma-4-26B-A4B | quiz | 2 | 2.28 | 2.83 | 3.74 |
| Gemma-4-26B-A4B | quiz | 4 | 2.43 | 2.91 | 3.75 |
| Gemma-4-26B-A4B | quiz | 8 | 2.83 | 2.83 | 3.53 |
| Gemma-4-26B-A4B | quiz | 16 | 2.48 | 2.75 | 3.85 |
| Gemma-4-26B-A4B | quiz | 32 | 2.58 | 2.76 | 3.91 |
| Gemma-4-26B-A4B | long | 0 | 1.93 | 2.79 | 4.28 |
| Gemma-4-26B-A4B | long | 32 | 2.58 | 2.76 | 3.91 |
| Gemma-4-26B-A4B | long | 64 | 2.61 | 2.71 | 3.79 |
| Gemma-4-26B-A4B | long | 128 | 2.76 | 2.85 | 3.78 |
| Gemma-4-26B-A4B | long | 256 | 2.84 | 2.92 | 3.67 |
| Gemma-4-26B-A4B | appdata | 0 | 1.93 | 2.79 | 4.28 |
| Gemma-4-26B-A4B | appdata | 1 | 1.71 | 2.75 | 4.13 |
| Gemma-4-26B-A4B | appdata | 2 | 1.85 | 2.66 | 4.20 |
| Gemma-4-26B-A4B | appdata | 8 | 1.84 | 2.67 | 4.22 |
| Gemma-4-26B-A4B | appdata | 32 | 2.89 | 2.98 | 3.60 |
