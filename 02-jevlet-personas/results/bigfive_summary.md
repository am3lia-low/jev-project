# Big Five personas (draft)

Written by `02_analyse_bigfive.py`. Numbers only; the reading is done by a person.

## Jev

- **Alpha** over 105 response sets from 21 condition(s): N 0.97, E 0.97, O 0.96, A 0.97, C 0.97 (interpret only >= 0.70).
- **Acquiescence** (baseline mean raw answer, plus- and minus-keyed averaged; 3 = none): 2.56.
- **Baseline noise** over 5 repeats: mean per-item SD 0.042.
- **Persona shifts** (target trait, high - low, 1-5 scale, 95% CI over items):
  - Neuroticism, state: +3.18 [+2.84, +3.50] (high 4.42, baseline 2.46, low 1.24); largest off-target A -1.12
  - Neuroticism, instructions: +2.81 [+2.51, +3.09] (high 4.26, baseline 2.46, low 1.45); largest off-target A -1.14
  - Extraversion, state: +3.63 [+3.46, +3.76] (high 4.71, baseline 2.76, low 1.08); largest off-target N -1.36
  - Extraversion, instructions: +3.29 [+3.14, +3.42] (high 4.44, baseline 2.76, low 1.15); largest off-target N -1.32
  - Openness, state: +3.06 [+2.64, +3.43] (high 4.57, baseline 3.21, low 1.52); largest off-target C +1.32
  - Openness, instructions: +2.67 [+2.31, +2.99] (high 4.42, baseline 3.21, low 1.75); largest off-target C +0.93
  - Agreeableness, state: +3.41 [+3.04, +3.69] (high 4.88, baseline 3.97, low 1.47); largest off-target N -1.24
  - Agreeableness, instructions: +3.03 [+2.68, +3.31] (high 4.78, baseline 3.97, low 1.75); largest off-target N -1.25
  - Conscientiousness, state: +3.69 [+3.46, +3.84] (high 4.90, baseline 3.75, low 1.21); largest off-target N -1.24
  - Conscientiousness, instructions: +3.22 [+3.02, +3.35] (high 4.73, baseline 3.75, low 1.51); largest off-target N -1.35

## Qwen3.8-27B

- **Alpha** over 101 response sets from 21 condition(s): N 0.97, E 0.97, O 0.96, A 0.96, C 0.97 (interpret only >= 0.70).
- **Acquiescence** (baseline mean raw answer, plus- and minus-keyed averaged; 3 = none): 2.73.
- **Persona shifts** (target trait, high - low, 1-5 scale, 95% CI over items):
  - Neuroticism, state: +3.69 [+3.49, +3.86] (high 4.72, baseline 2.65, low 1.03); largest off-target C -1.90
  - Neuroticism, instructions: +3.28 [+2.95, +3.57] (high 4.38, baseline 2.65, low 1.10); largest off-target C -1.49
  - Extraversion, state: +3.93 [+3.85, +3.99] (high 4.95, baseline 2.92, low 1.02); largest off-target N -1.24
  - Extraversion, instructions: +3.68 [+3.49, +3.83] (high 4.78, baseline 2.92, low 1.09); largest off-target N -1.17
  - Openness, state: +3.44 [+3.14, +3.70] (high 4.86, baseline 3.08, low 1.42); largest off-target C +1.67
  - Openness, instructions: +2.73 [+2.34, +3.09] (high 4.61, baseline 3.08, low 1.88); largest off-target C +0.91
  - Agreeableness, state: +3.56 [+3.27, +3.80] (high 4.97, baseline 3.78, low 1.41); largest off-target O +2.24
  - Agreeableness, instructions: +3.07 [+2.69, +3.42] (high 4.91, baseline 3.78, low 1.84); largest off-target O +1.55
  - Conscientiousness, state: +3.92 [+3.82, +3.98] (high 4.99, baseline 3.44, low 1.07); largest off-target N -2.37
  - Conscientiousness, instructions: +3.31 [+3.12, +3.49] (high 4.92, baseline 3.44, low 1.61); largest off-target N -1.68

## Gemma-4-26B-A4B

- **Alpha** over 101 response sets from 21 condition(s): N 0.95, E 0.95, O 0.95, A 0.97, C 0.95 (interpret only >= 0.70).
- **Acquiescence** (baseline mean raw answer, plus- and minus-keyed averaged; 3 = none): 2.37.
- **Persona shifts** (target trait, high - low, 1-5 scale, 95% CI over items):
  - Neuroticism, state: +3.69 [+3.42, +3.89] (high 4.81, baseline 1.89, low 1.12); largest off-target A -1.47
  - Neuroticism, instructions: +3.41 [+3.02, +3.74] (high 4.58, baseline 1.89, low 1.17); largest off-target A -1.46
  - Extraversion, state: +3.93 [+3.85, +3.99] (high 4.94, baseline 2.80, low 1.01); largest off-target A -1.00
  - Extraversion, instructions: +3.87 [+3.68, +4.00] (high 4.88, baseline 2.80, low 1.00); largest off-target N -1.13
  - Openness, state: +3.49 [+3.00, +3.86] (high 4.73, baseline 3.25, low 1.25); largest off-target A +0.98
  - Openness, instructions: +3.24 [+2.72, +3.68] (high 4.65, baseline 3.25, low 1.41); largest off-target A +1.13
  - Agreeableness, state: +3.65 [+3.23, +3.99] (high 4.83, baseline 4.21, low 1.18); largest off-target O +1.51
  - Agreeableness, instructions: +3.67 [+3.25, +4.00] (high 4.83, baseline 4.21, low 1.17); largest off-target N -1.03
  - Conscientiousness, state: +3.93 [+3.78, +4.00] (high 5.00, baseline 4.21, low 1.07); largest off-target N -2.42
  - Conscientiousness, instructions: +3.67 [+3.35, +3.93] (high 5.00, baseline 4.21, low 1.33); largest off-target N -1.92
