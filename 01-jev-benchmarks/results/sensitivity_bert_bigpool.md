# Sensitivity: BERT trained on a bigger pool (`bert_bigpool`)

Written by `10_analyse.py --sensitivity`; do not edit by hand. Pre-registered in PLAN A8 (2026-10-20): the verdict stays on the main BERT run; this checks whether it holds when BERT is trained on more data. Same test questions, same Jev and LLM results, same rules.

## BERT above-chance rule (val originals)

| Dataset | Main: accuracy, CI low | Valid | bert_bigpool: accuracy, CI low | Valid |
|---|---|---|---|---|
| hs | 37.5%, 32.1% | no | 38.4%, 32.6% | no |
| arce | 48.8%, 43.1% | yes | 53.4%, 47.2% | yes |
| arcc | 35.2%, 29.9% | no | 37.5%, 31.9% | no |
| csqa | 44.4%, 38.6% | yes | 47.6%, 41.9% | yes |
| swag | 63.0%, 57.5% | yes | 65.5%, 59.8% | yes |

BERT accuracy, all datasets: main Level 0 47.6%, Level 2 45.2%; bert_bigpool Level 0 49.3%, Level 2 47.4%

## Ladder rule, pooled over the datasets where each BERT run is valid

| Model | Run | Pool | Model drop | BERT drop | Ratio | Difference [95% CI] | Supported |
|---|---|---|---|---|---|---|---|
| Jev | main | (swag, arce, csqa) | 0.41 | 0.20 | 2.06x | 0.21 [0.03, 0.40] | yes |
| Qwen3.8-27B | main | (swag, arce, csqa) | 0.60 | 0.20 | 3.02x | 0.40 [0.21, 0.61] | yes |
| Gemma-4-26B-A4B | main | (swag, arce, csqa) | 0.34 | 0.20 | 1.70x | 0.14 [-0.03, 0.31] | no |
| Jev | bert_bigpool | (swag, arce, csqa) | 0.41 | 0.21 | 1.98x | 0.20 [0.03, 0.39] | no |
| Qwen3.8-27B | bert_bigpool | (swag, arce, csqa) | 0.60 | 0.21 | 2.90x | 0.39 [0.20, 0.60] | yes |
| Gemma-4-26B-A4B | bert_bigpool | (swag, arce, csqa) | 0.34 | 0.21 | 1.63x | 0.13 [-0.04, 0.30] | no |
