# Memorisation verdict (draft)

Written by `10_analyse.py`. It states the three signals of PLAN A8 with their numbers; the final judgement is made by a person, not by this script.

BERT difficulty control valid on: arce, csqa, swag; uninformative on: hs, arcc.

## Jev

- **Ladder** (pooled (swag, arce, csqa), n=1200): log-odds drop 0.41 vs BERT 0.20; difference 0.21 [0.03, 0.40] -> supports memorisation (needs CI > 0 and drop >= 2x BERT's).
- **Quiz:** original picked on 62.6% of benchmark items (p=3.76e-56 vs 25%) and 81.6% of control items (p=8.57e-32); gap p=0.000362 -> no evidence of memorised wording.
- **Level 3:** 100.0% on new questions vs 94.8% at Level 0 (p=0.000974) -> no significant drop.

Signals pointing to memorisation: 1 of 3.

## Qwen3.8-27B

- **Ladder** (pooled (swag, arce, csqa), n=1200): log-odds drop 0.60 vs BERT 0.20; difference 0.40 [0.21, 0.61] -> supports memorisation (needs CI > 0 and drop >= 2x BERT's).
- **Quiz:** original picked on 0.3% of benchmark items (p=1 vs 25%) and 1.0% of control items (p=1); gap p=0.284 -> no evidence of memorised wording.
- **Level 3:** 99.5% on new questions vs 94.1% at Level 0 (p=0.00125) -> no significant drop.

Signals pointing to memorisation: 1 of 3.

## Gemma-4-26B-A4B

- **Ladder** (pooled (swag, arce, csqa), n=1200): log-odds drop 0.34 vs BERT 0.20; difference 0.14 [-0.03, 0.31] -> does not support memorisation (needs CI > 0 and drop >= 2x BERT's).
- **Quiz:** original picked on 70.5% of benchmark items (p=3.17e-80 vs 25%) and 80.6% of control items (p=1.09e-30); gap p=0.0438 -> no evidence of memorised wording.
- **Level 3:** 100.0% on new questions vs 91.3% at Level 0 (p=1.43e-05) -> no significant drop.

Signals pointing to memorisation: 0 of 3.

