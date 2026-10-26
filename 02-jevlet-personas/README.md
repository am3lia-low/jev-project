# Jevlet personas

Part B of the project: can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs?

The full design is in [`docs/PLAN.md` → Part B](../docs/PLAN.md#part-b--jevlet-personas). It is independent of [jev benchmarks](../01-jev-benchmarks/) (Part A).

## The two axes

| Axis | What it tests | Plan section |
|---|---|---|
| Behavioural | Big Five personas with IPIP-NEO-120, using Jev's `score` type, plus scenario questions using `choice` | [B1](../docs/PLAN.md#b1-behavioural-axis--big-five) |
| Intellectual | "Pretend to be [level]" across a ladder from ARC-Easy up to Humanity's Last Exam | [B2](../docs/PLAN.md#b2-intellectual-axis--difficulty-ladder) |
| Persona decay | Does a Big Five persona fade as a history grows in the state? | [B4](../docs/PLAN.md#b4-persona-decay) |

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; scoring and analysis |
| API laptop | Every Jev and LLM call, and dataset downloads if HuggingFace needs a login or is blocked on the work network |

Code that needs the API laptop is still written on the work laptop; it just gets run there.

## Layout

Shared code (pinned model IDs, OpenRouter client, cache, parallel runner, stats) comes from [`src/jev_common/`](../src/jev_common/); this folder only holds Part B's own code. Scripts are numbered by their step in [PLAN B3](../docs/PLAN.md#b3-order-of-work). Run them from this folder.

```
02-jevlet-personas/
├── README.md
├── 01_run_bigfive.py       # B1 steps 1-3: check -> baseline -> personas (API laptop)
├── 02_analyse_bigfive.py   # B1 steps 1-3: scores, alpha, acquiescence, noise, persona shifts
├── 03_run_decay.py         # B4: estimate (local) -> check -> run (API laptop)
├── 04_analyse_decay.py     # B4: shift and retention per checkpoint, figure
├── 05_paraphrase_ipip.py   # B1 step 4: reword the IPIP items -> review sheet -> --apply-review
├── 06_analyse_paraphrase.py  # B1 step 4: original vs paraphrased answers
├── 07_make_scenarios.py    # B1 step 5: write scenario drafts -> review sheet -> --apply-review
├── 08_run_scenarios.py     # B1 step 5: estimate (local) -> check -> baseline -> personas
├── 09_analyse_scenarios.py # B1 step 5: behaviour shift next to self-report shift
├── 10_build_ladder.py      # B2: build the five benchmark sets (needs a HuggingFace login)
├── 11_run_ladder.py        # B2: estimate (local) -> check -> run
├── 12_analyse_ladder.py    # B2: accuracy vs chance and humans, persona drop, figure
├── helpers/                # Part B-only code
│   ├── config.py           # paths, seeds, run counts, persona settings
│   ├── ipip.py             # IPIP-NEO-120 items, scoring key, reverse-keyed items, Jev score questions
│   ├── reliability.py      # Cronbach's alpha per trait, acquiescence
│   ├── personas.py         # persona conditions and their wordings
│   ├── bigfive.py          # tasks, Jev requests, LLM prompts, result records
│   ├── decay.py            # B4 histories, conversations, probes, records
│   ├── choice.py           # choice questions (scenarios, ladder): requests, LLM prompts, picked letter
│   ├── costs.py            # cost estimates scaled from B1's real costs
│   ├── scenarios.py        # B1 step 5 scenarios, tasks, records
│   └── ladder.py           # B2 loaders, conditions, tasks, records
├── prompts/
│   ├── llm_system.txt      # LLM system line (same as Part A)
│   ├── ipip_paraphrase.txt # B1 step 4 generator prompt
│   ├── scenarios.txt       # B1 step 5 generator prompt
│   └── personas/
│       ├── markers.json    # Goldberg marker adjectives per trait and pole
│       └── persona.txt     # "You are a person who is {adjectives}."
├── data/                   # committed
│   ├── ipip/               # IPIP-NEO-120 items; paraphrased items after review (B1 step 4)
│   ├── scenarios/          # drafts and the reviewed scenarios.json (B1 step 5)
│   ├── ladder/             # <benchmark>.jsonl and build_report.json (B2)
│   └── reviews/            # sheets filled in by hand: ipip_paraphrases.csv, scenarios.csv
├── results/                # committed
│   ├── raw/<model slug>/bigfive/<condition>.jsonl   # 01
│   ├── raw/<model slug>/decay/<arm>/        # 03: <condition>.jsonl, histories/
│   ├── tables/bigfive_*.csv                         # 02
│   ├── figures/bigfive_shift_matrix.png             # 02
│   ├── figures/decay_retention.png                  # 04
│   ├── bigfive_summary.md                           # 02 (rewritten each run)
│   ├── decay_summary.md                             # 04 (rewritten each run)
│   ├── decay_findings.md                            # written by hand
│   └── bigfive_findings.md                          # written by hand
├── cache/                  # not committed
└── logs/                   # not committed
```

Results of the new steps go to `results/raw/<model slug>/bigfive_paraphrased/`, `scenarios/` and `ladder/<benchmark>/`, with `*_summary.md` files written by the analysis scripts.

## Big Five runs

A condition is `baseline` or `<trait>-<high|low>-<state|instructions>` (20 persona conditions). Each item is a separate request.

| | Baseline | Each persona condition |
|---|---|---|
| Jev | 5 fresh repeats (run-to-run noise) | 5 wordings x 1 call |
| LLMs | 1 run (temperature 0) | 5 wordings x 1 call |

About 12,600 Jev calls and 12,120 per LLM in total.

```
python 01_run_bigfive.py check       # one item on every model; prints Jev's raw score answer
python 01_run_bigfive.py baseline
python 01_run_bigfive.py personas    # refuses to run until markers.json is marked verified
python 02_analyse_bigfive.py         # any laptop
```

## Persona decay runs (B4)

Our script builds each conversation's history and sends it in the state with every request; no model remembers anything between requests. Two arms, 32 turns each: **own** (the model's own answers to O and A items) and **quiz** (fixed ARC-Easy Q&A). At 0, 2, 4, 8, 16 and 32 turns, the persona trait's 24 items are asked, one request each. Full design in [PLAN B4](../docs/PLAN.md#b4-persona-decay).

```
python 03_run_decay.py estimate      # local only, no API calls: call counts and cost
python 03_run_decay.py check         # API laptop, < $0.01: 2 turns + 1 probe per model
python 03_run_decay.py run           # API laptop, about $2.45 for all models and both arms
python 04_analyse_decay.py           # any laptop
```

Follow-up arms (2026-10-19), opt-in: `long` (quiz Q&A to 256 turns) and `appdata` (unrelated order records in the state). Run `estimate` with the same arguments first.

```
python 03_run_decay.py estimate --arms long appdata
python 03_run_decay.py run --arms appdata
python 03_run_decay.py run --arms long --placements state
```

## IPIP paraphrases (B1 step 4)

Does a model answer the original items from memory? Reword them and compare.

```
python 05_paraphrase_ipip.py                     # API laptop, ~$0.01; writes data/reviews/ipip_paraphrases.csv
python 05_paraphrase_ipip.py --apply-review      # after filling in same_meaning (and edit) for all 120
python 01_run_bigfive.py baseline --items paraphrased
python 01_run_bigfive.py personas --items paraphrased
python 06_analyse_paraphrase.py
```

## Scenarios (B1 step 5)

Does a persona move behaviour, not only self-report? Each scenario has a high-trait and a low-trait action.

```
python 07_make_scenarios.py                      # API laptop, ~$0.30; writes data/reviews/scenarios.csv
python 07_make_scenarios.py --apply-review       # after filling in keep (y/n) for every draft
python 08_run_scenarios.py estimate              # local, no API calls
python 08_run_scenarios.py check
python 08_run_scenarios.py baseline
python 08_run_scenarios.py personas
python 09_analyse_scenarios.py
```

## Difficulty ladder (B2)

Where does each model fall to chance, and does "pretend to be an average secondary-school student" lower its accuracy? GPQA and HLE are gated on HuggingFace: accept their terms on the website, then run `huggingface-cli login` once.

```
python 10_build_ladder.py
python 11_run_ladder.py estimate                 # local, no API calls
python 11_run_ladder.py check
python 11_run_ladder.py run
python 12_analyse_ladder.py
```

## Status

- Helpers, `01_run_bigfive.py` and `02_analyse_bigfive.py` written (2026-10-15). Dry-run end to end with a fake client: the analysis recovers planted persona shifts, alpha and Jev's noise.
- `check` run live 2026-10-15: Jev's `score` answer is a probability-weighted 0-4 position, as assumed. About $0.50 for the full B1 run.
- `markers.json` verified 2026-10-15 (all 100 Goldberg unipolar markers). `baseline` and `personas` run 2026-10-15 ($0.49 in total); findings in [`results/bigfive_findings.md`](results/bigfive_findings.md).
- B4 persona decay written and dry-run on a fake model 2026-10-18 (recovers a planted fade). `estimate`: ~14k calls per model, about $2.45 in total. Run 2026-10-18 ($1.33); findings in [`results/decay_findings.md`](results/decay_findings.md).
- 2026-10-19: B1 step 4 (05, 06), B1 step 5 (07-09), B2 (10-12) and the B4 follow-up arms written and dry-run end to end on a fake model; the analyses recover the planted effects.
- 2026-10-20: B4 follow-up arms run (`long` state only $1.06, `appdata` $0.43); results in PLAN B4 → Follow-up arms. `05_paraphrase_ipip.py` ($0.006) and `07_make_scenarios.py` ($0.20) run.
- 2026-10-24: both review sheets filled; `--apply-review` wrote `data/ipip/ipip_neo_120_paraphrased.json` (10 items replaced by an edit) and `data/scenarios/scenarios.json` (120 scenarios).
- Next: `01_run_bigfive.py --items paraphrased` → `06`; `08_run_scenarios.py` → `09`; B2 (needs a HuggingFace login for GPQA and HLE); the active support-chat decay arm (to design).
