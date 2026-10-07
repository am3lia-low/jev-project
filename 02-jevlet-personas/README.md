# Jevlet personas

Part B of the project: can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs?

The full design is in [`docs/PLAN.md` → Part B](../docs/PLAN.md#part-b--jevlet-personas). It is independent of [jev benchmarks](../01-jev-benchmarks/) (Part A).

## The two axes

| Axis | What it tests | Plan section |
|---|---|---|
| Behavioural | Big Five personas with IPIP-NEO-120, using Jev's `score` type, plus scenario questions using `choice` | [B1](../docs/PLAN.md#b1-behavioural-axis--big-five) |
| Intellectual | "Pretend to be [level]" across a ladder from ARC-Easy up to Humanity's Last Exam | [B2](../docs/PLAN.md#b2-intellectual-axis--difficulty-ladder) |

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; scoring and analysis |
| API laptop | Every Jev and LLM call, and dataset downloads if HuggingFace needs a login or is blocked on the work network |

Code that needs the API laptop is still written on the work laptop; it just gets run there.

## Layout (sketch)

To be detailed when Part B starts. Shared code (pinned model IDs, OpenRouter client, cache, stats) comes from [`src/jev_common/`](../src/jev_common/); this folder only holds Part B's own code. Scripts will be numbered by step, as in Part A, once Part B has an order of work in PLAN.md.

```
02-jevlet-personas/
├── README.md
├── helpers/          # Part B-only code
│   ├── ipip.py       # IPIP-NEO-120 items, scoring key, reverse-keyed items
│   └── reliability.py  # Cronbach's alpha per trait
├── prompts/
│   └── personas/     # persona text (in question instructions vs in the state)
├── data/             # committed
│   ├── ipip/         # items, scoring key, paraphrased items (B1 step 4)
│   ├── scenarios/    # behaviour-check questions per trait (B1 step 5)
│   └── ladder/<benchmark>/  # ARC-Easy ... HLE text-only MCQ (B2)
├── run_bigfive.py    # B1: baseline + personas on the IPIP items
├── run_scenarios.py  # B1: behaviour check
├── run_ladder.py     # B2: no persona vs "pretend to be [level]"
├── analyse.py
├── results/          # committed
├── cache/            # not committed
└── logs/             # not committed
```

## Status

- Helpers: `ipip.py` and `reliability.py` built; IPIP-NEO-120 items in `data/ipip/ipip_neo_120.json`.
- No scripts yet.
