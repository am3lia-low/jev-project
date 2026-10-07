# Jev benchmarks

Part A of the project: Jev vs BERT on multiple-choice questions, and how far Jev's memorisation of public benchmarks goes.

The full design is in [`docs/PLAN.md` → Part A](../docs/PLAN.md#part-a--jev-benchmarks). This README only covers how the code in this folder is organised and where each part runs.

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; sampling, validation and analysis on data already downloaded |
| API laptop | Anything that calls OpenRouter (paraphrases, new questions, swaps, Jev, LLM baselines). Also dataset downloads if HuggingFace needs a login or is blocked on the work network |
| Google Colab (T4) | BERT fine-tuning and evaluation |

Code that needs the API laptop is still written here; it just gets run there.

## Layout (planned)

```
01-jev-benchmarks/
├── README.md
├── helpers-to-build.txt   # build list for the shared helpers
├── helpers/               # shared code used by every script
├── sample.py              # A2: download, dedupe, sample, split
├── paraphrase.py          # A3.1 (API laptop)
├── validate.py            # A3.1 checks
├── generate_new.py        # A3.2 (API laptop)
├── swap.py                # A3.3 (API laptop)
├── validate_swaps.py      # A3.3 checks
└── data/                  # sampled sets and derived levels
```

## Status

Nothing built yet. Start with the helpers in [`helpers-to-build.txt`](helpers-to-build.txt), then `sample.py` (step 2 of [A10](../docs/PLAN.md#a10-order-of-work)).
