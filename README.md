# Jev project

Evaluating TypeSafe's **Jev** (a "System One" decision model) against BERT baselines on multiple-choice questions, and probing how far it can be steered into personas.

The full design is in [`docs/PLAN.md`](docs/PLAN.md). The project has two independent parts:

| Part | Folder | Question |
|---|---|---|
| A: jev benchmarks | [`01-jev-benchmarks/`](01-jev-benchmarks/) | Jev vs BERT on MCQ, and how deep Jev's memorisation of public benchmarks goes |
| B: jevlet personas | [`02-jevlet-personas/`](02-jevlet-personas/) | Can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs? |

## Layout

Each part's README has its own folder in full. [`dev_notes/TODO.md`](dev_notes/TODO.md) tracks work on the repo; progress on the experiment is in [PLAN A10](docs/PLAN.md#a10-order-of-work).

```
jev-project/
├── README.md
├── pyproject.toml           # dependencies; installs src/jev_common into the venv
├── requirements.txt         # generated from pyproject.toml (uv export), for Colab
├── .env.example             # copy to .env and add OPENROUTER_API_KEY
├── src/jev_common/          # code shared by both parts
│   ├── models.py            # pinned model IDs, LLM providers and request settings
│   ├── openrouter.py        # OpenRouter client (Jev decisions + chat completions), call log
│   ├── cache.py             # on-disk response cache, so API scripts can resume
│   └── stats.py             # bootstrap CI, McNemar, log-odds, ECE, binomial, two-proportion
├── docs/                    # PLAN.md, experiment flow diagram, API-laptop checklist
├── dev_notes/               # to-do list and build lists for the repo itself
├── 01-jev-benchmarks/       # Part A
└── 02-jevlet-personas/      # Part B
```

## Conventions

- **Scripts are numbered by their step** in the plan's order of work (Part A: [A10](docs/PLAN.md#a10-order-of-work)), e.g. `02_sample.py`, `06a_swap.py`. Run them from their part's folder.
- **Each data file is written by exactly one script** and only read by later ones, so any step can be re-run on its own.
- **Committed:** `data/` and `results/` (small JSONL / CSV files). Git is how data moves between the API laptop and the work laptop.
- **Not committed:** `.env`, `cache/` (raw API responses) and `logs/`.
- **Prompts live in `prompts/`** as text files, so the frozen versions are tracked in git.

## Where things run

| Machine | GPU | OpenRouter / HuggingFace | Use for |
|---|---|---|---|
| Work laptop | None | Blocked | Writing code; validation and analysis on data pulled from git |
| API laptop | None | Yes | Dataset downloads, every OpenRouter call (paraphrases, swaps, new questions, Jev, LLMs) |
| Google Colab | T4 (free) | — | BERT fine-tuning and evaluation |

## Setup

Python ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/).

```
uv sync                 # creates .venv, installs dependencies + src/jev_common
cp .env.example .env    # then add OPENROUTER_API_KEY (API laptop only)
```

Dependencies live in `pyproject.toml`. After changing them, regenerate `requirements.txt` (used on Colab) with `uv export --no-hashes -o requirements.txt`.
