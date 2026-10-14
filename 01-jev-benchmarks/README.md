# Jev benchmarks

Part A of the project: Jev vs BERT on multiple-choice questions, and how far Jev's memorisation of public benchmarks goes.

The full design is in [`docs/PLAN.md` → Part A](../docs/PLAN.md#part-a--jev-benchmarks). This README only covers how the code in this folder is organised and where each part runs.

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; validation and analysis on data pulled from git |
| API laptop | Dataset downloads (`02_sample.py`) and anything that calls OpenRouter (paraphrases, new questions, swaps, Jev, LLM baselines) |
| Google Colab (T4) | BERT fine-tuning and evaluation (`notebooks/07b_bert_colab.ipynb`) |

Code that needs the API laptop is still written on the work laptop; it just gets run there. Results come back through git.

## Layout

Scripts are numbered by their step in [A10](../docs/PLAN.md#a10-order-of-work) and run from this folder, e.g. `python 02_sample.py`.

```
01-jev-benchmarks/
├── README.md
├── helpers/                  # Part A-only shared code (cache and stats live in src/jev_common/)
│   ├── config.py             # paths, seeds, sample sizes, thresholds
│   ├── schema.py             # the one question format + JSONL read/write
│   ├── ids.py                # stable IDs, e.g. arcc-test-0417
│   ├── labels.py             # answer labels to A-E, letters <-> positions
│   ├── shuffle.py            # seeded balanced shuffle
│   ├── loaders.py            # HuggingFace datasets -> question format
│   ├── dedupe.py             # SWAG vs HellaSwag overlap
│   ├── text_checks.py        # automatic checks for 04 and 06b
│   ├── levels.py             # Level 0-3 views of the test set
│   ├── prompts.py            # loads and fills the templates in prompts/; Jev request builders
│   ├── generation.py         # LLM calls returning JSON, parallel runs, review sheets
│   └── evaluation.py         # tasks, LLM prompt, run loop, result records (08, 09, 10)
├── prompts/                  # frozen prompt text
│   ├── paraphrase.txt
│   ├── swap.txt
│   ├── quiz.txt
│   ├── llm_system.txt        # decision-index's system line for the LLM baselines
│   ├── generate_new/<dataset>.txt
│   └── eval/<dataset>.txt    # decision-index templates (A9)
│
├── 02_sample.py              # A2: download, dedupe, sample, split, shuffle  [API laptop]
├── 03_paraphrase.py          # A3.1: pilot (--pilot) and full run         [API laptop]
├── 04_validate.py            # A3.1: automatic checks + manual 50
├── 05_generate_new.py        # A3.2: 400 drafts -> 200 Level 3 questions  [API laptop]
├── 06a_swap.py               # A3.3: 3 synonym swaps per quiz item         [API laptop]
├── 06b_validate_swaps.py     # A3.3: automatic checks + manual 30
├── 06c_build_quiz.py         # A7: original + 3 swaps, balanced positions, option E
├── 07a_build_bert_data.py    # A5: originals + paraphrases for Colab
├── notebooks/
│   └── 07b_bert_colab.ipynb  # A4-A5: untrained + fine-tuned BERT, Levels 0-3   [Colab]
├── 08_run_jev.py             # A9: sanity check, repeatability, Levels 0-3, quiz   [API laptop]
├── 09_run_llm.py             # A9: LLM baselines, Levels 0-3 + quiz        [API laptop]
├── 10_analyse.py             # A8: tables, figures, McNemar, log-odds, quiz, verdict
│
├── data/                     # committed
│   ├── sampled/<dataset>/{train,val,test}.jsonl     # 02
│   ├── sampling_report.json                         # 02
│   ├── paraphrased/pilot/                           # 03 --pilot: 30 questions x 2 models + blind key
│   ├── paraphrased/raw/<dataset>/{train,val,test}.jsonl  # 03
│   ├── paraphrased/<dataset>/{train,val,test}.jsonl # 04: passed checks, shuffled
│   ├── paraphrased/{failed,validation_report}.json  # 04 (03 --retry-failed reads failed.json)
│   ├── new/drafts.jsonl                             # 05
│   ├── new/reviewed.jsonl                           # 05 --apply-review = Level 3
│   ├── swaps/{items,raw}.jsonl                      # 06a: quiz items, their 3 swaps
│   ├── swaps/swaps.jsonl, swaps/failed.json         # 06b: swaps that passed / rejected
│   ├── quiz/quiz.jsonl                              # 06c
│   ├── bert/{train,val,level0..level3}.jsonl, manifest.json  # 07a
│   └── reviews/              # manual-check sheets you fill in (CSV)
│       ├── pilot_30.csv
│       ├── paraphrase_50.csv
│       ├── new_questions.csv
│       └── swaps_30.csv
├── results/                  # committed
│   ├── raw/<author>__<model>/{level0,level1,level2,level3,quiz,repeat_1,repeat_2,repeat_3}.jsonl
│   │                         # per question: answer, probabilities, latency, cost
│   ├── bert/seed_<n>/        # downloaded from Colab
│   ├── tables/
│   ├── figures/
│   └── verdict.md
├── cache/                    # not committed: raw API responses, so scripts can resume
└── logs/                     # not committed: API call logs
```

Levels 0 and 1 are not separate files in `data/`: they are the original and shuffled orders of `data/sampled/<dataset>/test.jsonl`, built by `helpers/levels.py`. Only `07a` writes them out, into `data/bert/` for the Colab notebook.

## Status (2026-10-14)

- Helpers: all built and tested live. Build list: [`dev_notes/helpers-to-build.txt`](../dev_notes/helpers-to-build.txt).
- `02_sample.py`: run 2026-10-08; `data/sampled/` holds the 8,500 sampled questions.
- `03`-`06c` written and dry-run end to end. Both reviews done 2026-10-14: the paraphrase pilot (`pilot_30.csv`; gpt-6-luna chosen) and Level 3 (`new_questions.csv`; 200 kept in `data/new/reviewed.jsonl`). Next: the full `03` → `06c` run.
- `07a` run with `--originals-only` for the Colab smoke test; the notebook has not been run on Colab yet.
- `08_run_jev.py sanity` and `09_run_llm.py levels --models qwen/qwen3.8-27b --levels 0` run 2026-10-09 (Level 0 only); `10_analyse.py` run on those, so `results/tables/` and `results/figures/` cover Level 0 only and `verdict.md` has no signals yet.
- Remaining work, in order: [`docs/api-laptop-checks.md` → Run order](../docs/api-laptop-checks.md).
