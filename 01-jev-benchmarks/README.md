# Jev benchmarks

Part A of the project: Jev vs BERT on multiple-choice questions, and how far Jev's memorisation of public benchmarks goes.

The full design is in [`docs/PLAN.md` → Part A](../docs/PLAN.md#part-a--jev-benchmarks). This README only covers how the code in this folder is organised and where each part runs.

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; validation and analysis on data pulled from git |
| API laptop | Dataset downloads (`02_sample.py`) and anything that calls OpenRouter (paraphrases, new questions, swaps, Jev, LLM baselines) |
| Kaggle (T4; planned: Colab) | BERT fine-tuning and evaluation (`notebooks/07b_bert_colab.ipynb` runs on either) |

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
├── 07a_build_bert_data.py    # A5: originals + paraphrases for the notebook
├── notebooks/
│   └── 07b_bert_colab.ipynb  # A4-A5: untrained + fine-tuned BERT, Levels 0-3   [Kaggle]
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
│   ├── bert/seed_<n>/        # downloaded from Kaggle
│   ├── tables/
│   ├── figures/
│   ├── verdict.md
│   ├── memorisation_findings.md  # written by hand: reading of the results
│   └── bert_learning_curve/frac_<pct>_seed_<n>/  # 07b with LEARNING_CURVE = True (diagnostic)
├── cache/                    # not committed: raw API responses, so scripts can resume
└── logs/                     # not committed: API call logs
```

Levels 0 and 1 are not separate files in `data/`: they are the original and shuffled orders of `data/sampled/<dataset>/test.jsonl`, built by `helpers/levels.py`. Only `07a` writes them out, into `data/bert/` for the BERT notebook (uploaded to Kaggle as a dataset).

## Status (2026-10-18)

- Helpers: all built and tested live. Build list: [`dev_notes/helpers-to-build.txt`](../dev_notes/helpers-to-build.txt).
- `02_sample.py`: run 2026-10-08; `data/sampled/` holds the 8,500 sampled questions.
- `03`-`06c` written and dry-run end to end. Both reviews done 2026-10-14: the paraphrase pilot (`pilot_30.csv`; gpt-6-luna chosen) and Level 3 (`new_questions.csv`; 200 kept in `data/new/reviewed.jsonl`).
- `03` + `04` full run 2026-10-15/16: all 8,500 paraphrases pass ($0.79). `06a`-`06c` run 2026-10-17: quiz of 500 items. Review sheets filled 2026-10-17: `paraphrase_50.csv` 50/50 fine; `swaps_30.csv` flagged 6, which `06b` drops, so the quiz is now 494 items.
- `07a` rebuilt with paraphrases (train 11,000, val 2,000, Levels 0-3). The notebook runs on Kaggle (no Colab GPU was available): smoke test, originals-only ablation (`results/bert_original_only/`) and main run (`results/bert/`, 5 seeds + untrained) done. Details: [`results/bert_summary.md`](results/bert_summary.md).
- `08` and `09`: all levels and the quiz run for Jev, Qwen and Gemma (Levels 0, 1, 3 on 2026-10-17; Level 2 and quiz on 2026-10-18, 0 failed). `10_analyse.py` rerun on everything; draft verdict in [`results/verdict.md`](results/verdict.md).
- **Results so far:** Jev passes the pre-registered ladder rule only narrowly (pooled log-odds drop 2.05× BERT's, CI [0.03, 0.40]; no single dataset passes; Qwen drops more); the quiz is uninformative (originals picked more on unseen control items; Qwen answers "none" on 492/494); no drop on Level 3. Read as weak, inconclusive evidence of memorisation: [`results/memorisation_findings.md`](results/memorisation_findings.md).
- BERT learning curve run 2026-10-18 (`results/bert_learning_curve/`): HellaSwag flat; ARC-Challenge still rising but below the threshold, so it only rejoins via the RACE fallback; the other three still rising at 100%.
- Next: the final verdict by hand, the RACE-fallback decision, and the MMLU number-swap sub-experiment (PLAN A11).
- Remaining work, in order: [`docs/api-laptop-checks.md` → Run order](../docs/api-laptop-checks.md).
