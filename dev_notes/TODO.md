# To do

Working notes for the repo itself (not the experiment plan, which is in [`docs/PLAN.md`](../docs/PLAN.md)). The target layout is in the [root README](../README.md#layout) and each part's README.

## 1. [x] Code review all `.py` files in the helpers folder

`01-jev-benchmarks/helpers/`: `config.py`, `schema.py`, `ids.py`, `labels.py`, `shuffle.py`, `loaders.py`, `dedupe.py`, `text_checks.py`, `__init__.py`; plus `cache.py` and `stats.py` (now in `src/jev_common/`).

Check correctness against PLAN.md, edge cases, naming and formatting. The loaders' HuggingFace field names are unverified (HuggingFace is blocked on the work network); confirm them on the API laptop.

Done 2026-10-07 on the API laptop:

- [x] Loader field names checked against the live HuggingFace datasets; all match. Pool sizes and option counts match A1 (ARC has a few 3- and 5-option items, as expected).
- [x] **Fixed: SWAG vs HellaSwag video-ID dedupe never matched.** SWAG stores `anetv_<id>` lowercased, HellaSwag `activitynet~v_<ID>` in original case, so 0 SWAG questions were dropped by video. Both are now normalised to the bare, casefolded YouTube ID; HellaSwag WikiHow rows get no video ID. Result: 324 SWAG training questions dropped against the 400 sampled HellaSwag test questions.
- [x] `stats.two_proportion_test` now rejects empty groups instead of dividing by zero.
- [x] `stats.compare_drops`: documented that a negative BERT drop makes the 2× rule pass trivially, leaving the CI to decide.
- [ ] Decide: dedupe SWAG against the 400 sampled HellaSwag test questions (current, matches PLAN A1 wording) or against the full HellaSwag validation split (drops 3,489 + 31 by context of 73,546; removes any chance of related ActivityNet clips leaking into BERT's training set). Either way the pool is ample.

## 2. [x] Match the repo to the planned layout

Create the folders the layout needs, delete the ones it doesn't, and move every file to its assigned place.

- [x] Create `src/jev_common/` and make it installable with `uv` (build system + package in `pyproject.toml`; fill in the project description)
  - [x] Move `cache.py` and `stats.py` there from `01-jev-benchmarks/helpers/`
    - `DiskCache` now takes `root` explicitly (each part passes its own `cache/` folder).
    - Generic stats defaults (bootstrap count, confidence, ECE bins, bootstrap seed) live in `stats.py`; Part A thresholds (`QUIZ_CHANCE`, `MEMORISATION_DROP_RATIO`) stay in `helpers/config.py` and are passed in as required arguments.
  - [x] Move the pinned model IDs from `helpers/config.py` into `jev_common/models.py`
  - [x] Update imports in `helpers/` and `sample.py` (nothing in `helpers/` or `02_sample.py` used the moved code)
  - [x] Resolve the staged deletion of `src/jev_project/__init__.py` (already committed in `faa295d`)
- [x] Rename `01-jev-benchmarks/sample.py` → `02_sample.py`
- [x] Change its output path from `data/<dataset>/` to `data/sampled/<dataset>/` (`sampling_report.json` stays in `data/`)
- [x] Update old script names in docstrings to the numbered ones (`sample.py` usage lines; `validate.py` / `validate_swaps.py` in `helpers/text_checks.py`)
- [x] Create the empty Part A folders: `prompts/`, `notebooks/`, `data/`, `results/` (with a `.gitkeep` where empty)
- [x] Create the empty Part B folders: `helpers/`, `prompts/`, `data/`, `results/`
- [x] Generate `requirements.txt` from `pyproject.toml` (`uv export`) once the dependencies live in `pyproject.toml`
  - Regenerate with `uv export --no-hashes -o requirements.txt` after changing dependencies. `statsmodels` dropped (unused).
- [x] Delete any leftover folders and files that aren't in the layout (none found)

## 3. [x] Remaining helpers and the sampled dataset (2026-10-08)

- [x] `jev_common/openrouter.py`, `helpers/levels.py`, `helpers/prompts.py` + `prompts/eval/*.txt`, Part B `helpers/ipip.py` + `helpers/reliability.py` + `data/ipip/ipip_neo_120.json`
- [x] **Fixed: test questions leaking into train/val.** ARC's official splits share some question stems (one ARC-Easy item exactly, three ARC-Challenge stems with different options). `02_sample.py` now drops train/val candidates whose question text matches a sampled test question (4 ARC, 4 SWAG dropped; listed in `sampling_report.json`).
- [x] Ran `02_sample.py`: 8,500 questions in `data/sampled/`; re-runs are byte-identical.
- [x] Live checks ($0.00003): Jev returns a probability per option; Qwen and Gemma return letter logprobs with `max_tokens=1` and reasoning off. OpenRouter routed them to different providers (DekaLLM, Makora): pin one provider per baseline in `09_run_llm.py`.

## 4. [x] Synthetic-data scripts (2026-10-08)

- [x] `03_paraphrase.py`, `04_validate.py`, `05_generate_new.py`, `06a_swap.py`, `06b_validate_swaps.py`, `06c_build_quiz.py`, `helpers/generation.py`, prompts (`paraphrase.txt`, `swap.txt` and `quiz.txt` adapted from DCQ, `generate_new/`). Dry-run end to end with a fake model.
- [x] **Fixed: duplicate options.** Some CommonsenseQA items list the same option twice (sometimes the correct one, making them ambiguous); `02_sample.py` now drops them (24 test-pool, 122 train-pool; 1 SWAG). Option comparison is case-sensitive so ARC genotype options ("WW or Ww" vs "WW or ww") are not merged. Re-sampled.
- [x] Paraphrase pilot run ($0.005). Both candidates passed all automatic checks; ~$0.70–0.80 for the full 8,500.
- [x] Level 3 drafts run ($1.07 incl. a first attempt). Sonnet tends to write textbook questions ARC already contains, so drafts with similarity ≥ 0.75 to any benchmark question are now discarded and replaced (24); 60 more (0.6–0.75) are flagged for review.

## 5. [ ] Waiting on you

- [x] Rate the pilot blind (`data/reviews/pilot_30.csv`, 2026-10-14): both models kept meaning 30/30; gpt-6-luna 30 same difficulty, Mistral 29 same + 1 harder. `PARAPHRASE_MODEL = "openai/gpt-6-luna"`.
- [x] Review the Level 3 drafts (`data/reviews/new_questions.csv`, 2026-10-14): 272 of 400 reviewed, no text edits. `arcc-new-0095` and `csqa-new-0048` had keep=yes with a failed check, set to keep=no. `--apply-review` wrote `data/new/reviewed.jsonl`: 50 per dataset, 200 in total.
- [ ] Then: `03_paraphrase.py` → `04_validate.py` (+ `--retry-failed` loop, fill `paraphrase_50.csv`) → `06a_swap.py` → `06b_validate_swaps.py` (+ retry loop, fill `swaps_30.csv`) → `06c_build_quiz.py`.

## 6. [x] Docs out of date

- [x] `docs/api-laptop-checks.md` replaced with a record of the done checks and the on-hold list (2026-10-08).

## 7. [x] Experiment scripts (2026-10-08, work laptop)

- [x] `helpers/evaluation.py` (tasks, LLM prompt rendering copied from decision-index, run loop, result records), `prompts/llm_system.txt`.
- [x] `07a_build_bert_data.py`, `notebooks/07b_bert_colab.ipynb`, `08_run_jev.py`, `09_run_llm.py`, `10_analyse.py`. 08/09 dry-run with a fake API; 10 run on synthetic results (recovers planted effects). Notebook syntax-checked only.
- [x] `PAPER_JEV_ACCURACY` in `config.py`, incl. CommonsenseQA 0.882 (paper Table 2).
- [x] Decided 2026-10-09: LLM baselines stay on OpenRouter (top-20 logprobs; a letter outside them counts as 0%), no self-hosting for full probabilities. BERT's above-chance rule uses val originals only. Both recorded in PLAN A5/A9.

## 8. [ ] On hold for the API laptop / Colab

See [`docs/api-laptop-checks.md`](../docs/api-laptop-checks.md#on-hold-written-on-the-work-laptop-not-verifiable-there-2026-10-08): pin LLM providers, confirm the reasoning-off and provider-routing fields, check Jev's `usage.cost`, smoke-test 08/09, first Colab run of 07b.
