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
- [x] Decided 2026-10-14: dedupe SWAG against the 400 sampled HellaSwag test questions (current behaviour, PLAN A1), not the full HellaSwag validation split (which would drop 3,489 + 31 by context of 73,546).

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

## 5. [ ] Reviews, then the synthetic-data runs

- [x] Rate the pilot blind (`data/reviews/pilot_30.csv`, 2026-10-14): both models kept meaning 30/30; gpt-6-luna 30 same difficulty, Mistral 29 same + 1 harder. `PARAPHRASE_MODEL = "openai/gpt-6-luna"`.
- [x] Review the Level 3 drafts (`data/reviews/new_questions.csv`, 2026-10-14): 272 of 400 reviewed, no text edits. `arcc-new-0095` and `csqa-new-0048` had keep=yes with a failed check, set to keep=no. `--apply-review` wrote `data/new/reviewed.jsonl`: 50 per dataset, 200 in total.
- [x] Then: `03_paraphrase.py` → `04_validate.py` (+ `--retry-failed` loop, fill `paraphrase_50.csv`) → `06a_swap.py` → `06b_validate_swaps.py` (+ retry loop, fill `swaps_30.csv`) → `06c_build_quiz.py`. All scripts run 2026-10-16/17. Both review sheets filled 2026-10-17: paraphrases 50/50 fine; swaps 6/30 flagged, which `06b` now drops (not retried), quiz rebuilt with 494 items.

## 6. [x] Docs out of date

- [x] `docs/api-laptop-checks.md` replaced with a record of the done checks and the on-hold list (2026-10-08).
- [x] All Markdown files brought in line with each other after merging both laptops' work (2026-10-14).

## 7. [x] Experiment scripts (2026-10-08, work laptop)

- [x] `helpers/evaluation.py` (tasks, LLM prompt rendering copied from decision-index, run loop, result records), `prompts/llm_system.txt`.
- [x] `07a_build_bert_data.py`, `notebooks/07b_bert_colab.ipynb`, `08_run_jev.py`, `09_run_llm.py`, `10_analyse.py`. 08/09 dry-run with a fake API; 10 run on synthetic results (recovers planted effects). Notebook syntax-checked only.
- [x] `PAPER_JEV_ACCURACY` in `config.py`, incl. CommonsenseQA 0.882 (paper Table 2).
- [x] Decided 2026-10-09: LLM baselines stay on OpenRouter (top-20 logprobs; a letter outside them counts as 0%), no self-hosting for full probabilities. BERT's above-chance rule uses val originals only. Both recorded in PLAN A5/A9.

## 8. [ ] On hold for the API laptop / Colab (1-4 done 2026-10-09; Colab run left)

Details in [`docs/api-laptop-checks.md`](../docs/api-laptop-checks.md).

- [x] Pin LLM providers (Qwen → `dekallm`, Gemma → `makora`).
- [x] Confirm the request fields (reasoning off is now `effort: "none"`).
- [x] Check Jev's `usage.cost` (present).
- [x] Smoke-test 08/09 on Level 0 ($0.054): Jev sanity OK on every paper dataset; Qwen 2,000/2,000.
- [x] First run of `07b_bert_colab.ipynb` with `SMOKE_TEST = True`, on Kaggle (2026-10-15; no Colab GPU). See `01-jev-benchmarks/results/bert_summary.md`.

## 9. [ ] Full runs (after 5 and 8)

Same order as [`docs/api-laptop-checks.md` → Run order](../docs/api-laptop-checks.md#run-order-both-reviews-done-2026-10-14-paraphrase_model--gpt-6-luna) and PLAN A10:

- [x] `03_paraphrase.py` + `04_validate.py` full run (2026-10-16): 8,500/8,500 pass, $0.79.
- [x] Fill in `data/reviews/paraphrase_50.csv` (manual paraphrase check, 2026-10-17).
- [x] Optional originals-only BERT ablation on Kaggle (2026-10-15): `results/bert_original_only/`.
- [x] `07a_build_bert_data.py` without `--originals-only`, then the full notebook on Kaggle (5 seeds + untrained; 2026-10-17); results in `results/bert/`.
- [ ] `08_run_jev.py` repeat → levels → quiz. Done 2026-10-17: repeat (99.5% agreement, single runs), levels 0, 1, 3. Left: Level 2 (after the memorisation threshold) and quiz (ready, 494 items).
- [ ] `09_run_llm.py` levels → quiz (Qwen and Gemma). Done 2026-10-17: levels 0, 1, 3 for both. Left: Level 2 and quiz, same gates as 08.
- [ ] `10_analyse.py`, then the final verdict by hand.

## 9b. [ ] BERT learning curve (2026-10-18)

- [x] `LEARNING_CURVE` option in `notebooks/07b_bert_colab.ipynb` (nested 25/50/100% subsets per dataset, original + paraphrase kept together, 2 seeds, ~80 min on a T4). Subset logic and summary cell tested locally on `data/bert/train.jsonl`; training not run here (no GPU).
- [ ] Run on Kaggle (new kernel, `LEARNING_CURVE = True`), download to `results/bert_learning_curve/`, then decide: enlarge the train pool and/or the RACE fallback for ARC-Challenge.

## 10. [ ] Open decisions

See PLAN.md → Open questions: confirm the memorisation threshold (before Jev's Level 2 run), and whether to add an MMLU math number-swap slice.

## 11. [ ] Part B: Big Five (B1 steps 1-3)

- [x] Decided 2026-10-15: personas as Goldberg marker adjectives (5 wordings per condition); Jev baseline 5 repeats, LLMs 1 run; no item shuffling (each item is its own request). Recorded in PLAN B1.
- [x] `helpers/config.py`, `personas.py`, `bigfive.py`; `01_run_bigfive.py`, `02_analyse_bigfive.py`. Dry-run with a fake client (recovers planted shifts).
- [x] `run_parallel` moved to `jev_common/parallel.py` (Part A's `helpers.generation` re-exports it).
- [x] `prompts/personas/markers.json` checked 2026-10-15 against two secondary sources for Goldberg's 100 unipolar markers (psytests.org list + psychTools scoring key). The draft had 14 non-Goldberg words (e.g. C: responsible, reliable, hardworking, thrifty, lazy; N-low: calm, at ease, secure); now all 100 Goldberg words are used, by pole.
- [x] `01_run_bigfive.py check` live (2026-10-15, $0.0001): Jev's `score` is the probability-weighted position 0-4, with per-position probabilities; all 3 models move up under an E-high persona (Jev 2.73 → 4.25/4.37, LLMs 3.0 → ~5). About $0.50 for the full B1 run.
- [x] `baseline` run 2026-10-15 ($0.011; one Gemma 520 retried). Jev's repeats barely differ (mean per-item SD 0.04).
- [x] `personas` run 2026-10-15 ($0.48, 0 failed) and analysed: alpha 0.95-0.97 for every model and trait; every persona moves its trait (Jev high - low +2.7 to +3.7; LLMs +2.7 to +3.9); state placement > instructions; large off-target shifts (~1-2 points, a desirability halo). Written up in `02-jevlet-personas/results/bigfive_findings.md` with the shift-matrix figure.
- [ ] Then B1 step 4 (IPIP paraphrases), step 5 (scenarios), B2 (ladder).
