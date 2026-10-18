# API-laptop checks (OpenRouter)

**For the Claude agent on the API laptop.** The work laptop's network blocks openrouter.ai and huggingface.co, so anything below that touches them is done there. Read `docs/PLAN.md` first (Part A, A3, A4, A9).

Do not spend money beyond the tiny test calls listed without asking the user, and **show the user the code changes before any full run**.

## Done (2026-10-07/08, $0.00003)

- Model IDs, prices and logprobs support checked; the lineup is in PLAN.md → Decided.
- Jev (`typesafe/jev-1.13`) on `POST /api/alpha/decisions` returns a probability per option.
- Qwen (`qwen/qwen3.8-27b`) and Gemma (`google/gemma-4-26b-a4b-it`) return option-letter logprobs with `max_tokens=1` and reasoning off. OpenRouter routed them to different providers (DekaLLM, Makora), so each baseline must be pinned to one provider.

## Done (2026-10-09, $0.054)

1. **Providers pinned** (`LLM_PROVIDERS`): Qwen → `dekallm`, Gemma → `makora`. The live-check responses were never cached or logged (the log's `served_by` is the model, not the provider), but Makora serves only Gemma, so DekaLLM served Qwen. Both endpoints list quantization as "unknown". `order` takes lowercase slugs, not display names. DekaLLM's Qwen price is $0.049 / $3.00 per M, not the $0.425 / $2.55 list price. The call log now records `provider`.
2. **Request fields:** provider routing (`order`, `allow_fallbacks`, `require_parameters`) matches the docs. Reasoning off is documented as `{"effort": "none"}`; `{"enabled": false}` is not documented for chat completions. Both gave 0 reasoning tokens live; switched to `effort: "none"`.
3. **Jev `usage.cost`:** present (`usage: {input_tokens, output_tokens, cost}`), plus `provider: "TypeSafe"`. Jev rounds probabilities to 2 decimals, so `helpers/evaluation.py` now breaks ties with Jev's own `choice`. 1,745 of 2,000 Level 0 answers have some letter at exactly 0. Bear this in mind when reading ECE.
4. **Smoke test:** `08_run_jev.py sanity` 2,000/2,000, 0 failed, $0.033, every paper dataset OK (hs 94.0, arce 99.2, arcc 97.5, csqa 88.5; swag 85.0). `09_run_llm.py levels --models qwen/qwen3.8-27b --levels 0`: 2,000/2,000, 0 failed, $0.021, all served by DekaLLM. 6 rows had a letter outside the top 20 and 4 rows tied (falls back to the earlier letter).
5. Done 2026-10-15 on **Kaggle** (no Colab GPU was available); see [`results/bert_summary.md`](../01-jev-benchmarks/results/bert_summary.md). The notebook has `SMOKE_TEST = True` for the small first run (1 seed, 2,000 train rows, 1 epoch, into `results_smoke/`). `data/bert/` is built with `--originals-only`; rebuild it without the flag after 04.

## On hold: written on the work laptop, not verifiable there (2026-10-08). Items 1-4 done 2026-10-09, item 5 2026-10-15, see above

Kept as the original brief. All items are done.

| # | What | Where | How to check |
|---|---|---|---|
| 1 | **Pin a provider per baseline.** Which of DekaLLM / Makora served which model was not recorded. | `LLM_PROVIDERS` in `src/jev_common/models.py` (both `None`; `09_run_llm.py` refuses to run until set) | Look in `01-jev-benchmarks/cache/` (or `logs/openrouter.jsonl`) for the live-check responses; or `GET /api/v1/models/{author}/{slug}/endpoints` and pick a provider that lists `logprobs` + `top_logprobs`. |
| 2 | **Request fields written from memory:** `"reasoning": {"enabled": false}` and provider routing `{"order": [...], "allow_fallbacks": false, "require_parameters": true}`. | `LLM_EVAL_PARAMS` / `llm_eval_params` in `src/jev_common/models.py` | Compare with the request the live check sent (cache file `request` field) and the OpenRouter docs (reasoning tokens, provider routing). |
| 3 | **Does a Jev decisions response carry `usage.cost`?** `CallResult.cost_usd` reads it; if absent, cost per 1,000 questions shows $0. | `src/jev_common/openrouter.py` | Look at one cached Jev response. |
| 4 | **Smoke test 08 and 09 on Level 0** before the other runs. There is no "first N questions" option and a run stopped by `--max-cost` writes nothing, so run `08_run_jev.py sanity` (~$0.01) and `09_run_llm.py levels --models qwen/qwen3.8-27b --levels 0` (~$0.2); check `results/raw/<model>/level0.jsonl` has probabilities, few or no errors, and sanity says OK. The calls are cached, so later full runs reuse them. | `08_run_jev.py`, `09_run_llm.py` | ~$0.25 |
| 5 | **First run of the BERT notebook** (planned for Colab, run on Kaggle) (needs a GPU and the HuggingFace model download). | `notebooks/07b_bert_colab.ipynb` | Run on Colab T4 with a small data set first; check the truncation count, loss falls, val accuracy above chance. |

## Done (2026-10-18, $0.13): Level 2 and quiz runs (Jev + LLMs)

All four runs done 2026-10-18, 0 failed, 2,000 / 494 rows each; providers still DekaLLM / Makora. Jev $0.05, LLMs $0.08. `10_analyse.py` rerun; results committed. Reading: [`results/memorisation_findings.md`](../01-jev-benchmarks/results/memorisation_findings.md). Kept below as the brief that was followed.

Both gates cleared: `paraphrase_50.csv` filled 2026-10-17 (50/50 fine), memorisation threshold confirmed as written 2026-10-18 (PLAN A8), quiz rebuilt 2026-10-17 with 494 items (6 swaps flagged in `swaps_30.csv` dropped by `06b`). Independent of the BERT learning curve below; run in any order.

Run from `01-jev-benchmarks/` after `git pull`:

1. `python 08_run_jev.py levels --levels 2` — 2,000 questions; Level 0 cost $0.033, so expect about the same. 8b's 99.5% agreement means one run (no `--runs`).
2. `python 08_run_jev.py quiz` — 494 items.
3. `python 09_run_llm.py levels --levels 2` — both Qwen and Gemma (default `--models`).
4. `python 09_run_llm.py quiz` — both models.
5. `python 10_analyse.py`, then commit `results/` (raw, tables, figures, `verdict.md`).

All four runs together should stay well under $1 (previous estimate for the whole of 08 + 09: about $1.10, most of it already spent). The `--max-cost` defaults ($0.50 for 08, $2.00 for 09) are enough; a run stopped by `--max-cost` writes nothing, but its calls are cached, so a re-run picks up where it stopped.

**Check after each run:** `results/raw/<model>/level2.jsonl` and `quiz.jsonl` have 2,000 / 494 rows, probabilities present, few or no errors (`08`/`09` print only progress and spend; accuracies come from `10_analyse.py`). LLMs: still served by the pinned providers (DekaLLM for Qwen, Makora for Gemma; the call log records `provider`).

**Do not** change `MEMORISATION_DROP_RATIO`, the ladder pool or the above-chance rule after seeing these results; they are fixed (PLAN A8). ARC-Challenge rejoins the pool only on BERT's evidence (learning curve / RACE fallback).

## Done (2026-10-18): BERT learning curve on Kaggle

Run 2026-10-18 as `shiyalow/jev-bert-learning-curve` (a first push with `LEARNING_CURVE = False` was deleted before it started). All 6 folders downloaded to `results/bert_learning_curve/`; summary in [`bert_summary.md` → Learning curve](../01-jev-benchmarks/results/bert_summary.md#learning-curve). Kept below as the brief that was followed.

Can't be run from the cloud Claude session: kaggle.com is blocked there and it has no Kaggle token. Free (Kaggle GPU quota only, ~80 min of T4); no OpenRouter calls.

**Why:** test whether BERT is data-limited before enlarging the train pool (PLAN A5 → Learning curve). BERT failed the above-chance rule on HellaSwag and ARC-Challenge, and val stops improving after epoch 1.

1. `git pull`; the option is in `notebooks/07b_bert_colab.ipynb` (commit "Add a learning-curve option to the BERT notebook"). No data rebuild: the current `shiyalow/jev-bert-data` version is the main-run data and is what the curve needs.
2. In a new folder, copy the notebook and set `LEARNING_CURVE = True` in the settings cell (leave `CURVE_FRACTIONS = (0.25, 0.5, 1.0)`, `CURVE_SEEDS = SEEDS[:2]`, `SMOKE_TEST = False`). Do not commit that change; the repo copy stays `False`.
3. `kernel-metadata.json` as for the main run, with id `shiyalow/jev-bert-learning-curve`, private, `enable_gpu`, `enable_internet`, `dataset_sources: ["shiyalow/jev-bert-data"]`. Push with `uvx kaggle kernels push --accelerator NvidiaTeslaT4`.
4. When it finishes, `uvx kaggle kernels output shiyalow/jev-bert-learning-curve` and copy `results_learning_curve/` to `01-jev-benchmarks/results/bert_learning_curve/` (outside `results/bert/`, so `10_analyse.py` never reads it). Check all 6 folders arrived (`frac_025|050|100_seed_11|22`, each with `train_log.json`); the download can time out half-way.
5. Commit the results with the printed summary table pasted into `results/bert_summary.md` under a new "Learning curve" heading, and tick TODO 9b.

**Reading it:** one fraction's val number per dataset has a 95% CI of about ±7 points, so read the trend over the three fractions (Level 0, 400 per dataset, is steadier). HellaSwag and ARC-Challenge flat while SWAG / CommonsenseQA rise → more data only strengthens the datasets that already pass, and ARC-Challenge needs the RACE fallback (its train split is already 98% used). Everything still rising at 100% → enlarging the pool is worth it.

## Run order (both reviews done 2026-10-14; `PARAPHRASE_MODEL` = gpt-6-luna)

0. Item 5 above: smoke test of the notebook. Done 2026-10-15 (Kaggle).
1. `03_paraphrase.py` → `04_validate.py` (retry loop; fill `paraphrase_50.csv`) → `06a_swap.py` → `06b_validate_swaps.py` (retry loop; fill `swaps_30.csv`) → `06c_build_quiz.py`. About $0.90. All run 2026-10-16/17; both review sheets filled 2026-10-17 (quiz rebuilt with 494 items after dropping 6 flagged swaps).
2. `07a_build_bert_data.py` (without `--originals-only`), upload `data/bert/` as the Kaggle dataset `shiyalow/jev-bert-data`, run the notebook on Kaggle, download results into `results/bert/`. Done 2026-10-17.
3. `08_run_jev.py sanity` (cached, free) → `repeat` → `levels` → `quiz`. Under $0.10. Done: Levels 0, 1, 3 on 2026-10-17, Level 2 and quiz on 2026-10-18.
4. `09_run_llm.py levels` → `quiz`. About $1. Levels 0, 1, 3 done for both 2026-10-17; Level 2 and quiz 2026-10-18.
5. `10_analyse.py` (any laptop). Run on all results 2026-10-18; final verdict by hand left.
