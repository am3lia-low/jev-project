# API-laptop checks (OpenRouter)

**For the Claude agent on the API laptop.** The work laptop's network blocks openrouter.ai and huggingface.co, so anything below that touches them is done there. Read `docs/PLAN.md` first (Part A, A3, A4, A9).

Do not spend money beyond the tiny test calls listed without asking the user, and **show the user the code changes before any full run**.

## Done (2026-10-07/08, $0.00003)

- Model IDs, prices and logprobs support checked; the lineup is in PLAN.md → Decided.
- Jev (`typesafe/jev-1.13`) on `POST /api/alpha/decisions` returns a probability per option.
- Qwen (`qwen/qwen3.8-27b`) and Gemma (`google/gemma-4-26b-a4b-it`) return option-letter logprobs with `max_tokens=1` and reasoning off. OpenRouter routed them to different providers (DekaLLM, Makora), so each baseline must be pinned to one provider.

## On hold: written on the work laptop, not verifiable there (2026-10-08)

| # | What | Where | How to check |
|---|---|---|---|
| 1 | **Pin a provider per baseline.** Which of DekaLLM / Makora served which model was not recorded. | `LLM_PROVIDERS` in `src/jev_common/models.py` (both `None`; `09_run_llm.py` refuses to run until set) | Look in `01-jev-benchmarks/cache/` (or `logs/openrouter.jsonl`) for the live-check responses; or `GET /api/v1/models/{author}/{slug}/endpoints` and pick a provider that lists `logprobs` + `top_logprobs`. |
| 2 | **Request fields written from memory:** `"reasoning": {"enabled": false}` and provider routing `{"order": [...], "allow_fallbacks": false, "require_parameters": true}`. | `LLM_EVAL_PARAMS` / `llm_eval_params` in `src/jev_common/models.py` | Compare with the request the live check sent (cache file `request` field) and the OpenRouter docs (reasoning tokens, provider routing). |
| 3 | **Does a Jev decisions response carry `usage.cost`?** `CallResult.cost_usd` reads it; if absent, cost per 1,000 questions shows $0. | `src/jev_common/openrouter.py` | Look at one cached Jev response. |
| 4 | **Smoke test 08 and 09 on Level 0** before the other runs. There is no "first N questions" option and a run stopped by `--max-cost` writes nothing, so run `08_run_jev.py sanity` (~$0.01) and `09_run_llm.py levels --models qwen/qwen3.8-27b --levels 0` (~$0.2); check `results/raw/<model>/level0.jsonl` has probabilities, few or no errors, and sanity says OK. The calls are cached, so later full runs reuse them. | `08_run_jev.py`, `09_run_llm.py` | ~$0.25 |
| 5 | **First run of the Colab notebook** (needs a GPU and the HuggingFace model download). | `notebooks/07b_bert_colab.ipynb` | Run on Colab T4 with a small data set first; check the truncation count, loss falls, val accuracy above chance. |

## Run order once the reviews are in (needs you)

1. `03_paraphrase.py` → `04_validate.py` (retry loop; fill `paraphrase_50.csv`) → `06a_swap.py` → `06b_validate_swaps.py` (retry loop; fill `swaps_30.csv`) → `06c_build_quiz.py`. About $0.90.
2. `07a_build_bert_data.py`, copy `data/bert/` to `MyDrive/jev-bert/data/`, run the notebook on Colab, download results into `results/bert/`.
3. `08_run_jev.py sanity` → `repeat` → `levels` → `quiz`. Under $0.10.
4. `09_run_llm.py levels` → `quiz`. About $1.
5. `10_analyse.py` (any laptop).
