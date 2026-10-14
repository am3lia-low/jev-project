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
5. Not done (Colab). The notebook now has `SMOKE_TEST = True` for the small first run (1 seed, 2,000 train rows, 1 epoch, into `results_smoke/`). `data/bert/` is built with `--originals-only`; rebuild it without the flag after 04.

## On hold: written on the work laptop, not verifiable there (2026-10-08). Items 1-4 done 2026-10-09, see above

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
