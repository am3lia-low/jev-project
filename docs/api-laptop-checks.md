# API-laptop checks (OpenRouter)

**For the Claude agent on the API laptop.** The work laptop's network blocks openrouter.ai, so these checks could not be done there. Read `docs/PLAN.md` first for context (Part A, sections A3, A4, A9 and "Open questions").

Written 2026-10-06. Do not spend money beyond the tiny test calls in step 3 without asking the user.

## Goal

Confirm which models the experiment can actually use on OpenRouter, at what price, and whether each one returns **token log-probabilities** (needed to read option-letter probabilities, as the arXiv paper did).

## Models to check

| Role | Model | What we need to know |
|---|---|---|
| Subject | `typesafe/jev-1.13` | Still listed? Price still $0.042/M input, output free? Context 32k? |
| LLM baseline | **Qwen3.8-27B** (paper's model) | Exact OpenRouter ID, if listed. Price. Logprobs supported? |
| LLM baseline | **Gemma-4-E4B** (paper's model) | Exact OpenRouter ID, if listed. Price. Logprobs supported? |
| Paraphraser candidate | `openai/gpt-5-nano` | Listed? Price |
| Paraphraser candidate | `google/gemini-2.0-flash-001` | Listed? Price |
| Paraphraser candidate | `mistralai/mistral-small-3.1-24b-instruct` | Listed? Price |
| Paraphraser candidate | `anthropic/claude-haiku-4.5` | Listed? Price |

Prices in PLAN.md for the paraphraser candidates came from a third-party tracker with no date, so treat them as unconfirmed.

## Steps

1. **List models** (no API key needed): `GET https://openrouter.ai/api/v1/models`.
   For each model above, record: `id`, `pricing.prompt` and `pricing.completion` (per token → convert to $ per million), `context_length`, and whether `supported_parameters` contains `logprobs` / `top_logprobs`.
   For Qwen and Gemma, search the list for `qwen3.8` and `gemma-4`; record every close match if the exact model is missing.
2. **Check providers** for the two baselines: `GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints`. Logprobs support can differ by provider; note which providers support it.
3. **Live logprobs test** (needs `OPENROUTER_API_KEY`; costs fractions of a cent). For each baseline that claims logprobs support, send one MCQ prompt with options A–D, `max_tokens: 1`, `logprobs: true`, `top_logprobs: 10`. Confirm the response actually contains probabilities for the letters A–D. If it doesn't, try pinning a provider that claims support.
4. **Jev smoke test**: one `choice` question to `typesafe/jev-1.13` with 4 options; confirm it returns a probability per option.

## Fallbacks (if a paper model is missing or has no logprobs)

- Pick the nearest model in the **same family** (Qwen / Gemma) that does support logprobs, and record the size difference.
- If no option in a family has logprobs on OpenRouter: report it. The user will decide between generated-letter answers (accuracy only, no like-for-like ECE) or running the model on Colab (Gemma-4-E4B may fit a T4 with 4-bit quantisation; unverified).
- The paraphraser must **not** be a Qwen or Gemma model.

## Report back

Write the results to `docs/openrouter-check.md`:
- one table: model · OpenRouter ID · $/M input · $/M output · context · logprobs (listed / tested OK / failed) · notes
- the date checked
- a short recommendation: final baseline IDs, and 2 paraphraser candidates for the pilot

Then update the "Open questions" section of `docs/PLAN.md` to match, and tell the user before committing.
