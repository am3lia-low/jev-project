"""Pinned model IDs for both parts of the project.

Source of truth: the "Decided" table in docs/PLAN.md. Pin exact IDs: no
~...-latest aliases, no :free variants.
"""

JEV_MODEL = "typesafe/jev-1.13"
LLM_BASELINE_MODELS = ("qwen/qwen3.8-27b", "google/gemma-4-26b-a4b-it")
NEW_QUESTION_MODEL = "anthropic/claude-sonnet-5.5"

# Paraphrases (A3.1), swaps (A3.3) and IPIP paraphrases (B1). Placeholder until
# the pilot picks one of PARAPHRASE_CANDIDATES.
PARAPHRASE_CANDIDATES = ("openai/gpt-6-luna", "mistralai/mistral-small-2603")
PARAPHRASE_MODEL: str | None = None

# Request settings for the generator models, checked against each model's
# `supported_parameters` on OpenRouter (2026-10-08). gpt-6-luna takes no
# temperature; its max_tokens must leave room for reasoning tokens.
GENERATION_PARAMS: dict[str, dict] = {
    "openai/gpt-6-luna": {"reasoning": {"effort": "low"}, "max_tokens": 4000},
    "mistralai/mistral-small-2603": {"temperature": 0.3, "max_tokens": 2000},
    "anthropic/claude-sonnet-5.5": {"temperature": 0.8, "max_tokens": 8000},
}

BERT_MODEL = "bert-base-uncased"
BERT_OPTIONAL_MODELS = ("bert-large-uncased", "roberta-base")


def require_paraphrase_model() -> str:
    """Return the paraphrase model, or fail loudly if the pilot hasn't picked one."""
    if PARAPHRASE_MODEL is None:
        raise RuntimeError(
            "PARAPHRASE_MODEL is not set. Run the pilot on "
            f"{' vs '.join(PARAPHRASE_CANDIDATES)} and set it in src/jev_common/models.py."
        )
    return PARAPHRASE_MODEL
