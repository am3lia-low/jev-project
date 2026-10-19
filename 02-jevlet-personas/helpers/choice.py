"""Choice questions: requests, answering and the picked letter (B1 step 5 scenarios, B2 ladder).

Jev gets one `choice` question, `{type: "choice", instructions, criteria:
{letter: option text}}`, as in Part A. The persona goes into the state as
`{"persona": text}` or in front of the question instructions, as in B1. The
LLM baselines get the same request rendered as a chat prompt the way
decision-index renders a choice question (Part A's `render_prompt`); their
answer is the option letters' probabilities from the first token's top
logprobs.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jev_common.models import JEV_MODEL, llm_eval_params
from jev_common.openrouter import CallResult, OpenRouterClient, choice_probabilities, letter_probabilities

from .bigfive import QUESTION_KEY, Request
from .config import PROMPTS_DIR

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# (probabilities per letter, call result) for one request
ChoiceAnswer = tuple[dict[str, float], CallResult]
ChoiceAnswerer = Callable[[Request], ChoiceAnswer]


def choice_request(instructions: str, options: Sequence[str], persona: str | None = None,
                   placement: str | None = None) -> dict[str, Any]:
    """`{state, questions}` for one Jev choice request, with the persona placed as in B1."""
    state: dict[str, Any] = {}
    if placement == "state":
        state = {"persona": persona}
    elif placement == "instructions":
        instructions = f"{persona} {instructions}"
    question = {"type": "choice", "instructions": instructions,
                "criteria": {LETTERS[i]: text for i, text in enumerate(options)}}
    return {"state": state, "questions": {QUESTION_KEY: question}}


def letters_of(request: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(request["questions"][QUESTION_KEY]["criteria"])


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------
def jev_choice_answerer(client: OpenRouterClient, model: str) -> ChoiceAnswerer:
    def answer(call: Request) -> ChoiceAnswer:
        request = call.request
        result = client.decide(model, request["state"], request["questions"], cache_tag=call.cache_tag)
        return choice_probabilities(result.response, QUESTION_KEY), result

    return answer


def render_choice_prompt(call: Request) -> str:
    """The user message, as decision-index renders a choice question (Part A's `render_prompt`)."""
    request = call.request
    question = request["questions"][QUESTION_KEY]
    state = request["state"]
    lines = ["State:", json.dumps(state, ensure_ascii=False) if state else "(empty)", "",
             "Question:", question["instructions"], "", "Options:"]
    lines += [f"{key}: {text}" for key, text in question["criteria"].items()]
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def choice_chat_messages(call: Request) -> list[dict[str, str]]:
    system = (PROMPTS_DIR / "llm_system.txt").read_text(encoding="utf-8").strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": render_choice_prompt(call)}]


def llm_choice_answerer(client: OpenRouterClient, model: str, params: dict[str, Any]) -> ChoiceAnswerer:
    def answer(call: Request) -> ChoiceAnswer:
        result = client.chat(model, choice_chat_messages(call), cache_tag=call.cache_tag, **params)
        return letter_probabilities(result.response, letters_of(call.request)), result

    return answer


def model_choice_answerer(client: OpenRouterClient, model: str) -> ChoiceAnswerer:
    """Jev's answerer for Jev, else the LLM answerer with that model's pinned settings."""
    if model == JEV_MODEL:
        return jev_choice_answerer(client, model)
    return llm_choice_answerer(client, model, llm_eval_params(model))


def picked_letter(letters: Sequence[str], probabilities: Mapping[str, float] | None,
                  result: CallResult | None) -> str | None:
    """Letter with the highest probability; on a tie, Jev's own `choice` (as in Part A).

    Jev rounds probabilities to 2 decimals, so two options can tie; `max`
    alone would then always favour the earlier letter.
    """
    if not probabilities:
        return None
    best = max(probabilities.get(k, 0.0) for k in letters)
    tied = [k for k in letters if probabilities.get(k, 0.0) == best]
    if len(tied) > 1 and result is not None:
        choice = ((result.response.get("answers") or {}).get(QUESTION_KEY) or {}).get("choice")
        if choice in tied:
            return choice
    return tied[0]
