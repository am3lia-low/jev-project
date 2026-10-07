"""Load and fill the frozen prompt templates in prompts/, and build eval requests.

Templates are plain text with `$name` placeholders (`string.Template`), so
JSON examples with braces can sit in a prompt unescaped. Filling fails if a
placeholder is left without a value.

Jev eval requests follow decision-index (A9): empty `state`, one `choice`
question `q1` whose instructions are `prompts/eval/<dataset>.txt` filled with
the question, and options keyed by single letters A-E. decision-index has
templates for HellaSwag and ARC only; CommonsenseQA copies ARC's (the bare
question) and SWAG copies HellaSwag's.
"""

from functools import cache
from pathlib import Path
from string import Template
from typing import Any

from .config import DATASETS, PROMPTS_DIR
from .labels import letter_from_position
from .levels import LevelItem

EVAL_QUESTION_KEY = "q1"


@cache
def load(name: str, root: Path = PROMPTS_DIR) -> Template:
    """`load("eval/hs")` reads prompts/eval/hs.txt."""
    path = root / f"{name}.txt"
    return Template(path.read_text(encoding="utf-8").strip())


def fill(name: str, **fields: Any) -> str:
    """Fill a template; raises KeyError naming any placeholder left unfilled."""
    return load(name).substitute(**{key: str(value) for key, value in fields.items()})


def options_block(options: tuple[str, ...]) -> str:
    """"A. first\\nB. second ..." for prompts that list the options inline."""
    return "\n".join(f"{letter_from_position(i)}. {text}" for i, text in enumerate(options))


def quiz_request(record: dict[str, Any]) -> dict[str, Any]:
    """`{state, questions}` for one quiz item (a record of data/quiz/quiz.jsonl).

    The prompt names the split the dataset's benchmark items come from, for
    control items too: the quiz only works if both sets are asked the same way.
    """
    spec = DATASETS[record["dataset"]]
    return {
        "state": {},
        "questions": {
            EVAL_QUESTION_KEY: {
                "type": "choice",
                "instructions": fill("quiz", split_name=spec.test_source, dataset_name=spec.name),
                "criteria": dict(record["options"]),
            }
        },
    }


def jev_eval_request(item: LevelItem) -> dict[str, Any]:
    """`{state, questions}` for one Jev choice request; pass to `OpenRouterClient.decide`."""
    return {
        "state": {},
        "questions": {
            EVAL_QUESTION_KEY: {
                "type": "choice",
                "instructions": fill(f"eval/{item.dataset}", question=item.question),
                "criteria": {letter_from_position(i): text for i, text in enumerate(item.options)},
            }
        },
    }
