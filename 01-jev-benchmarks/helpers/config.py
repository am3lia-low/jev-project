"""Paths, seeds, sample sizes and pinned model IDs for the jev benchmarks.

Source of truth for the values: the "Decided" table and sections A1-A9 of
docs/PLAN.md. Change a value here, never inside a script.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_DIR / "data"
CACHE_DIR = BENCH_DIR / "cache"  # gitignored; raw API responses
LOG_DIR = BENCH_DIR / "logs"

# ---------------------------------------------------------------------------
# Seeds (one per purpose so changing one step never moves another)
# ---------------------------------------------------------------------------
SAMPLE_SEED = 20261007  # which questions are drawn (A2)
SHUFFLE_SEED = 20261008  # balanced option shuffle (A2 rule 3)
QUIZ_SEED = 20261009  # position of the original in the quiz (A7)
BOOTSTRAP_SEED = 20261010  # bootstrap confidence intervals (A8)
BERT_SEEDS = (11, 22, 33, 44, 55)  # five fine-tuning seeds (A5)

# ---------------------------------------------------------------------------
# Sample sizes (A2, A3.2, A7)
# ---------------------------------------------------------------------------
N_TRAIN = 1_100  # per dataset
N_VAL = 200  # per dataset
N_TEST = 400  # per dataset
N_NEW_PER_DATASET = 100  # Level 3, paper datasets only (A3.2)
N_QUIZ_BENCHMARK_PER_DATASET = 100  # quiz benchmark items (A7)
N_QUIZ_CONTROL_PER_DATASET = 25  # quiz control items, taken from Level 3 (A7)
N_SWAPS_PER_ITEM = 3  # synonym-swap versions per quiz item (A3.3)
N_REPEATABILITY = 200  # questions re-run 3 times (A9)
N_REPEATABILITY_RUNS = 3

# ---------------------------------------------------------------------------
# Datasets (A1)
# ---------------------------------------------------------------------------
# Output splits are named train / val / test; "source_split" values are the
# HuggingFace split names. Field names of each HuggingFace dataset are handled
# in loaders.py.
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Where one dataset comes from and how it is split."""

    key: str  # used in IDs, file names and the `dataset` field
    name: str  # display name
    hf_path: str
    hf_config: str | None
    test_source: str  # HuggingFace split the test questions are drawn from
    val_source: str  # HuggingFace split the val questions are drawn from
    paper_dataset: bool  # False = BERT-fair control, Levels 0-2 only


DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        DatasetSpec("hs", "HellaSwag", "Rowan/hellaswag", None, "validation", "train", True),
        DatasetSpec("arce", "ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "test", "train", True),
        # ARC-Challenge only has 1,119 train questions; val comes from the
        # official validation split so 1,100 train + 200 val fit (A2 rule 1).
        DatasetSpec(
            "arcc", "ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "validation", True
        ),
        DatasetSpec("csqa", "CommonsenseQA", "tau/commonsense_qa", None, "validation", "train", True),
        DatasetSpec("swag", "SWAG", "allenai/swag", "regular", "validation", "train", False),
    )
}
PAPER_DATASETS = tuple(k for k, spec in DATASETS.items() if spec.paper_dataset)

# Train questions always come from the official train split (A2 rule 1).
TRAIN_SOURCE = "train"

# ---------------------------------------------------------------------------
# Models (pinned; no ~...-latest aliases, no :free variants)
# ---------------------------------------------------------------------------
JEV_MODEL = "typesafe/jev-1.13"
LLM_BASELINE_MODELS = ("qwen/qwen3.8-27b", "google/gemma-4-26b-a4b-it")
NEW_QUESTION_MODEL = "anthropic/claude-sonnet-5.5"

# Paraphrases (A3.1), swaps (A3.3) and IPIP paraphrases (B1). Placeholder until
# the pilot picks one of PARAPHRASE_CANDIDATES.
PARAPHRASE_CANDIDATES = ("openai/gpt-6-luna", "mistralai/mistral-small-2603")
PARAPHRASE_MODEL: str | None = None

BERT_MODEL = "bert-base-uncased"
BERT_OPTIONAL_MODELS = ("bert-large-uncased", "roberta-base")

# ---------------------------------------------------------------------------
# Validation thresholds (A3.1, A3.3)
# ---------------------------------------------------------------------------
WORD_COUNT_TOLERANCE = 2  # swaps: word count may differ by at most this
MAX_EDIT_SPAN = 3  # swaps: longest run of changed words on either side

# ---------------------------------------------------------------------------
# Analysis (A5, A8)
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
ECE_BINS = 10
QUIZ_CHANCE = 0.25  # original is one of 4 options, A-D
ABOVE_CHANCE_MARGIN = 0.10  # BERT control valid if CI lower bound >= chance + this
MEMORISATION_DROP_RATIO = 2.0  # Jev's log-odds drop must be >= this x BERT's


def require_paraphrase_model() -> str:
    """Return the paraphrase model, or fail loudly if the pilot hasn't picked one."""
    if PARAPHRASE_MODEL is None:
        raise RuntimeError(
            "PARAPHRASE_MODEL is not set. Run the pilot on "
            f"{' vs '.join(PARAPHRASE_CANDIDATES)} and set it in helpers/config.py."
        )
    return PARAPHRASE_MODEL
