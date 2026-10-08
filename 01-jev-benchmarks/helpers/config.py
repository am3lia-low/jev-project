"""Paths, seeds, sample sizes and thresholds for the jev benchmarks.

Source of truth for the values: sections A1-A9 of docs/PLAN.md. Change a value
here, never inside a script. Pinned model IDs are in `jev_common.models`;
generic statistics defaults (bootstrap count, confidence, ECE bins, bootstrap
seed) are in `jev_common.stats`.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_DIR / "data"
SAMPLED_DIR = DATA_DIR / "sampled"  # 02_sample.py: <dataset>/{train,val,test}.jsonl
PARAPHRASED_RAW_DIR = DATA_DIR / "paraphrased" / "raw"  # 03: <dataset>/{train,val,test}.jsonl
PARAPHRASED_DIR = DATA_DIR / "paraphrased"  # 04, passed checks: <dataset>/{train,val,test}.jsonl
PILOT_DIR = DATA_DIR / "paraphrased" / "pilot"  # 03 --pilot
NEW_DIR = DATA_DIR / "new"
NEW_REVIEWED_PATH = NEW_DIR / "reviewed.jsonl"  # human-checked 05 output = Level 3
SWAPS_DIR = DATA_DIR / "swaps"
QUIZ_DIR = DATA_DIR / "quiz"
REVIEW_DIR = DATA_DIR / "reviews"  # manual-check sheets (CSV) you fill in
BERT_DATA_DIR = DATA_DIR / "bert"  # 07a: flat files for the Colab notebook
PROMPTS_DIR = BENCH_DIR / "prompts"
RESULTS_DIR = BENCH_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"  # 08, 09: <model slug>/<run>.jsonl
BERT_RESULTS_DIR = RESULTS_DIR / "bert"  # 07b, downloaded from Colab: <seed_n | untrained>/<run>.jsonl
TABLES_DIR = RESULTS_DIR / "tables"  # 10
FIGURES_DIR = RESULTS_DIR / "figures"  # 10
VERDICT_PATH = RESULTS_DIR / "verdict.md"  # 10
CACHE_DIR = BENCH_DIR / "cache"  # gitignored; raw API responses
LOG_DIR = BENCH_DIR / "logs"

# ---------------------------------------------------------------------------
# Seeds (one per purpose so changing one step never moves another)
# ---------------------------------------------------------------------------
SAMPLE_SEED = 20261007  # which questions are drawn (A2)
SHUFFLE_SEED = 20261008  # balanced option shuffle (A2 rule 3)
QUIZ_SEED = 20261009  # position of the original in the quiz (A7)
# 20261010 is the bootstrap seed, kept in jev_common.stats (A8)
BERT_SEEDS = (11, 22, 33, 44, 55)  # five fine-tuning seeds (A5)

# ---------------------------------------------------------------------------
# Sample sizes (A2, A3.2, A7)
# ---------------------------------------------------------------------------
N_TRAIN = 1_100  # per dataset
N_VAL = 200  # per dataset
N_TEST = 400  # per dataset
N_NEW_PER_DATASET = 50  # Level 3 questions kept per paper dataset (A3.2)
N_NEW_DRAFTS_PER_DATASET = 100  # drafted per paper dataset; spares for rejected drafts
N_QUIZ_BENCHMARK_PER_DATASET = 100  # quiz benchmark items (A7)
N_QUIZ_CONTROL_PER_DATASET = 25  # quiz control items, taken from Level 3 (A7)
N_SWAPS_PER_ITEM = 3  # synonym-swap versions per quiz item (A3.3)
N_REPEATABILITY = 200  # questions re-run 3 times (A9)
N_REPEATABILITY_RUNS = 3
N_PILOT = 30  # paraphrase pilot questions, same for both candidate models (A3.1)
N_MANUAL_PARAPHRASE = 50  # manual check of test paraphrases (A3.1)
N_MANUAL_SWAPS = 30  # manual check of swap sets (A3.3)
N_NEW_PER_CALL = 10  # Level 3 drafts per request (A3.2)
N_STYLE_EXAMPLES = 5  # train questions shown as style examples per request (A3.2)
MAX_ATTEMPTS = 3  # generations per item before giving up on it (03, 06a)

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
# Validation thresholds (A3.1, A3.3)
# ---------------------------------------------------------------------------
WORD_COUNT_TOLERANCE = 2  # swaps: word count may differ by at most this
MAX_EDIT_SPAN = 3  # swaps: longest run of changed words on either side
NEAR_COPY_SIMILARITY = 0.6  # 05: flag a draft this similar to any benchmark question
NEAR_COPY_REJECT = 0.75  # 05: discard and replace a draft this similar (clear near-copies)

# ---------------------------------------------------------------------------
# Analysis thresholds (A5, A7, A8); pass these to jev_common.stats
# ---------------------------------------------------------------------------
QUIZ_CHANCE = 0.25  # original is one of 4 options, A-D
ABOVE_CHANCE_MARGIN = 0.10  # BERT control valid if CI lower bound >= chance + this
MEMORISATION_DROP_RATIO = 2.0  # Jev's log-odds drop must be >= this x BERT's
SIGNIFICANCE = 0.05  # two-sided tests in 10_analyse.py's verdict
REPEAT_AGREEMENT_SINGLE_RUN = 0.99  # 8b: at or above this, one run per level is enough (A9)

# Jev accuracy reported by Deußer et al. (arXiv 2609.37647), for the Level 0
# sanity check (A9). HellaSwag and CommonsenseQA: Table 2; ARC-Easy and
# ARC-Challenge: section 4.1 (Table 2 only has ARC E+C combined, 0.988).
# Checked 2026-10-08.
PAPER_JEV_ACCURACY = {"hs": 0.955, "arce": 0.993, "arcc": 0.977, "csqa": 0.882}

