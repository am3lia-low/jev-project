"""Paths, seeds, run counts and persona settings for the jevlet personas.

Source of truth for the values: Part B of docs/PLAN.md. Change a value here,
never inside a script. Pinned model IDs are in `jev_common.models`.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PART_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PART_DIR / "data"
PROMPTS_DIR = PART_DIR / "prompts"
PERSONAS_DIR = PROMPTS_DIR / "personas"  # markers.json (adjectives), persona.txt (template)
RESULTS_DIR = PART_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"  # 01: <model slug>/bigfive/<condition>.jsonl
TABLES_DIR = RESULTS_DIR / "tables"  # 02
FIGURES_DIR = RESULTS_DIR / "figures"  # 02
BIGFIVE_SUMMARY_PATH = RESULTS_DIR / "bigfive_summary.md"  # 02
DECAY_SUMMARY_PATH = RESULTS_DIR / "decay_summary.md"  # 04
# B4 quiz history: ARC-Easy questions from Part A's validation split (never used as a Jev test set)
QUIZ_SOURCE_PATH = PART_DIR.parent / "01-jev-benchmarks" / "data" / "sampled" / "arce" / "val.jsonl"
CACHE_DIR = PART_DIR / "cache"  # gitignored; raw API responses
LOG_DIR = PART_DIR / "logs"

# ---------------------------------------------------------------------------
# Seeds (one per purpose)
# ---------------------------------------------------------------------------
PERSONA_SEED = 20261015  # which marker adjectives each persona variant uses, and their order
DECAY_SEED = 20261018  # which IPIP items and quiz questions fill the B4 histories, and their order

# ---------------------------------------------------------------------------
# B1 Big Five runs
# ---------------------------------------------------------------------------
# Every IPIP item is its own stateless request, so item order cannot matter and
# is not shuffled. Jev's baseline is repeated to measure its run-to-run noise
# (each repeat is a fresh request); the LLMs run at temperature 0, so a repeat
# would return the same answer and they run once (decided 2026-10-15).
N_BASELINE_RUNS_JEV = 5
N_BASELINE_RUNS_LLM = 1

# Personas: high/low on one trait at a time (B1 step 3), written as Goldberg
# marker adjectives rather than the trait name (decided 2026-10-15). Each
# condition has N_PERSONA_VARIANTS wordings, each a different seeded draw of
# N_PERSONA_ADJECTIVES markers, and every model answers each wording once. The
# wordings give the response sets real trait variance for alpha (B1 step 2).
POLES = ("high", "low")
PLACEMENTS = ("state", "instructions")  # where the persona text goes (B1 step 3)
N_PERSONA_VARIANTS = 5
N_PERSONA_ADJECTIVES = 5
PERSONA_QUALIFIER = "very"  # "very talkative, very bold, ..."

BASELINE = "baseline"

# ---------------------------------------------------------------------------
# B4 persona decay
# ---------------------------------------------------------------------------
# The persona comes first, then a history of earlier turns, then the probe
# question. Our script builds the history and sends it with every request;
# no model keeps anything between requests. Probes branch off the history at
# each checkpoint and are never added to it (decided 2026-10-18).
DECAY_ARMS = ("own", "quiz")  # own: the model's own answers to IPIP items; quiz: fixed ARC-Easy Q&A
DECAY_DOMAINS = ("E", "C", "N")  # persona traits: E and C pulled hardest in B1, N had the largest side effects
DECAY_VARIANTS = (1, 2, 3)  # B1 persona wordings 1-3
DECAY_TURNS = 32
DECAY_CHECKPOINTS = (0, 2, 4, 8, 16, 32)  # history length (turns) when the probes are asked
# History IPIP items come from traits that are never probed, so a probe item
# never appears in a history. Same items, same order, for every condition.
DECAY_HISTORY_DOMAINS = ("O", "A")
# Re-reminder (persona in the state only): the persona repeated as a history
# entry right after turn DECAY_REMINDER_TURN, inserted into the same history
# (later turns are not regenerated), and probed at these checkpoints.
DECAY_REMINDER_TURN = 16
DECAY_REMINDER_CHECKPOINTS = (16, 32)

# ---------------------------------------------------------------------------
# B4 follow-ups (added 2026-10-19)
# ---------------------------------------------------------------------------
# Each arm's history: how many turns, when to probe, and which state key holds
# it. "own" and "quiz" are the main run (unchanged). "long" continues the quiz
# history to DECAY_LONG_TURNS turns (its first 32 turns are the quiz arm's, so
# those requests come from the cache). "appdata" fills the state with
# unrelated app records instead of Q&A, to test whether any non-empty state
# weakens a persona in the instructions (decay_findings.md, finding 2).
DECAY_LONG_TURNS = 256
DECAY_ARM_SETTINGS: dict[str, dict] = {
    "own": {"turns": DECAY_TURNS, "checkpoints": DECAY_CHECKPOINTS, "key": "history", "reminder": True},
    "quiz": {"turns": DECAY_TURNS, "checkpoints": DECAY_CHECKPOINTS, "key": "history", "reminder": True},
    "long": {"turns": DECAY_LONG_TURNS, "checkpoints": (0, 32, 64, 128, 256), "key": "history", "reminder": False},
    "appdata": {"turns": DECAY_TURNS, "checkpoints": (0, 1, 2, 8, 32), "key": "orders", "reminder": False},
}
DECAY_ALL_ARMS = tuple(DECAY_ARM_SETTINGS)
# The long arm needs more ARC-Easy questions than the validation split has, so
# it continues with Part A's ARC-Easy train questions (never a Jev test set).
QUIZ_EXTRA_SOURCE_PATH = QUIZ_SOURCE_PATH.with_name("train.jsonl")

# ---------------------------------------------------------------------------
# B1 step 4: IPIP paraphrases (contamination check)
# ---------------------------------------------------------------------------
IPIP_PARAPHRASED_PATH = DATA_DIR / "ipip" / "ipip_neo_120_paraphrased.json"
IPIP_PARAPHRASE_DRAFTS_PATH = DATA_DIR / "ipip" / "paraphrase_drafts.jsonl"
REVIEWS_DIR = DATA_DIR / "reviews"
IPIP_REVIEW_PATH = REVIEWS_DIR / "ipip_paraphrases.csv"  # all 120, filled in by hand
PARAPHRASE_SUMMARY_PATH = RESULTS_DIR / "paraphrase_summary.md"  # 06
# Item sets 01_run_bigfive.py can run, and the results folder of each.
ITEM_SETS = {"original": "bigfive", "paraphrased": "bigfive_paraphrased"}

# ---------------------------------------------------------------------------
# B1 step 5: scenario questions (behaviour check)
# ---------------------------------------------------------------------------
# Per IPIP facet, N_SCENARIOS_PER_FACET everyday situations with two actions:
# one a high-trait person would take, one a low-trait person would take, both
# acceptable, so a desirability halo can't pick the answer on its own.
N_SCENARIOS_PER_FACET = 4  # 30 facets x 4 = 120 scenarios, like the 120 IPIP items
N_SCENARIO_DRAFTS_PER_FACET = 5  # one spare per facet, for drafts the review drops
SCENARIO_SEED = 20261019  # which option is shown as A
SCENARIO_DRAFTS_PATH = DATA_DIR / "scenarios" / "drafts.jsonl"
SCENARIOS_PATH = DATA_DIR / "scenarios" / "scenarios.json"  # after review
SCENARIO_REVIEW_PATH = REVIEWS_DIR / "scenarios.csv"
SCENARIO_SUMMARY_PATH = RESULTS_DIR / "scenarios_summary.md"  # 09
SCENARIO_MAX_LENGTH_RATIO = 1.6  # longer option / shorter option, in characters

# ---------------------------------------------------------------------------
# B2 difficulty ladder
# ---------------------------------------------------------------------------
LADDER_DIR = DATA_DIR / "ladder"  # <benchmark>.jsonl, built by 10
LADDER_SUMMARY_PATH = RESULTS_DIR / "ladder_summary.md"  # 12
LADDER_SEED = 20261019  # which questions are drawn, and option order for GPQA
LADDER_N = 200  # questions per benchmark (GPQA Diamond has 198, so all of them)
LADDER_MIN_HLE = 200  # fewer text-only multiple-choice HLE questions -> report HLE as exploratory
LADDER_BENCHMARKS = ("arce", "arcc", "mmlu", "gpqa", "hle")  # easiest to hardest
LADDER_NAMES = {"arce": "ARC-Easy", "arcc": "ARC-Challenge", "mmlu": "MMLU", "gpqa": "GPQA Diamond",
                "hle": "HLE (text-only MCQ)"}
# "Pretend to be [level]": one level to start with (PLAN B2), in the state and
# in the instructions, as in B1. "none" is the actual-ability row.
LADDER_PERSONAS = {
    "secondary": "You are an average secondary-school student. Answer every question the way such a "
                 "student would, including getting it wrong when the student would not know the answer.",
}
# Published human accuracy, for the comparison row. Only values with a source;
# None where no comparable human figure is published. Check before citing.
HUMAN_ACCURACY: dict[str, dict[str, float | str] | None] = {
    "arce": None,
    "arcc": None,
    # Hendrycks et al. 2021 (MMLU paper): Amazon Mechanical Turk workers 34.5%; expert-level estimate 89.8%.
    "mmlu": {"non-expert": 0.345, "expert": 0.898, "source": "Hendrycks et al. 2021"},
    # Rein et al. 2023 (GPQA paper), main set: domain experts 65%, skilled non-experts with web access 34%.
    # Diamond keeps questions experts got right and most non-experts got wrong, so it is harsher on non-experts.
    "gpqa": {"non-expert": 0.34, "expert": 0.65, "source": "Rein et al. 2023 (main set)"},
    "hle": None,
}
