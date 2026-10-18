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
