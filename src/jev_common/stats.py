"""Statistics for the analysis (A7, A8).

Correctness is passed around as arrays with one entry per question: 1/0 (or
True/False) for a single run. BERT's per-question correctness may be averaged
over its 5 seeds, giving fractions in [0, 1]; the bootstrap and log-odds
functions accept that, the paired tests do not.

Generic defaults live here. Thresholds that belong to one part's design (e.g.
Part A's quiz chance rate and memorisation drop ratio) are required arguments,
passed in from that part's config.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as scipy_stats

N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
ECE_BINS = 10
BOOTSTRAP_SEED = 20261010

_CHUNK = 500  # bootstrap resamples drawn at once, to bound memory


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class McNemarResult:
    a_only: int  # model A right, model B wrong
    b_only: int  # model B right, model A wrong
    p_value: float  # exact two-sided


@dataclass(frozen=True)
class ProportionTestResult:
    z: float
    p_value: float  # two-sided


@dataclass(frozen=True)
class DropComparison:
    """Level 0 -> Level 2 log-odds drop of a subject model vs a reference (A8)."""

    subject_drop: float
    reference_drop: float
    difference: ConfidenceInterval  # subject drop - reference drop
    supported: bool  # CI entirely above 0 and subject drop >= ratio x reference drop


def _as_array(values: ArrayLike, name: str = "values") -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if array.min() < 0 or array.max() > 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _as_binary(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    array = _as_array(values, name)
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be 0/1 or True/False")
    return array.astype(bool)


def _interval_bounds(samples: NDArray[np.float64], confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Accuracy and bootstrap
# ---------------------------------------------------------------------------
def accuracy(correct: ArrayLike) -> float:
    return float(_as_array(correct, "correct").mean())


def bootstrap_ci(
    correct: ArrayLike,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Accuracy with a percentile bootstrap CI, resampling questions."""
    values = _as_array(correct, "correct")
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            values[rng.integers(0, values.size, size=(min(_CHUNK, n_boot - start), values.size))].mean(
                axis=1
            )
            for start in range(0, n_boot, _CHUNK)
        ]
    )
    low, high = _interval_bounds(means, confidence)
    return ConfidenceInterval(float(values.mean()), low, high)


def bootstrap_statistic(
    arrays: Sequence[ArrayLike],
    statistic: Callable[..., float],
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Percentile bootstrap CI of `statistic(*arrays)`.

    All arrays are per-question and paired, so every resample draws the same
    questions from each of them.
    """
    values = [_as_array(array, f"arrays[{i}]") for i, array in enumerate(arrays)]
    n = values[0].size
    if any(array.size != n for array in values):
        raise ValueError("all arrays must have the same length (one entry per question)")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, n, size=n)
        samples[i] = statistic(*(array[index] for array in values))
    low, high = _interval_bounds(samples, confidence)
    return ConfidenceInterval(float(statistic(*values)), low, high)


# ---------------------------------------------------------------------------
# Log-odds drop (A8)
# ---------------------------------------------------------------------------
def log_odds(correct: ArrayLike) -> float:
    """Log-odds of accuracy, with +0.5 / +1 smoothing so 0% and 100% stay finite."""
    values = _as_array(correct, "correct")
    p = (values.sum() + 0.5) / (values.size + 1)
    return float(np.log(p / (1 - p)))


def log_odds_drop(level_a: ArrayLike, level_b: ArrayLike) -> float:
    """log-odds(level A) - log-odds(level B), e.g. Level 0 minus Level 2."""
    return log_odds(level_a) - log_odds(level_b)


def compare_drops(
    subject_a: ArrayLike,
    subject_b: ArrayLike,
    reference_a: ArrayLike,
    reference_b: ArrayLike,
    ratio: float,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> DropComparison:
    """Compare the subject's A -> B log-odds drop with the reference's (A8).

    For the memorisation test: subject = Jev, reference = BERT, A = Level 0,
    B = Level 2, all arrays over the same questions in the same order (pool
    datasets by concatenating). Supported if the bootstrap CI of the
    difference is entirely above 0 and the subject's drop is at least
    `ratio` times the reference's. A negative reference drop (the reference
    did better on B) makes the ratio condition hold for any subject drop >= 0,
    so the CI condition decides.
    """

    def difference(sa: NDArray, sb: NDArray, ra: NDArray, rb: NDArray) -> float:
        return log_odds_drop(sa, sb) - log_odds_drop(ra, rb)

    interval = bootstrap_statistic(
        [subject_a, subject_b, reference_a, reference_b], difference, n_boot, confidence, seed
    )
    subject_drop = log_odds_drop(subject_a, subject_b)
    reference_drop = log_odds_drop(reference_a, reference_b)
    return DropComparison(
        subject_drop=subject_drop,
        reference_drop=reference_drop,
        difference=interval,
        supported=interval.low > 0 and subject_drop >= ratio * reference_drop,
    )


# ---------------------------------------------------------------------------
# Paired and unpaired tests
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a: ArrayLike, correct_b: ArrayLike) -> McNemarResult:
    """Exact paired McNemar test on two 0/1 correctness arrays over the same questions.

    Only questions where the two disagree matter; under the null each
    disagreement is equally likely to favour either side.
    """
    a, b = _as_binary(correct_a, "correct_a"), _as_binary(correct_b, "correct_b")
    if a.size != b.size:
        raise ValueError("correct_a and correct_b must have the same length")
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n_disagree = a_only + b_only
    if n_disagree == 0:
        return McNemarResult(0, 0, 1.0)
    p_value = scipy_stats.binomtest(min(a_only, b_only), n_disagree, 0.5).pvalue
    return McNemarResult(a_only, b_only, float(p_value))


def binomial_test_vs_chance(
    successes: int, n: int, chance: float, alternative: str = "greater"
) -> float:
    """p-value that the success rate differs from `chance` (default: is above it)."""
    return float(scipy_stats.binomtest(successes, n, chance, alternative=alternative).pvalue)


def two_proportion_test(
    successes_1: int, n_1: int, successes_2: int, n_2: int
) -> ProportionTestResult:
    """Two-sided pooled z-test that two proportions are equal."""
    if n_1 <= 0 or n_2 <= 0:
        raise ValueError(f"both groups need at least one trial, got n_1={n_1}, n_2={n_2}")
    pooled = (successes_1 + successes_2) / (n_1 + n_2)
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / n_1 + 1 / n_2))
    if standard_error == 0:  # both groups are all-success or all-failure
        return ProportionTestResult(0.0, 1.0)
    z = (successes_1 / n_1 - successes_2 / n_2) / standard_error
    return ProportionTestResult(float(z), float(2 * scipy_stats.norm.sf(abs(z))))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(
    confidence: ArrayLike, correct: ArrayLike, n_bins: int = ECE_BINS
) -> float:
    """Expected calibration error with equal-width bins on the top-answer probability.

    `confidence` is the probability the model gave its chosen answer;
    `correct` is whether that answer was right.
    """
    conf, hits = _as_array(confidence, "confidence"), _as_array(correct, "correct")
    if conf.size != hits.size:
        raise ValueError("confidence and correct must have the same length")
    # Bins are (lo, hi]; a confidence of exactly 0 joins the first bin.
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.sum() / conf.size * abs(hits[in_bin].mean() - conf[in_bin].mean())
    return float(error)
