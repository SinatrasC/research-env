"""Paired statistics for comparing two conditions measured on the same tasks.

Moved unchanged from the v0 General Agent adapter: a deterministic bootstrap (fixed seed, 2000
resamples) and an exact two-sided sign test, so v0 study numbers reproduce.
"""

from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Any

BOOTSTRAP_SEED = 20260520
BOOTSTRAP_RESAMPLES = 2000


def bootstrap_ci95(values: list[float]) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(BOOTSTRAP_SEED)
    boot = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(BOOTSTRAP_RESAMPLES))
    return [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]]


def sign_test_two_sided(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def paired_summary(left: list[float], right: list[float]) -> dict[str, Any]:
    """Statistics of `right - left` over paired values (same task, two conditions)."""
    if len(left) != len(right):
        raise ValueError("paired_summary needs equal-length lists.")
    deltas = [r - lo for lo, r in zip(left, right)]
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    return {
        "n_pairs": len(deltas),
        "left_mean": mean(left) if left else None,
        "right_mean": mean(right) if right else None,
        "mean_delta": mean(deltas) if deltas else None,
        "median_delta": median(deltas) if deltas else None,
        "right_better_count": better,
        "right_worse_count": worse,
        "tied_count": len(deltas) - better - worse,
        "sign_test_non_tie_n": better + worse,
        "sign_test_two_sided_p": sign_test_two_sided(better, worse),
        "bootstrap_mean_delta_ci95": bootstrap_ci95(deltas),
    }
