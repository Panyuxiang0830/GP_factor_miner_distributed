"""Epsilon-lexicase on a precomputed matrix.

``gp_factor_miner.gp_engine._epsilon_lexicase`` rebuilds a Python list of
signed segment ICs for the whole parent pool on every selection: O(pool) Python
work per draw, i.e. ~10^9 interpreted operations per generation with a
20,000-member pool and 100,000 draws. This class builds the matrix once per
generation and filters index arrays with NumPy. It draws exactly the same
random numbers (``rng.shuffle(cases)``, ``rng.choice(survivors)``, and the
tournament fallback) and applies the same median/MAD cutoff, so the selected
expression is identical.
"""
from __future__ import annotations

import numpy as np

from gp_factor_miner.gp_engine import _tournament


class LexicaseSelector:
    def __init__(self, scored):
        self.scored = scored
        survivors = [candidate for candidate, _ in scored if candidate.segment_rank_ics]
        self.survivors = survivors
        if survivors:
            cases = len(survivors[0].segment_rank_ics)
            self.matrix = np.asarray(
                [
                    [
                        candidate.segment_rank_ics[case] * (1.0 if candidate.rank_ic > 0 else -1.0)
                        for case in range(cases)
                    ]
                    for candidate in survivors
                ],
                dtype=float,
            ).reshape(len(survivors), cases)
            self.cases = cases

    def select(self, rng, epsilon_scale=1.0):
        if not self.survivors:
            return _tournament(self.scored, rng, min(3, len(self.scored)))
        cases = list(range(self.cases))
        rng.shuffle(cases)
        index = np.arange(len(self.survivors))
        for case in cases:
            values = self.matrix[index, case]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            finite_values = values[finite]
            median = float(np.median(finite_values))
            epsilon = epsilon_scale * float(np.median(np.abs(finite_values - median)))
            cutoff = float(finite_values.max()) - epsilon
            with np.errstate(invalid="ignore"):
                keep = finite & (values >= cutoff)
            index = index[keep]
            if len(index) <= 1:
                break
        survivors = [self.survivors[i] for i in index]
        return rng.choice(survivors).expression


def _check_equivalence(scored, seeds=range(50), epsilon_scale=1.0):
    """Test helper: compare with the original implementation."""
    import random

    from gp_factor_miner.gp_engine import _epsilon_lexicase

    selector = LexicaseSelector(scored)
    for seed in seeds:
        a, b = random.Random(seed), random.Random(seed)
        left = _epsilon_lexicase(scored, a, epsilon_scale)
        right = selector.select(b, epsilon_scale)
        if left != right or a.getstate() != b.getstate():
            return False
    return True

