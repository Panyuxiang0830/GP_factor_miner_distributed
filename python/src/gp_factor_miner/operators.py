from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _clean(value):
    return value.replace([np.inf, -np.inf], np.nan) if isinstance(value, pd.DataFrame) else value


def _divide(a, b):
    return _clean(a / b.where(b.abs() > 1e-12))


def _log(a):
    return np.log(a.where(a > 0))


def _rank(a):
    return a.rank(axis=1, method="average", pct=True)


def _zscore(a):
    clean = _clean(a)
    means = clean.mean(axis=1)
    stds = clean.std(axis=1).replace(0, np.nan)
    return clean.sub(means, axis=0).div(stds, axis=0)


def _constant_window(a, window):
    """True where every observation in the window is equal.

    The variance of such a window is exactly 0, but the rolling algorithm
    leaves a rounding residue that depends on where the recursion started.
    Setting it to 0 makes the result independent of the start, which the
    date-distributed C++ engine needs.
    """
    rolling = a.rolling(int(window), min_periods=1)
    return rolling.max() == rolling.min()


def _rolling_var(a, window):
    var = a.rolling(int(window), min_periods=min(3, int(window))).var()
    return var.mask(_constant_window(a, window) & var.notna(), 0.0)


def _rolling(method: str):
    def apply(a, window):
        minimum = 1 if method in {"mean", "sum", "median"} else min(3, int(window))
        result = getattr(a.rolling(window=int(window), min_periods=minimum), method)()
        if method == "std":
            result = result.mask(_constant_window(a, window) & result.notna(), 0.0)
        return result

    return apply


def _ema(a, window):
    return a.ewm(span=int(window), adjust=False, min_periods=1).mean()


def _correlation(a, b, window):
    corr = a.rolling(int(window), min_periods=min(3, int(window))).corr(b)
    # Pairwise-complete windows as in pandas. A constant window has no
    # correlation; its covariance is only a residue, so return NaN, not +-inf.
    variance = _rolling_var(a + 0 * b, window) * _rolling_var(b + 0 * a, window)
    return corr.mask(variance == 0)


def _beta(a, b, window):
    rolling_a = a.rolling(int(window), min_periods=min(3, int(window)))
    variance = _rolling_var(b, window)
    return _clean(rolling_a.cov(b) / variance.where(variance.abs() > 1e-12))


OPERATORS: dict[str, Callable] = {
    "+": lambda a, b: _clean(a + b),
    "-": lambda a, b: _clean(a - b),
    "*": lambda a, b: _clean(a * b),
    "/": _divide,
    "log": _log,
    "abs": np.abs,
    "rank": _rank,
    "zscore": _zscore,
    "mean": _rolling("mean"),
    "std": _rolling("std"),
    "max": _rolling("max"),
    "min": _rolling("min"),
    "sum": _rolling("sum"),
    "median": _rolling("median"),
    "ema": _ema,
    "lag": lambda a, periods: a.shift(int(periods)),
    "delta": lambda a, periods: a.diff(int(periods)),
    "pct_change": lambda a, periods: _clean(a.pct_change(periods=int(periods))),
    "correlation": _correlation,
    "beta": _beta,
}
