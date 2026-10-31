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


def _rolling(method: str):
    def apply(a, window):
        minimum = 1 if method in {"mean", "sum", "median"} else min(3, int(window))
        return getattr(a.rolling(window=int(window), min_periods=minimum), method)()

    return apply


def _ema(a, window):
    return a.ewm(span=int(window), adjust=False, min_periods=1).mean()


def _correlation(a, b, window):
    return a.rolling(int(window), min_periods=min(3, int(window))).corr(b)


def _beta(a, b, window):
    rolling_a = a.rolling(int(window), min_periods=min(3, int(window)))
    variance = b.rolling(int(window), min_periods=min(3, int(window))).var()
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
