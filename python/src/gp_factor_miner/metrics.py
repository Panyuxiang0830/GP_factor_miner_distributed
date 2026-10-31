from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankICResult:
    mean: float
    valid_times: int
    valid_ratio: float
    missing_ratio: float


def daily_rank_ic(factor: pd.DataFrame, target: pd.DataFrame):
    factor, target = factor.align(target, join="inner", axis=None)
    if factor.empty:
        return pd.Series(dtype=float), 1.0, 0
    eligible_values = target.notna()
    eligible_count = int(eligible_values.to_numpy().sum())
    valid_values = factor.notna() & eligible_values
    missing_ratio = (
        1.0
        if eligible_count == 0
        else 1.0 - float(valid_values.to_numpy().sum() / eligible_count)
    )
    factor_rank = factor.rank(axis=1, method="average", pct=True)
    target_rank = target.rank(axis=1, method="average", pct=True)
    daily = factor_rank.corrwith(target_rank, axis=1).replace([np.inf, -np.inf], np.nan)
    return daily, missing_ratio, len(factor.index)


def rank_ic(factor: pd.DataFrame, target: pd.DataFrame) -> RankICResult:
    daily, missing_ratio, time_count = daily_rank_ic(factor, target)
    if time_count == 0:
        return RankICResult(np.nan, 0, 0.0, missing_ratio)
    valid = daily.dropna()
    return RankICResult(
        mean=float(valid.mean()) if not valid.empty else np.nan,
        valid_times=int(valid.size),
        valid_ratio=float(valid.size / time_count),
        missing_ratio=missing_ratio,
    )
