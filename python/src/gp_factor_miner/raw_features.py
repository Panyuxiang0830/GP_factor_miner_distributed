from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import rank_ic


def profile_raw_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    names = list(feature_names)
    wide = frame.pivot(
        index="time_id",
        columns="asset_id",
        values=["target", *names],
    ).sort_index()
    target = wide["target"]
    records = []
    for feature in names:
        result = rank_ic(wide[feature], target)
        records.append(
            {
                "feature": feature,
                "rank_ic": result.mean,
                "abs_rank_ic": abs(result.mean) if np.isfinite(result.mean) else np.nan,
                "valid_times": result.valid_times,
                "valid_ratio": result.valid_ratio,
                "missing_ratio": result.missing_ratio,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("feature").reset_index(drop=True)


def profile_raw_panels(
    panels: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Profile already-wide EOD panels without a costly long-to-wide pivot."""

    records = []
    for feature, panel in panels.items():
        result = rank_ic(panel, target)
        records.append(
            {
                "feature": feature,
                "rank_ic": result.mean,
                "abs_rank_ic": abs(result.mean) if np.isfinite(result.mean) else np.nan,
                "valid_times": result.valid_times,
                "valid_ratio": result.valid_ratio,
                "missing_ratio": result.missing_ratio,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("feature").reset_index(drop=True)


def lock_raw_threshold(
    evolution_profile: pd.DataFrame,
    preferred: float = 0.03,
    fallback: float = 0.01,
    minimum_count: int = 30,
) -> float:
    values = evolution_profile["rank_ic"].abs()
    return preferred if int((values >= preferred).sum()) >= minimum_count else fallback


def feature_sampling_probabilities(
    profile: pd.DataFrame,
    uniform_share: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= uniform_share <= 1.0:
        raise ValueError("uniform_share must be between 0 and 1")
    count = len(profile)
    if count == 0:
        raise ValueError("profile must contain at least one feature")
    uniform = np.full(count, 1.0 / count)
    weights = profile["rank_ic"].abs().fillna(0.0).to_numpy(dtype=float)
    weighted = uniform if weights.sum() == 0 else weights / weights.sum()
    probabilities = uniform_share * uniform + (1.0 - uniform_share) * weighted
    return probabilities / probabilities.sum()
