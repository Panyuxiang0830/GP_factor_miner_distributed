from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .expressions import evaluate_expression, expression_key
from .metrics import rank_ic


@dataclass(frozen=True)
class ScreenedFactor:
    name: str
    expression: object
    evolution_rank_ic: float
    screening_rank_ic: float
    valid_ratio: float
    missing_ratio: float
    depth: int
    nodes: int
    fitness: float
    segment_rank_ic_std: float
    same_sign_ratio: float


def _same_direction(left: float, right: float) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left * right > 0


def screen_raw_features(evolution_profile, screening_profile, threshold):
    evolution = evolution_profile.rename(
        columns={"rank_ic": "evolution_rank_ic"}
    )
    screening = screening_profile.rename(
        columns={"rank_ic": "screening_rank_ic"}
    )
    merged = evolution.merge(screening, on="feature", suffixes=("_evolution", "_screening"))
    keep = (
        merged["screening_rank_ic"].abs().ge(threshold)
        & (merged["evolution_rank_ic"] * merged["screening_rank_ic"]).gt(0)
    )
    return merged.loc[keep].sort_values(
        "screening_rank_ic", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def _absolute_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_values = left.to_numpy(dtype=float).ravel()
    right_values = right.to_numpy(dtype=float).ravel()
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 3:
        return np.nan
    if np.std(left_values[valid]) == 0 or np.std(right_values[valid]) == 0:
        return np.nan
    return float(abs(np.corrcoef(left_values[valid], right_values[valid])[0, 1]))


def screen_gp_factors(
    candidates,
    screening_data,
    screening_target,
    threshold=0.04,
    corr_threshold=0.9,
    limit=300,
):
    evaluated = []
    seen = set()
    for index, candidate in enumerate(candidates):
        key = expression_key(candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        try:
            values = evaluate_expression(candidate.expression, screening_data)
            result = rank_ic(values, screening_target)
        except Exception:
            continue
        if abs(result.mean) < threshold or not _same_direction(candidate.rank_ic, result.mean):
            continue
        evaluated.append((candidate, result, values, index))
    evaluated.sort(
        key=lambda item: (-abs(item[1].mean), -item[0].abs_rank_ic, expression_key(item[0].expression))
    )
    selected = []
    selected_values = []
    for candidate, result, values, index in evaluated:
        if any(
            np.isfinite(correlation) and correlation > corr_threshold
            for correlation in (_absolute_correlation(values, other) for other in selected_values)
        ):
            continue
        name = candidate.expression if isinstance(candidate.expression, str) else f"gp_{index:04d}"
        selected.append(
            ScreenedFactor(
                name=name,
                expression=candidate.expression,
                evolution_rank_ic=candidate.rank_ic,
                screening_rank_ic=result.mean,
                valid_ratio=result.valid_ratio,
                missing_ratio=result.missing_ratio,
                depth=candidate.depth,
                nodes=candidate.nodes,
                fitness=candidate.fitness,
                segment_rank_ic_std=candidate.segment_rank_ic_std,
                same_sign_ratio=candidate.same_sign_ratio,
            )
        )
        selected_values.append(values)
        if len(selected) >= limit:
            break
    return selected
