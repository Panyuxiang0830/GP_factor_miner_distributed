from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .expressions import evaluate_expression, expression_key
from .gp_engine import Candidate
from .metrics import RankICResult, rank_ic
from .screening import ScreenedFactor


_VALIDATION_COLUMNS = (
    "expression",
    "evolution_rank_ic",
    "screening_rank_ic",
    "abs_screening_rank_ic",
    "same_direction",
    "valid_times",
    "valid_ratio",
    "missing_ratio",
    "fitness",
    "error",
)
_CORRELATION_COLUMNS = (
    *_VALIDATION_COLUMNS,
    "selected",
    "max_abs_correlation",
    "correlated_with",
    "correlation_threshold",
    "downsample_step",
)


@dataclass(frozen=True)
class _EligibleFactor:
    candidate: Candidate
    validation: RankICResult
    key: str
    name: str


def _atomic_csv(path: Path, rows, columns) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _same_direction(left: float, right: float) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left * right > 0


def _factor_name(candidate, index: int) -> str:
    if isinstance(candidate.expression, str):
        return candidate.expression
    return f"gp_{index:04d}"


def _validation_row(candidate, key: str, result: RankICResult, error: str = ""):
    screening_rank_ic = result.mean
    return {
        "expression": key,
        "evolution_rank_ic": candidate.rank_ic,
        "screening_rank_ic": screening_rank_ic,
        "abs_screening_rank_ic": abs(screening_rank_ic),
        "same_direction": int(_same_direction(candidate.rank_ic, screening_rank_ic)),
        "valid_times": result.valid_times,
        "valid_ratio": result.valid_ratio,
        "missing_ratio": result.missing_ratio,
        "fitness": candidate.fitness,
        "error": error,
    }


def _sampled_rank_vector(name: str, values: pd.DataFrame, step: int) -> np.ndarray:
    """Return one factor's sampled cross-sectional ranks as a flat float32 vector."""
    del name
    sampled = values.iloc[::step]
    return sampled.rank(axis=1, method="average", pct=True).to_numpy(
        dtype=np.float32
    ).ravel()


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    left_values = left[valid].astype(np.float64, copy=False)
    right_values = right[valid].astype(np.float64, copy=False)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0.0:
        return np.nan
    return float(abs(np.dot(left_centered, right_centered) / denominator))


def screen_hierarchical_factors(
    candidates,
    screening_data,
    screening_target,
    output_dir,
    rankic_threshold=0.07,
    corr_threshold=0.75,
    sample_step=20,
    limit=None,
    progress_callback=None,
) -> tuple[ScreenedFactor, ...]:
    """Validate, quality-order, and greedily correlation-prune V4 candidates."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    validation_path = destination / "validation_rankic.csv"
    correlation_path = destination / "correlation_decisions.csv"

    def report(event, **values):
        if progress_callback is not None:
            progress_callback({"event": event, **values})

    total_validation = (
        len({expression_key(candidate.expression) for candidate in candidates})
        if isinstance(candidates, Sequence)
        else None
    )
    seen = set()
    validation_rows = []
    eligible = []
    completed = 0
    for index, candidate in enumerate(candidates):
        key = expression_key(candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        completed += 1
        values = None
        try:
            values = evaluate_expression(candidate.expression, screening_data)
            result = rank_ic(values, screening_target)
            row = _validation_row(candidate, key, result)
            if (
                abs(result.mean) >= rankic_threshold
                and _same_direction(candidate.rank_ic, result.mean)
            ):
                eligible.append(
                    _EligibleFactor(
                        candidate=candidate,
                        validation=result,
                        key=key,
                        name=_factor_name(candidate, index),
                    )
                )
        except Exception as error:
            result = RankICResult(np.nan, 0, 0.0, 1.0)
            row = _validation_row(candidate, key, result, repr(error))
        finally:
            del values
        validation_rows.append(row)
        if completed % 100 == 0:
            _atomic_csv(validation_path, validation_rows, _VALIDATION_COLUMNS)
            report(
                "validation_progress",
                completed=completed,
                total=total_validation,
            )
    _atomic_csv(validation_path, validation_rows, _VALIDATION_COLUMNS)
    report(
        "validation_completed",
        completed=completed,
        total=completed,
    )

    eligible.sort(
        key=lambda item: (
            -abs(item.validation.mean),
            -abs(item.candidate.rank_ic),
            item.candidate.nodes,
            item.key,
        )
    )

    selected = []
    selected_vectors = []
    selected_keys = []
    correlation_rows = []
    for item in eligible:
        if limit is not None and len(selected) >= limit:
            break
        values = evaluate_expression(item.candidate.expression, screening_data)
        try:
            vector = _sampled_rank_vector(item.name, values, sample_step)
        finally:
            del values
        max_correlation = 0.0
        correlated_with = ""
        has_finite_correlation = False
        keep = True
        for selected_key, selected_vector in zip(selected_keys, selected_vectors):
            correlation = _absolute_correlation(vector, selected_vector)
            # Selected vectors are quality-ordered, so strict improvement makes
            # the earliest selected factor the deterministic partner on ties.
            if np.isfinite(correlation) and (
                not has_finite_correlation or correlation > max_correlation
            ):
                max_correlation = correlation
                correlated_with = selected_key
                has_finite_correlation = True
            if np.isfinite(correlation) and correlation >= corr_threshold:
                keep = False

        if keep:
            candidate = item.candidate
            selected.append(
                ScreenedFactor(
                    name=item.name,
                    expression=candidate.expression,
                    evolution_rank_ic=candidate.rank_ic,
                    screening_rank_ic=item.validation.mean,
                    valid_ratio=item.validation.valid_ratio,
                    missing_ratio=item.validation.missing_ratio,
                    depth=candidate.depth,
                    nodes=candidate.nodes,
                    fitness=candidate.fitness,
                    segment_rank_ic_std=candidate.segment_rank_ic_std,
                    same_sign_ratio=candidate.same_sign_ratio,
                )
            )
            selected_vectors.append(vector)
            selected_keys.append(item.key)

        validation_row = _validation_row(
            item.candidate, item.key, item.validation
        )
        correlation_rows.append(
            {
                **validation_row,
                "selected": int(keep),
                "max_abs_correlation": max_correlation,
                "correlated_with": correlated_with,
                "correlation_threshold": corr_threshold,
                "downsample_step": sample_step,
            }
        )
        if len(correlation_rows) % 100 == 0:
            _atomic_csv(correlation_path, correlation_rows, _CORRELATION_COLUMNS)
        report(
            "correlation_progress",
            completed=len(correlation_rows),
            total=len(eligible),
            selected=len(selected),
        )
        del vector

    _atomic_csv(correlation_path, correlation_rows, _CORRELATION_COLUMNS)
    report(
        "correlation_completed",
        completed=len(correlation_rows),
        total=len(eligible),
        selected=len(selected),
    )
    return tuple(selected)
