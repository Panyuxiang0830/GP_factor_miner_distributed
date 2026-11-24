"""Native version of screen_hierarchical_factors with identical outputs.

Same rules, same CSV files (validation_rankic.csv, correlation_decisions.csv)
and the same returned ScreenedFactor tuple; the work happens in one call of
the C++ screening (shared DAG, single evaluation per candidate, blocked
correlation products, serial greedy scan).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from gp_factor_miner.expressions import expression_key
from gp_factor_miner.hierarchical_screening import (
    _CORRELATION_COLUMNS,
    _VALIDATION_COLUMNS,
    _factor_name,
    _same_direction,
)
from gp_factor_miner.screening import ScreenedFactor

from .native import native_screen


def _atomic_csv(path: Path, rows, columns) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _row(candidate, key, mean, valid_times, valid_ratio, missing_ratio, error=""):
    return {
        "expression": key,
        "evolution_rank_ic": candidate.rank_ic,
        "screening_rank_ic": mean,
        "abs_screening_rank_ic": abs(mean),
        "same_direction": int(_same_direction(candidate.rank_ic, mean)),
        "valid_times": valid_times,
        "valid_ratio": valid_ratio,
        "missing_ratio": missing_ratio,
        "fitness": candidate.fitness,
        "error": error,
    }


def screen_hierarchical_factors_native(
    candidates,
    screening_panel_dir,
    output_dir,
    rankic_threshold=0.07,
    corr_threshold=0.75,
    sample_step=20,
    limit=None,
    progress_callback=None,
    *,
    dates=0,
    stocks=0,
    engine_options=None,
):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    def report(event, **values):
        if progress_callback is not None:
            progress_callback({"event": event, **values})

    # Validation order and de-duplication exactly as the Python loop.
    unique, keys, names = [], [], []
    seen = set()
    for index, candidate in enumerate(candidates):
        key = expression_key(candidate.expression)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        keys.append(key)
        names.append(_factor_name(candidate, index))

    options = dict(engine_options or {})
    options.update(rankic_threshold=rankic_threshold, corr_threshold=corr_threshold, sample_step=sample_step,
                   limit=-1 if limit is None else int(limit))
    result = native_screen(
        screening_panel_dir,
        [c.expression for c in unique],
        [c.rank_ic for c in unique],
        [c.nodes for c in unique],
        keys,
        dates=dates,
        stocks=stocks,
        options=options,
    )
    validation_rows = []
    for i, candidate in enumerate(unique):
        if result["status"][i]:
            # Python: RankICResult(nan, 0, 0.0, 1.0) and repr(error).
            validation_rows.append(_row(candidate, keys[i], np.nan, 0, 0.0, 1.0,
                                        repr(ValueError(result["errors"][i]))))
        else:
            validation_rows.append(_row(candidate, keys[i], float(result["mean"][i]), int(result["valid_times"][i]),
                                        float(result["valid_ratio"][i]), float(result["missing_ratio"][i])))
    _atomic_csv(destination / "validation_rankic.csv", validation_rows, _VALIDATION_COLUMNS)
    report("validation_completed", completed=len(unique), total=len(unique))

    order = result["order"]
    selected = []
    correlation_rows = []
    for position in range(len(result["selected"])):
        i = int(order[position])
        candidate = unique[i]
        keep = bool(result["selected"][position])
        partner = int(result["correlated_with"][position])
        correlation_rows.append({
            **validation_rows[i],
            "selected": int(keep),
            "max_abs_correlation": float(result["max_correlation"][position]),
            "correlated_with": keys[partner] if partner >= 0 else "",
            "correlation_threshold": corr_threshold,
            "downsample_step": sample_step,
        })
        if keep:
            selected.append(ScreenedFactor(
                name=names[i],
                expression=candidate.expression,
                evolution_rank_ic=candidate.rank_ic,
                screening_rank_ic=float(result["mean"][i]),
                valid_ratio=float(result["valid_ratio"][i]),
                missing_ratio=float(result["missing_ratio"][i]),
                depth=candidate.depth,
                nodes=candidate.nodes,
                fitness=candidate.fitness,
                segment_rank_ic_std=candidate.segment_rank_ic_std,
                same_sign_ratio=candidate.same_sign_ratio,
            ))
    _atomic_csv(destination / "correlation_decisions.csv", correlation_rows, _CORRELATION_COLUMNS)
    report("correlation_completed", completed=len(correlation_rows), total=len(order), selected=len(selected),
           stats=result["stats"])
    return tuple(selected)
