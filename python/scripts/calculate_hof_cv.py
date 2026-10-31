from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from gp_factor_miner.data import compute_time_split, load_research_rows
from gp_factor_miner.expressions import evaluate_expression, expression_key
from gp_factor_miner.gp_engine import terminal_names
from gp_factor_miner.metrics import daily_rank_ic


_DATA = None
_TARGET = None


def _initialize_worker(data, target):
    global _DATA, _TARGET
    _DATA = data
    _TARGET = target


def _calculate(item):
    expression, evolution_rank_ic, fitness = item
    try:
        values = evaluate_expression(expression, _DATA)
        daily, missing_ratio, time_count = daily_rank_ic(values, _TARGET)
        valid = daily.dropna()
        ic = float(valid.mean())
        rolling_ic = daily.rolling(20, min_periods=5).mean().dropna()
        rolling_min = float(rolling_ic.min())
        rolling_max = float(rolling_ic.max())
        cv = (rolling_min if ic > 0 else rolling_max) / ic
        return {
            "expression": expression_key(expression),
            "evolution_rank_ic_saved": evolution_rank_ic,
            "ic": ic,
            "abs_ic": abs(ic),
            "ic_rolling20_min": rolling_min,
            "ic_rolling20_max": rolling_max,
            "cv": float(cv),
            "valid_times": int(valid.size),
            "valid_ratio": float(valid.size / time_count),
            "missing_ratio": missing_ratio,
            "fitness": fitness,
            "error": "",
        }
    except Exception as error:
        return {
            "expression": expression_key(expression),
            "evolution_rank_ic_saved": evolution_rank_ic,
            "ic": np.nan,
            "abs_ic": np.nan,
            "ic_rolling20_min": np.nan,
            "ic_rolling20_max": np.nan,
            "cv": np.nan,
            "valid_times": 0,
            "valid_ratio": 0.0,
            "missing_ratio": 1.0,
            "fitness": fitness,
            "error": repr(error),
        }


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    candidates = {}
    for run in range(1, 11):
        path = args.run_dir / "eval" / f"run_{run:02d}" / "completed.pkl"
        with path.open("rb") as handle:
            result = pickle.load(handle)
        for candidate in result.hall_of_fame:
            key = expression_key(candidate.expression)
            current = candidates.get(key)
            if current is None or candidate.fitness > current[2]:
                candidates[key] = (
                    candidate.expression,
                    candidate.rank_ic,
                    candidate.fitness,
                )
    work = list(candidates.values())
    features = sorted(set().union(*(terminal_names(item[0]) for item in work)))
    print(f"[setup] unique_hof={len(work)} source_features={len(features)}", flush=True)

    split = compute_time_split(args.data_dir)
    frame = load_research_rows(
        args.data_dir,
        split,
        ["time_id", "asset_id", "target", *features],
    )
    frame = frame[frame["time_id"] <= split.evolution_end]
    wide = frame.pivot(
        index="time_id", columns="asset_id", values=["target", *features]
    ).sort_index()
    target = wide["target"]
    data = {feature: wide[feature] for feature in features}
    del frame, wide
    print(f"[setup] evolution_times={len(target)} workers={args.workers}", flush=True)

    output = args.run_dir / "hof_cv_rolling20_metrics.csv"
    rows = []
    context = mp.get_context("fork")
    with context.Pool(
        args.workers, initializer=_initialize_worker, initargs=(data, target)
    ) as pool:
        for completed, row in enumerate(pool.imap(_calculate, work, chunksize=1), 1):
            rows.append(row)
            if completed % 100 == 0 or completed == len(work):
                _atomic_csv(pd.DataFrame(rows), output)
                print(f"[progress] {completed}/{len(work)}", flush=True)

    result = pd.DataFrame(rows).sort_values("cv", ascending=False)
    _atomic_csv(result, output)
    valid = result["error"].eq("") & np.isfinite(result["cv"])
    print(f"[done] valid={int(valid.sum())} errors={int((~valid).sum())}", flush=True)
    print(f"[done] output={output}", flush=True)


if __name__ == "__main__":
    main()
