from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from gp_factor_miner.data import compute_time_split, load_research_rows
from gp_factor_miner.expressions import evaluate_expression, expression_key
from gp_factor_miner.gp_engine import terminal_names
from gp_factor_miner.metrics import rank_ic


_DATA = None
_TARGET = None


def _initialize(data, target):
    global _DATA, _TARGET
    _DATA, _TARGET = data, target


def _calculate(item):
    expression, evolution_rank_ic, fitness = item
    try:
        result = rank_ic(evaluate_expression(expression, _DATA), _TARGET)
        return {
            "expression": expression_key(expression),
            "evolution_rank_ic": evolution_rank_ic,
            "screening_rank_ic": result.mean,
            "abs_screening_rank_ic": abs(result.mean),
            "same_direction": int(evolution_rank_ic * result.mean > 0),
            "valid_times": result.valid_times,
            "valid_ratio": result.valid_ratio,
            "missing_ratio": result.missing_ratio,
            "fitness": fitness,
            "error": "",
        }
    except Exception as error:
        return {
            "expression": expression_key(expression),
            "evolution_rank_ic": evolution_rank_ic,
            "screening_rank_ic": np.nan,
            "abs_screening_rank_ic": np.nan,
            "same_direction": 0,
            "valid_times": 0,
            "valid_ratio": 0.0,
            "missing_ratio": 1.0,
            "fitness": fitness,
            "error": repr(error),
        }


def _write(frame, path):
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
            payload = pickle.load(handle)
        result = payload["result"] if isinstance(payload, dict) else payload
        for candidate in result.hall_of_fame:
            key = expression_key(candidate.expression)
            if key not in candidates or candidate.fitness > candidates[key][2]:
                candidates[key] = (
                    candidate.expression,
                    candidate.rank_ic,
                    candidate.fitness,
                )
    work = list(candidates.values())
    features = sorted(set().union(*(terminal_names(item[0]) for item in work)))
    print(f"[setup] factors={len(work)} features={len(features)}", flush=True)

    split = compute_time_split(args.data_dir)
    frame = load_research_rows(
        args.data_dir, split, ["time_id", "asset_id", "target", *features]
    )
    frame = frame[
        (frame["time_id"] > split.evolution_end)
        & (frame["time_id"] <= split.screening_end)
    ]
    wide = frame.pivot(
        index="time_id", columns="asset_id", values=["target", *features]
    ).sort_index()
    target = wide["target"]
    data = {feature: wide[feature] for feature in features}
    print(f"[setup] screening_times={len(target)}", flush=True)

    rows = []
    output = args.run_dir / "hof_screening_rankic.csv"
    with mp.get_context("fork").Pool(
        args.workers, initializer=_initialize, initargs=(data, target)
    ) as pool:
        for completed, row in enumerate(pool.imap(_calculate, work, chunksize=1), 1):
            rows.append(row)
            if completed % 100 == 0 or completed == len(work):
                _write(pd.DataFrame(rows), output)
                print(f"[progress] {completed}/{len(work)}", flush=True)

    result = pd.DataFrame(rows)
    _write(result.sort_values("abs_screening_rank_ic", ascending=False), output)
    above = result["abs_screening_rank_ic"] > 0.02
    same = above & result["same_direction"].eq(1)
    print(f"[done] abs_gt_002={int(above.sum())}", flush=True)
    print(f"[done] abs_gt_002_same_direction={int(same.sum())}", flush=True)
    print(f"[done] output={output}", flush=True)


if __name__ == "__main__":
    main()
