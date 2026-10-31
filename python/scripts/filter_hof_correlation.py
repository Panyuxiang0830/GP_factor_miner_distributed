from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gp_factor_miner.data import compute_time_split, load_research_rows
from gp_factor_miner.expressions import evaluate_expression, normalize_expression
from gp_factor_miner.gp_engine import terminal_names


_DATA = None
_DOWNSAMPLE = None


def _initialize(data, downsample):
    global _DATA, _DOWNSAMPLE
    _DATA, _DOWNSAMPLE = data, downsample


def _evaluate(expression):
    values = evaluate_expression(expression, _DATA).iloc[::_DOWNSAMPLE]
    return values.to_numpy(dtype=np.float32).ravel()


def _absolute_correlation(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    left_valid = left[valid].astype(np.float64, copy=False)
    right_valid = right[valid].astype(np.float64, copy=False)
    if left_valid.std() == 0 or right_valid.std() == 0:
        return np.nan
    return float(abs(np.corrcoef(left_valid, right_valid)[0, 1]))


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--rankic-threshold", type=float, default=0.03)
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    metrics = pd.read_csv(args.run_dir / "hof_screening_rankic.csv")
    metrics = metrics[
        metrics["abs_screening_rank_ic"].ge(args.rankic_threshold)
        & metrics["same_direction"].eq(1)
        & metrics["error"].isna()
    ].sort_values(
        ["abs_screening_rank_ic", "fitness"], ascending=[False, False]
    ).reset_index(drop=True)
    expressions = [normalize_expression(json.loads(value)) for value in metrics["expression"]]
    features = sorted(set().union(*(terminal_names(value) for value in expressions)))

    split = compute_time_split(args.data_dir)
    frame = load_research_rows(
        args.data_dir, split, ["time_id", "asset_id", *features]
    )
    frame = frame[
        (frame["time_id"] > split.evolution_end)
        & (frame["time_id"] <= split.screening_end)
    ]
    wide = frame.pivot(
        index="time_id", columns="asset_id", values=features
    ).sort_index()
    data = {feature: wide[feature] for feature in features}
    sampled_times = len(next(iter(data.values())).iloc[:: args.downsample])
    print(
        f"[setup] factors={len(expressions)} threshold={args.threshold} "
        f"downsample=1/{args.downsample} sampled_times={sampled_times}",
        flush=True,
    )

    context = mp.get_context("fork")
    with context.Pool(
        args.workers,
        initializer=_initialize,
        initargs=(data, args.downsample),
    ) as pool:
        vectors = list(pool.imap(_evaluate, expressions, chunksize=1))
    print("[setup] factor values computed", flush=True)

    selected_indices = []
    decisions = []
    for index, vector in enumerate(vectors):
        max_correlation = 0.0
        correlated_with = ""
        keep = True
        for selected_index in selected_indices:
            correlation = _absolute_correlation(vector, vectors[selected_index])
            if np.isfinite(correlation) and correlation > max_correlation:
                max_correlation = correlation
                correlated_with = metrics.iloc[selected_index]["expression"]
            if np.isfinite(correlation) and correlation >= args.threshold:
                keep = False
                break
        if keep:
            selected_indices.append(index)
        decisions.append(
            {
                **metrics.iloc[index].to_dict(),
                "selected": int(keep),
                "max_abs_correlation": max_correlation,
                "correlated_with": correlated_with,
                "correlation_threshold": args.threshold,
                "downsample_step": args.downsample,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(vectors):
            print(
                f"[progress] {index + 1}/{len(vectors)} selected={len(selected_indices)}",
                flush=True,
            )

    decisions_frame = pd.DataFrame(decisions)
    selected_frame = decisions_frame[decisions_frame["selected"].eq(1)].copy()
    threshold_tag = f"{args.threshold:.2f}".replace(".", "")
    decisions_path = args.run_dir / f"hof_correlation_{threshold_tag}_decisions.csv"
    selected_path = args.run_dir / f"selected_hof_corr{threshold_tag}.csv"
    json_path = args.run_dir / f"selected_gp_factors_corr{threshold_tag}.json"
    _atomic_csv(decisions_frame, decisions_path)
    _atomic_csv(selected_frame, selected_path)
    payload = []
    for selected_number, row in enumerate(
        selected_frame.to_dict(orient="records"), start=1
    ):
        payload.append(
            {
                "name": f"gp_{selected_number:04d}",
                "expression": json.loads(row["expression"]),
                "evolution_rank_ic": row["evolution_rank_ic"],
                "screening_rank_ic": row["screening_rank_ic"],
                "abs_screening_rank_ic": row["abs_screening_rank_ic"],
                "fitness": row["fitness"],
                "max_abs_correlation": row["max_abs_correlation"],
                "correlation_threshold": row["correlation_threshold"],
                "downsample_step": row["downsample_step"],
            }
        )
    _atomic_json(payload, json_path)
    print(f"[done] selected={len(selected_frame)}/{len(metrics)}", flush=True)
    print(f"[done] selected_output={selected_path}", flush=True)
    print(f"[done] decisions_output={decisions_path}", flush=True)
    print(f"[done] json_output={json_path}", flush=True)


if __name__ == "__main__":
    main()
