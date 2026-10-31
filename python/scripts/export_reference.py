"""Produce reference outputs from the original pandas implementation.

Two modes:

``normalize``
    For a list of raw expressions, record the normalized key, depth and node
    count (or the rejection reason).  Used to test the C++ parser/DAG (E1–E3).

``values``
    Evaluate expressions on an exported panel directory (see export_panels.py)
    and record each factor panel, its daily RankIC series and the GP candidate
    statistics.  Used to test operators, RankIC and fitness (E5–E8).

Mined factors are research output and stay outside the repository: pass them
with ``--factors`` from the data directory, never commit the results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gp_factor_miner.cache import ExpressionCache
from gp_factor_miner.expressions import (
    evaluate_expression,
    expression_depth,
    expression_key,
    expression_nodes,
    normalize_expression,
)
from gp_factor_miner.gp_engine import _evaluate_with_values
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.hierarchical_engine import _evaluation_config
from gp_factor_miner.metrics import daily_rank_ic


def crafted_expressions(f1: str, f2: str) -> list:
    """One case per operator plus a few nested and normalization cases."""
    cases = [
        ["log", f1], ["abs", ["-", f1, f2]], ["rank", f1], ["zscore", f1],
        ["+", f1, f2], ["-", f1, f2], ["*", f1, f2], ["/", f1, f2],
    ]
    for op in ("mean", "std", "sum", "max", "min", "median", "ema", "lag", "delta", "pct_change"):
        for window in (1, 5, 20):
            cases.append([op, f1, window])
    for op in ("correlation", "beta"):
        for window in (5, 20):
            cases.append([op, f1, f2, window])
    cases += [
        ["mean", ["std", f1, 10], 5],
        ["*", ["std", f1, 10], f2],
        ["correlation", ["std", f1, 10], f2, 20],
        ["ema", ["*", ["+", f1, f2], f2], 5],
        ["+", f2, f1],                       # commutative reorder
        ["rank", ["rank", f1]],              # rank(rank(x)) -> rank(x)
        ["-", f1, f1],                       # degenerate
        ["/", ["+", f1, f2], ["+", f2, f1]],  # degenerate after normalization
    ]
    return cases


def normalize_record(expression) -> dict:
    try:
        normalized = normalize_expression(expression)
    except (TypeError, ValueError) as error:
        return {"input": expression, "error": str(error)}
    return {
        "input": expression,
        "key": expression_key(normalized),
        "depth": expression_depth(normalized),
        "nodes": expression_nodes(normalized),
    }


def load_panels(panel_dir: Path):
    meta = json.loads((panel_dir / "meta.json").read_text())
    shape = (meta["T"], meta["N"])

    def read(path: Path) -> pd.DataFrame:
        values = np.fromfile(path, dtype="<f4").reshape(shape)
        return pd.DataFrame(values, index=meta["dates"], columns=meta["assets"])

    data = {name: read(panel_dir / "fields" / f"{name}.f32") for name in meta["fields"]}
    return meta, data, read(panel_dir / "target.f32")


def _finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def run_values(args) -> None:
    panel_dir = Path(args.panel_dir)
    meta, data, target = load_panels(panel_dir)
    f1, f2 = args.field1, args.field2
    expressions = crafted_expressions(f1, f2)
    if args.factors:
        factors = json.loads(Path(args.factors).read_text())
        usable = [item["expression"] for item in factors
                  if _terminals(item["expression"]) <= set(data)]
        expressions += usable[: args.max_factors]

    config = _evaluation_config(HierarchicalGPConfig())
    out = Path(args.out)
    (out / "values").mkdir(parents=True, exist_ok=True)
    records = []
    for index, expression in enumerate(expressions):
        record = normalize_record(expression)
        record["id"] = index
        if "error" not in record:
            values = evaluate_expression(normalize_expression(expression), data)
            values.to_numpy(dtype="<f8").tofile(out / "values" / f"{index:04d}.f8")
            daily, missing_ratio, _ = daily_rank_ic(values, target)
            record["daily_ic"] = [_finite_or_none(v) for v in daily.to_numpy(dtype=float)]
            record["missing_ratio"] = float(missing_ratio)
            # The C++ engine stores panels in float32. Rounding creates ties
            # that change ranks, so the RankIC of the rounded values is kept
            # too: it checks the RankIC code apart from the storage precision.
            daily32, _, _ = daily_rank_ic(values.astype("float32"), target)
            record["daily_ic_f32"] = [_finite_or_none(v) for v in daily32.to_numpy(dtype=float)]
            candidate, _ = _evaluate_with_values(expression, data, target, config, ExpressionCache())
            record["candidate"] = None if candidate is None else {
                "rank_ic": candidate.rank_ic,
                "valid_ratio": candidate.valid_ratio,
                "missing_ratio": candidate.missing_ratio,
                "fitness": candidate.fitness,
                "segment_rank_ics": [_finite_or_none(v) for v in candidate.segment_rank_ics],
                "segment_rank_ic_std": candidate.segment_rank_ic_std,
                "same_sign_ratio": candidate.same_sign_ratio,
            }
        records.append(record)
    reference = {"panel_dir": str(panel_dir), "T": meta["T"], "N": meta["N"],
                 "value_dtype": "float64", "cases": records}
    (out / "reference.json").write_text(json.dumps(reference, indent=1))
    print(f"wrote {len(records)} reference cases to {out}")


def _terminals(expression) -> set[str]:
    if isinstance(expression, str):
        return {expression}
    if isinstance(expression, list):
        return set().union(*(_terminals(item) for item in expression[1:]))
    return set()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    norm = sub.add_parser("normalize")
    norm.add_argument("--out", required=True)
    norm.add_argument("--factors", default=None, help="selected_gp_factors.json (kept outside the repo)")

    vals = sub.add_parser("values")
    vals.add_argument("--panel-dir", required=True)
    vals.add_argument("--out", required=True)
    vals.add_argument("--field1", required=True)
    vals.add_argument("--field2", required=True)
    vals.add_argument("--factors", default=None)
    vals.add_argument("--max-factors", type=int, default=40)

    args = parser.parse_args(argv)
    if args.mode == "normalize":
        expressions = crafted_expressions("a", "b")
        if args.factors:
            expressions += [item["expression"] for item in json.loads(Path(args.factors).read_text())]
        Path(args.out).write_text(json.dumps([normalize_record(e) for e in expressions], indent=1))
        print(f"wrote {len(expressions)} normalization cases to {args.out}")
    else:
        run_values(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
