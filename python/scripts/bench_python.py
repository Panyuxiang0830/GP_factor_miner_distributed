"""Time the original Python batch evaluation (baseline for the C++ engine).

Evaluates the same expressions on the same exported panels as
``apps/gpe_bench`` with the miner's own worker functions: a fork-based
multiprocessing pool, one whole expression per task (``_worker_evaluate``:
factor values, daily RankIC, fitness; no PCA signature).

    PYTHONPATH=src python scripts/bench_python.py --panel-dir <dir> \\
        --expressions <file.json> --exprs 200 --workers 1 8 16 32 --csv out.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gp_factor_miner.cache import ExpressionCache
from gp_factor_miner.gp_engine import _evaluate, _initialize_worker, _worker_evaluate
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.hierarchical_engine import _evaluation_config


def _terminals(expression) -> set[str]:
    if isinstance(expression, str):
        return {expression}
    if isinstance(expression, list):
        return set().union(*(_terminals(item) for item in expression[1:]))
    return set()


def load_panels(panel_dir: Path, fields: set[str]):
    meta = json.loads((panel_dir / "meta.json").read_text())
    T, N = meta["T"], meta["N"]
    index = pd.Index(meta["dates"])
    columns = pd.Index(meta["assets"])

    def read(path):
        values = np.fromfile(path, dtype="<f4").reshape(T, N)
        return pd.DataFrame(values, index=index, columns=columns)

    data = {f: read(panel_dir / "fields" / f"{f}.f32") for f in sorted(fields)}
    return data, read(panel_dir / "target.f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-dir", required=True, type=Path)
    parser.add_argument("--expressions", required=True, type=Path)
    parser.add_argument("--exprs", type=int, default=200)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    items = json.loads(args.expressions.read_text())
    expressions = [item["expression"] if isinstance(item, dict) else item for item in items][: args.exprs]
    fields = set().union(*(_terminals(e) for e in expressions))

    start = time.perf_counter()
    data, target = load_panels(args.panel_dir, fields)
    load_time = time.perf_counter() - start
    config = _evaluation_config(HierarchicalGPConfig())
    print(f"loaded {len(fields)} fields, {target.shape[0]} dates x {target.shape[1]} assets in {load_time:.1f}s")

    rows = []
    for workers in args.workers:
        start = time.perf_counter()
        if workers == 1:
            pool_start = 0.0
            candidates = [_evaluate(e, data, target, config, ExpressionCache()) for e in expressions]
        else:
            context = mp.get_context("fork")
            pool = context.Pool(processes=workers, initializer=_initialize_worker,
                                initargs=(data, target, config))
            pool.map(abs, range(workers))  # wait until every worker is up
            pool_start = time.perf_counter() - start
            candidates = [c for c, _ in pool.imap(_worker_evaluate, expressions, chunksize=1)]
            pool.close()
            pool.join()
        total = time.perf_counter() - start
        accepted = sum(c is not None for c in candidates)
        print(f"workers={workers:3d}  total {total:8.2f}s  (pool start {pool_start:.2f}s)  accepted {accepted}")
        rows.append({"label": "python", "workers": workers, "exprs": len(expressions),
                     "T": target.shape[0], "N": target.shape[1], "total": total,
                     "pool_start": pool_start, "accepted": accepted})

    if args.csv:
        frame = pd.DataFrame(rows)
        frame.to_csv(args.csv, mode="a", header=not args.csv.exists(), index=False)


if __name__ == "__main__":
    main()
