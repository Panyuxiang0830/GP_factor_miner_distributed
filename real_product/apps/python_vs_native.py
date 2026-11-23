"""Python (course) evaluation vs the native engine on the same expressions.

Production-shaped synthetic panels (T=1461, N=5876 by default). The Python
side is the course's per-expression path: _evaluate (pandas operators, daily
RankIC, fitness) plus factor_signature, without the expression cache. The
native side is NativeBackend.evaluate (one hash-consed DAG for the batch) with
a given thread count and number of GPUs.

    PYTHONPATH=python:../python/src python3 apps/python_vs_native.py OUT_DIR \
        [--T 1461] [--N 5876] [--per-depth 8] [--python-limit 40] [--runs 1:0 32:0 32:1]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from gp_factor_miner.cache import ExpressionCache
from gp_factor_miner.expressions import evaluate_expression, normalize_expression
from gp_factor_miner.gp_engine import _evaluate
from gp_factor_miner.pca_qd import factor_signature

from gpp.backend import NativeBackend
from gpp.config import ProductionGPConfig
from gpp.export import write_panels
from gpp.hier_engine import _evaluation_config
from gpp.hier_ops import balanced_stage_proposals
from gpp.synthetic import FIELDS, make_frames


def expressions_for(config, per_depth, seed=3):
    pools, out = [], []
    for depth in range(1, 6):
        batch = balanced_stage_proposals(FIELDS, config.windows, depth, per_depth, seed + depth, tuple(pools))
        out.extend(batch)
        pools.append(tuple(batch[: max(1, per_depth // 2)]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--T", type=int, default=1461)
    p.add_argument("--N", type=int, default=5876)
    p.add_argument("--per-depth", type=int, default=8)
    p.add_argument("--python-limit", type=int, default=40)
    p.add_argument("--runs", nargs="+", default=["1:0", "32:0", "32:1"], help="threads:gpus")
    a = p.parse_args(argv)
    frames, target = make_frames(a.T, a.N, seed=11)
    write_panels(a.out / "panels", frames, target, period="evolution")
    base = replace(ProductionGPConfig(), stage_depths=(1, 2, 3, 4, 5), max_nodes=64)
    expressions = expressions_for(base, a.per_depth)
    evaluation = _evaluation_config(base)
    report = {"T": a.T, "N": a.N, "expressions": len(expressions)}

    python = expressions[: a.python_limit]
    t = time.perf_counter()
    for e in python:
        candidate = _evaluate(e, frames, target, evaluation, ExpressionCache())
        if candidate is not None:
            try:
                factor_signature(evaluate_expression(normalize_expression(e), frames), base.pca_sample_step,
                                 base.pca_power_iterations, seed=1234)
            except ValueError:
                pass
    py = time.perf_counter() - t
    report["python"] = {"expressions": len(python), "seconds": py, "per_expression": py / len(python)}
    print(json.dumps(report["python"]), flush=True)

    for run in a.runs:
        threads, gpus = (int(x) for x in run.split(":"))
        cfg = replace(base, engine_threads=threads, gpus=gpus)
        backend = NativeBackend(a.out / "panels", cfg)
        backend.evaluate(expressions[:2], evaluation, cfg, pca_seed=1234)  # load fields, start GPUs
        batch = expressions if threads > 1 else python
        t = time.perf_counter()
        backend.evaluate(batch, evaluation, cfg, pca_seed=1234)
        s = time.perf_counter() - t
        row = {"threads": threads, "gpus": gpus, "expressions": len(batch), "seconds": s,
               "per_expression": s / len(batch), "speedup_vs_python": (py / len(python)) / (s / len(batch))}
        report[f"native_{threads}t_{gpus}g"] = row
        print(json.dumps(row), flush=True)
    (a.out / "python_vs_native.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
