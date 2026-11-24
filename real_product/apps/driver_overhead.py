"""Python driver overhead at production population sizes.

The evaluation backend is replaced by an instant fake (deterministic random
statistics and dim-293 signatures, like real 1461-day/step-5 signatures);
QD admission uses the real NativeArchive. The wall time is then the cost of
everything that stays in Python: proposals, variation, normalization, the
expression cache, epsilon-lexicase, archive bookkeeping and checkpoints.

    PYTHONPATH=python:../python/src python3 apps/driver_overhead.py OUT_DIR \
        [--population 100000] [--stages 1 2] [--generations 1]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gp_factor_miner.expressions import expression_depth, expression_key, expression_nodes, normalize_expression
from gp_factor_miner.gp_engine import Candidate

from gpp import hier_engine
from gpp.backend import NativeBackend
from gpp.config import ProductionGPConfig
from gpp.export import write_panels


class FakeEvaluation(NativeBackend):
    dim = 293
    # Real signatures cluster (many expressions are near-copies of a few
    # behaviours); independent random vectors would all be admitted and make
    # the archive, hence QD time, grow without bound. 400 behaviours, most
    # candidates close to one (|corr| ~0.95-0.99), 5% far from all.
    centres = np.random.default_rng(0).normal(size=(400, dim))

    def signature(self, rng):
        centre = self.centres[rng.integers(len(self.centres))]
        noise = 2.0 if rng.random() < 0.05 else rng.choice([0.1, 0.3])
        return (centre + noise * rng.normal(size=self.dim)).astype(np.float32)

    def evaluate(self, expressions, evaluation_config, config, pca_seed):
        t = time.perf_counter()
        out = []
        for expression in expressions:
            e = normalize_expression(expression)
            h = int(hashlib.blake2b(expression_key(e).encode(), digest_size=8).hexdigest(), 16)
            rng = np.random.default_rng(h)
            if rng.random() < 0.15:
                out.append((None, None))
                continue
            seg = tuple(float(x) for x in rng.normal(0.02, 0.02, config.stability_segments))
            ic = float(np.mean(seg))
            out.append((Candidate(expression=e, rank_ic=ic, abs_rank_ic=abs(ic), valid_ratio=1.0, missing_ratio=0.0,
                                  depth=expression_depth(e), nodes=expression_nodes(e),
                                  fitness=abs(ic) - 0.0005 * expression_nodes(e), segment_rank_ic_std=0.01,
                                  same_sign_ratio=1.0, segment_rank_ics=seg),
                        self.signature(rng)))
        self.fake_seconds = getattr(self, "fake_seconds", 0.0) + time.perf_counter() - t
        return out

    def consider(self, archive, records, config, pca_seed):
        t = time.perf_counter()
        answer = super().consider(archive, records, config, pca_seed)
        self.qd_seconds = getattr(self, "qd_seconds", 0.0) + time.perf_counter() - t
        return answer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--population", type=int, default=100_000)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--fields", type=int, default=132)
    a = p.parse_args(argv)
    names = [f"field_{i:03d}" for i in range(a.fields)]
    dates = [f"2016{i:04d}" for i in range(12)]
    frames = {n: pd.DataFrame(np.zeros((12, 3), np.float32), index=dates, columns=list("abc")) for n in names}
    write_panels(a.out / "panels", frames, frames[names[0]], period="evolution")
    profile = pd.DataFrame({"feature": names, "rank_ic": np.linspace(0.05, -0.03, a.fields)})
    pop = a.population
    cfg = replace(ProductionGPConfig(), stage_depths=tuple(a.stages),
                  stage_max_nodes=ProductionGPConfig().stage_max_nodes[: len(a.stages)],
                  population_size=pop, gene_pool_size=pop // 5, parent_pool_size=pop // 5,
                  generations_per_stage=a.generations, progress_chunk_size=pop)
    backend = FakeEvaluation(a.out / "panels", cfg)
    events = []
    t0 = time.perf_counter()

    def log(event):
        events.append({"t": round(time.perf_counter() - t0, 2), "event": event["event"], "stage": event.get("stage"),
                       "generation": event.get("generation"), "phase": event.get("phase"),
                       "fake_evaluate": round(getattr(backend, "fake_seconds", 0.0), 2),
                       "qd": round(getattr(backend, "qd_seconds", 0.0), 2),
                       "archive": event.get("archive_size")})
        if event["event"] in ("warm_start_completed", "generation_completed", "stage_completed"):
            print(events[-1], flush=True)

    result = hier_engine.run_hierarchical_evolution({n: None for n in names}, None, profile, cfg, seed=1,
                                                    progress_callback=log, batch_backend=backend,
                                                    checkpoint_dir=a.out / "ckpt")
    total = time.perf_counter() - t0
    summary = dict(population=pop, stages=a.stages, generations=a.generations, wall=total,
                   fake_evaluate=backend.fake_seconds, qd=getattr(backend, "qd_seconds", 0.0),
                   evaluated=result.evaluated_count, archive=len(result.archive_records),
                   archive_timing=next((c for c in reversed(backend.calls) if c["kind"] == "consider"), None),
                   events=events)
    (a.out / "driver_overhead.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}))


if __name__ == "__main__":
    main()
