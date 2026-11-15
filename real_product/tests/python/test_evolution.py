"""Whole evolutions: native backend vs Python path vs the course engine."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import gp_factor_miner.hierarchical_engine as course_engine
from gp_factor_miner.expressions import expression_key
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.raw_features import profile_raw_panels

from gpp import hier_engine
from gpp.config import ProductionGPConfig
from test_parity import float32_reference  # noqa: F401  (fixture)


def small(cls, **changes):
    values = dict(population_size=24, generations_per_stage=1, warm_start_multiplier=2, gene_pool_size=10,
                  parent_pool_size=10, progress_chunk_size=16, windows=(1, 3, 5), pca_sample_step=3,
                  pca_qd_block_size=8)
    values.update(changes)
    return replace(cls().smoke(), **values)


def summary(result):
    return dict(
        pools=[[expression_key(c.expression) for c in stage.gene_pool] for stage in result.stages],
        archive=[r.key for r in result.archive_records],
        evaluated=result.evaluated_count,
        history=[(row["qd_admitted"], row["qd_rejected"], row["qd_replaced"], row["archive_size"])
                 for row in result.history],
    )


def test_gpp_python_path_equals_course_engine(evolution_panels):
    """Same configuration, depths 1-3: lexicase rewrite, per-stage caps
    (set to the course cap) and throttled checkpoints change nothing."""
    _, frames, target = evolution_panels
    profile = profile_raw_panels(frames, target)
    course = small(HierarchicalGPConfig, pca_qd_backend="cpu")
    ours = small(ProductionGPConfig, stage_depths=(1, 2, 3), stage_max_nodes=(11, 11, 11), max_nodes=11,
                 pca_qd_backend="cpu")
    a = course_engine.run_hierarchical_evolution(frames, target, profile, course, seed=7)
    b = hier_engine.run_hierarchical_evolution(frames, target, profile, ours, seed=7)
    assert summary(a) == summary(b)


def test_native_backend_equals_python_path(native_lib, evolution_panels, float32_reference, tmp_path, monkeypatch):
    """Depths 1-5, whole run.

    Candidate statistics differ from pandas by ~1e-17 (numpy's pairwise mean
    over days vs the course C++ sequential sum; pandas itself switches to a
    sequential sum when bottleneck is installed). Over a whole run one such
    difference eventually flips a near-tie, so the run is checked in two
    parts: (1) every candidate both paths evaluate agrees to 1e-14; (2) with
    the Python path fed the native candidate statistics (signatures and the
    PCA-QD archive stay pure Python), the whole trajectory - gene pools,
    archive, QD history - is identical, which checks the backend protocol,
    archive mirroring and signature path exactly.
    """
    from gpp.backend import NativeBackend
    from gp_factor_miner.expressions import normalize_expression

    panel_dir, frames, target = evolution_panels
    profile = profile_raw_panels(frames, target)
    cfg = small(ProductionGPConfig, pca_qd_backend="cpu", engine_threads=2)

    backend = NativeBackend(panel_dir, cfg)
    native_seen = {}
    evaluate = backend.evaluate

    def recording(expressions, *args):
        out = evaluate(expressions, *args)
        for candidate, _ in out:
            if candidate is not None:
                native_seen[expression_key(candidate.expression)] = candidate
        return out

    backend.evaluate = recording
    native = hier_engine.run_hierarchical_evolution({k: None for k in frames}, None, profile, cfg, seed=11,
                                                    batch_backend=backend, checkpoint_dir=tmp_path / "ckpt")
    assert [s.depth for s in native.stages] == [1, 2, 3, 4, 5]
    assert sum(row["qd_rejected"] for row in native.history) > 0

    # (1) statistics, on everything the native run evaluated
    python_evaluate = hier_engine._evaluate
    evaluation = hier_engine._evaluation_config(cfg)
    # Known course-engine difference (src/core/operators.cpp, not changed
    # here): on a window where beta's first input is constant, the C++
    # covariance leaves a ~1e-16 residue where pandas gives exactly 0, which
    # can reorder a cross-section full of zeros. Such expressions are held to
    # a loose tolerance and counted.
    loose = 0
    for key, candidate in native_seen.items():
        reference = python_evaluate(json.loads(key), frames, target, evaluation, course_cache())
        assert reference is not None, key
        tolerance = 1e-14
        if '"beta"' in key and abs(candidate.rank_ic - reference.rank_ic) > tolerance:
            tolerance = 1e-4
            loose += 1
        for name in ("rank_ic", "fitness", "valid_ratio", "missing_ratio", "segment_rank_ic_std", "same_sign_ratio"):
            assert abs(getattr(candidate, name) - getattr(reference, name)) <= tolerance, (key, name)
    assert len(native_seen) > 300 and loose <= 0.02 * len(native_seen)

    # (2) trajectory with native statistics in the Python path
    lookup = NativeBackend(panel_dir, cfg)

    def native_statistics(expression, data_dict, target_, config, cache):
        normalized = normalize_expression(expression)
        key = expression_key(normalized)
        cached = cache.get(key)
        if cached is not None:
            return cached
        candidate, _ = lookup.evaluate([normalized], config, cfg, 0)[0]
        if candidate is not None:
            cache.put(key, candidate)
        return candidate

    monkeypatch.setattr(hier_engine, "_evaluate", native_statistics)
    python = hier_engine.run_hierarchical_evolution(frames, target, profile, cfg, seed=11)
    assert summary(native) == summary(python)
    backend.close()
    lookup.close()


def course_cache():
    from gp_factor_miner.cache import ExpressionCache

    return ExpressionCache()


def test_cli_smoke_depth_1_to_5(native_lib, evolution_panels, screening_panels, tmp_path):
    from gpp import cli

    out = tmp_path / "run"
    code = cli.main([
        "--evolution-panels", str(evolution_panels[0]), "--screening-panels", str(screening_panels[0]),
        "--output-dir", str(out), "--smoke", "--threads", "2",
    ])
    assert code == 0
    for depth in range(1, 6):
        assert json.loads((out / f"gene_pool_depth_{depth:02d}.json").read_text())
    assert (out / "validation_rankic.csv").exists() and (out / "correlation_decisions.csv").exists()
    run = json.loads((out / "run_config.json").read_text())
    assert run["completed_depths"] == [1, 2, 3, 4, 5]
    assert pd.read_csv(out / "validation_rankic.csv").shape[0] > 0
