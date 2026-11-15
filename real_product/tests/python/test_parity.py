"""libgpp_native vs the course Python implementation, item by item."""
from dataclasses import replace
import random

import numpy as np
import pandas as pd
import pytest

from gp_factor_miner.cache import ExpressionCache
from gp_factor_miner.expressions import evaluate_expression, expression_key, normalize_expression
from gp_factor_miner.gp_engine import Candidate, _evaluate
from gp_factor_miner.hierarchical_screening import screen_hierarchical_factors
from gp_factor_miner.metrics import rank_ic
from gp_factor_miner.pca_qd import PCAQDArchive, QDRecord, factor_signature

from conftest import FIELDS
from gpp.config import ProductionGPConfig
from gpp.hier_engine import _evaluation_config
from gpp.hier_ops import balanced_stage_proposals
from gpp.selection import LexicaseSelector, _check_equivalence

WINDOWS = (1, 3, 5, 10)


@pytest.fixture
def float32_reference(monkeypatch):
    """Course Python operators with float32 storage after every node.

    The existing C++ engine (src/core) and libgpp_native compute each operator
    in double from float32 inputs and store its output panel in float32;
    pandas keeps float64 throughout. The rounding can create or break a tie
    in a cross-sectional rank (here: depth <= 2 identical, ~2-11% of depth 3-4
    expressions differ by up to 3e-5 in mean RankIC; report/progress.md
    documents the same effect for the course engine). Rounding every Python
    operator output to float32 removes the difference exactly (|diff| ~1e-17),
    so this is the item-by-item reference (plus pct_change in float32, see
    below); test_float64_difference_is_small
    bounds the gap to unmodified pandas.
    """
    import gp_factor_miner.expressions as expressions

    for name, function in list(expressions.OPERATORS.items()):
        def rounded(*args, _f=function, _name=name):
            if _name == "pct_change":
                # src/core/operators.cpp computes pct_change in float32 (as
                # pandas does on the float32 raw fields) for every input.
                args = (args[0].astype(np.float32), *args[1:])
            out = _f(*args)
            return out.astype(np.float32).astype(np.float64) if hasattr(out, "astype") else out
        monkeypatch.setitem(expressions.OPERATORS, name, rounded)

    # A day whose paired-valid factor (or target) cross-section is constant
    # has no rank correlation. The C++ engines return NaN; pandas returns NaN
    # or a 1e-16 residue depending on N and its summation layout (about half
    # of N values each). The reference uses the defined value, NaN.
    import gp_factor_miner.gp_engine as gp_engine
    import gp_factor_miner.metrics as metrics

    original = metrics.daily_rank_ic

    def daily_rank_ic(factor, target):
        daily, missing_ratio, count = original(factor, target)
        f, t = factor.align(target, join="inner", axis=None)
        both = f.notna() & t.notna()
        constant = (f.where(both).nunique(axis=1) <= 1) | (t.where(both).nunique(axis=1) <= 1)
        return daily.mask(constant.reindex(daily.index, fill_value=False)), missing_ratio, count

    monkeypatch.setattr(metrics, "daily_rank_ic", daily_rank_ic)
    monkeypatch.setattr(gp_engine, "daily_rank_ic", daily_rank_ic)


def config(**changes):
    values = dict(stage_depths=(1, 2, 3, 4, 5), windows=WINDOWS, pca_sample_step=5, max_nodes=21,
                  engine_threads=2, memory_budget_gb=0.0)
    values.update(changes)
    return replace(ProductionGPConfig(), **values)


def hierarchical_expressions(count_per_depth=40, seed=3):
    pools, out = [], []
    for depth in range(1, 6):
        batch = balanced_stage_proposals(FIELDS, WINDOWS, depth, count_per_depth, seed + depth, tuple(pools))
        out.extend(batch)
        pools.append(tuple(batch[: count_per_depth // 2]))
    return out


# -- factor values, RankIC, fitness, signatures --------------------------------

def test_engine_matches_python(native_lib, evolution_panels, float32_reference):
    from gpp.backend import NativeBackend

    panel_dir, frames, target = evolution_panels
    cfg = config()
    backend = NativeBackend(panel_dir, cfg)
    backend.validate_inputs(frames, target, cfg)  # exported values == Python frames
    expressions = hierarchical_expressions()
    evaluation = _evaluation_config(cfg)
    native = backend.evaluate(expressions, evaluation, cfg, pca_seed=1234)
    compared = signatures = 0
    for expression, (candidate, signature) in zip(expressions, native):
        reference = _evaluate(expression, frames, target, evaluation, ExpressionCache())
        assert (candidate is None) == (reference is None), expression
        if reference is None:
            continue
        compared += 1
        assert candidate.expression == reference.expression
        assert (candidate.depth, candidate.nodes) == (reference.depth, reference.nodes)
        for name in ("rank_ic", "valid_ratio", "missing_ratio", "fitness", "segment_rank_ic_std", "same_sign_ratio"):
            assert getattr(candidate, name) == pytest.approx(getattr(reference, name), abs=1e-9, rel=1e-9), name
        np.testing.assert_allclose(candidate.segment_rank_ics, reference.segment_rank_ics, atol=1e-9)
        values = evaluate_expression(normalize_expression(expression), frames)
        try:
            expected = factor_signature(values, cfg.pca_sample_step, cfg.pca_power_iterations, seed=1234)
        except ValueError:
            expected = None
        assert (signature is None) == (expected is None)
        if expected is not None:
            signatures += 1
            np.testing.assert_allclose(signature, expected, atol=2e-5)
    assert compared > 100 and signatures > 100
    assert backend.calls[-1]["kind"] == "evaluate"
    backend.close()


def test_daily_rank_ic_on_values(native_lib, evolution_panels, float32_reference):
    """RankIC itself, independent of the fitness filter, on deep expressions."""
    from gpp.backend import NativeBackend

    panel_dir, frames, target = evolution_panels
    cfg = config(min_valid_time_ratio=0.0, max_missing_ratio=1.0, minimum_same_sign_ratio=0.0)
    backend = NativeBackend(panel_dir, cfg)
    deep = [e for e in hierarchical_expressions(20, seed=9)][-40:]
    native = backend.evaluate(deep, _evaluation_config(cfg), cfg, pca_seed=5)
    for expression, (candidate, _) in zip(deep, native):
        result = rank_ic(evaluate_expression(normalize_expression(expression), frames), target)
        if candidate is None:
            assert not np.isfinite(result.mean) or result.valid_times == 0
            continue
        assert candidate.rank_ic == pytest.approx(result.mean, abs=1e-12)
        assert candidate.valid_ratio == pytest.approx(result.valid_ratio)
    backend.close()


def test_float64_difference_is_small(native_lib, evolution_panels):
    """Against unmodified (float64) pandas: shallow trees identical, deep ones close."""
    from gpp.backend import NativeBackend

    panel_dir, frames, target = evolution_panels
    cfg = config(min_valid_time_ratio=0.0, max_missing_ratio=1.0, minimum_same_sign_ratio=0.0)
    backend = NativeBackend(panel_dir, cfg)
    expressions = hierarchical_expressions(30, seed=9)
    gaps = {}
    for expression, (candidate, _) in zip(expressions, backend.evaluate(expressions, _evaluation_config(cfg), cfg, 5)):
        if candidate is not None:
            mean = rank_ic(evaluate_expression(normalize_expression(expression), frames), target).mean
            gaps.setdefault(min(candidate.depth, 3), []).append(abs(candidate.rank_ic - mean))
    assert max(max(gaps[1]), max(gaps[2])) < 1e-4
    assert max(gaps[3]) < 1e-3
    assert np.mean(np.asarray(gaps[3]) < 1e-12) > 0.8
    backend.close()


# -- PCA-QD admission ------------------------------------------------------------

def _records(rng, count, dim, prefix, centers):
    out = []
    for i in range(count):
        center = centers[rng.integers(len(centers))]
        signature = (center + rng.normal(size=dim) * rng.choice([0.05, 0.3, 1.0])).astype(np.float32)
        quality = float(rng.uniform(0.01, 0.12))
        key = f"{prefix}{rng.integers(count // 2)}" if rng.random() < 0.15 else f"{prefix}_{i}"
        out.append(QDRecord(key=key, expression=key, abs_rank_ic=quality, nodes=3, stage_depth=1,
                            signature=signature))
    return out


@pytest.mark.parametrize("block", [1, 16, 4096])
def test_qd_admission_matches_python(native_lib, evolution_panels, block):
    from gpp.backend import NativeBackend

    cfg = config(pca_qd_block_size=block)
    backend = NativeBackend(evolution_panels[0], cfg)
    rng = np.random.default_rng(block)
    dim = 30
    centers = rng.normal(size=(25, dim))
    python, native = PCAQDArchive(0.90, 0.002), PCAQDArchive(0.90, 0.002)
    for batch in range(4):
        records = _records(rng, 300, dim, f"b{batch % 2}", centers)
        expected = python.consider_batch(records, block_size=block, backend="cpu")
        got = backend.consider(native, records, cfg, pca_seed=0)
        assert [d.action for d in got] == [d.action for d in expected]
        assert [d.closest_key for d in got] == [d.closest_key for d in expected]
        assert [d.replaced_key for d in got] == [d.replaced_key for d in expected]
        np.testing.assert_allclose([d.max_similarity for d in got], [d.max_similarity for d in expected],
                                   atol=1e-12, equal_nan=True)
        assert list(native._records) == list(python._records)
        assert list(native) == list(python)
    # The Python archive is edited elsewhere (e.g. after a resume): resync.
    del python._records[next(iter(python._records))]
    del native._records[next(iter(native._records))]
    records = _records(rng, 200, dim, "c", centers)
    expected = python.consider_batch(records, block_size=block, backend="cpu")
    got = backend.consider(native, records, cfg, pca_seed=0)
    assert [(d.action, d.closest_key, d.replaced_key) for d in got] == [
        (d.action, d.closest_key, d.replaced_key) for d in expected]
    backend.close()


# -- 2022 out-of-sample screening ---------------------------------------------------

def _screening_candidates(frames, target, count=160):
    """Candidates with plausible evolution RankIC signs (some flipped)."""
    rng = random.Random(4)
    out = []
    for i, expression in enumerate(hierarchical_expressions(count // 5, seed=21)):
        values = evaluate_expression(normalize_expression(expression), frames)
        ic = rank_ic(values, target).mean
        if not np.isfinite(ic):
            ic = 0.05
        sign = -1 if rng.random() < 0.15 else 1
        out.append(Candidate(expression=normalize_expression(expression), rank_ic=sign * float(ic) * 1.1,
                             abs_rank_ic=abs(float(ic)), valid_ratio=1.0, missing_ratio=0.0, depth=1,
                             nodes=3 + i % 7, fitness=abs(float(ic)), segment_rank_ics=()))
    out.append(out[3])  # duplicate key: skipped like the Python loop
    out.append(replace(out[0], expression=("ts_mean", "no_such_field", 5)))  # evaluation error row
    return out


@pytest.mark.parametrize("step,threshold,limit", [(20, 0.02, None), (5, 0.02, None), (20, 0.0, 7), (5, 0.07, None)])
def test_screening_matches_python(native_lib, screening_panels, tmp_path, float32_reference, step, threshold,
                                  limit):
    """validation_rankic.csv and correlation_decisions.csv row by row."""
    from gpp.screening import screen_hierarchical_factors_native

    panel_dir, frames, target = screening_panels
    candidates = _screening_candidates(frames, target)
    expected = screen_hierarchical_factors(candidates, frames, target, tmp_path / "py", rankic_threshold=threshold,
                                           corr_threshold=0.75, sample_step=step, limit=limit)
    got = screen_hierarchical_factors_native(candidates, panel_dir, tmp_path / "cpp", rankic_threshold=threshold,
                                             corr_threshold=0.75, sample_step=step, limit=limit,
                                             engine_options=dict(threads=2, block=8))
    assert [f.name for f in got] == [f.name for f in expected]
    assert [expression_key(f.expression) for f in got] == [expression_key(f.expression) for f in expected]
    for a, b in zip(got, expected):
        assert a.screening_rank_ic == pytest.approx(b.screening_rank_ic, abs=1e-10)
    for name in ("validation_rankic.csv", "correlation_decisions.csv"):
        left = pd.read_csv(tmp_path / "cpp" / name, keep_default_na=False, na_values=[""])
        right = pd.read_csv(tmp_path / "py" / name, keep_default_na=False, na_values=[""])
        assert list(left.columns) == list(right.columns)
        assert len(left) == len(right)
        exact = [c for c in left.columns if not pd.api.types.is_numeric_dtype(right[c]) or c in (
            "same_direction", "valid_times", "selected", "downsample_step")]
        for column in exact:
            if column == "error":  # message text differs; presence must not
                assert (left[column].fillna("") == "").tolist() == (right[column].fillna("") == "").tolist()
                continue
            assert left[column].fillna("").tolist() == right[column].fillna("").tolist(), (name, column)
        for column in set(left.columns) - set(exact):
            np.testing.assert_allclose(left[column].to_numpy(float), right[column].to_numpy(float), atol=1e-9,
                                       equal_nan=True, err_msg=f"{name}:{column}")
    decisions = pd.read_csv(tmp_path / "py" / "correlation_decisions.csv")
    if threshold < 0.07:  # the synthetic signals are weak; 0.07 checks the empty path
        assert len(got) > 0
        if limit is None:
            assert (decisions["selected"] == 0).sum() > 0  # de-duplication exercised


# -- epsilon-lexicase ------------------------------------------------------------

def test_lexicase_identical_draws():
    rng = np.random.default_rng(0)
    scored = []
    for i in range(300):
        ics = tuple(float(x) if rng.random() > 0.05 else float("nan") for x in rng.normal(0.03, 0.02, 4))
        rank = float(rng.normal(0.03, 0.02))
        scored.append((Candidate(expression=f"e{i}", rank_ic=rank, abs_rank_ic=abs(rank), valid_ratio=1.0,
                                 missing_ratio=0.0, depth=1, nodes=1, fitness=float(rng.random()),
                                 segment_rank_ics=ics if i % 13 else ()), None))
    assert _check_equivalence(scored, seeds=range(200))
    assert _check_equivalence(scored, seeds=range(50), epsilon_scale=0.5)
    assert LexicaseSelector([(c, None) for c, _ in scored if not c.segment_rank_ics]).survivors == []


# -- factor values ---------------------------------------------------------------

def test_factor_values_match_python(native_lib, evolution_panels, float32_reference, tmp_path):
    """Root panels of the C++ operators (gpp_values) vs the Python operators."""
    import json
    import subprocess

    binary = native_lib.library_path().parent / "gpp_values"
    if not binary.exists():
        pytest.skip("gpp_values not built")
    panel_dir, frames, _ = evolution_panels
    expressions = [normalize_expression(e) for e in hierarchical_expressions(30, seed=17)]
    (tmp_path / "e.json").write_text(json.dumps(expressions))
    (tmp_path / "out").mkdir()
    subprocess.run([str(binary), str(panel_dir), str(tmp_path / "e.json"), str(tmp_path / "out")], check=True)
    exact = compared = 0
    for i, expression in enumerate(expressions):
        path = tmp_path / "out" / f"{i}.f32"
        if not path.exists():
            continue
        native = np.fromfile(path, dtype="<f4").reshape(frames["close"].shape).astype(np.float64)
        python = evaluate_expression(expression, frames).to_numpy(np.float64)
        compared += 1
        assert (np.isnan(native) == np.isnan(python)).all(), expression
        both = ~np.isnan(native)
        scale = np.maximum(1.0, np.abs(python[both]))
        gap = np.abs(native[both] - python[both]) / scale
        if gap.size == 0 or gap.max() == 0.0:
            exact += 1
        else:
            # A few float32 ulps (zscore/std accumulate differently); a later
            # pct_change or division by a near-zero value can amplify them
            # locally, and beta keeps its covariance residue (test_evolution).
            assert np.mean(gap <= 1e-5) >= 0.99, (expression, float(np.mean(gap <= 1e-5)))
    assert compared > 100 and exact >= 0.5 * compared
