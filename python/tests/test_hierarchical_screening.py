from collections import Counter
from dataclasses import replace
import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from gp_factor_miner import hierarchical_screening
from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.hierarchical_engine import run_hierarchical_evolution
from gp_factor_miner.hierarchical_screening import screen_hierarchical_factors


def _candidate(name, rank_ic):
    return Candidate(
        expression=name,
        rank_ic=rank_ic,
        abs_rank_ic=abs(rank_ic),
        valid_ratio=1.0,
        missing_ratio=0.0,
        depth=0,
        nodes=1,
    )


def _target():
    return pd.DataFrame(
        [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
        index=pd.Index([0, 1], name="time_id"),
        columns=pd.Index([0, 1, 2], name="asset_id"),
    )


def test_screening_requires_rankic_threshold_and_same_direction(tmp_path):
    base = [1.0, 2.0, 3.0]
    reverse = [3.0, 2.0, 1.0]
    target = pd.DataFrame([base] * 80)
    too_low = pd.DataFrame([base] * 41 + [reverse] * 39)

    selected = screen_hierarchical_factors(
        candidates=[
            _candidate("good", 0.04),
            _candidate("wrong", -0.04),
            _candidate("too_low", 0.04),
            _candidate("good", 0.04),
        ],
        screening_data={"good": target, "wrong": target, "too_low": too_low},
        screening_target=target,
        output_dir=tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    assert [item.expression for item in selected] == ["good"]
    validation = pd.read_csv(tmp_path / "validation_rankic.csv")
    decisions = pd.read_csv(tmp_path / "correlation_decisions.csv")
    assert len(validation) == 3
    assert validation.loc[
        validation["expression"].eq('"too_low"'), "abs_screening_rank_ic"
    ].item() < 0.03
    assert decisions["selected"].tolist() == [1]


def test_ranked_sample_is_computed_once_per_eligible_factor(tmp_path, monkeypatch):
    target = _target()
    calls = Counter()
    original = hierarchical_screening._sampled_rank_vector

    def counted(name, values, step):
        calls[name] += 1
        return original(name, values, step)

    monkeypatch.setattr(hierarchical_screening, "_sampled_rank_vector", counted)

    selected = screen_hierarchical_factors(
        candidates=[_candidate("a", 0.04), _candidate("b", 0.05)],
        screening_data={"a": target, "b": target * 2.0},
        screening_target=target,
        output_dir=tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    assert calls == {"a": 1, "b": 1}
    assert [item.expression for item in selected] == ["b"]


def test_default_correlation_threshold_is_point_seven_five(tmp_path):
    target = pd.DataFrame(np.random.default_rng(3).normal(size=(80, 12)))
    data = {
        "a": target
        + np.random.default_rng(103).normal(scale=0.35, size=target.shape),
        "b": target
        + np.random.default_rng(203).normal(scale=0.35, size=target.shape),
    }

    selected = screen_hierarchical_factors(
        candidates=[_candidate("a", 0.04), _candidate("b", 0.05)],
        screening_data=data,
        screening_target=target,
        output_dir=tmp_path,
        rankic_threshold=0.03,
        sample_step=1,
    )

    assert [item.expression for item in selected] == ["b"]
    decisions = pd.read_csv(tmp_path / "correlation_decisions.csv")
    assert decisions["correlation_threshold"].unique().tolist() == [0.75]


def test_default_screening_keeps_all_distinct_factors_above_point_zero_seven(
    tmp_path,
):
    rng = np.random.default_rng(11)
    target = pd.DataFrame(rng.normal(size=(80, 40)))
    data = {
        f"factor_{index}": target + rng.normal(scale=4.0, size=target.shape)
        for index in range(5)
    }
    candidates = [_candidate(name, 0.10) for name in data]
    candidates.append(_candidate("weak", 0.10))
    data["weak"] = pd.DataFrame(rng.normal(size=target.shape))

    selected = screen_hierarchical_factors(
        candidates=candidates,
        screening_data=data,
        screening_target=target,
        output_dir=tmp_path,
        sample_step=1,
    )

    validation = pd.read_csv(tmp_path / "validation_rankic.csv")
    assert (validation["abs_screening_rank_ic"] >= 0.07).sum() == 5
    assert sorted(item.expression for item in selected) == [
        f"factor_{index}" for index in range(5)
    ]


def test_screening_reports_incremental_validation_and_correlation_progress(
    tmp_path,
):
    target = _target()
    candidates = [_candidate(f"factor_{index:03d}", 0.04) for index in range(205)]
    candidates.append(candidates[0])
    data = {candidate.expression: target for candidate in candidates}
    events = []

    screen_hierarchical_factors(
        candidates,
        data,
        target,
        tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
        progress_callback=events.append,
    )

    validation_progress = [
        event for event in events if event["event"] == "validation_progress"
    ]
    correlation_progress = [
        event for event in events if event["event"] == "correlation_progress"
    ]
    assert [event["completed"] for event in validation_progress] == [100, 200]
    assert [event["total"] for event in validation_progress] == [205, 205]
    assert [
        event for event in events if event["event"] == "validation_completed"
    ] == [{"event": "validation_completed", "completed": 205, "total": 205}]
    assert len(correlation_progress) == 205
    assert correlation_progress[0]["completed"] == 1
    assert correlation_progress[-1] == {
        "event": "correlation_progress",
        "completed": 205,
        "total": 205,
        "selected": 1,
    }
    assert [
        event for event in events if event["event"] == "correlation_completed"
    ] == [
        {
            "event": "correlation_completed",
            "completed": 205,
            "total": 205,
            "selected": 1,
        }
    ]


def test_correlation_decision_reports_true_maximum_and_deterministic_tie(
    tmp_path, monkeypatch
):
    target = _target()
    vector_codes = {
        "first": 0.0,
        "second": 1.0,
        "probe": 2.0,
        "tie": 3.0,
        "zero": 4.0,
    }
    correlations = {
        (1.0, 0.0): 0.20,
        (2.0, 0.0): 0.86,
        (2.0, 1.0): 0.95,
        (3.0, 0.0): 0.90,
        (3.0, 1.0): 0.90,
        (4.0, 0.0): 0.0,
        (4.0, 1.0): 0.0,
    }

    monkeypatch.setattr(
        hierarchical_screening,
        "_sampled_rank_vector",
        lambda name, values, step: np.array([vector_codes[name]], dtype=np.float32),
    )
    monkeypatch.setattr(
        hierarchical_screening,
        "_absolute_correlation",
        lambda left, right: correlations[(float(left[0]), float(right[0]))],
    )

    selected = screen_hierarchical_factors(
        candidates=[
            _candidate("first", 0.07),
            _candidate("second", 0.06),
            _candidate("probe", 0.05),
            _candidate("tie", 0.04),
            _candidate("zero", 0.03),
        ],
        screening_data={name: target for name in vector_codes},
        screening_target=target,
        output_dir=tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    assert [item.expression for item in selected] == ["first", "second", "zero"]
    decisions = pd.read_csv(tmp_path / "correlation_decisions.csv").set_index(
        "expression"
    )
    assert decisions.loc['"probe"', "max_abs_correlation"] == pytest.approx(0.95)
    assert decisions.loc['"probe"', "correlated_with"] == '"second"'
    assert decisions.loc['"tie"', "max_abs_correlation"] == pytest.approx(0.90)
    assert decisions.loc['"tie"', "correlated_with"] == '"first"'
    assert decisions.loc['"zero"', "max_abs_correlation"] == pytest.approx(0.0)
    assert decisions.loc['"zero"', "correlated_with"] == '"first"'


def test_validation_rankic_exactly_at_threshold_is_admitted(tmp_path, monkeypatch):
    target = _target()
    monkeypatch.setattr(
        hierarchical_screening,
        "rank_ic",
        lambda values, screening_target: hierarchical_screening.RankICResult(
            mean=0.03,
            valid_times=2,
            valid_ratio=1.0,
            missing_ratio=0.0,
        ),
    )

    selected = screen_hierarchical_factors(
        [_candidate("boundary", 0.04)],
        {"boundary": target},
        target,
        tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    assert [item.expression for item in selected] == ["boundary"]


def test_correlation_exactly_at_threshold_is_rejected(tmp_path, monkeypatch):
    target = _target()
    monkeypatch.setattr(
        hierarchical_screening,
        "_absolute_correlation",
        lambda left, right: 0.85,
    )

    selected = screen_hierarchical_factors(
        [_candidate("first", 0.05), _candidate("boundary", 0.04)],
        {"first": target, "boundary": target},
        target,
        tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    assert [item.expression for item in selected] == ["first"]
    decisions = pd.read_csv(tmp_path / "correlation_decisions.csv")
    assert decisions.iloc[-1]["max_abs_correlation"] == pytest.approx(0.85)
    assert decisions.iloc[-1]["selected"] == 0


def test_releases_each_full_panel_before_evaluating_the_next(tmp_path, monkeypatch):
    target = _target()
    panel_references = []

    def tracked_evaluation(expression, data):
        del expression, data
        gc.collect()
        assert not panel_references or panel_references[-1]() is None
        panel = target.copy()
        panel_references.append(weakref.ref(panel))
        return panel

    monkeypatch.setattr(
        hierarchical_screening, "evaluate_expression", tracked_evaluation
    )

    screen_hierarchical_factors(
        candidates=[_candidate("a", 0.05), _candidate("b", 0.04)],
        screening_data={},
        screening_target=target,
        output_dir=tmp_path,
        rankic_threshold=0.03,
        corr_threshold=0.85,
        sample_step=1,
        limit=300,
    )

    gc.collect()
    assert panel_references[-1]() is None


def test_validation_results_are_persisted_every_hundred_candidates(tmp_path):
    target = _target()
    screening_data = {f"signal_{index}": target for index in range(100)}

    def interrupted_candidates():
        for index in range(100):
            yield _candidate(f"signal_{index}", 0.04)
        raise RuntimeError("candidate source interrupted")

    with pytest.raises(RuntimeError, match="candidate source interrupted"):
        screen_hierarchical_factors(
            candidates=interrupted_candidates(),
            screening_data=screening_data,
            screening_target=target,
            output_dir=tmp_path,
            rankic_threshold=0.03,
            corr_threshold=0.85,
            sample_step=1,
            limit=300,
        )

    persisted = pd.read_csv(tmp_path / "validation_rankic.csv")
    assert len(persisted) == 100
    assert not (tmp_path / "validation_rankic.csv.tmp").exists()


def test_final_screening_runs_only_after_all_evolution_stages(tmp_path):
    rng = np.random.default_rng(15)
    index = pd.Index(range(61), name="time_id")
    columns = pd.Index(range(5), name="asset_id")
    data = {
        f"f{number}": pd.DataFrame(
            rng.normal(size=(len(index), len(columns))),
            index=index,
            columns=columns,
        )
        for number in range(4)
    }
    target = data["f0"] - 0.25 * data["f1"]
    profile = pd.DataFrame(
        {"feature": list(data), "rank_ic": [0.10, 0.05, 0.03, 0.01]}
    )
    config = replace(
        HierarchicalGPConfig().smoke(),
        population_size=4,
        generations_per_stage=0,
        warm_start_multiplier=1,
        gene_pool_size=4,
        parent_pool_size=4,
        progress_chunk_size=4,
        windows=(1, 2),
        pca_sample_step=10,
    )

    result = run_hierarchical_evolution(
        data,
        target,
        profile,
        config,
        seed=3,
        checkpoint_dir=tmp_path,
    )

    assert [stage.depth for stage in result.stages] == [1, 2, 3]
    assert not (tmp_path / "validation_rankic.csv").exists()
    assert not (tmp_path / "correlation_decisions.csv").exists()

    candidates = [
        candidate
        for stage in result.stages
        for candidate in stage.gene_pool
    ]
    screen_hierarchical_factors(
        candidates,
        data,
        target,
        tmp_path,
        rankic_threshold=config.final_rankic_threshold,
        corr_threshold=config.final_correlation_threshold,
        sample_step=config.pca_sample_step,
        limit=config.final_factor_count,
    )

    assert (tmp_path / "validation_rankic.csv").exists()
    assert (tmp_path / "correlation_decisions.csv").exists()
