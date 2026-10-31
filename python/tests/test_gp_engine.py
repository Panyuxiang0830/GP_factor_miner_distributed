import pandas as pd
import pytest
from dataclasses import replace

import gp_factor_miner.gp_engine as gp_engine
from gp_factor_miner.config import GPConfig
from gp_factor_miner.gp_engine import (
    _fitness_from_segments,
    build_initial_population,
    run_evolution,
    terminal_names,
)


def test_initial_population_covers_every_terminal():
    population = build_initial_population(
        ["a", "b", "c"],
        probabilities=[0.8, 0.1, 0.1],
        windows=(2, 4),
        size=20,
        init_min_depth=1,
        init_max_depth=2,
        seed=7,
    )

    leaves = set().union(*(terminal_names(individual) for individual in population))

    assert leaves == {"a", "b", "c"}


def test_smoke_population_can_sample_from_more_terminals_than_individuals():
    features = [f"f{i}" for i in range(60)]
    population = build_initial_population(
        features,
        probabilities=[1 / 60] * 60,
        windows=(2, 4),
        size=20,
        init_min_depth=1,
        init_max_depth=2,
        seed=7,
    )

    assert len(population) == 20


def _synthetic_problem():
    index = pd.Index(range(8), name="time_id")
    columns = pd.Index([0, 1, 2], name="asset_id")
    signal = pd.DataFrame(
        [[1, 2, 3], [3, 2, 1]] * 4,
        index=index,
        columns=columns,
        dtype=float,
    )
    noise = pd.DataFrame(
        [[2, 1, 3], [1, 3, 2]] * 4,
        index=index,
        columns=columns,
        dtype=float,
    )
    profile = pd.DataFrame(
        {"feature": ["signal", "noise"], "rank_ic": [1.0, 0.0]}
    )
    config = GPConfig(
        population_size=20,
        generations=2,
        independent_runs=1,
        init_min_depth=0,
        init_max_depth=2,
        max_depth=3,
        hall_of_fame_size=5,
        windows=(2, 4),
        max_workers=1,
    )
    return {"signal": signal, "noise": noise}, signal, profile, config


def test_same_seed_returns_same_hall_of_fame():
    problem = _synthetic_problem()

    first = run_evolution(*problem, seed=2026)
    second = run_evolution(*problem, seed=2026)

    assert first.hall_of_fame == second.hall_of_fame
    assert first.history == second.history


def test_evolution_retains_known_signal():
    result = run_evolution(*_synthetic_problem(), seed=2026)

    assert result.hall_of_fame[0].abs_rank_ic == pytest.approx(1.0)
    assert "signal" in terminal_names(result.hall_of_fame[0].expression)
    assert result.evaluated_count > 0


def test_parallel_evaluation_uses_multiple_worker_processes():
    data, target, profile, config = _synthetic_problem()
    parallel = replace(config, generations=1, max_workers=2)

    result = run_evolution(data, target, profile, parallel, seed=2026)

    assert len(result.worker_pids) == 2


def test_combined_worker_reuses_values_for_candidate_and_signature(monkeypatch):
    data, target, _, config = _synthetic_problem()
    gp_engine._initialize_worker(data, target, config)
    original = gp_engine.evaluate_expression
    calls = 0

    def counted(expression, panels):
        nonlocal calls
        calls += 1
        return original(expression, panels)

    monkeypatch.setattr(gp_engine, "evaluate_expression", counted)

    candidate, signature, _ = gp_engine._worker_evaluate_with_signature(
        ("signal", 2, 4, 7)
    )

    assert candidate is not None
    assert signature is not None
    assert signature.size == 4
    assert calls == 1


def test_fitness_penalizes_segment_instability_and_complexity():
    fitness, segment_std, same_sign_ratio = _fitness_from_segments(
        rank_ic_mean=0.04,
        segment_means=(0.02, 0.04, 0.06, 0.04),
        nodes=7,
        stability_penalty=0.5,
        complexity_penalty=0.0005,
        minimum_same_sign_ratio=0.75,
        inconsistent_sign_multiplier=0.25,
    )

    expected_std = pd.Series([0.02, 0.04, 0.06, 0.04]).std(ddof=0)
    assert segment_std == pytest.approx(expected_std)
    assert same_sign_ratio == pytest.approx(1.0)
    assert fitness == pytest.approx(0.04 - 0.5 * expected_std - 0.0005 * 4)


def test_fitness_strongly_penalizes_inconsistent_segment_signs():
    fitness, _, same_sign_ratio = _fitness_from_segments(
        rank_ic_mean=0.04,
        segment_means=(0.04, -0.01, -0.02, 0.03),
        nodes=3,
        stability_penalty=0.0,
        complexity_penalty=0.0,
        minimum_same_sign_ratio=0.75,
        inconsistent_sign_multiplier=0.25,
    )

    assert same_sign_ratio == pytest.approx(0.5)
    assert fitness == pytest.approx(0.01)


def test_evolution_reports_chunk_and_generation_progress():
    data, target, profile, config = _synthetic_problem()
    config = replace(
        config,
        generations=1,
        population_size=8,
        progress_chunk_size=3,
        max_workers=1,
    )
    events = []

    run_evolution(
        data, target, profile, config, seed=2026, progress_callback=events.append
    )

    assert events[0]["event"] == "generation_started"
    chunk_events = [event for event in events if event["event"] == "chunk_completed"]
    assert chunk_events
    assert chunk_events[-1]["completed"] == chunk_events[-1]["total"]
    assert chunk_events[-1]["chunk"] == chunk_events[-1]["chunks"]
    assert events[-1]["event"] == "generation_completed"
    assert "best_fitness" in events[-1]


def test_checkpoint_writes_one_csv_per_population_chunk(tmp_path):
    data, target, profile, config = _synthetic_problem()
    config = replace(
        config,
        generations=2,
        population_size=8,
        progress_chunk_size=3,
        max_workers=1,
    )

    run_evolution(
        data,
        target,
        profile,
        config,
        seed=2026,
        checkpoint_dir=tmp_path / "eval" / "run_01",
    )

    generations = sorted((tmp_path / "eval" / "run_01").glob("generation_*"))
    assert [path.name for path in generations] == ["generation_01", "generation_02"]
    assert len(list(generations[0].glob("chunk_*.csv"))) == 3
    assert len(list(generations[1].glob("chunk_*.csv"))) == 3
    assert (tmp_path / "eval" / "run_01" / "completed.pkl").exists()


def test_completed_checkpoint_resumes_without_evaluation(tmp_path, monkeypatch):
    data, target, profile, config = _synthetic_problem()
    config = replace(
        config,
        generations=1,
        population_size=8,
        progress_chunk_size=3,
        max_workers=1,
    )
    checkpoint_dir = tmp_path / "eval" / "run_01"
    expected = run_evolution(
        data, target, profile, config, seed=2026, checkpoint_dir=checkpoint_dir
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed run must not evaluate again")

    monkeypatch.setattr("gp_factor_miner.gp_engine._evaluate", fail_if_called)
    resumed = run_evolution(
        data, target, profile, config, seed=2026, checkpoint_dir=checkpoint_dir
    )

    assert resumed == expected


def test_incomplete_generation_resumes_after_existing_chunk(tmp_path):
    data, target, profile, config = _synthetic_problem()
    config = replace(
        config,
        generations=1,
        population_size=8,
        progress_chunk_size=3,
        max_workers=1,
    )
    checkpoint_dir = tmp_path / "eval" / "run_01"

    def interrupt_after_first_chunk(event):
        if event["event"] == "chunk_completed":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_evolution(
            data,
            target,
            profile,
            config,
            seed=2026,
            progress_callback=interrupt_after_first_chunk,
            checkpoint_dir=checkpoint_dir,
        )

    generation_dir = checkpoint_dir / "generation_01"
    assert [path.name for path in generation_dir.glob("chunk_*.csv")] == [
        "chunk_01.csv"
    ]

    resumed_events = []
    result = run_evolution(
        data,
        target,
        profile,
        config,
        seed=2026,
        progress_callback=resumed_events.append,
        checkpoint_dir=checkpoint_dir,
    )

    assert len(result.history) == 1
    assert len(list(generation_dir.glob("chunk_*.csv"))) == 3
    assert [
        event["chunk"]
        for event in resumed_events
        if event["event"] == "chunk_completed"
    ] == [2, 3]
