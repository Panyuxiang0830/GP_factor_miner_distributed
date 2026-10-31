import random

import gp_factor_miner.gp_engine as engine

from gp_factor_miner.gp_engine import (
    Candidate,
    _build_capped_parent_pool,
    _epsilon_lexicase,
    _mutate_feature,
    _mutate_operator,
    _mutate_window,
    _random_expression,
    BINARY,
)


def _candidate(expression, fitness, segments=(0.03, 0.03, 0.03, 0.03)):
    return Candidate(
        expression=expression,
        rank_ic=0.03,
        abs_rank_ic=0.03,
        valid_ratio=1.0,
        missing_ratio=0.0,
        depth=1,
        nodes=3,
        fitness=fitness,
        segment_rank_ic_std=0.0,
        same_sign_ratio=1.0,
        segment_rank_ics=segments,
    )


def test_epsilon_lexicase_preserves_different_segment_specialists():
    first = _candidate("first", 0.03, (0.08, 0.01, 0.01, 0.01))
    second = _candidate("second", 0.03, (0.01, 0.08, 0.01, 0.01))
    average = _candidate("average", 0.03, (0.04, 0.04, 0.00, 0.00))
    scored = [(first, first.fitness), (second, second.fitness), (average, average.fitness)]

    winners = {
        _epsilon_lexicase(scored, random.Random(seed), epsilon_scale=0.0)
        for seed in range(20)
    }

    assert winners == {"first", "second"}


def test_parent_pool_caps_hot_feature_presence():
    scored = [
        (_candidate(("+", "hot", f"x{i}"), 1.0 - i / 100), 1.0 - i / 100)
        for i in range(6)
    ] + [
        (_candidate(("+", f"a{i}", f"b{i}"), 0.5 - i / 100), 0.5 - i / 100)
        for i in range(6)
    ]

    pool, used_share = _build_capped_parent_pool(
        scored, target_size=8, cap_shares=(0.25,)
    )

    assert len(pool) == 8
    assert sum("hot" in str(candidate.expression) for candidate, _ in pool) == 2
    assert used_share == 0.25


def test_feature_mutation_replaces_one_terminal():
    expression = ("+", "feature_001", "feature_002")

    mutated = _mutate_feature(
        expression,
        random.Random(3),
        ["feature_001", "feature_002", "feature_003"],
        [1, 1, 1],
    )

    assert mutated != expression
    assert "feature_003" in str(mutated)


def test_operator_mutation_preserves_arity():
    expression = ("+", "feature_001", "feature_002")

    mutated = _mutate_operator(expression, random.Random(2))

    assert mutated[0] in {"-", "*", "/"}
    assert mutated[1:] == expression[1:]


def test_window_mutation_changes_only_window():
    expression = ("lag", "feature_001", 5)

    mutated = _mutate_window(expression, random.Random(2), (1, 5, 10))

    assert mutated[0:2] == expression[0:2]
    assert mutated[2] in {1, 10}


def test_soft_island_can_force_preferred_operator_family():
    expression = _random_expression(
        random.Random(7),
        ["feature_001", "feature_002"],
        [0.5, 0.5],
        (5, 10),
        depth=1,
        min_depth=1,
        preferred_operators=BINARY,
        preferred_probability=1.0,
    )

    assert expression[0] in BINARY


def test_random_normalized_expression_retries_degenerate_tree(monkeypatch):
    generated = iter([("/", "feature_001", "feature_001"), "feature_002"])
    monkeypatch.setattr(engine, "_random_expression", lambda *args, **kwargs: next(generated))

    expression = engine._random_normalized_expression(
        random.Random(7),
        ["feature_001", "feature_002"],
        [0.5, 0.5],
        (5, 10),
        depth=1,
    )

    assert expression == "feature_002"
