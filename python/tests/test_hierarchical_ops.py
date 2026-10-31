import random
from dataclasses import dataclass

import pytest

from gp_factor_miner.expressions import (
    expression_depth,
    expression_key,
    expression_nodes,
)
from gp_factor_miner.gp_engine import BINARY, PAIR_ROLLING, ROLLING, UNARY, terminal_names
from gp_factor_miner.hierarchical_ops import (
    balanced_stage_proposals,
    bounded_mutation,
    expression_operators,
    family_survivors,
    same_depth_crossover,
)


@dataclass(frozen=True)
class _ScoredCandidate:
    expression: object
    fitness: float
    nodes: int = 1


def candidate(expression, fitness, nodes=1):
    return _ScoredCandidate(expression, fitness, nodes)


def _contains_subtree(expression, target):
    if expression == target:
        return True
    return isinstance(expression, tuple) and any(
        _contains_subtree(child, target) for child in expression[1:]
    )


def test_depth_one_proposals_cover_every_feature_and_operator():
    features = ["f0", "f1", "f2"]
    operators = set(UNARY + BINARY + ROLLING + PAIR_ROLLING)

    proposals = balanced_stage_proposals(
        features, (1, 5), stage_depth=1, size=200, seed=7, gene_pools=()
    )

    assert set().union(*(terminal_names(item) for item in proposals)) == set(features)
    assert operators <= set().union(*(expression_operators(item) for item in proposals))
    assert all(expression_depth(item) <= 1 for item in proposals)


def test_deeper_proposals_reuse_prior_gene_pool_without_losing_raw_terminals():
    pool = (("ema", "f0", 5), ("delta", "f1", 1))

    proposals = balanced_stage_proposals(
        ["f0", "f1", "f2"],
        (1, 5),
        stage_depth=2,
        size=100,
        seed=9,
        gene_pools=(pool,),
    )

    assert all(expression_depth(item) <= 2 for item in proposals)
    assert any("f2" in terminal_names(item) for item in proposals)
    assert any(
        any(gene == child for child in item[1:] if isinstance(child, tuple))
        for item in proposals
        for gene in pool
    )


def test_depth_one_exhaustion_returns_only_unique_candidates():
    proposals = balanced_stage_proposals(
        ["f0", "f1", "f2"], (1, 5), stage_depth=1, size=200, seed=7
    )

    assert len(proposals) < 200
    assert len(proposals) == len({expression_key(item) for item in proposals})


def test_depth_one_adequate_search_space_reaches_requested_size():
    proposals = balanced_stage_proposals(
        [f"f{index}" for index in range(10)],
        (1, 5, 10, 20, 60),
        stage_depth=1,
        size=200,
        seed=7,
    )

    assert len(proposals) == 200
    assert len(proposals) == len({expression_key(item) for item in proposals})


@pytest.mark.parametrize("stage_depth", (0, 4))
def test_stage_depth_outside_hierarchical_stages_is_rejected(stage_depth):
    with pytest.raises(ValueError, match="stage_depth"):
        balanced_stage_proposals(["f0"], (1,), stage_depth, size=1, seed=1)


@pytest.mark.parametrize(
    "malformed_gene",
    (("bogus", "f0"), ("+", "f0"), ("rank",)),
)
def test_malformed_promoted_genes_are_skipped(malformed_gene):
    proposals = balanced_stage_proposals(
        ["f0", "f1"],
        (1,),
        stage_depth=2,
        size=100,
        seed=1,
        gene_pools=((malformed_gene,),),
    )

    assert not any(_contains_subtree(item, malformed_gene) for item in proposals)


def test_stage_two_completion_reaches_requested_unique_size():
    proposals = balanced_stage_proposals(
        ["f0", "f1"],
        (1,),
        stage_depth=2,
        size=40,
        seed=7,
        gene_pools=((("ema", "f0", 1),),),
    )

    assert len(proposals) == 40
    assert len(proposals) == len({expression_key(item) for item in proposals})


def test_stage_three_completion_reaches_requested_unique_size():
    proposals = balanced_stage_proposals(
        ["f0", "f1"],
        (1,),
        stage_depth=3,
        size=40,
        seed=7,
        gene_pools=((("ema", ("delta", "f0", 1), 1),),),
    )

    assert len(proposals) == 40
    assert len(proposals) == len({expression_key(item) for item in proposals})


def test_promoted_gene_with_unknown_terminal_is_skipped():
    unknown_gene = ("ema", "unknown", 1)
    proposals = balanced_stage_proposals(
        ["f0", "f1"],
        (1,),
        stage_depth=2,
        size=100,
        seed=1,
        gene_pools=((unknown_gene,),),
    )

    assert not any(_contains_subtree(item, unknown_gene) for item in proposals)
    assert all(terminal_names(item) <= {"f0", "f1"} for item in proposals)


def test_valid_list_form_promoted_gene_is_normalized_and_reused():
    list_gene = ["ema", ["delta", "f0", 1], 1]
    normalized_gene = ("ema", ("delta", "f0", 1), 1)
    proposals = balanced_stage_proposals(
        ["f0", "f1"],
        (1,),
        stage_depth=3,
        size=100,
        seed=1,
        gene_pools=((list_gene,),),
    )

    assert any(_contains_subtree(item, normalized_gene) for item in proposals)


def test_same_depth_crossover_never_exceeds_stage():
    left = ("+", ("ema", "f0", 5), "f1")
    right = ("*", ("delta", "f2", 1), "f3")

    for seed in range(20):
        children = same_depth_crossover(
            left, right, random.Random(seed), stage_depth=2
        )

        assert all(expression_depth(child) <= 2 for child in children)


def test_same_depth_crossover_swaps_nodes_from_equal_levels():
    left = ("+", ("ema", "f0", 5), "f1")
    right = ("*", ("delta", "f2", 1), "f3")

    children = same_depth_crossover(
        left, right, random.Random(4), stage_depth=2
    )

    assert children != (left, right)


def test_same_depth_crossover_respects_node_bound():
    left = ("+", ("rank", ("abs", "f4")), "f0")
    right = ("*", ("rank", ("abs", "f2")), "f3")

    children = same_depth_crossover(
        left, right, random.Random(1), stage_depth=3, max_nodes=5
    )

    assert all(expression_nodes(child) <= 5 for child in children)


def test_same_depth_crossover_falls_back_only_over_node_sibling():
    left = ("+", ("rank", ("abs", "f4")), "f0")
    right = ("*", ("rank", ("abs", "f2")), "f3")

    children = same_depth_crossover(
        left, right, random.Random(1), stage_depth=3, max_nodes=5
    )

    assert children == (("+", "f0", "f3"), right)


def test_same_depth_crossover_exchanges_subtrees_from_equal_levels():
    left = ("+", ("rank", "f0"), "f1")
    right = ("*", "f2", ("abs", "f3"))

    children = same_depth_crossover(
        left, right, random.Random(7), stage_depth=3
    )

    assert children == (
        ("+", "f1", ("abs", "f3")),
        ("*", "f2", ("rank", "f0")),
    )


@pytest.mark.parametrize(
    ("mutation_probability", "expression"),
    (
        ("feature_mutation_probability", ("+", "f0", "f1")),
        ("operator_mutation_probability", ("+", "f0", "f1")),
        ("window_mutation_probability", ("ema", "f0", 5)),
        ("subtree_mutation_probability", ("ema", "f0", 5)),
    ),
)
def test_bounded_mutation_performs_each_selected_mutation(
    mutation_probability, expression
):
    probabilities = {
        "feature_mutation_probability": 0.0,
        "operator_mutation_probability": 0.0,
        "window_mutation_probability": 0.0,
        "subtree_mutation_probability": 0.0,
    }
    probabilities[mutation_probability] = 1.0

    mutated = bounded_mutation(
        expression,
        random.Random(7),
        features=("f0", "f1", "f2"),
        feature_probabilities=(1.0, 1.0, 1.0),
        windows=(1, 5, 10),
        stage_depth=2,
        max_nodes=11,
        **probabilities,
    )

    assert expression_key(mutated) != expression_key(expression)
    assert expression_depth(mutated) <= 2


def test_bounded_mutation_is_deterministic_for_seed():
    kwargs = dict(
        features=("f0", "f1", "f2"),
        feature_probabilities=(1.0, 2.0, 3.0),
        windows=(1, 5, 10),
        stage_depth=2,
        max_nodes=11,
        feature_mutation_probability=1.0,
        operator_mutation_probability=1.0,
        window_mutation_probability=1.0,
        subtree_mutation_probability=1.0,
    )

    first = bounded_mutation(("ema", "f0", 5), random.Random(11), **kwargs)
    second = bounded_mutation(("ema", "f0", 5), random.Random(11), **kwargs)

    assert first == second


def test_bounded_mutation_returns_parent_when_node_bound_is_exceeded():
    expression = ("+", "f0", "f1")

    mutated = bounded_mutation(
        expression,
        random.Random(7),
        features=("f0", "f1", "f2"),
        feature_probabilities=(1.0, 1.0, 1.0),
        windows=(1, 5),
        stage_depth=2,
        max_nodes=1,
        operator_mutation_probability=1.0,
        feature_mutation_probability=0.0,
        window_mutation_probability=0.0,
        subtree_mutation_probability=0.0,
    )

    assert mutated == expression


def test_subtree_mutation_resamples_invalid_candidate_for_valid_variation():
    expression = ("rank", ("abs", "f0"))

    mutated = bounded_mutation(
        expression,
        random.Random(3),
        features=("f0", "f1"),
        feature_probabilities=(1.0, 1.0),
        windows=(1, 5),
        stage_depth=3,
        max_nodes=11,
        feature_mutation_probability=0.0,
        operator_mutation_probability=0.0,
        window_mutation_probability=0.0,
        subtree_mutation_probability=1.0,
    )

    assert mutated == (
        "rank",
        ("-", ("correlation", "f1", "f0", 1), ("rank", "f0")),
    )


def test_family_retains_best_two_distinct_expressions():
    parents = [candidate("p1", 0.03), candidate("p2", 0.02)]
    children = [candidate("c1", 0.04), candidate("p1", 0.05)]

    survivors = family_survivors(parents, children, limit=2)

    assert [item.expression for item in survivors] == ["p1", "c1"]


def test_family_breaks_equal_fitness_ties_by_nodes_then_expression_key():
    parents = [candidate("z", 0.03, nodes=3), candidate("a", 0.03, nodes=3)]
    children = [candidate("m", 0.03, nodes=2)]

    survivors = family_survivors(parents, children, limit=2)

    assert [item.expression for item in survivors] == ["m", "a"]


def test_rejected_qd_child_leaves_parent_family_intact():
    parents = [candidate("p1", 0.03), candidate("p2", 0.02)]

    assert family_survivors(parents, [], limit=2) == tuple(parents)
