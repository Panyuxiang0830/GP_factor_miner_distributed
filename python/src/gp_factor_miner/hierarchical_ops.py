from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from .expressions import (
    expression_depth,
    expression_key,
    expression_nodes,
    normalize_expression,
)
from .gp_engine import (
    BINARY,
    PAIR_ROLLING,
    ROLLING,
    UNARY,
    _mutate,
    _mutate_feature,
    _mutate_operator,
    _mutate_window,
)


OPERATORS = UNARY + BINARY + ROLLING + PAIR_ROLLING
STAGE_DEPTHS = frozenset((1, 2, 3))


def _validate_stage_depth(stage_depth: int) -> None:
    if stage_depth not in STAGE_DEPTHS:
        raise ValueError("stage_depth must be one of 1, 2, or 3")


def _path_levels(expression: Any, path=(), level=0):
    result = [(path, level)]
    if isinstance(expression, tuple):
        for index, child in enumerate(expression[1:], start=1):
            if not isinstance(child, (int, float)):
                result.extend(
                    _path_levels(child, path + (index,), level + 1)
                )
    return result


def _at_path(expression: Any, path: tuple[int, ...]) -> Any:
    for index in path:
        expression = expression[index]
    return expression


def _replace_path(expression: Any, path: tuple[int, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    values = list(expression)
    values[path[0]] = _replace_path(values[path[0]], path[1:], replacement)
    return tuple(values)


def _bounded_child(
    child: Any, parent: Any, stage_depth: int, max_nodes: int
) -> Any:
    try:
        normalized = normalize_expression(child)
    except (TypeError, ValueError):
        return parent
    if (
        expression_depth(normalized) > stage_depth
        or expression_nodes(normalized) > max_nodes
    ):
        return parent
    return normalized


def same_depth_crossover(
    left: Any,
    right: Any,
    rng: random.Random,
    stage_depth: int,
    max_nodes: int = 11,
) -> tuple[Any, Any]:
    """Swap subtrees selected from one shared tree level."""
    _validate_stage_depth(stage_depth)
    left_levels = _path_levels(left)
    right_levels = _path_levels(right)
    common_levels = tuple(
        sorted(
            {level for _, level in left_levels}
            & {level for _, level in right_levels}
        )
    )
    if not common_levels:
        return left, right

    level = rng.choice(common_levels)
    left_path = rng.choice(
        [path for path, path_level in left_levels if path_level == level]
    )
    right_path = rng.choice(
        [path for path, path_level in right_levels if path_level == level]
    )
    left_subtree = _at_path(left, left_path)
    right_subtree = _at_path(right, right_path)
    left_child = _replace_path(left, left_path, right_subtree)
    right_child = _replace_path(right, right_path, left_subtree)
    return (
        _bounded_child(left_child, left, stage_depth, max_nodes),
        _bounded_child(right_child, right, stage_depth, max_nodes),
    )


def _actual_mutation(
    expression: Any,
    mutate,
    stage_depth: int,
    max_nodes: int,
    attempts: int = 32,
) -> Any:
    for _ in range(attempts):
        try:
            normalized = normalize_expression(mutate(expression))
        except (TypeError, ValueError):
            continue
        if (
            normalized != expression
            and expression_depth(normalized) <= stage_depth
            and expression_nodes(normalized) <= max_nodes
        ):
            return normalized
    return expression


def bounded_mutation(
    expression: Any,
    rng: random.Random,
    features,
    feature_probabilities,
    windows,
    stage_depth: int,
    max_nodes: int,
    feature_mutation_probability: float = 0.15,
    operator_mutation_probability: float = 0.10,
    window_mutation_probability: float = 0.10,
    subtree_mutation_probability: float = 0.10,
) -> Any:
    """Apply independently sampled V4 mutations within stage and node bounds."""
    _validate_stage_depth(stage_depth)
    features = tuple(features)
    feature_probabilities = tuple(feature_probabilities)
    windows = tuple(windows)
    try:
        mutated = normalize_expression(expression)
    except (TypeError, ValueError):
        return expression

    if rng.random() < feature_mutation_probability:
        mutated = _actual_mutation(
            mutated,
            lambda value: _mutate_feature(
                value, rng, features, feature_probabilities
            ),
            stage_depth,
            max_nodes,
        )
    if rng.random() < operator_mutation_probability:
        mutated = _actual_mutation(
            mutated,
            lambda value: _mutate_operator(value, rng),
            stage_depth,
            max_nodes,
        )
    if rng.random() < window_mutation_probability:
        mutated = _actual_mutation(
            mutated,
            lambda value: _mutate_window(value, rng, windows),
            stage_depth,
            max_nodes,
        )
    if rng.random() < subtree_mutation_probability:
        mutated = _actual_mutation(
            mutated,
            lambda value: _mutate(
                value,
                rng,
                features,
                feature_probabilities,
                windows,
                stage_depth,
            ),
            stage_depth,
            max_nodes,
        )

    if (
        expression_depth(mutated) > stage_depth
        or expression_nodes(mutated) > max_nodes
    ):
        return expression
    return mutated


def family_survivors(parents, children, limit: int = 2) -> tuple[Any, ...]:
    """Return the fittest distinct members of one parent-offspring family."""
    if limit < 0:
        raise ValueError("limit must not be negative")
    best_by_expression = {}
    for candidate in (*parents, *children):
        key = expression_key(candidate.expression)
        incumbent = best_by_expression.get(key)
        if incumbent is None or candidate.fitness > incumbent.fitness:
            best_by_expression[key] = candidate
    ranked = sorted(
        best_by_expression.values(),
        key=lambda candidate: (
            -candidate.fitness,
            candidate.nodes,
            expression_key(candidate.expression),
        ),
    )
    return tuple(ranked[:limit])


def expression_operators(expression: Any) -> set[str]:
    """Return every operator used by an expression tree."""
    if not isinstance(expression, (tuple, list)):
        return set()
    return {expression[0]} | set().union(
        *(expression_operators(child) for child in expression[1:])
    )


def _feature_cycle(rng: random.Random, features: tuple[str, ...]):
    while True:
        shuffled = list(features)
        rng.shuffle(shuffled)
        yield from shuffled


def _normalized_prior_genes(
    gene_pools: Iterable[Iterable[Any]], stage_depth: int, features: tuple[str, ...]
) -> tuple[Any, ...]:
    genes = []
    seen = set()
    for pool in gene_pools:
        for gene in pool:
            try:
                normalized = normalize_expression(gene)
                if not _valid_gene(normalized, features):
                    continue
            except Exception:
                continue
            if expression_depth(normalized) >= stage_depth:
                continue
            key = expression_key(normalized)
            if key not in seen:
                seen.add(key)
                genes.append(normalized)
    return tuple(genes)


def _valid_gene(expression: Any, features: tuple[str, ...]) -> bool:
    if isinstance(expression, str):
        return expression in features
    if not isinstance(expression, tuple) or not expression:
        return False
    operator = expression[0]
    if operator in UNARY:
        return len(expression) == 2 and _valid_gene(expression[1], features)
    if operator in BINARY:
        return (
            len(expression) == 3
            and _valid_gene(expression[1], features)
            and _valid_gene(expression[2], features)
        )
    if operator in ROLLING:
        return (
            len(expression) == 3
            and _valid_gene(expression[1], features)
            and isinstance(expression[2], (int, float))
        )
    if operator in PAIR_ROLLING:
        return (
            len(expression) == 4
            and _valid_gene(expression[1], features)
            and _valid_gene(expression[2], features)
            and isinstance(expression[3], (int, float))
        )
    return False


def _proposal(
    operator: str,
    next_feature,
    windows: tuple[int, ...],
    prior_genes: tuple[Any, ...],
    rng: random.Random,
) -> Any:
    def child():
        if prior_genes and rng.random() < 0.70:
            return rng.choice(prior_genes)
        return next(next_feature)

    if operator in UNARY:
        return operator, child()
    if operator in BINARY:
        return operator, child(), child()
    if operator in ROLLING:
        return operator, child(), rng.choice(windows)
    return operator, child(), child(), rng.choice(windows)


def _child_domain(features: tuple[str, ...], prior_genes: tuple[Any, ...]):
    yield from features
    yield from prior_genes


def _completion_expressions(
    features: tuple[str, ...], windows: tuple[int, ...], prior_genes: tuple[Any, ...]
):
    for operator in UNARY:
        for child in _child_domain(features, prior_genes):
            yield operator, child
    for operator in BINARY:
        for left in _child_domain(features, prior_genes):
            for right in _child_domain(features, prior_genes):
                yield operator, left, right
    for operator in ROLLING:
        for child in _child_domain(features, prior_genes):
            for window in windows:
                yield operator, child, window
    for operator in PAIR_ROLLING:
        for left in _child_domain(features, prior_genes):
            for right in _child_domain(features, prior_genes):
                for window in windows:
                    yield operator, left, right, window


def balanced_stage_proposals(
    features,
    windows,
    stage_depth: int,
    size: int,
    seed: int,
    gene_pools=(),
) -> tuple[Any, ...]:
    """Build deterministic, stage-bounded candidates with balanced roots.

    Stage one starts from raw terminals.  At later stages each child samples a
    promoted lower-depth gene 70% of the time, falling back to a shuffled raw
    feature cycle otherwise.
    """
    features = tuple(dict.fromkeys(features))
    windows = tuple(windows)
    if not features:
        raise ValueError("at least one raw feature is required")
    if not windows:
        raise ValueError("at least one rolling window is required")
    _validate_stage_depth(stage_depth)
    if size < 0:
        raise ValueError("size must not be negative")
    if size == 0:
        return ()

    rng = random.Random(seed)
    next_feature = _feature_cycle(rng, features)
    prior_genes = _normalized_prior_genes(gene_pools, stage_depth, features)
    proposals = []
    seen = set()

    def add(expression: Any) -> bool:
        try:
            normalized = normalize_expression(expression)
        except (TypeError, ValueError):
            return False
        if expression_depth(normalized) > stage_depth:
            return False
        key = expression_key(normalized)
        if key in seen:
            return False
        seen.add(key)
        proposals.append(normalized)
        return True

    if stage_depth == 1:
        for _ in range(min(size, len(features))):
            add(next(next_feature))

    operator_index = 0
    attempts = 0
    max_attempts = max(size * 100, len(OPERATORS) * len(features) * 10)
    while len(proposals) < size and attempts < max_attempts:
        operator = OPERATORS[operator_index % len(OPERATORS)]
        operator_index += 1
        attempts += 1
        add(_proposal(operator, next_feature, windows, prior_genes, rng))

    if len(proposals) < size:
        for expression in _completion_expressions(features, windows, prior_genes):
            add(expression)
            if len(proposals) == size:
                break
    return tuple(proposals)
