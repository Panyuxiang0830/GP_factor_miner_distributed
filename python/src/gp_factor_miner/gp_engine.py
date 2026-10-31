from __future__ import annotations

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd

from .cache import ExpressionCache
from .config import GPConfig
from .expressions import (
    evaluate_expression,
    expression_depth,
    expression_key,
    expression_nodes,
    normalize_expression,
)
from .metrics import daily_rank_ic
from .pca_qd import factor_signature
from .raw_features import feature_sampling_probabilities


UNARY = ("log", "abs", "rank", "zscore")
BINARY = ("+", "-", "*", "/")
ROLLING = ("mean", "std", "max", "min", "sum", "median", "ema", "lag", "delta", "pct_change")
PAIR_ROLLING = ("correlation", "beta")
ALL_OPERATOR_GROUPS = (UNARY, BINARY, ROLLING, PAIR_ROLLING)
ISLAND_SPECS = {
    "arithmetic": BINARY,
    "smoothing": ("mean", "std", "max", "min", "sum", "median", "ema", "rank", "zscore", "abs"),
    "change": ("lag", "delta", "pct_change"),
    "pair": PAIR_ROLLING,
}


@dataclass(frozen=True)
class Candidate:
    expression: Any
    rank_ic: float
    abs_rank_ic: float
    valid_ratio: float
    missing_ratio: float
    depth: int
    nodes: int
    fitness: float = 0.0
    segment_rank_ic_std: float = 0.0
    same_sign_ratio: float = 0.0
    segment_rank_ics: tuple[float, ...] = ()


def _fitness_from_segments(
    rank_ic_mean,
    segment_means,
    nodes,
    stability_penalty,
    complexity_penalty,
    minimum_same_sign_ratio,
    inconsistent_sign_multiplier,
):
    finite = np.asarray([value for value in segment_means if math.isfinite(value)])
    if finite.size == 0:
        return -math.inf, math.inf, 0.0
    segment_std = float(np.std(finite, ddof=0))
    direction = 1.0 if rank_ic_mean > 0 else -1.0
    same_sign_ratio = float(np.mean(finite * direction > 0))
    fitness = (
        abs(rank_ic_mean)
        - stability_penalty * segment_std
        - complexity_penalty * max(nodes - 3, 0)
    )
    if same_sign_ratio < minimum_same_sign_ratio:
        fitness *= inconsistent_sign_multiplier
    return float(fitness), segment_std, same_sign_ratio


@dataclass(frozen=True)
class EvolutionResult:
    hall_of_fame: tuple[Candidate, ...]
    history: tuple[dict, ...]
    evaluated_count: int
    cache_hits: int
    worker_pids: tuple[int, ...]


_WORKER_DATA = None
_WORKER_TARGET = None
_WORKER_CONFIG = None


def _initialize_worker(data_dict, target, config):
    global _WORKER_DATA, _WORKER_TARGET, _WORKER_CONFIG
    _WORKER_DATA = data_dict
    _WORKER_TARGET = target
    _WORKER_CONFIG = config


def _worker_evaluate(expression):
    candidate = _evaluate(
        expression,
        _WORKER_DATA,
        _WORKER_TARGET,
        _WORKER_CONFIG,
        ExpressionCache(),
    )
    return candidate, os.getpid()


def _worker_evaluate_with_signature(task):
    expression, sample_step, power_iterations, seed = task
    candidate, values = _evaluate_with_values(
        expression,
        _WORKER_DATA,
        _WORKER_TARGET,
        _WORKER_CONFIG,
        ExpressionCache(),
    )
    signature = None
    if candidate is not None and values is not None:
        try:
            signature = factor_signature(
                values,
                sample_step=sample_step,
                power_iterations=power_iterations,
                seed=seed,
            )
        except Exception:
            signature = None
    return candidate, signature, os.getpid()


def _worker_signature(task):
    expression, sample_step, power_iterations, seed = task
    try:
        values = evaluate_expression(expression, _WORKER_DATA)
        signature = factor_signature(
            values,
            sample_step=sample_step,
            power_iterations=power_iterations,
            seed=seed,
        )
    except Exception:
        signature = None
    return expression_key(expression), signature, os.getpid()


def terminal_names(expression) -> set[str]:
    if isinstance(expression, str):
        return {expression}
    if not isinstance(expression, (tuple, list)):
        return set()
    result = set()
    for item in expression[1:]:
        result.update(terminal_names(item))
    return result


def _choose_feature(rng, features, probabilities):
    return rng.choices(features, weights=probabilities, k=1)[0]


def _random_expression(
    rng,
    features,
    probabilities,
    windows,
    depth,
    min_depth=0,
    preferred_operators=(),
    preferred_probability=0.0,
):
    if depth <= 0 or (min_depth <= 0 and rng.random() < 0.28):
        return _choose_feature(rng, features, probabilities)
    preferred = tuple(preferred_operators)
    operator = (
        rng.choice(preferred)
        if preferred and rng.random() < preferred_probability
        else None
    )
    category = rng.choice(("unary", "binary", "rolling", "pair")) if operator is None else None
    if operator in UNARY:
        category = "unary"
    elif operator in BINARY:
        category = "binary"
    elif operator in ROLLING:
        category = "rolling"
    elif operator in PAIR_ROLLING:
        category = "pair"
    next_min = max(0, min_depth - 1)
    if category == "unary":
        return (
            operator or rng.choice(UNARY),
            _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
        )
    if category == "binary":
        return (
            operator or rng.choice(BINARY),
            _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
            _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
        )
    if category == "rolling":
        return (
            operator or rng.choice(ROLLING),
            _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
            rng.choice(windows),
        )
    return (
        operator or rng.choice(PAIR_ROLLING),
        _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
        _random_expression(rng, features, probabilities, windows, depth - 1, next_min, preferred, preferred_probability),
        rng.choice(windows),
    )


def _random_normalized_expression(
    rng,
    features,
    probabilities,
    windows,
    depth,
    min_depth=0,
    preferred_operators=(),
    preferred_probability=0.0,
):
    while True:
        try:
            return normalize_expression(
                _random_expression(
                    rng, features, probabilities, windows, depth, min_depth,
                    preferred_operators, preferred_probability,
                )
            )
        except (ValueError, TypeError):
            continue


def build_initial_population(
    features,
    probabilities,
    windows,
    size,
    init_min_depth,
    init_max_depth,
    seed,
    preferred_operators=(),
    preferred_probability=0.0,
):
    rng = random.Random(seed)
    population = list(features) if size >= len(features) else rng.sample(list(features), size)
    while len(population) < size:
        depth = rng.randint(max(0, init_min_depth), max(init_min_depth, init_max_depth))
        try:
            expression = normalize_expression(
                _random_expression(
                    rng,
                    features,
                    probabilities,
                    windows,
                    depth,
                    init_min_depth,
                    preferred_operators,
                    preferred_probability,
                )
            )
        except (ValueError, TypeError):
            continue
        population.append(expression)
    rng.shuffle(population)
    return population


def _paths(expression, prefix=()):
    result = [prefix]
    if isinstance(expression, tuple):
        for index, child in enumerate(expression[1:], start=1):
            if not isinstance(child, (int, float)):
                result.extend(_paths(child, prefix + (index,)))
    return result


def _at(expression, path):
    for index in path:
        expression = expression[index]
    return expression


def _replace(expression, path, replacement):
    if not path:
        return replacement
    values = list(expression)
    values[path[0]] = _replace(values[path[0]], path[1:], replacement)
    return tuple(values)


def _crossover(left, right, rng):
    left_path = rng.choice(_paths(left))
    right_path = rng.choice(_paths(right))
    return _replace(left, left_path, _at(right, right_path))


def _mutate(
    expression, rng, features, probabilities, windows, max_depth,
    preferred_operators=(), preferred_probability=0.0,
):
    path = rng.choice(_paths(expression))
    replacement = _random_expression(
        rng, features, probabilities, windows, rng.randint(0, 2),
        preferred_operators=preferred_operators,
        preferred_probability=preferred_probability,
    )
    mutated = _replace(expression, path, replacement)
    if expression_depth(mutated) > max_depth:
        return expression
    return mutated


def _mutate_feature(expression, rng, features, probabilities):
    paths = [path for path in _paths(expression) if isinstance(_at(expression, path), str)]
    if not paths:
        return expression
    path = rng.choice(paths)
    current = _at(expression, path)
    alternatives = [(feature, weight) for feature, weight in zip(features, probabilities) if feature != current]
    if not alternatives:
        return expression
    replacement = rng.choices(
        [item[0] for item in alternatives],
        weights=[item[1] for item in alternatives],
        k=1,
    )[0]
    return _replace(expression, path, replacement)


def _mutate_operator(expression, rng):
    paths = [path for path in _paths(expression) if isinstance(_at(expression, path), tuple)]
    mutable = []
    for path in paths:
        operator = _at(expression, path)[0]
        group = next((group for group in ALL_OPERATOR_GROUPS if operator in group), ())
        alternatives = tuple(value for value in group if value != operator)
        if alternatives:
            mutable.append((path, alternatives))
    if not mutable:
        return expression
    path, alternatives = rng.choice(mutable)
    node = _at(expression, path)
    return _replace(expression, path, (rng.choice(alternatives), *node[1:]))


def _numeric_paths(expression, prefix=()):
    result = []
    if isinstance(expression, tuple):
        for index, child in enumerate(expression[1:], start=1):
            path = prefix + (index,)
            if isinstance(child, (int, float)):
                result.append(path)
            elif isinstance(child, tuple):
                result.extend(_numeric_paths(child, path))
    return result


def _mutate_window(expression, rng, windows):
    paths = _numeric_paths(expression)
    if not paths:
        return expression
    path = rng.choice(paths)
    current = _at(expression, path)
    alternatives = [window for window in windows if window != current]
    return _replace(expression, path, rng.choice(alternatives)) if alternatives else expression


def _epsilon_lexicase(scored, rng, epsilon_scale=1.0):
    survivors = [candidate for candidate, _ in scored if candidate.segment_rank_ics]
    if not survivors:
        return _tournament(scored, rng, min(3, len(scored)))
    cases = list(range(len(survivors[0].segment_rank_ics)))
    rng.shuffle(cases)
    for case in cases:
        values = np.asarray(
            [
                candidate.segment_rank_ics[case] * (1.0 if candidate.rank_ic > 0 else -1.0)
                for candidate in survivors
            ],
            dtype=float,
        )
        finite = np.isfinite(values)
        if not finite.any():
            continue
        finite_values = values[finite]
        median = float(np.median(finite_values))
        epsilon = epsilon_scale * float(np.median(np.abs(finite_values - median)))
        cutoff = float(finite_values.max()) - epsilon
        survivors = [
            candidate
            for candidate, value in zip(survivors, values)
            if math.isfinite(value) and value >= cutoff
        ]
        if len(survivors) <= 1:
            break
    return rng.choice(survivors).expression


def _build_capped_parent_pool(scored, target_size, cap_shares=(0.15, 0.20, 0.30)):
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    selected = []
    selected_keys = set()
    counts = {}
    used_share = cap_shares[-1]
    for share in cap_shares:
        cap = max(1, int(target_size * share))
        for item in ordered:
            key = expression_key(item[0].expression)
            if key in selected_keys:
                continue
            terminals = terminal_names(item[0].expression)
            if any(counts.get(terminal, 0) >= cap for terminal in terminals):
                continue
            selected.append(item)
            selected_keys.add(key)
            for terminal in terminals:
                counts[terminal] = counts.get(terminal, 0) + 1
            if len(selected) >= target_size:
                return selected, share
        used_share = share
    return selected, used_share


def _evaluate_with_values(expression, data_dict, target, config, cache):
    try:
        normalized = normalize_expression(expression)
        key = expression_key(normalized)
    except (ValueError, TypeError):
        return None, None
    cached = cache.get(key)
    if cached is not None:
        return cached, None
    if expression_depth(normalized) > config.max_depth or expression_nodes(normalized) > config.max_nodes:
        return None, None
    try:
        values = evaluate_expression(normalized, data_dict)
        daily, missing_ratio, time_count = daily_rank_ic(values, target)
    except Exception:
        return None, None
    valid = daily.dropna()
    rank_ic_mean = float(valid.mean()) if not valid.empty else math.nan
    valid_ratio = float(valid.size / time_count) if time_count else 0.0
    if (
        not math.isfinite(rank_ic_mean)
        or valid_ratio < config.min_valid_time_ratio
        or missing_ratio > config.max_missing_ratio
    ):
        return None, None
    segment_means = []
    for segment in np.array_split(daily.to_numpy(dtype=float), config.stability_segments):
        finite_segment = segment[np.isfinite(segment)]
        segment_means.append(
            float(finite_segment.mean()) if finite_segment.size else math.nan
        )
    segment_means = tuple(segment_means)
    nodes = expression_nodes(normalized)
    fitness, segment_std, same_sign_ratio = _fitness_from_segments(
        rank_ic_mean,
        segment_means,
        nodes,
        config.stability_penalty,
        config.complexity_penalty,
        config.minimum_same_sign_ratio,
        config.inconsistent_sign_multiplier,
    )
    candidate = Candidate(
        expression=normalized,
        rank_ic=rank_ic_mean,
        abs_rank_ic=abs(rank_ic_mean),
        valid_ratio=valid_ratio,
        missing_ratio=missing_ratio,
        depth=expression_depth(normalized),
        nodes=nodes,
        fitness=fitness,
        segment_rank_ic_std=segment_std,
        same_sign_ratio=same_sign_ratio,
        segment_rank_ics=segment_means,
    )
    cache.put(key, candidate)
    return candidate, values


def _evaluate(expression, data_dict, target, config, cache):
    candidate, _ = _evaluate_with_values(
        expression, data_dict, target, config, cache
    )
    return candidate


def _tournament(scored, rng, size):
    entrants = rng.choices(scored, k=size)
    winner, _ = max(entrants, key=lambda item: item[1])
    return winner.expression


def _atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _candidate_to_row(position, key, candidate, source, island):
    row = {
        "position": position,
        "expression": key,
        "valid": int(candidate is not None),
        "source": source,
        "island": island,
        "segment_rank_ics": json.dumps(candidate.segment_rank_ics if candidate else ()),
    }
    for field in (
        "rank_ic",
        "abs_rank_ic",
        "valid_ratio",
        "missing_ratio",
        "depth",
        "nodes",
        "fitness",
        "segment_rank_ic_std",
        "same_sign_ratio",
    ):
        row[field] = getattr(candidate, field) if candidate is not None else np.nan
    return row


def _candidate_from_row(row):
    if int(row["valid"]) != 1:
        return None
    return Candidate(
        expression=normalize_expression(__import__("json").loads(row["expression"])),
        rank_ic=float(row["rank_ic"]),
        abs_rank_ic=float(row["abs_rank_ic"]),
        valid_ratio=float(row["valid_ratio"]),
        missing_ratio=float(row["missing_ratio"]),
        depth=int(row["depth"]),
        nodes=int(row["nodes"]),
        fitness=float(row["fitness"]),
        segment_rank_ic_std=float(row["segment_rank_ic_std"]),
        same_sign_ratio=float(row["same_sign_ratio"]),
        segment_rank_ics=tuple(json.loads(row.get("segment_rank_ics", "[]"))),
    )


def _operator_counts(expressions):
    counts = {}
    for expression in expressions:
        for path in _paths(expression):
            node = _at(expression, path)
            if isinstance(node, tuple):
                counts[node[0]] = counts.get(node[0], 0) + 1
    return counts


def _feature_counts(expressions):
    counts = {}
    for expression in expressions:
        for terminal in terminal_names(expression):
            counts[terminal] = counts.get(terminal, 0) + 1
    return counts


def _select_parent(scored, rng, config, counters):
    if rng.random() < config.tournament_selection_probability:
        counters["tournament"] += 1
        return _tournament(scored, rng, min(config.tournament_size, len(scored)))
    counters["lexicase"] += 1
    return _epsilon_lexicase(scored, rng, config.lexicase_epsilon_scale)


def run_evolution(
    data_dict,
    target,
    feature_profile,
    config: GPConfig,
    seed: int,
    progress_callback=None,
    checkpoint_dir=None,
):
    def report(event, **values):
        if progress_callback is not None:
            progress_callback({"event": event, **values})

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    completed_path = checkpoint_dir / "completed.pkl" if checkpoint_dir else None
    state_path = checkpoint_dir / "run_state.pkl" if checkpoint_dir else None
    if completed_path is not None and completed_path.exists():
        with completed_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != config.checkpoint_version:
            raise RuntimeError("checkpoint format is incompatible; use a new output directory")
        return payload["result"]

    features = feature_profile["feature"].tolist()
    probabilities = feature_sampling_probabilities(
        feature_profile, uniform_share=config.uniform_feature_probability
    )
    island_names = tuple(ISLAND_SPECS)
    if config.population_size % len(island_names):
        raise ValueError("population_size must be divisible by four islands")
    island_size = config.population_size // len(island_names)
    rng = random.Random(seed)
    if state_path is not None and state_path.exists():
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != config.checkpoint_version:
            raise RuntimeError("checkpoint format is incompatible; use a new output directory")
        generation = state["generation"]
        populations = state["populations"]
        cache = state["cache"]
        history = state["history"]
        archives = state["archives"]
        rng.setstate(state["rng_state"])
    else:
        generation = 0
        populations = {
            name: build_initial_population(
                features, probabilities, config.windows, island_size,
                config.init_min_depth, config.init_max_depth, seed + index * 1009,
                ISLAND_SPECS[name], config.island_preferred_operator_probability,
            )
            for index, name in enumerate(island_names)
        }
        cache = ExpressionCache()
        history = []
        archives = {name: {} for name in island_names}

    def save_state():
        if state_path is not None:
            _atomic_pickle(state_path, {
                "version": config.checkpoint_version,
                "generation": generation,
                "populations": populations,
                "cache": cache,
                "history": history,
                "archives": archives,
                "rng_state": rng.getstate(),
            })

    save_state()
    worker_pids = set()
    pool = None
    if config.max_workers > 1:
        context = mp.get_context("fork")
        pool = context.Pool(
            processes=config.max_workers,
            initializer=_initialize_worker,
            initargs=(data_dict, target, config),
        )
    try:
        while generation < config.generations:
            generation_number = generation + 1
            flat_population = [
                (name, expression)
                for name in island_names
                for expression in populations[name]
            ]
            generation_dir = (
                checkpoint_dir / f"generation_{generation_number:02d}"
                if checkpoint_dir
                else None
            )
            total_chunks = math.ceil(len(flat_population) / config.progress_chunk_size)
            completed_chunk_paths = (
                sorted(generation_dir.glob("chunk_*.csv"))
                if generation_dir is not None and generation_dir.exists()
                else []
            )
            completed_chunks = {int(path.stem.split("_")[-1]): path for path in completed_chunk_paths}
            generation_candidates = {name: {} for name in island_names}
            for path in completed_chunk_paths:
                for row in pd.read_csv(path).to_dict(orient="records"):
                    candidate = _candidate_from_row(row)
                    if candidate is not None:
                        key = expression_key(candidate.expression)
                        cache.put(key, candidate)
                        generation_candidates[row["island"]][key] = candidate
            cached_rows = sum(len(pd.read_csv(path)) for path in completed_chunk_paths)
            report(
                "generation_started",
                generation=generation_number,
                generations=config.generations,
                population=len(flat_population), cached=cached_rows,
                pending=len(flat_population) - cached_rows,
                chunks=total_chunks,
            )
            for chunk_number in range(1, total_chunks + 1):
                if chunk_number in completed_chunks:
                    continue
                start = (chunk_number - 1) * config.progress_chunk_size
                stop = min(start + config.progress_chunk_size, len(flat_population))
                chunk_expressions = []
                pending_by_key = {}
                for position, (island, expression) in enumerate(flat_population[start:stop], start=start):
                    try:
                        normalized = normalize_expression(expression)
                        key = expression_key(normalized)
                    except (ValueError, TypeError):
                        chunk_expressions.append((position, island, "", None))
                        continue
                    chunk_expressions.append((position, island, key, normalized))
                    cached = cache.get(key)
                    if cached is not None:
                        generation_candidates[island][key] = cached
                    elif key not in pending_by_key:
                        pending_by_key[key] = normalized
                pending = list(pending_by_key.values())
                if pool is None:
                    evaluated = (
                        (_evaluate(expression, data_dict, target, config, ExpressionCache()), os.getpid())
                        for expression in pending
                    )
                else:
                    evaluated = pool.imap(_worker_evaluate, pending, chunksize=1)
                for candidate, worker_pid in evaluated:
                    worker_pids.add(worker_pid)
                    if candidate is not None:
                        key = expression_key(candidate.expression)
                        cache.put(key, candidate)
                        for _, island, candidate_key, _ in chunk_expressions:
                            if candidate_key == key:
                                generation_candidates[island][key] = candidate
                rows = []
                for position, island, key, normalized in chunk_expressions:
                    candidate = cache.get(key) if key else None
                    if candidate is not None:
                        generation_candidates[island][key] = candidate
                    rows.append(
                        _candidate_to_row(
                            position, key, candidate,
                            "cache" if key not in pending_by_key else "evaluated", island,
                        )
                    )
                if generation_dir is not None:
                    _atomic_csv(generation_dir / f"chunk_{chunk_number:02d}.csv", rows)
                report(
                    "chunk_completed",
                    generation=generation_number,
                    completed=stop,
                    total=len(flat_population),
                    chunk=chunk_number,
                    chunks=total_chunks,
                    valid_candidates=sum(len(items) for items in generation_candidates.values()),
                )
            scored_by_island = {}
            island_summaries = []
            for name in island_names:
                scored = [(candidate, candidate.fitness) for candidate in generation_candidates[name].values()]
                if not scored:
                    raise RuntimeError(f"GP island {name} produced no valid candidates")
                scored_by_island[name] = scored
                for candidate, _ in scored:
                    archives[name][expression_key(candidate.expression)] = candidate
                expressions = [candidate.expression for candidate, _ in scored]
                feature_counts = _feature_counts(expressions)
                operator_counts = _operator_counts(expressions)
                island_metrics = {
                    "generation": generation_number, "island": name,
                    "best_fitness": max(score for _, score in scored),
                    "best_abs_rank_ic": max(candidate.abs_rank_ic for candidate, _ in scored),
                    "valid_candidates": len(scored),
                    "top_features": json.dumps(sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))[:20]),
                    "operator_counts": json.dumps(operator_counts, sort_keys=True),
                }
                island_summaries.append(island_metrics)
                report("island_generation_completed", **island_metrics)
            all_scored = [item for name in island_names for item in scored_by_island[name]]
            scores = np.array([score for _, score in all_scored], dtype=float)
            absolute_rank_ics = np.array([candidate.abs_rank_ic for candidate, _ in all_scored], dtype=float)
            generation_metrics = {
                    "generation": generation_number, "island": "all",
                    "best_abs_rank_ic": float(absolute_rank_ics.max()),
                    "mean_abs_rank_ic": float(absolute_rank_ics.mean()),
                    "best_fitness": float(scores.max()),
                    "mean_fitness": float(scores.mean()),
                    "valid_candidates": int(len(all_scored)),
                    "unique_evaluated": int(len(cache)),
                    "cache_hits": int(cache.hits),
                    "island_metrics": json.dumps(island_summaries, sort_keys=True),
                }
            history.append(generation_metrics)
            report("generation_completed", **generation_metrics)
            generation += 1
            if generation < config.generations:
                migrate = generation % config.migration_interval == 0
                migrants = {}
                if migrate:
                    for index, name in enumerate(island_names):
                        source = island_names[index - 1]
                        migrants[name] = [item[0].expression for item in sorted(scored_by_island[source], key=lambda item: item[1], reverse=True)[:config.migration_count]]
                    report("migration_completed", generation=generation_number, count=sum(map(len, migrants.values())))
                next_populations = {}
                reproduction_summaries = []
                for name in island_names:
                    scored = sorted(scored_by_island[name], key=lambda item: item[1], reverse=True)
                    target_pool_size = min(config.parent_pool_size, len(scored))
                    parent_pool, used_cap = _build_capped_parent_pool(scored, target_pool_size, config.parent_feature_cap_shares)
                    if not parent_pool:
                        parent_pool = scored
                    elite_count = max(1, int(island_size * config.elitism_ratio))
                    immigrant_count = int(island_size * config.random_immigrant_ratio)
                    next_population = [item[0].expression for item in scored[:elite_count]]
                    next_population.extend(migrants.get(name, ()))
                    counters = {"tournament": 0, "lexicase": 0}
                    offspring_target = island_size - immigrant_count
                    while len(next_population) < offspring_target:
                        parent = _select_parent(parent_pool, rng, config, counters)
                        child = parent
                        if rng.random() < config.crossover_probability:
                            child = _crossover(child, _select_parent(parent_pool, rng, config, counters), rng)
                        if rng.random() < config.feature_mutation_probability:
                            child = _mutate_feature(child, rng, features, probabilities)
                        if rng.random() < config.operator_mutation_probability:
                            child = _mutate_operator(child, rng)
                        if rng.random() < config.window_mutation_probability:
                            child = _mutate_window(child, rng, config.windows)
                        if rng.random() < config.subtree_mutation_probability:
                            child = _mutate(
                                child, rng, features, probabilities, config.windows,
                                config.max_depth, ISLAND_SPECS[name],
                                config.island_preferred_operator_probability,
                            )
                        try:
                            child = normalize_expression(child)
                        except (ValueError, TypeError):
                            child = parent
                        if expression_depth(child) <= config.max_depth and expression_nodes(child) <= config.max_nodes:
                            next_population.append(child)
                    while len(next_population) < island_size:
                        next_population.append(_random_normalized_expression(
                            rng, features, probabilities, config.windows,
                            rng.randint(config.init_min_depth, config.init_max_depth),
                            config.init_min_depth, ISLAND_SPECS[name],
                            config.island_preferred_operator_probability,
                        ))
                    next_populations[name] = next_population[:island_size]
                    reproduction_summaries.append({
                        "generation": generation_number, "island": name,
                        "event": "reproduction", "parent_cap": used_cap,
                        "parent_pool": len(parent_pool), **counters,
                    })
                    report(
                        "island_reproduction_completed",
                        generation=generation_number, island=name,
                        parent_cap=used_cap, parent_pool=len(parent_pool), **counters,
                    )
                populations = next_populations
                history[-1]["reproduction"] = json.dumps(reproduction_summaries, sort_keys=True)
            save_state()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    hall = tuple(candidate for name in island_names for candidate in sorted(
        archives[name].values(), key=lambda item: (-item.fitness, expression_key(item.expression))
    )[:config.island_hall_of_fame_size])
    hall = tuple(sorted(hall, key=lambda item: (-item.fitness, expression_key(item.expression))))
    result = EvolutionResult(
        hall,
        tuple(history),
        len(cache),
        cache.hits,
        tuple(sorted(worker_pids)),
    )
    if completed_path is not None:
        _atomic_pickle(completed_path, {"version": config.checkpoint_version, "result": result})
    return result
