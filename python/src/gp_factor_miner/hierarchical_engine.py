from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np

from .cache import ExpressionCache
from .expressions import (
    evaluate_expression,
    expression_depth,
    expression_key,
    expression_nodes,
    normalize_expression,
)
from .gp_engine import (
    Candidate,
    _atomic_pickle,
    _build_capped_parent_pool,
    _epsilon_lexicase,
    _evaluate,
    _initialize_worker,
    _tournament,
    _worker_evaluate_with_signature,
    _worker_signature,
    terminal_names,
)
from .hierarchical_config import HierarchicalGPConfig
from .hierarchical_ops import (
    OPERATORS as HIERARCHICAL_OPERATORS,
    balanced_stage_proposals,
    bounded_mutation,
    expression_operators,
    family_survivors,
    same_depth_crossover,
)
from .pca_qd import PCAQDArchive, QDRecord, factor_signature
from .raw_features import feature_sampling_probabilities


CHECKPOINT_VERSION = 4
_NUMPY_SEED_MODULUS = 1 << 64


@dataclass(frozen=True)
class StageResult:
    depth: int
    gene_pool: tuple[Candidate, ...]
    generations_completed: int


@dataclass(frozen=True)
class OperatorDiagnostic:
    operator: str
    candidate: Candidate
    stage_depth: int
    signature: np.ndarray


@dataclass(frozen=True)
class HierarchicalResult:
    stages: tuple[StageResult, ...]
    archive_records: tuple[QDRecord, ...]
    archive_candidates: tuple[Candidate, ...]
    operator_diagnostics: tuple[OperatorDiagnostic, ...]
    history: tuple[dict, ...]
    evaluated_count: int
    cache_hits: int


@dataclass(frozen=True)
class _EvaluationConfig:
    max_depth: int
    max_nodes: int
    min_valid_time_ratio: float
    max_missing_ratio: float
    stability_segments: int
    stability_penalty: float
    complexity_penalty: float
    minimum_same_sign_ratio: float
    inconsistent_sign_multiplier: float


class _ComparableSignature(np.ndarray):
    """An ndarray whose scalar equality keeps result records comparable."""

    def __new__(cls, values):
        return np.asarray(values, dtype=np.float32).view(cls)

    def __eq__(self, other):
        try:
            return bool(
                np.array_equal(
                    np.asarray(self), np.asarray(other), equal_nan=True
                )
            )
        except (TypeError, ValueError):
            return False


def _config_fingerprint(config: HierarchicalGPConfig) -> str:
    def serializable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): serializable(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value

    encoded = json.dumps(
        serializable(asdict(config)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: HierarchicalGPConfig) -> None:
    if config.checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError("hierarchical checkpoints require version 4")
    if tuple(config.stage_depths) != (1, 2, 3):
        raise ValueError("stage_depths must be exactly (1, 2, 3)")
    if config.population_size <= 0:
        raise ValueError("population_size must be positive")
    if config.warm_start_multiplier <= 0:
        raise ValueError("warm_start_multiplier must be positive")
    if config.generations_per_stage < 0:
        raise ValueError("generations_per_stage must not be negative")
    if config.gene_pool_size <= 0 or config.parent_pool_size <= 0:
        raise ValueError("gene_pool_size and parent_pool_size must be positive")
    if config.max_nodes <= 0 or config.progress_chunk_size <= 0:
        raise ValueError("max_nodes and progress_chunk_size must be positive")
    if config.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if config.pca_qd_block_size <= 0:
        raise ValueError("pca_qd_block_size must be positive")
    if config.pca_qd_backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("pca_qd_backend must be auto, cpu, or cuda")
    if config.final_factor_count is not None and config.final_factor_count <= 0:
        raise ValueError("final_factor_count must be positive or None")
    if not 0.0 <= config.random_immigrant_ratio <= 1.0:
        raise ValueError("random_immigrant_ratio must be between zero and one")


def _validate_checkpoint(payload, fingerprint: str, seed: int) -> None:
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "checkpoint format is incompatible; use a new output directory"
        )
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            "checkpoint configuration is incompatible; use a new output directory"
        )
    if (
        "seed" not in payload
        or not isinstance(payload["seed"], int)
        or isinstance(payload["seed"], bool)
        or payload["seed"] != seed
    ):
        raise RuntimeError(
            "checkpoint seed is incompatible; use a new output directory"
        )


def _checkpoint_error() -> RuntimeError:
    return RuntimeError(
        "checkpoint format is incompatible; use a new output directory"
    )


def _validate_loaded_structure(validator, *args) -> None:
    """Normalize malformed nested checkpoint failures at the load boundary."""
    try:
        validator(*args)
    except RuntimeError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _checkpoint_error() from error


def _valid_int(value, minimum=0, maximum=None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def _validate_candidate(
    candidate, config: HierarchicalGPConfig | None = None
) -> bool:
    if not isinstance(candidate, Candidate):
        return False
    try:
        normalized = normalize_expression(candidate.expression)
        depth = expression_depth(normalized)
        nodes = expression_nodes(normalized)
    except (IndexError, TypeError, ValueError):
        return False
    if candidate.expression != normalized:
        return False
    if not (
        _finite_number(candidate.rank_ic)
        and _finite_number(candidate.abs_rank_ic)
        and candidate.abs_rank_ic == abs(candidate.rank_ic)
        and _finite_number(candidate.valid_ratio)
        and 0.0 <= candidate.valid_ratio <= 1.0
        and _finite_number(candidate.missing_ratio)
        and 0.0 <= candidate.missing_ratio <= 1.0
        and _valid_int(candidate.depth, 0)
        and candidate.depth == depth
        and _valid_int(candidate.nodes, 1)
        and candidate.nodes == nodes
        and _finite_number(candidate.fitness)
        and _finite_number(candidate.segment_rank_ic_std)
        and candidate.segment_rank_ic_std >= 0.0
        and _finite_number(candidate.same_sign_ratio)
        and 0.0 <= candidate.same_sign_ratio <= 1.0
        and isinstance(candidate.segment_rank_ics, tuple)
        and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and not math.isinf(value)
            for value in candidate.segment_rank_ics
        )
    ):
        return False
    if config is None:
        return True
    return (
        candidate.depth <= max(config.stage_depths)
        and candidate.nodes <= config.max_nodes
        and len(candidate.segment_rank_ics) == config.stability_segments
    )


def _validate_qd_record(record) -> bool:
    if not isinstance(record, QDRecord):
        return False
    try:
        key = expression_key(record.expression)
        nodes = expression_nodes(record.expression)
        depth = expression_depth(record.expression)
    except (IndexError, TypeError, ValueError):
        return False
    return (
        isinstance(record.key, str)
        and record.key == key
        and _finite_number(record.abs_rank_ic)
        and record.abs_rank_ic >= 0.0
        and _valid_int(record.nodes, 1)
        and record.nodes == nodes
        and record.stage_depth in (1, 2, 3)
        and depth <= record.stage_depth
        and _signature_is_comparable(record.signature)
    )


def _validate_operator_diagnostic(operator, diagnostic) -> bool:
    return (
        isinstance(operator, str)
        and isinstance(diagnostic, OperatorDiagnostic)
        and diagnostic.operator == operator
        and _validate_candidate(diagnostic.candidate)
        and diagnostic.stage_depth in (1, 2, 3)
        and diagnostic.candidate.depth <= diagnostic.stage_depth
        and operator in expression_operators(diagnostic.candidate.expression)
        and isinstance(diagnostic.signature, np.ndarray)
        and _signature_is_comparable(diagnostic.signature)
    )


def _validate_qd_stats(stats) -> bool:
    count_keys = {
        "valid_candidates",
        "qd_admitted",
        "qd_rejected",
        "qd_replaced",
        "pca_failures",
    }
    if not isinstance(stats, dict) or set(stats) != count_keys | {"similarities"}:
        return False
    return (
        all(_valid_int(stats[key], 0) for key in count_keys)
        and isinstance(stats["similarities"], list)
        and all(
            _finite_number(similarity) and similarity >= 0.0
            for similarity in stats["similarities"]
        )
    )


def _validate_expression(expression, config: HierarchicalGPConfig) -> bool:
    try:
        normalized = normalize_expression(expression)
        return (
            expression == normalized
            and expression_depth(expression) <= max(config.stage_depths)
            and expression_nodes(expression) <= config.max_nodes
        )
    except (IndexError, TypeError, ValueError):
        return False


def _validate_batch(batch, config: HierarchicalGPConfig) -> bool:
    if batch is None:
        return True
    if not isinstance(batch, dict):
        return False
    expressions = batch.get("expressions")
    cursor = batch.get("cursor")
    results = batch.get("results")
    signatures = batch.get("signatures", {})
    result_keys = {
        expression_key(candidate.expression)
        for candidate in results or ()
        if isinstance(candidate, Candidate)
    }
    return (
        isinstance(expressions, tuple)
        and _valid_int(cursor, 0, len(expressions))
        and isinstance(results, list)
        and len(results) == cursor
        and isinstance(signatures, dict)
        and set(signatures) <= result_keys
        and all(
            isinstance(key, str)
            and (signature is None or _signature_is_comparable(signature))
            for key, signature in signatures.items()
        )
        and all(_validate_expression(expression, config) for expression in expressions)
        and all(
            candidate is None
            or (
                _validate_candidate(candidate, config)
                and expression_key(candidate.expression)
                == expression_key(expressions[index])
            )
            for index, candidate in enumerate(results)
        )
    )


def _valid_string_set(value) -> bool:
    return isinstance(value, set) and all(isinstance(item, str) for item in value)


def _validate_work(
    work,
    stage_index: int,
    generation: int,
    config: HierarchicalGPConfig,
) -> bool:
    if work is None:
        return True
    if not isinstance(work, dict) or work.get("stage_index") != stage_index:
        return False
    kind = work.get("kind")
    if kind == "warm_start":
        return (
            generation == 0
            and _valid_int(work.get("round"), 0)
            and _validate_batch(work.get("batch"), config)
            and all(
                _valid_string_set(work.get(key))
                for key in ("seen_keys", "accepted_keys", "attempted_features")
            )
            and _validate_qd_stats(work.get("initial_stats"))
            and _validate_qd_stats(work.get("fill_stats"))
            and _valid_int(work.get("initial_valid"), 0)
            and _finite_number(work.get("initial_parent_cap"))
            and _valid_int(work.get("fill_proposals"), 0)
            and isinstance(work.get("space_exhausted"), bool)
        )
    if kind == "generation":
        phase = work.get("phase")
        families = work.get("families")
        child_pairs = work.get("child_pairs")
        family_population = work.get("family_population")
        if not (
            work.get("generation") == generation
            and generation < config.generations_per_stage
            and phase in {"children", "immigrants"}
            and _validate_batch(work.get("batch"), config)
            and isinstance(families, tuple)
            and isinstance(child_pairs, tuple)
            and len(families) == len(child_pairs)
            and all(
                isinstance(family, tuple)
                and len(family) == 2
                and all(_validate_candidate(candidate, config) for candidate in family)
                for family in families
            )
            and all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(_validate_expression(expression, config) for expression in pair)
                for pair in child_pairs
            )
            and _validate_qd_stats(work.get("stats"))
            and all(
                _valid_string_set(work.get(key))
                for key in (
                    "accepted_child_keys",
                    "immigrant_seen_keys",
                    "accepted_immigrant_keys",
                    "attempted_features",
                )
            )
            and _valid_int(work.get("immigrant_target"), 0, config.population_size)
            and _valid_int(work.get("immigrant_round"), 0)
            and isinstance(work.get("selection"), dict)
            and set(work["selection"]) == {"tournament", "lexicase"}
            and all(_valid_int(value, 0) for value in work["selection"].values())
            and _finite_number(work.get("parent_cap"))
            and _valid_int(work.get("parent_pool"), 1, config.parent_pool_size)
            and _valid_int(work.get("family_replacements"), 0, len(families))
            and isinstance(work.get("immigrant_space_exhausted"), bool)
        ):
            return False
        if phase == "children":
            return work.get("batch") is not None and family_population is None
        return (
            isinstance(family_population, list)
            and len(family_population) == config.population_size
            and all(
                _validate_candidate(candidate, config)
                for candidate in family_population
            )
        )
    return False


def _validate_history(
    history, config: HierarchicalGPConfig, *, complete: bool = False
) -> bool:
    if not isinstance(history, (list, tuple)):
        return False
    expected = []
    for stage, depth in enumerate(config.stage_depths, start=1):
        expected.append(("warm_start_completed", stage, depth, 0))
        expected.extend(
            ("generation_completed", stage, depth, generation)
            for generation in range(1, config.generations_per_stage + 1)
        )
    actual = []
    for row in history:
        if not isinstance(row, dict):
            return False
        identity = (
            row.get("event"),
            row.get("stage"),
            row.get("depth"),
            row.get("generation"),
        )
        if (
            not _valid_int(row.get("population"), config.population_size)
            or row["population"] != config.population_size
        ):
            return False
        actual.append(identity)
    return actual == expected if complete else actual == expected[: len(actual)]


def _validate_in_progress_state(state, config: HierarchicalGPConfig) -> None:
    required = {
        "stage_index",
        "generation",
        "population",
        "gene_pools",
        "archive",
        "cache",
        "history",
        "rng_state",
        "work",
        "operator_diagnostics",
    }
    if not required <= state.keys():
        raise _checkpoint_error()
    stage_index = state["stage_index"]
    generation = state["generation"]
    population = state["population"]
    gene_pools = state["gene_pools"]
    archive = state["archive"]
    cache = state["cache"]
    history = state["history"]
    operator_diagnostics = state["operator_diagnostics"]
    if not _valid_int(stage_index, 0, len(config.stage_depths)):
        raise _checkpoint_error()
    if not _valid_int(generation, 0, config.generations_per_stage):
        raise _checkpoint_error()
    if not isinstance(population, list) or not all(
        _validate_candidate(candidate, config) for candidate in population
    ):
        raise _checkpoint_error()
    if population and len(population) != config.population_size:
        raise _checkpoint_error()
    if not isinstance(gene_pools, list) or len(gene_pools) != stage_index:
        raise _checkpoint_error()
    if not all(
        isinstance(gene_pool, tuple)
        and 0 < len(gene_pool) <= config.gene_pool_size
        and all(
            _validate_candidate(candidate, config)
            and candidate.depth <= config.stage_depths[index]
            for candidate in gene_pool
        )
        for index, gene_pool in enumerate(gene_pools)
    ):
        raise _checkpoint_error()
    active_depth = config.stage_depths[min(stage_index, len(config.stage_depths) - 1)]
    if any(candidate.depth > active_depth for candidate in population):
        raise _checkpoint_error()
    if not isinstance(archive, PCAQDArchive) or not isinstance(
        archive._records, dict
    ):
        raise _checkpoint_error()
    if (
        archive.similarity_threshold != config.pca_similarity_threshold
        or archive.replacement_margin != config.qd_replacement_rankic_margin
        or any(
            not _validate_qd_record(record) or key != record.key
            for key, record in archive._records.items()
        )
        or any(
            record.stage_depth > active_depth
            for record in archive._records.values()
        )
    ):
        raise _checkpoint_error()
    if not isinstance(cache, ExpressionCache) or not isinstance(
        cache._values, dict
    ):
        raise _checkpoint_error()
    if not _valid_int(cache.hits, 0) or any(
        not isinstance(key, str)
        or not _validate_candidate(candidate, config)
        or key != expression_key(candidate.expression)
        or candidate.depth > active_depth
        for key, candidate in cache._values.items()
    ):
        raise _checkpoint_error()
    if not _validate_history(history, config):
        raise _checkpoint_error()
    if not isinstance(operator_diagnostics, dict) or any(
        not _validate_operator_diagnostic(operator, diagnostic)
        for operator, diagnostic in operator_diagnostics.items()
    ):
        raise _checkpoint_error()
    if any(
        cache._values.get(expression_key(candidate.expression)) != candidate
        for candidate in (
            population
            + [candidate for gene_pool in gene_pools for candidate in gene_pool]
            + [
                diagnostic.candidate
                for diagnostic in operator_diagnostics.values()
            ]
        )
    ):
        raise _checkpoint_error()
    if any(
        diagnostic.stage_depth > active_depth
        for diagnostic in operator_diagnostics.values()
    ):
        raise _checkpoint_error()
    if any(
        key not in cache._values
        or cache._values[key].expression != record.expression
        or cache._values[key].abs_rank_ic != record.abs_rank_ic
        or cache._values[key].nodes != record.nodes
        for key, record in archive._records.items()
    ):
        raise _checkpoint_error()
    try:
        random.Random().setstate(state["rng_state"])
    except (TypeError, ValueError):
        raise _checkpoint_error()
    if not _validate_work(state["work"], stage_index, generation, config):
        raise _checkpoint_error()
    completed_rows = stage_index * (1 + config.generations_per_stage)
    if population:
        completed_rows += 1 + generation
    if len(history) != completed_rows:
        raise _checkpoint_error()
    if not population and generation != 0:
        raise _checkpoint_error()
    if state["work"] is not None and state["work"]["kind"] == "warm_start":
        if population:
            raise _checkpoint_error()
    if stage_index == len(config.stage_depths) and (
        population or generation != 0 or state["work"] is not None
    ):
        raise _checkpoint_error()
    if state["work"] is not None and state["work"]["kind"] == "generation":
        if len(population) != config.population_size:
            raise _checkpoint_error()


def _validate_completed_result(result, config: HierarchicalGPConfig) -> None:
    if not isinstance(result, HierarchicalResult):
        raise _checkpoint_error()
    if (
        not isinstance(result.stages, tuple)
        or len(result.stages) != len(config.stage_depths)
        or any(
            not isinstance(stage, StageResult)
            or stage.depth != depth
            or stage.generations_completed != config.generations_per_stage
            or not isinstance(stage.gene_pool, tuple)
            or not stage.gene_pool
            or len(stage.gene_pool) > config.gene_pool_size
            or not all(
                _validate_candidate(candidate, config)
                and candidate.depth <= stage.depth
                for candidate in stage.gene_pool
            )
            for stage, depth in zip(result.stages, config.stage_depths)
        )
    ):
        raise _checkpoint_error()
    if (
        not isinstance(result.archive_records, tuple)
        or not all(_validate_qd_record(record) for record in result.archive_records)
        or tuple(record.key for record in result.archive_records)
        != tuple(sorted({record.key for record in result.archive_records}))
    ):
        raise _checkpoint_error()
    if (
        not isinstance(result.archive_candidates, tuple)
        or not all(
            _validate_candidate(candidate, config)
            for candidate in result.archive_candidates
        )
        or tuple(
            expression_key(candidate.expression)
            for candidate in result.archive_candidates
        )
        != tuple(record.key for record in result.archive_records)
        or any(
            candidate.expression != record.expression
            or candidate.abs_rank_ic != record.abs_rank_ic
            or candidate.nodes != record.nodes
            or candidate.depth > record.stage_depth
            for record, candidate in zip(
                result.archive_records, result.archive_candidates
            )
        )
    ):
        raise _checkpoint_error()
    if not isinstance(result.operator_diagnostics, tuple) or any(
        not isinstance(diagnostic, OperatorDiagnostic)
        or not _validate_operator_diagnostic(diagnostic.operator, diagnostic)
        for diagnostic in result.operator_diagnostics
    ):
        raise _checkpoint_error()
    diagnostic_operators = tuple(
        diagnostic.operator for diagnostic in result.operator_diagnostics
    )
    if diagnostic_operators != tuple(sorted(set(diagnostic_operators))):
        raise _checkpoint_error()
    if not _validate_history(result.history, config, complete=True):
        raise _checkpoint_error()
    if not _valid_int(result.evaluated_count, 0) or not _valid_int(
        result.cache_hits, 0
    ):
        raise _checkpoint_error()


def validate_hierarchical_checkpoint_payload(
    payload,
    config: HierarchicalGPConfig,
    seed: int,
    *,
    completed: bool,
    backend_identity=None,
) -> None:
    """Validate one V4 checkpoint payload without mutating engine state."""
    _validate_config(config)
    _validate_checkpoint(payload, _config_fingerprint(config), seed)
    if payload.get("batch_backend_identity") != backend_identity:
        raise RuntimeError("checkpoint batch backend or input data changed; use a new output directory")
    if completed:
        if "result" not in payload:
            raise _checkpoint_error()
        _validate_loaded_structure(
            _validate_completed_result, payload["result"], config
        )
    else:
        _validate_loaded_structure(_validate_in_progress_state, payload, config)


def _evaluation_config(config: HierarchicalGPConfig) -> _EvaluationConfig:
    return _EvaluationConfig(
        max_depth=max(config.stage_depths),
        max_nodes=config.max_nodes,
        min_valid_time_ratio=config.min_valid_time_ratio,
        max_missing_ratio=config.max_missing_ratio,
        stability_segments=config.stability_segments,
        stability_penalty=config.stability_penalty,
        complexity_penalty=config.complexity_penalty,
        minimum_same_sign_ratio=config.minimum_same_sign_ratio,
        inconsistent_sign_multiplier=config.inconsistent_sign_multiplier,
    )


def _ordered_candidates(candidates) -> list[Candidate]:
    best_by_key = {}
    for candidate in candidates:
        key = expression_key(candidate.expression)
        incumbent = best_by_key.get(key)
        if incumbent is None or candidate.fitness > incumbent.fitness:
            best_by_key[key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda item: (
            -item.fitness,
            item.nodes,
            expression_key(item.expression),
        ),
    )


def _ranked_population(candidates) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.fitness,
            item.nodes,
            expression_key(item.expression),
        ),
    )


def _normalized_unique(expressions, depth: int, max_nodes: int) -> tuple[Any, ...]:
    normalized = []
    seen = set()
    for expression in expressions:
        try:
            candidate = normalize_expression(expression)
        except (TypeError, ValueError):
            continue
        if expression_depth(candidate) > depth or expression_nodes(candidate) > max_nodes:
            continue
        key = expression_key(candidate)
        if key not in seen:
            seen.add(key)
            normalized.append(candidate)
    return tuple(normalized)


def _request_unseen_proposals(
    *,
    features,
    windows,
    stage_depth: int,
    needed: int,
    seed: int,
    gene_pools,
    seen_keys: set[str],
    max_nodes: int,
) -> tuple[tuple[Any, ...], bool]:
    """Return unseen bounded proposals or prove the finite space exhausted."""
    if needed <= 0:
        return (), False
    selected = []
    while len(selected) < needed:
        request_size = len(seen_keys) + (needed - len(selected))
        proposals = balanced_stage_proposals(
            features,
            windows,
            stage_depth=stage_depth,
            size=request_size,
            seed=seed,
            gene_pools=gene_pools,
        )
        finite_space_complete = len(proposals) < request_size
        for expression in proposals:
            key = expression_key(expression)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if expression_nodes(expression) <= max_nodes:
                selected.append(expression)
                if len(selected) == needed:
                    break
        if len(selected) == needed:
            return tuple(selected), False
        if finite_space_complete:
            return tuple(selected), True
    return tuple(selected), False


def _proposal_seed(seed: int, stage_index: int, round_number: int, offset=0) -> int:
    return seed + (stage_index + 1) * 1_000_003 + offset + round_number


def _signature_is_comparable(signature) -> bool:
    try:
        values = np.asarray(signature, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return False
    if values.size < 2 or not np.isfinite(values).all():
        return False
    centered = values - values.mean()
    return bool(np.linalg.norm(centered) > 0.0)


def _numpy_pca_seed(seed: int) -> int:
    """Map any Python integer seed into NumPy's unsigned 64-bit seed domain."""
    return seed % _NUMPY_SEED_MODULUS


def _empty_qd_stats() -> dict[str, Any]:
    return {
        "valid_candidates": 0,
        "qd_admitted": 0,
        "qd_rejected": 0,
        "qd_replaced": 0,
        "pca_failures": 0,
        "similarities": [],
    }


def _merge_qd_stats(target: dict, source: dict) -> None:
    for key in (
        "valid_candidates",
        "qd_admitted",
        "qd_rejected",
        "qd_replaced",
        "pca_failures",
    ):
        target[key] += source[key]
    target["similarities"].extend(source["similarities"])


def _qd_summary(stats: dict) -> dict[str, Any]:
    similarities = np.asarray(stats["similarities"], dtype=float)
    finite = similarities[np.isfinite(similarities)]
    return {
        key: stats[key]
        for key in (
            "valid_candidates",
            "qd_admitted",
            "qd_rejected",
            "qd_replaced",
            "pca_failures",
        )
    } | {
        "max_similarity": float(finite.max()) if finite.size else None,
        "median_similarity": float(np.median(finite)) if finite.size else None,
    }


def _archive_records(archive: PCAQDArchive) -> tuple[QDRecord, ...]:
    return tuple(archive._records[key] for key in archive)


def _frequencies(candidates) -> tuple[dict[str, int], dict[str, int]]:
    feature_counts = {}
    operator_counts = {}
    for candidate in candidates:
        for feature in terminal_names(candidate.expression):
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        for operator in expression_operators(candidate.expression):
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
    return (
        dict(sorted(feature_counts.items())),
        dict(sorted(operator_counts.items())),
    )


def _update_operator_diagnostics(
    diagnostics: dict[str, OperatorDiagnostic],
    candidate: Candidate,
    stage_depth: int,
    signature: np.ndarray,
) -> None:
    comparable_signature = _ComparableSignature(signature)
    new_key = (
        -candidate.abs_rank_ic,
        candidate.nodes,
        expression_key(candidate.expression),
    )
    for operator in sorted(expression_operators(candidate.expression)):
        incumbent = diagnostics.get(operator)
        if incumbent is None:
            diagnostics[operator] = OperatorDiagnostic(
                operator, candidate, stage_depth, comparable_signature
            )
            continue
        incumbent_key = (
            -incumbent.candidate.abs_rank_ic,
            incumbent.candidate.nodes,
            expression_key(incumbent.candidate.expression),
        )
        if new_key < incumbent_key:
            diagnostics[operator] = OperatorDiagnostic(
                operator, candidate, stage_depth, comparable_signature
            )


def run_hierarchical_evolution(
    data_dict,
    target,
    feature_profile,
    config: HierarchicalGPConfig,
    seed: int,
    progress_callback=None,
    checkpoint_dir=None,
    batch_backend=None,
) -> HierarchicalResult:
    """Run deterministic depth-wise PCA-QD genetic-program evolution."""
    _validate_config(config)
    fingerprint = _config_fingerprint(config)
    backend_identity = None
    if batch_backend is not None:
        batch_backend.validate_inputs(data_dict, target, config)
        backend_identity = batch_backend.identity
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    completed_path = checkpoint_dir / "completed.pkl" if checkpoint_dir else None
    state_path = checkpoint_dir / "run_state.pkl" if checkpoint_dir else None

    def load_pickle(path):
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError) as error:
            raise RuntimeError(
                "checkpoint format is incompatible; use a new output directory"
            ) from error

    if completed_path is not None and completed_path.exists():
        completed = load_pickle(completed_path)
        validate_hierarchical_checkpoint_payload(
            completed, config, seed, completed=True, backend_identity=backend_identity
        )
        return completed["result"]

    features = tuple(feature_profile["feature"].tolist())
    if not features:
        raise ValueError("feature profile must contain at least one feature")
    missing_features = set(features) - set(data_dict)
    if missing_features:
        raise ValueError(f"missing feature panels: {sorted(missing_features)}")
    probabilities = tuple(feature_sampling_probabilities(feature_profile))
    rng = random.Random(seed)
    pca_seed = _numpy_pca_seed(seed)

    if state_path is not None and state_path.exists():
        state = load_pickle(state_path)
        validate_hierarchical_checkpoint_payload(
            state, config, seed, completed=False, backend_identity=backend_identity
        )
        try:
            stage_index = state["stage_index"]
            generation = state["generation"]
            population = list(state["population"])
            gene_pools = list(state["gene_pools"])
            archive = state["archive"]
            cache = state["cache"]
            history = list(state["history"])
            operator_diagnostics = dict(state["operator_diagnostics"])
            work = state.get("work")
            rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "checkpoint format is incompatible; use a new output directory"
            ) from error
    else:
        stage_index = 0
        generation = 0
        population = []
        gene_pools = []
        archive = PCAQDArchive(
            config.pca_similarity_threshold,
            config.qd_replacement_rankic_margin,
        )
        cache = ExpressionCache()
        history = []
        operator_diagnostics = {}
        work = None

    if not isinstance(archive, PCAQDArchive) or not isinstance(cache, ExpressionCache):
        raise RuntimeError(
            "checkpoint format is incompatible; use a new output directory"
        )
    if not 0 <= stage_index <= len(config.stage_depths):
        raise RuntimeError(
            "checkpoint format is incompatible; use a new output directory"
        )

    def report(event, **values):
        if progress_callback is not None:
            progress_callback({"event": event, **values})

    def save_state():
        if state_path is None:
            return
        _atomic_pickle(
            state_path,
            {
                "version": CHECKPOINT_VERSION,
                "config_fingerprint": fingerprint,
                "batch_backend_identity": backend_identity,
                "seed": seed,
                "stage_index": stage_index,
                "generation": generation,
                "population": population,
                "gene_pools": gene_pools,
                "archive": archive,
                "cache": cache,
                "history": history,
                "rng_state": rng.getstate(),
                "work": work,
                "operator_diagnostics": operator_diagnostics,
            },
        )

    evaluation_config = _evaluation_config(config)
    save_state()
    pool = None
    if config.max_workers > 1 and batch_backend is None:
        context = mp.get_context("fork")
        pool = context.Pool(
            processes=config.max_workers,
            initializer=_initialize_worker,
            initargs=(data_dict, target, evaluation_config),
        )

    def evaluate_batch(batch, depth, stage_number, generation_number, phase):
        expressions = batch["expressions"]
        precomputed_signatures = batch.setdefault("signatures", {})
        total = len(expressions)
        total_chunks = math.ceil(total / config.progress_chunk_size) if total else 0
        while batch["cursor"] < total:
            start = batch["cursor"]
            stop = min(start + config.progress_chunk_size, total)
            chunk = expressions[start:stop]
            results = []
            if batch_backend is not None:
                pending = []
                resolved = {}
                for expression in chunk:
                    key = expression_key(expression)
                    candidate = cache.get(key)
                    if candidate is None:
                        pending.append(expression)
                    else:
                        resolved[key] = candidate
                returned = batch_backend.evaluate(pending, evaluation_config, config, pca_seed)
                if len(returned) != len(pending):
                    raise RuntimeError("batch backend result count mismatch")
                for expression, (candidate, signature) in zip(pending, returned):
                    key = expression_key(expression)
                    if candidate is not None:
                        if expression_key(candidate.expression) != key:
                            raise RuntimeError("batch backend candidate order mismatch")
                        cache.put(key, candidate)
                        resolved[key] = candidate
                        precomputed_signatures[key] = signature
                results = [resolved.get(expression_key(expression)) for expression in chunk]
            elif pool is None:
                for expression in chunk:
                    results.append(
                        _evaluate(
                            expression,
                            data_dict,
                            target,
                            evaluation_config,
                            cache,
                        )
                    )
            else:
                resolved = {}
                pending = []
                for expression in chunk:
                    key = expression_key(expression)
                    candidate = cache.get(key)
                    if candidate is None:
                        pending.append(expression)
                    else:
                        resolved[key] = candidate
                tasks = (
                    (
                        expression,
                        config.pca_sample_step,
                        config.pca_power_iterations,
                        pca_seed,
                    )
                    for expression in pending
                )
                for candidate, signature, _ in pool.imap(
                    _worker_evaluate_with_signature, tasks, chunksize=1
                ):
                    if candidate is not None:
                        key = expression_key(candidate.expression)
                        cache.put(key, candidate)
                        resolved[key] = candidate
                        precomputed_signatures[key] = signature
                results = [
                    resolved.get(expression_key(expression)) for expression in chunk
                ]
            batch["results"].extend(results)
            batch["cursor"] = stop
            save_state()
            report(
                "chunk_completed",
                stage=stage_number,
                depth=depth,
                generation=generation_number,
                phase=phase,
                completed=stop,
                total=total,
                chunk=math.ceil(stop / config.progress_chunk_size),
                chunks=total_chunks,
                valid_candidates=sum(
                    candidate is not None for candidate in batch["results"]
                ),
            )

    def candidate_signatures(candidates, depth, precomputed=None):
        raw_signatures = dict(precomputed or {})
        ordered = _ordered_candidates(
            candidate for candidate in candidates if candidate is not None
        )
        for candidate in ordered:
            key = expression_key(candidate.expression)
            record = archive._records.get(key)
            if key not in raw_signatures and record is not None:
                raw_signatures[key] = record.signature
        missing = [
            candidate
            for candidate in ordered
            if expression_key(candidate.expression) not in raw_signatures
        ]
        if batch_backend is not None:
            returned = batch_backend.evaluate([item.expression for item in missing], evaluation_config, config, pca_seed)
            if len(returned) != len(missing):
                raise RuntimeError("batch backend signature count mismatch")
            computed = [(expression_key(item.expression), signature) for item, (_, signature) in zip(missing, returned)]
        elif pool is None:
            computed = []
            for candidate in missing:
                try:
                    values = evaluate_expression(candidate.expression, data_dict)
                    signature_values = factor_signature(
                        values,
                        sample_step=config.pca_sample_step,
                        power_iterations=config.pca_power_iterations,
                        seed=pca_seed,
                    )
                except Exception:
                    signature_values = None
                computed.append((expression_key(candidate.expression), signature_values))
        else:
            tasks = (
                (
                    candidate.expression,
                    config.pca_sample_step,
                    config.pca_power_iterations,
                    pca_seed,
                )
                for candidate in missing
            )
            computed = [
                (key, signature_values)
                for key, signature_values, _ in pool.imap(
                    _worker_signature, tasks, chunksize=1
                )
            ]

        for candidate, (key, signature_values) in zip(missing, computed):
            expected_key = expression_key(candidate.expression)
            if key != expected_key:
                raise RuntimeError("parallel signature result order mismatch")
            raw_signatures[key] = signature_values

        signatures = {}
        for candidate in ordered:
            key = expression_key(candidate.expression)
            signature_values = raw_signatures.get(key)
            try:
                if not _signature_is_comparable(signature_values):
                    raise ValueError("incomparable PCA-QD signature")
                signature = _ComparableSignature(signature_values)
            except Exception:
                signatures[key] = None
                continue
            signatures[key] = signature

            _update_operator_diagnostics(
                operator_diagnostics,
                candidate,
                depth,
                signature,
            )
        return signatures

    def gate_candidates(candidates, depth, signatures):
        ordered = _ordered_candidates(
            candidate for candidate in candidates if candidate is not None
        )
        stats = _empty_qd_stats()
        stats["valid_candidates"] = len(ordered)
        admitted = {}
        replaced_keys = set()
        comparable = []
        for candidate in ordered:
            key = expression_key(candidate.expression)
            signature = signatures.get(key)
            if signature is None:
                stats["pca_failures"] += 1
                stats["qd_rejected"] += 1
                continue
            comparable.append(
                (candidate, QDRecord(
                    key=key,
                    expression=candidate.expression,
                    abs_rank_ic=candidate.abs_rank_ic,
                    nodes=candidate.nodes,
                    stage_depth=depth,
                    signature=signature,
                ))
            )
        records = [record for _, record in comparable]
        if batch_backend is not None:
            decisions = batch_backend.consider(archive, records, config, pca_seed)
        else:
            decisions = archive.consider_batch(
                records, block_size=config.pca_qd_block_size, backend=config.pca_qd_backend,
            )
        for (candidate, _), decision in zip(comparable, decisions):
            key = expression_key(candidate.expression)
            if math.isfinite(decision.max_similarity):
                stats["similarities"].append(decision.max_similarity)
            if decision.action == "reject":
                stats["qd_rejected"] += 1
                continue
            if decision.action == "replace":
                stats["qd_replaced"] += 1
                if decision.replaced_key is not None:
                    replaced_keys.add(decision.replaced_key)
                    admitted.pop(decision.replaced_key, None)
            else:
                stats["qd_admitted"] += 1
            admitted[key] = candidate
        return stats, admitted, replaced_keys

    def candidates_for_keys(keys):
        return _ordered_candidates(
            cache._values[key]
            for key in keys
            if key in archive._records and key in cache._values
        )

    def prior_gene_expressions():
        return tuple(
            tuple(candidate.expression for candidate in gene_pool)
            for gene_pool in gene_pools
        )

    try:
        while stage_index < len(config.stage_depths):
            depth = config.stage_depths[stage_index]
            stage_number = stage_index + 1

            if not population:
                if work is None:
                    work = {
                        "kind": "warm_start",
                        "stage_index": stage_index,
                        "round": 0,
                        "batch": None,
                        "seen_keys": set(),
                        "accepted_keys": set(),
                        "attempted_features": set(),
                        "initial_stats": _empty_qd_stats(),
                        "fill_stats": _empty_qd_stats(),
                        "initial_valid": 0,
                        "initial_parent_cap": config.parent_feature_cap_shares[-1],
                        "fill_proposals": 0,
                        "space_exhausted": False,
                    }
                    save_state()
                    report(
                        "stage_started",
                        stage=stage_number,
                        depth=depth,
                        stages=len(config.stage_depths),
                    )
                elif (
                    work.get("kind") != "warm_start"
                    or work.get("stage_index") != stage_index
                ):
                    raise RuntimeError(
                        "checkpoint format is incompatible; use a new output directory"
                    )

                requested = config.population_size * config.warm_start_multiplier
                while len(candidates_for_keys(work["accepted_keys"])) < config.population_size:
                    if work["batch"] is None:
                        if work["space_exhausted"]:
                            raise RuntimeError(
                                f"depth {depth} exhausted its finite proposal space before filling the population"
                            )
                        batch_size = requested if work["round"] == 0 else config.population_size
                        proposals, exhausted = _request_unseen_proposals(
                            features=features,
                            windows=config.windows,
                            stage_depth=depth,
                            needed=batch_size,
                            seed=_proposal_seed(seed, stage_index, work["round"]),
                            gene_pools=prior_gene_expressions(),
                            seen_keys=work["seen_keys"],
                            max_nodes=config.max_nodes,
                        )
                        work["space_exhausted"] = exhausted
                        work["attempted_features"].update(
                            *(
                                terminal_names(expression)
                                for expression in proposals
                            )
                        )
                        if not proposals:
                            raise RuntimeError(
                                f"depth {depth} exhausted its finite proposal space before filling the population"
                            )
                        if work["round"] > 0:
                            work["fill_proposals"] += len(proposals)
                        work["batch"] = {
                            "expressions": proposals,
                            "cursor": 0,
                            "results": [],
                            "signatures": {},
                        }
                        save_state()

                    evaluate_batch(
                        work["batch"], depth, stage_number, 0, "warm_start"
                    )
                    valid = [
                        candidate
                        for candidate in work["batch"]["results"]
                        if candidate is not None
                    ]
                    signatures = candidate_signatures(
                        valid, depth, work["batch"].get("signatures")
                    )
                    capped, used_cap = _build_capped_parent_pool(
                        [(candidate, candidate.fitness) for candidate in valid],
                        min(config.population_size, len(valid)),
                        config.parent_feature_cap_shares,
                    )
                    gated_stats, admitted, replaced_keys = gate_candidates(
                        [candidate for candidate, _ in capped],
                        depth,
                        signatures,
                    )
                    work["accepted_keys"].difference_update(replaced_keys)
                    work["accepted_keys"].update(admitted)
                    if work["round"] == 0:
                        work["initial_valid"] = len(valid)
                        work["initial_parent_cap"] = used_cap
                        _merge_qd_stats(work["initial_stats"], gated_stats)
                    else:
                        _merge_qd_stats(work["fill_stats"], gated_stats)
                    work["batch"] = None
                    work["round"] += 1
                    save_state()
                    report(
                        "qd_batch_completed",
                        stage=stage_number,
                        depth=depth,
                        generation=0,
                        phase="warm_start",
                        archive_size=len(archive),
                        **_qd_summary(gated_stats),
                    )

                population = candidates_for_keys(work["accepted_keys"])[
                    : config.population_size
                ]
                initial_summary = _qd_summary(work["initial_stats"])
                fill_summary = _qd_summary(work["fill_stats"])
                feature_counts, operator_counts = _frequencies(population)
                warm_row = {
                    "event": "warm_start_completed",
                    "stage": stage_number,
                    "depth": depth,
                    "generation": 0,
                    "warm_start_proposals": requested,
                    "warm_start_valid": work["initial_valid"],
                    "qd_admitted": initial_summary["qd_admitted"],
                    "qd_rejected": initial_summary["qd_rejected"],
                    "qd_replaced": initial_summary["qd_replaced"],
                    "pca_failures": initial_summary["pca_failures"],
                    "fill_proposals": work["fill_proposals"],
                    "fill_qd_admitted": fill_summary["qd_admitted"],
                    "fill_qd_replaced": fill_summary["qd_replaced"],
                    "parent_cap": work["initial_parent_cap"],
                    "population": len(population),
                    "archive_size": len(archive),
                    "attempted_features": tuple(sorted(work["attempted_features"])),
                    "feature_frequencies": feature_counts,
                    "operator_frequencies": operator_counts,
                    "best_fitness": population[0].fitness,
                    "best_abs_rank_ic": max(
                        candidate.abs_rank_ic for candidate in population
                    ),
                }
                history.append(warm_row)
                work = None
                save_state()
                report(**{"event": "warm_start_completed", **warm_row})

            while generation < config.generations_per_stage:
                generation_number = generation + 1
                if work is None:
                    scored = [(candidate, candidate.fitness) for candidate in population]
                    target_pool_size = min(config.parent_pool_size, len(scored))
                    parent_pool, used_cap = _build_capped_parent_pool(
                        scored,
                        target_pool_size,
                        config.parent_feature_cap_shares,
                    )
                    if not parent_pool:
                        parent_pool = scored
                    by_key = {
                        expression_key(candidate.expression): candidate
                        for candidate, _ in parent_pool
                    }
                    counters = {"tournament": 0, "lexicase": 0}
                    selected = []
                    parent_count = 2 * math.ceil(config.population_size / 2)
                    for _ in range(parent_count):
                        if rng.random() < config.tournament_selection_probability:
                            counters["tournament"] += 1
                            expression = _tournament(
                                parent_pool,
                                rng,
                                min(config.tournament_size, len(parent_pool)),
                            )
                        else:
                            counters["lexicase"] += 1
                            expression = _epsilon_lexicase(
                                parent_pool, rng, config.lexicase_epsilon_scale
                            )
                        selected.append(by_key[expression_key(expression)])
                    indices = list(range(len(selected)))
                    rng.shuffle(indices)
                    selected = [selected[index] for index in indices]
                    ranked_parents = _ordered_candidates(
                        candidate for candidate, _ in parent_pool
                    )
                    families = []
                    child_pairs = []
                    for index in range(0, len(selected), 2):
                        left = selected[index]
                        right = selected[index + 1]
                        if (
                            expression_key(left.expression)
                            == expression_key(right.expression)
                            and len(ranked_parents) > 1
                        ):
                            right = next(
                                candidate
                                for candidate in ranked_parents
                                if expression_key(candidate.expression)
                                != expression_key(left.expression)
                            )
                        families.append((left, right))
                        children = (left.expression, right.expression)
                        if rng.random() < config.crossover_probability:
                            children = same_depth_crossover(
                                left.expression,
                                right.expression,
                                rng,
                                stage_depth=depth,
                                max_nodes=config.max_nodes,
                            )
                        children = tuple(
                            bounded_mutation(
                                child,
                                rng,
                                features,
                                probabilities,
                                config.windows,
                                stage_depth=depth,
                                max_nodes=config.max_nodes,
                                feature_mutation_probability=config.feature_mutation_probability,
                                operator_mutation_probability=config.operator_mutation_probability,
                                window_mutation_probability=config.window_mutation_probability,
                                subtree_mutation_probability=config.subtree_mutation_probability,
                            )
                            for child in children
                        )
                        child_pairs.append(children)
                    child_expressions = _normalized_unique(
                        (
                            child
                            for pair in child_pairs
                            for child in pair
                        ),
                        depth,
                        config.max_nodes,
                    )
                    work = {
                        "kind": "generation",
                        "stage_index": stage_index,
                        "generation": generation,
                        "phase": "children",
                        "families": tuple(families),
                        "child_pairs": tuple(child_pairs),
                        "batch": {
                            "expressions": child_expressions,
                            "cursor": 0,
                            "results": [],
                            "signatures": {},
                        },
                        "accepted_child_keys": set(),
                        "family_population": None,
                        "immigrant_target": int(
                            config.population_size * config.random_immigrant_ratio
                        ),
                        "immigrant_round": 0,
                        "immigrant_seen_keys": set(),
                        "accepted_immigrant_keys": set(),
                        "attempted_features": set().union(
                            *(
                                terminal_names(expression)
                                for expression in child_expressions
                            )
                        )
                        if child_expressions
                        else set(),
                        "stats": _empty_qd_stats(),
                        "selection": counters,
                        "parent_cap": used_cap,
                        "parent_pool": len(parent_pool),
                        "family_replacements": 0,
                        "immigrant_space_exhausted": False,
                    }
                    save_state()
                    report(
                        "generation_started",
                        stage=stage_number,
                        depth=depth,
                        generation=generation_number,
                        generations=config.generations_per_stage,
                        population=len(population),
                    )
                elif (
                    work.get("kind") != "generation"
                    or work.get("stage_index") != stage_index
                    or work.get("generation") != generation
                ):
                    raise RuntimeError(
                        "checkpoint format is incompatible; use a new output directory"
                    )

                if work["phase"] == "children":
                    evaluate_batch(
                        work["batch"],
                        depth,
                        stage_number,
                        generation_number,
                        "children",
                    )
                    signatures = candidate_signatures(
                        work["batch"]["results"],
                        depth,
                        work["batch"].get("signatures"),
                    )
                    child_stats, admitted, replaced_keys = gate_candidates(
                        work["batch"]["results"], depth, signatures
                    )
                    work["accepted_child_keys"].difference_update(replaced_keys)
                    work["accepted_child_keys"].update(admitted)
                    _merge_qd_stats(work["stats"], child_stats)

                    admitted_children = {
                        key: cache._values[key]
                        for key in work["accepted_child_keys"]
                        if key in archive._records and key in cache._values
                    }
                    family_population = []
                    family_replacements = 0
                    for parents, child_pair in zip(
                        work["families"], work["child_pairs"]
                    ):
                        children = []
                        for expression in child_pair:
                            key = expression_key(expression)
                            if key in admitted_children:
                                children.append(admitted_children[key])
                        survivors = family_survivors(parents, children, limit=2)
                        parent_keys = {
                            expression_key(candidate.expression) for candidate in parents
                        }
                        survivor_keys = {
                            expression_key(candidate.expression)
                            for candidate in survivors
                        }
                        if survivor_keys != parent_keys:
                            family_replacements += 1
                        family_population.extend(survivors)
                    if len(family_population) < config.population_size:
                        fallback = _ranked_population(population)
                        missing = config.population_size - len(family_population)
                        family_population.extend(
                            fallback[index % len(fallback)]
                            for index in range(missing)
                        )
                    work["family_population"] = family_population[
                        : config.population_size
                    ]
                    work["family_replacements"] = family_replacements
                    work["batch"] = None
                    work["phase"] = "immigrants"
                    save_state()
                    report(
                        "family_replacement_completed",
                        stage=stage_number,
                        depth=depth,
                        generation=generation_number,
                        families=len(work["families"]),
                        replaced=family_replacements,
                    )

                while (
                    work["phase"] == "immigrants"
                    and len(candidates_for_keys(work["accepted_immigrant_keys"]))
                    < work["immigrant_target"]
                ):
                    if work["batch"] is None:
                        if work["immigrant_space_exhausted"]:
                            raise RuntimeError(
                                f"depth {depth} exhausted its finite immigrant proposal space"
                            )
                        proposal_count = len(features) + max(
                            len(HIERARCHICAL_OPERATORS),
                            work["immigrant_target"]
                            * config.warm_start_multiplier,
                        )
                        proposals, exhausted = _request_unseen_proposals(
                            features=features,
                            windows=config.windows,
                            stage_depth=depth,
                            needed=proposal_count,
                            seed=_proposal_seed(
                                seed,
                                stage_index,
                                work["immigrant_round"],
                                offset=100_000_019 + generation * 10_007,
                            ),
                            gene_pools=prior_gene_expressions(),
                            seen_keys=work["immigrant_seen_keys"],
                            max_nodes=config.max_nodes,
                        )
                        work["immigrant_space_exhausted"] = exhausted
                        work["attempted_features"].update(
                            *(
                                terminal_names(expression)
                                for expression in proposals
                            )
                        )
                        if not proposals:
                            raise RuntimeError(
                                f"depth {depth} exhausted its finite immigrant proposal space"
                            )
                        work["batch"] = {
                            "expressions": proposals,
                            "cursor": 0,
                            "results": [],
                            "signatures": {},
                        }
                        save_state()

                    evaluate_batch(
                        work["batch"],
                        depth,
                        stage_number,
                        generation_number,
                        "immigrants",
                    )
                    signatures = candidate_signatures(
                        work["batch"]["results"],
                        depth,
                        work["batch"].get("signatures"),
                    )
                    immigrant_stats, admitted, replaced_keys = gate_candidates(
                        work["batch"]["results"], depth, signatures
                    )
                    work["accepted_immigrant_keys"].difference_update(
                        replaced_keys
                    )
                    work["accepted_immigrant_keys"].update(admitted)
                    _merge_qd_stats(work["stats"], immigrant_stats)
                    work["batch"] = None
                    work["immigrant_round"] += 1
                    save_state()
                    report(
                        "qd_batch_completed",
                        stage=stage_number,
                        depth=depth,
                        generation=generation_number,
                        phase="immigrants",
                        archive_size=len(archive),
                        **_qd_summary(immigrant_stats),
                    )

                immigrants = candidates_for_keys(
                    work["accepted_immigrant_keys"]
                )[: work["immigrant_target"]]
                base_population = _ranked_population(work["family_population"])
                keep = max(0, config.population_size - len(immigrants))
                population = (base_population[:keep] + immigrants)[
                    : config.population_size
                ]
                scores = np.asarray(
                    [candidate.fitness for candidate in population], dtype=float
                )
                rank_ics = np.asarray(
                    [candidate.abs_rank_ic for candidate in population], dtype=float
                )
                feature_counts, operator_counts = _frequencies(population)
                generation_row = {
                    "event": "generation_completed",
                    "stage": stage_number,
                    "depth": depth,
                    "generation": generation_number,
                    "population": len(population),
                    "valid_candidates": work["stats"]["valid_candidates"],
                    "qd_admitted": work["stats"]["qd_admitted"],
                    "qd_rejected": work["stats"]["qd_rejected"],
                    "qd_replaced": work["stats"]["qd_replaced"],
                    "pca_failures": work["stats"]["pca_failures"],
                    "tournament": work["selection"]["tournament"],
                    "lexicase": work["selection"]["lexicase"],
                    "parent_cap": work["parent_cap"],
                    "parent_pool": work["parent_pool"],
                    "parent_child_replacements": work[
                        "family_replacements"
                    ],
                    "random_immigrants": len(immigrants),
                    "archive_size": len(archive),
                    "attempted_features": tuple(
                        sorted(work["attempted_features"])
                    ),
                    "feature_frequencies": feature_counts,
                    "operator_frequencies": operator_counts,
                    "best_fitness": float(scores.max()),
                    "mean_fitness": float(scores.mean()),
                    "best_abs_rank_ic": float(rank_ics.max()),
                    "mean_abs_rank_ic": float(rank_ics.mean()),
                    **{
                        key: value
                        for key, value in _qd_summary(work["stats"]).items()
                        if key in {"max_similarity", "median_similarity"}
                    },
                }
                history.append(generation_row)
                generation += 1
                work = None
                save_state()
                report(**{"event": "generation_completed", **generation_row})

            stage_candidates = _ordered_candidates(
                cache._values[key]
                for key, record in archive._records.items()
                if record.stage_depth == depth and key in cache._values
            )
            gene_pool = tuple(stage_candidates[: config.gene_pool_size])
            if not gene_pool:
                raise RuntimeError(f"depth {depth} produced no QD-admitted gene pool")
            gene_pools.append(gene_pool)
            stage_index += 1
            generation = 0
            population = []
            work = None
            save_state()
            report(
                "stage_completed",
                stage=stage_number,
                depth=depth,
                gene_pool_size=len(gene_pool),
                archive_size=len(archive),
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    archive_records = _archive_records(archive)
    result = HierarchicalResult(
        stages=tuple(
            StageResult(depth, tuple(gene_pool), config.generations_per_stage)
            for depth, gene_pool in zip(config.stage_depths, gene_pools)
        ),
        archive_records=archive_records,
        archive_candidates=tuple(
            cache._values[record.key] for record in archive_records
        ),
        operator_diagnostics=tuple(
            operator_diagnostics[operator]
            for operator in sorted(operator_diagnostics)
        ),
        history=tuple(history),
        evaluated_count=len(cache),
        cache_hits=cache.hits,
    )
    if completed_path is not None:
        _atomic_pickle(
            completed_path,
            {
                "version": CHECKPOINT_VERSION,
                "config_fingerprint": fingerprint,
                "batch_backend_identity": backend_identity,
                "seed": seed,
                "result": result,
            },
        )
    return result
