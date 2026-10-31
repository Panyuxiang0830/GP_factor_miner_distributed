from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

import gp_factor_miner.hierarchical_engine as hierarchical_engine
from gp_factor_miner.expressions import expression_key
from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.hierarchical_engine import run_hierarchical_evolution
from gp_factor_miner.hierarchical_ops import balanced_stage_proposals


def synthetic_problem(feature_count=6):
    rng = np.random.default_rng(2026)
    index = pd.Index(range(201), name="time_id")
    columns = pd.Index(range(8), name="asset_id")
    panels = {
        f"f{feature_index}": pd.DataFrame(
            rng.normal(size=(len(index), len(columns)))
            + np.linspace(-0.2, 0.2, len(columns)),
            index=index,
            columns=columns,
        )
        for feature_index in range(feature_count)
    }
    target = panels["f0"] - 0.4 * panels["f1"]
    profile = pd.DataFrame(
        {
            "feature": list(panels),
            "rank_ic": np.linspace(0.08, -0.04, feature_count),
        }
    )
    return panels, target, profile


def small_config(**changes):
    values = dict(
        population_size=8,
        generations_per_stage=1,
        warm_start_multiplier=2,
        gene_pool_size=6,
        parent_pool_size=6,
        progress_chunk_size=4,
        windows=(1, 2, 3),
    )
    values.update(changes)
    return replace(HierarchicalGPConfig().smoke(), **values)


def interrupt_after_event(event_name):
    interrupted = False

    def callback(event):
        nonlocal interrupted
        if not interrupted and event["event"] == event_name:
            interrupted = True
            raise RuntimeError("simulated interrupt")

    return callback


def candidate(expression, abs_rank_ic, nodes):
    return Candidate(
        expression=expression,
        rank_ic=abs_rank_ic,
        abs_rank_ic=abs_rank_ic,
        valid_ratio=1.0,
        missing_ratio=0.0,
        depth=1,
        nodes=nodes,
        fitness=abs_rank_ic,
        segment_rank_ics=(abs_rank_ic,) * 4,
    )


def contains_factor_panel(value, seen=None):
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.ndarray)):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        return any(
            contains_factor_panel(item, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set)):
        return any(contains_factor_panel(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return contains_factor_panel(vars(value), seen)
    return False


def test_smoke_runs_all_depths_and_keeps_raw_features_available():
    data, target, profile = synthetic_problem(feature_count=6)
    config = HierarchicalGPConfig().smoke()

    result = run_hierarchical_evolution(
        data, target, profile, config, seed=2026
    )

    assert [stage.depth for stage in result.stages] == [1, 2, 3]
    assert all(stage.gene_pool for stage in result.stages)
    assert tuple(
        diagnostic.operator for diagnostic in result.operator_diagnostics
    ) == tuple(
        sorted(diagnostic.operator for diagnostic in result.operator_diagnostics)
    )
    assert all(
        isinstance(diagnostic.signature, np.ndarray)
        and diagnostic.signature.size >= 2
        and np.isfinite(diagnostic.signature).all()
        and np.linalg.norm(
            diagnostic.signature - diagnostic.signature.mean()
        )
        > 0.0
        for diagnostic in result.operator_diagnostics
    )
    assert all(row["population"] == config.population_size for row in result.history)
    assert sum(row["qd_rejected"] for row in result.history) > 0
    attempted = set().union(
        *(set(row["attempted_features"]) for row in result.history)
    )
    assert attempted == set(profile["feature"])


def test_warm_start_evaluates_multiplier_times_population():
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(population_size=8, warm_start_multiplier=3)

    result = run_hierarchical_evolution(data, target, profile, config, seed=7)

    assert result.history[0]["warm_start_proposals"] == 24
    assert result.history[0]["qd_admitted"] <= 24


def test_v4_rejects_old_checkpoint(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    checkpoint = tmp_path / "run_state.pkl"
    checkpoint.write_bytes(pickle.dumps({"version": 3}))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            small_config(),
            7,
            checkpoint_dir=tmp_path,
        )


def test_interrupted_stage_resumes_deterministically(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config()

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("generation_completed"),
        )

    resumed = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    fresh = run_hierarchical_evolution(data, target, profile, config, 7)

    assert resumed.stages == fresh.stages
    assert resumed.archive_records == fresh.archive_records
    assert resumed.operator_diagnostics == fresh.operator_diagnostics
    assert resumed.history == fresh.history


def test_checkpoint_without_precomputed_signatures_still_resumes(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(max_workers=2)

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("chunk_completed"),
        )

    state_path = tmp_path / "run_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    assert state["work"]["batch"]["signatures"]
    del state["work"]["batch"]["signatures"]
    state_path.write_bytes(pickle.dumps(state))

    resumed = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    fresh = run_hierarchical_evolution(data, target, profile, config, 7)

    assert resumed == fresh


@pytest.mark.parametrize(
    "event_name",
    (
        "qd_batch_completed",
        "family_replacement_completed",
        "stage_completed",
    ),
)
def test_resume_is_exact_from_each_persisted_boundary(tmp_path, event_name):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config()

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event(event_name),
        )

    if event_name == "qd_batch_completed":
        with (tmp_path / "run_state.pkl").open("rb") as handle:
            state = pickle.load(handle)
        assert state["operator_diagnostics"]
        assert all(
            isinstance(diagnostic.signature, np.ndarray)
            and hierarchical_engine._signature_is_comparable(
                diagnostic.signature
            )
            for diagnostic in state["operator_diagnostics"].values()
        )

    resumed = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    fresh = run_hierarchical_evolution(data, target, profile, config, 7)

    assert resumed == fresh


def test_resume_rejects_changed_configuration_fingerprint(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )

    with pytest.raises(RuntimeError, match="configuration.*new output directory"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            replace(config, max_nodes=config.max_nodes + 1),
            7,
            checkpoint_dir=tmp_path,
        )


def test_valid_completed_checkpoint_round_trips_exactly(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)

    completed = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    loaded = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )

    assert loaded == completed


def test_result_retains_complete_candidate_metrics_for_every_archive_record():
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(
        population_size=4,
        generations_per_stage=0,
        warm_start_multiplier=2,
        gene_pool_size=1,
        parent_pool_size=4,
    )

    result = run_hierarchical_evolution(data, target, profile, config, seed=7)

    archive_keys = tuple(record.key for record in result.archive_records)
    candidate_keys = tuple(
        expression_key(candidate.expression)
        for candidate in result.archive_candidates
    )
    gene_pool_keys = {
        expression_key(candidate.expression)
        for stage in result.stages
        for candidate in stage.gene_pool
    }
    assert candidate_keys == archive_keys
    assert set(archive_keys) - gene_pool_keys
    assert all(
        len(candidate.segment_rank_ics) == config.stability_segments
        for candidate in result.archive_candidates
    )
    for record, candidate in zip(
        result.archive_records, result.archive_candidates
    ):
        assert candidate.abs_rank_ic == record.abs_rank_ic
        assert candidate.nodes == record.nodes


def test_resume_rejects_archive_candidate_key_coverage_mismatch(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    result = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = replace(
        result, archive_candidates=result.archive_candidates[:-1]
    )
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_checkpoint_contains_resumable_v4_state_after_each_chunk(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=1)

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("chunk_completed"),
        )

    with (tmp_path / "run_state.pkl").open("rb") as handle:
        state = pickle.load(handle)

    assert state["version"] == 4
    assert {
        "config_fingerprint",
        "stage_index",
        "generation",
        "population",
        "gene_pools",
        "archive",
        "cache",
        "history",
        "rng_state",
        "operator_diagnostics",
    } <= state.keys()
    assert not contains_factor_panel(state)

    resumed = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    fresh = run_hierarchical_evolution(data, target, profile, config, 7)
    assert resumed == fresh


@pytest.mark.parametrize(
    "bad_signature",
    (
        np.array([1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([np.nan, 1.0], dtype=np.float32),
    ),
)
def test_incomparable_pca_signature_is_rejected_before_archive_consideration(
    monkeypatch, bad_signature, tmp_path
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(
        population_size=5,
        generations_per_stage=0,
        warm_start_multiplier=1,
        gene_pool_size=5,
        parent_pool_size=5,
    )
    monkeypatch.setattr(
        "gp_factor_miner.hierarchical_engine.factor_signature",
        lambda *args, **kwargs: bad_signature,
    )
    proposals = ("f0", "f1", "f2", "f3", ("rank", "f0"))
    monkeypatch.setattr(
        "gp_factor_miner.hierarchical_engine.balanced_stage_proposals",
        lambda *args, **kwargs: proposals[: kwargs["size"]],
    )
    monkeypatch.setattr(
        "gp_factor_miner.hierarchical_engine.PCAQDArchive.consider",
        lambda *args, **kwargs: pytest.fail(
            "incomparable signature reached archive consideration"
        ),
    )

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            seed=7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("qd_batch_completed"),
        )
    with (tmp_path / "run_state.pkl").open("rb") as handle:
        state = pickle.load(handle)

    assert any(
        hierarchical_engine.expression_operators(candidate.expression)
        for candidate in state["cache"]._values.values()
    )
    assert state["operator_diagnostics"] == {}


def test_public_engine_rejects_partial_stage_contract():
    data, target, profile = synthetic_problem(feature_count=4)

    with pytest.raises(ValueError, match="exactly.*1, 2, 3"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            replace(small_config(), stage_depths=(1,)),
            seed=7,
        )


def test_identical_panels_are_rejected_as_qd_duplicates_with_default_sampling():
    data, target, profile = synthetic_problem(feature_count=4)
    data["f1"] = data["f0"].copy()
    config = small_config(
        population_size=4,
        generations_per_stage=0,
        warm_start_multiplier=2,
        gene_pool_size=4,
        parent_pool_size=4,
    )

    result = run_hierarchical_evolution(data, target, profile, config, seed=7)

    assert len(target.iloc[:: config.pca_sample_step]) >= 2
    archive_keys = {record.key for record in result.archive_records}
    assert not {expression_key("f0"), expression_key("f1")} <= archive_keys
    assert sum(row["qd_rejected"] for row in result.history) > 0


def test_unseen_proposal_request_sizes_past_seen_canonical_prefix(monkeypatch):
    features = ("f0", "f1", "f2", "f3")
    windows = (1, 2)
    first = balanced_stage_proposals(
        features, windows, stage_depth=1, size=2, seed=7
    )
    seen = {expression_key(expression) for expression in first}
    requested_sizes = []
    original = balanced_stage_proposals

    def tracked(*args, **kwargs):
        requested_sizes.append(kwargs["size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(hierarchical_engine, "balanced_stage_proposals", tracked)

    proposals, exhausted = hierarchical_engine._request_unseen_proposals(
        features=features,
        windows=windows,
        stage_depth=1,
        needed=1,
        seed=7,
        gene_pools=(),
        seen_keys=seen,
        max_nodes=11,
    )

    assert requested_sizes == [3]
    assert len(proposals) == 1
    assert expression_key(proposals[0]) not in {
        expression_key(expression) for expression in first
    }
    assert exhausted is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("seed", None),
        ("archive", {}),
        ("cache", {}),
        ("gene_pools", {}),
        ("history", {}),
        ("rng_state", "not-rng-state"),
        ("work", []),
        ("operator_diagnostics", []),
    ),
)
def test_resume_rejects_structurally_invalid_in_progress_checkpoint(
    tmp_path, field, invalid_value
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config()
    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("chunk_completed"),
        )
    state_path = tmp_path / "run_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    if field == "seed":
        state.pop("seed")
    else:
        state[field] = invalid_value
    state_path.write_bytes(pickle.dumps(state))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "archive_record",
        "archive_object",
        "archive_signature",
        "cache_candidate",
        "operator_diagnostic",
        "operator_diagnostic_object",
        "operator_diagnostic_signature",
        "work_stats",
    ),
)
def test_resume_rejects_invalid_nested_checkpoint_invariants(
    tmp_path, corruption
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config()
    event_name = (
        "chunk_completed"
        if corruption == "work_stats"
        else "qd_batch_completed"
    )
    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event(event_name),
        )
    state_path = tmp_path / "run_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)

    if corruption == "archive_record":
        key = next(iter(state["archive"]._records))
        record = state["archive"]._records[key]
        state["archive"]._records[key] = replace(record, expression="wrong")
    elif corruption == "archive_object":
        key = next(iter(state["archive"]._records))
        state["archive"]._records[key] = object()
    elif corruption == "archive_signature":
        key = next(iter(state["archive"]._records))
        record = state["archive"]._records[key]
        state["archive"]._records[key] = replace(record, signature=object())
    elif corruption == "cache_candidate":
        key = next(iter(state["cache"]._values))
        cached = state["cache"]._values[key]
        state["cache"]._values[key] = replace(cached, nodes=cached.nodes + 1)
    elif corruption == "operator_diagnostic":
        operator = next(iter(state["operator_diagnostics"]))
        diagnostic = state["operator_diagnostics"][operator]
        state["operator_diagnostics"][operator] = replace(
            diagnostic, stage_depth=diagnostic.stage_depth + 1
        )
    elif corruption == "operator_diagnostic_object":
        operator = next(iter(state["operator_diagnostics"]))
        state["operator_diagnostics"][operator] = object()
    elif corruption == "operator_diagnostic_signature":
        operator = next(iter(state["operator_diagnostics"]))
        diagnostic = state["operator_diagnostics"][operator]
        state["operator_diagnostics"][operator] = replace(
            diagnostic, signature=object()
        )
    else:
        state["work"]["initial_stats"] = {"qd_admitted": "invalid"}
    state_path.write_bytes(pickle.dumps(state))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


@pytest.mark.parametrize("corruption", ("gene_pool", "history"))
def test_resume_rejects_invalid_gene_pool_and_history_invariants(
    tmp_path, corruption
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config()
    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("stage_completed"),
        )
    state_path = tmp_path / "run_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    if corruption == "gene_pool":
        first = state["gene_pools"][0][0]
        state["gene_pools"][0] = (
            replace(first, depth=first.depth + 1),
            *state["gene_pools"][0][1:],
        )
    else:
        state["history"][0]["stage"] = 99
    state_path.write_bytes(pickle.dumps(state))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_resume_rejects_completed_checkpoint_with_invalid_result(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = {"not": "a hierarchical result"}
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_resume_rejects_completed_checkpoint_with_duplicate_diagnostics(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    result = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    first = result.operator_diagnostics[0]
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = replace(
        result,
        operator_diagnostics=(
            first,
            first,
            *result.operator_diagnostics[1:],
        ),
    )
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_resume_rejects_completed_checkpoint_with_malformed_diagnostic(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    result = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = replace(
        result, operator_diagnostics=(object(),)
    )
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_resume_rejects_completed_checkpoint_with_invalid_diagnostic_signature(
    tmp_path,
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    result = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    first = result.operator_diagnostics[0]
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = replace(
        result,
        operator_diagnostics=(
            replace(first, signature=object()),
            *result.operator_diagnostics[1:],
        ),
    )
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_resume_rejects_completed_checkpoint_with_invalid_candidate(tmp_path):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(generations_per_stage=0)
    result = run_hierarchical_evolution(
        data, target, profile, config, 7, checkpoint_dir=tmp_path
    )
    first_stage = result.stages[0]
    first_candidate = first_stage.gene_pool[0]
    invalid_stage = replace(
        first_stage,
        gene_pool=(
            replace(first_candidate, nodes=first_candidate.nodes + 1),
            *first_stage.gene_pool[1:],
        ),
    )
    completed_path = tmp_path / "completed.pkl"
    with completed_path.open("rb") as handle:
        completed = pickle.load(handle)
    completed["result"] = replace(
        result, stages=(invalid_stage, *result.stages[1:])
    )
    completed_path.write_bytes(pickle.dumps(completed))

    with pytest.raises(RuntimeError, match="new output directory"):
        run_hierarchical_evolution(
            data, target, profile, config, 7, checkpoint_dir=tmp_path
        )


def test_qd_rejected_candidate_can_be_operator_diagnostic_best(
    monkeypatch, tmp_path
):
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(
        population_size=2,
        generations_per_stage=0,
        warm_start_multiplier=1,
        gene_pool_size=2,
        parent_pool_size=2,
    )
    proposals = ("f0", ("rank", "f2"))
    signature = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    monkeypatch.setattr(
        "gp_factor_miner.hierarchical_engine.balanced_stage_proposals",
        lambda *args, **kwargs: proposals[: kwargs["size"]],
    )
    monkeypatch.setattr(
        "gp_factor_miner.hierarchical_engine.factor_signature",
        lambda *args, **kwargs: signature,
    )

    with pytest.raises(RuntimeError, match="interrupt"):
        run_hierarchical_evolution(
            data,
            target,
            profile,
            config,
            seed=7,
            checkpoint_dir=tmp_path,
            progress_callback=interrupt_after_event("qd_batch_completed"),
        )
    with (tmp_path / "run_state.pkl").open("rb") as handle:
        state = pickle.load(handle)

    rejected_key = expression_key(("rank", "f2"))
    assert rejected_key not in state["archive"]._records
    diagnostic = state["operator_diagnostics"]["rank"]
    assert expression_key(diagnostic.candidate.expression) == rejected_key
    assert np.array_equal(diagnostic.signature, signature)


def test_operator_diagnostic_ties_prefer_fewer_nodes_then_expression_key():
    diagnostics = {}
    first_signature = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    second_signature = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
    winning_signature = np.array([0.0, -1.0, 1.0], dtype=np.float32)
    hierarchical_engine._update_operator_diagnostics(
        diagnostics,
        candidate(("rank", ("abs", "f2")), abs_rank_ic=0.05, nodes=3),
        stage_depth=2,
        signature=first_signature,
    )
    hierarchical_engine._update_operator_diagnostics(
        diagnostics,
        candidate(("rank", "f2"), abs_rank_ic=0.05, nodes=2),
        stage_depth=1,
        signature=second_signature,
    )
    hierarchical_engine._update_operator_diagnostics(
        diagnostics,
        candidate(("rank", "f1"), abs_rank_ic=0.05, nodes=2),
        stage_depth=1,
        signature=winning_signature,
    )

    assert diagnostics["rank"].candidate.expression == ("rank", "f1")
    assert np.array_equal(
        diagnostics["rank"].signature, winning_signature
    )


def test_multiprocess_and_single_process_runs_are_deterministic():
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(
        population_size=4,
        generations_per_stage=0,
        warm_start_multiplier=2,
        gene_pool_size=4,
        parent_pool_size=4,
    )

    single = run_hierarchical_evolution(
        data, target, profile, replace(config, max_workers=1), seed=17
    )
    parallel = run_hierarchical_evolution(
        data, target, profile, replace(config, max_workers=2), seed=17
    )

    assert parallel == single


def test_negative_integer_seed_is_deterministic_across_worker_counts():
    data, target, profile = synthetic_problem(feature_count=4)
    config = small_config(
        population_size=4,
        generations_per_stage=0,
        warm_start_multiplier=2,
        gene_pool_size=4,
        parent_pool_size=4,
    )
    seed = -(1 << 130) - 17

    single = run_hierarchical_evolution(
        data, target, profile, replace(config, max_workers=1), seed=seed
    )
    parallel = run_hierarchical_evolution(
        data, target, profile, replace(config, max_workers=2), seed=seed
    )

    assert parallel == single
    assert single.archive_records
    assert single.operator_diagnostics
