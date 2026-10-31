import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from gp_factor_miner.pca_qd import PCAQDArchive, QDRecord, factor_signature, signature_similarity


def test_signature_is_deterministic_normalized_and_sign_fixed():
    values = pd.DataFrame([[1, 2, 3], [3, 1, 2], [2, 3, 1], [1, 3, 2]])
    first = factor_signature(values, sample_step=1, power_iterations=4, seed=7)
    second = factor_signature(values, sample_step=1, power_iterations=4, seed=7)

    np.testing.assert_allclose(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert first[np.argmax(np.abs(first))] > 0


def test_daily_scale_equivalent_factors_have_high_similarity():
    base = pd.DataFrame([[1, 2, 3], [3, 1, 2], [2, 3, 1], [1, 3, 2]])
    scaled = base.mul(pd.Series([1.0, 100.0, 0.01, 5.0]), axis=0)

    left = factor_signature(base, 1, 4, 7)
    right = factor_signature(scaled, 1, 4, 7)

    assert signature_similarity(left, right) > 0.99


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),
        (np.array([1.0, np.inf]), np.array([1.0, 2.0])),
        (np.array([1.0, -np.inf]), np.array([1.0, 2.0])),
        (np.array([1.0, 1.0]), np.array([1.0, 2.0])),
    ],
)
def test_signature_similarity_returns_nan_for_unequal_nonfinite_or_constant_inputs(left, right):
    assert np.isnan(signature_similarity(left, right))


def record(key, rank_ic, signature, nodes=3):
    return QDRecord(key, key, rank_ic, nodes, 1, signature)


def test_archive_rejects_similar_equal_quality_candidate():
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)
    assert archive.consider(record("a", 0.030, np.array([1.0, 0.0]))).action == "admit"

    decision = archive.consider(record("b", 0.031, np.array([0.999, 0.001])))

    assert decision.action == "reject"
    assert set(archive) == {"a"}


def test_archive_replaces_similar_candidate_with_materially_better_rankic():
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)
    archive.consider(record("a", 0.030, np.array([1.0, 0.0])))

    decision = archive.consider(record("b", 0.033, np.array([0.999, 0.001])))

    assert decision.action == "replace"
    assert decision.replaced_key == "a"
    assert set(archive) == {"b"}


@pytest.mark.parametrize("rank_ic", [np.nan, np.inf, -np.inf])
def test_archive_rejects_nonfinite_rankic_before_admission_or_replacement(rank_ic):
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)

    decision = archive.consider(record("bad", rank_ic, np.array([1.0, 0.0])))

    assert decision.action == "reject"
    assert list(archive) == []

    archive.consider(record("a", 0.030, np.array([1.0, 0.0])))
    decision = archive.consider(record("bad", rank_ic, np.array([0.999, 0.001])))

    assert decision.action == "reject"
    assert list(archive) == ["a"]


def test_archive_reconsidered_key_rejects_similarity_to_another_record():
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)
    incumbent_a = record("a", 0.030, np.array([-1.0, 0.0, 1.0]))
    incumbent_b = record("b", 0.030, np.array([1.0, -1.0, 0.0]))
    archive.consider(incumbent_a)
    archive.consider(incumbent_b)

    decision = archive.consider(record("b", 0.033, np.array([-1.0, 0.0, 1.0])))

    assert decision.action == "reject"
    assert decision.closest_key == "a"
    assert list(archive) == ["a", "b"]
    assert archive._records["a"] is incumbent_a
    assert archive._records["b"] is incumbent_b


def test_archive_reconsidered_key_replaces_own_incumbent_when_distinct():
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)
    archive.consider(record("a", 0.030, np.array([-1.0, 0.0, 1.0])))
    archive.consider(record("b", 0.030, np.array([1.0, -1.0, 0.0])))

    refreshed_b = record("b", 0.033, np.array([1.0, -1.0, 0.0]))
    decision = archive.consider(refreshed_b)

    assert decision.action == "replace"
    assert decision.replaced_key == "b"
    assert list(archive) == ["a", "b"]
    assert archive._records["b"] is refreshed_b


def test_archive_iteration_is_sorted_by_canonical_key():
    archive = PCAQDArchive(similarity_threshold=0.90, replacement_margin=0.002)
    archive.consider(record("zeta", 0.030, np.array([-1.0, 0.0, 1.0])))
    archive.consider(record("alpha", 0.030, np.array([1.0, -1.0, 0.0])))

    assert list(archive) == ["alpha", "zeta"]


def _decision_tuple(decision):
    return (
        decision.action,
        decision.closest_key,
        decision.replaced_key,
        decision.max_similarity,
    )


def test_consider_batch_matches_sequential_for_admit_reject_replace_same_key_and_ties():
    initial = [
        record("a", 0.030, np.array([-1.0, 0.0, 1.0, 0.0])),
        record("z", 0.030, np.array([0.0, 1.0, 0.0, -1.0])),
    ]
    candidates = [
        record("b", 0.029, np.array([-0.99, 0.0, 1.0, 0.0])),  # reject near a
        record("c", 0.034, np.array([-0.99, 0.0, 1.0, 0.0])),  # replace a
        record("d", 0.040, np.array([1.0, 1.0, -1.0, -1.0])),  # admit
        record("d", 0.045, np.array([1.0, 1.0, -1.0, -1.0])),  # same-key replace
        record("x", 0.020, np.array([1.0, -1.0, -1.0, 1.0])),  # equal tie by key
    ]
    sequential = PCAQDArchive(0.90, 0.002)
    batched = PCAQDArchive(0.90, 0.002)
    for item in initial:
        sequential.consider(item)
        batched.consider(item)

    expected = [_decision_tuple(sequential.consider(item)) for item in candidates]
    actual = [
        _decision_tuple(item)
        for item in batched.consider_batch(candidates, block_size=2, backend="cpu")
    ]

    assert [item[:3] for item in actual] == [item[:3] for item in expected]
    np.testing.assert_allclose(
        [item[3] for item in actual], [item[3] for item in expected], rtol=2e-6, atol=2e-6
    )
    assert list(batched) == list(sequential)
    assert {key: value.abs_rank_ic for key, value in batched._records.items()} == {
        key: value.abs_rank_ic for key, value in sequential._records.items()
    }


def test_consider_batch_auto_falls_back_to_cpu_without_torch(monkeypatch):
    import gp_factor_miner.pca_qd as pca_qd

    monkeypatch.setattr(pca_qd, "_load_torch", lambda: None)
    archive = PCAQDArchive(0.90, 0.002)

    decisions = archive.consider_batch(
        [record("a", 0.03, np.array([-1.0, 0.0, 1.0]))],
        backend="auto",
    )

    assert decisions[0].action == "admit"


def test_batched_archive_pickle_contains_only_numpy_signatures():
    archive = PCAQDArchive(0.90, 0.002)
    archive.consider_batch(
        [record("a", 0.03, np.array([-1.0, 0.0, 1.0]))], backend="cpu"
    )

    restored = pickle.loads(pickle.dumps(archive))

    assert isinstance(restored._records["a"].signature, np.ndarray)


def test_cuda_batch_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    candidates = [
        record("a", 0.030, np.array([-1.0, 0.0, 1.0, 0.0])),
        record("b", 0.034, np.array([-0.99, 0.0, 1.0, 0.0])),
        record("c", 0.040, np.array([1.0, 1.0, -1.0, -1.0])),
    ]
    cpu = PCAQDArchive(0.90, 0.002)
    cuda = PCAQDArchive(0.90, 0.002)

    cpu_decisions = cpu.consider_batch(candidates, block_size=2, backend="cpu")
    cuda_decisions = cuda.consider_batch(candidates, block_size=2, backend="cuda")

    assert [item.action for item in cuda_decisions] == [item.action for item in cpu_decisions]
    assert list(cuda) == list(cpu)


def _orthogonal_signatures(length=3554, seed=1, correlation=0.8999999758):
    rng = np.random.default_rng(seed)
    left = rng.normal(size=length)
    left -= left.mean()
    left /= np.linalg.norm(left)
    right_basis = rng.normal(size=length)
    right_basis -= right_basis.mean()
    right_basis -= left * np.dot(left, right_basis)
    right_basis /= np.linalg.norm(right_basis)
    right = correlation * left + np.sqrt(1.0 - correlation**2) * right_basis
    return left, right


def test_batch_refines_float32_threshold_flip_before_mutation():
    left, right = _orthogonal_signatures()
    assert signature_similarity(left, right) == pytest.approx(0.8999999758)
    matrix, _ = __import__("gp_factor_miner.pca_qd", fromlist=["_normalized_matrix"])._normalized_matrix(
        [left, right], left.size
    )
    assert float(abs(matrix[0] @ matrix[1])) > 0.90
    legacy = PCAQDArchive(0.90, 0.002)
    batched = PCAQDArchive(0.90, 0.002)
    incumbent = record("old", 0.03, left)
    candidate = record("new", 0.031, right)
    legacy.consider(incumbent)
    batched.consider(incumbent)

    expected = legacy.consider(candidate)
    actual = batched.consider_batch([candidate], backend="cpu")[0]

    assert _decision_tuple(actual) == _decision_tuple(expected)
    assert list(batched) == list(legacy) == ["new", "old"]


def test_dimension_and_normalization_error_bound_covers_observed_dot_error():
    import gp_factor_miner.pca_qd as pca_qd

    left, right = _orthogonal_signatures()
    matrix, valid = pca_qd._normalized_matrix([left, right], left.size)
    errors = pca_qd._normalization_error_bounds([left, right], matrix, valid)
    bound = pca_qd._similarity_error_bound(errors[0], errors[1], left.size)

    observed = abs(float(matrix[0] @ matrix[1]) - signature_similarity(left, right))
    assert observed <= bound
    assert pca_qd._float32_dot_error_bound(left.size) > 0.0


def test_batch_refines_near_tie_for_exact_closest_and_replaced_key():
    rng = np.random.default_rng(7)
    candidate_signature = rng.normal(size=3554)
    candidate_signature -= candidate_signature.mean()
    candidate_signature /= np.linalg.norm(candidate_signature)
    archived = []
    for seed, correlation in ((8, 0.8), (9, 0.800000003)):
        basis = np.random.default_rng(seed).normal(size=3554)
        basis -= basis.mean()
        basis -= candidate_signature * np.dot(candidate_signature, basis)
        basis /= np.linalg.norm(basis)
        archived.append(
            correlation * candidate_signature
            + np.sqrt(1.0 - correlation**2) * basis
        )
    legacy = PCAQDArchive(0.79, 0.002)
    batched = PCAQDArchive(0.79, 0.002)
    # Float32 ties choose lexicographically larger z, while float64 says a is closer.
    for item in (record("z", 0.03, archived[0]), record("a", 0.03, archived[1])):
        legacy._records[item.key] = item
        batched._records[item.key] = item
    candidate = record("new", 0.04, candidate_signature)

    expected = legacy.consider(candidate)
    actual = batched.consider_batch([candidate], backend="cpu")[0]

    assert _decision_tuple(actual) == _decision_tuple(expected)
    assert actual.replaced_key == "a"
    assert list(batched) == list(legacy)


def test_batch_compares_candidate_to_matching_signature_dimension():
    archive = PCAQDArchive(0.90, 0.002)
    archive._records = {
        "short": record("short", 0.03, np.array([-1.0, 0.0, 1.0])),
        "long": record("long", 0.03, np.array([-1.0, 0.0, 1.0, 2.0])),
    }
    candidate = record("candidate", 0.031, np.array([-0.99, 0.0, 1.0, 2.0]))
    legacy = pickle.loads(pickle.dumps(archive))

    expected = legacy.consider(candidate)
    actual = archive.consider_batch([candidate], backend="cpu")[0]

    assert _decision_tuple(actual) == _decision_tuple(expected)
    assert list(archive) == list(legacy)


def test_non_oom_cuda_runtime_error_propagates(monkeypatch):
    import gp_factor_miner.pca_qd as pca_qd

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            raise AssertionError("must not fall back for unrelated runtime errors")

        class OutOfMemoryError(RuntimeError):
            pass

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(pca_qd, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(
        pca_qd, "_cuda_similarity_pair", lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        pca_qd._block_similarities(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), "auto")


def test_second_cuda_matrix_oom_recomputes_both_matrices_on_cpu(monkeypatch):
    import gp_factor_miner.pca_qd as pca_qd

    calls = []

    class FakeCuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls.append("empty")

    class FakeTorch:
        cuda = FakeCuda()

    def cuda_pair(*args):
        calls.extend(("cuda-cross", "cuda-internal"))
        raise FakeCuda.OutOfMemoryError("second matrix")

    def cpu_pair(candidates, archive):
        calls.extend(("cpu-cross", "cpu-internal"))
        return np.abs(candidates @ archive.T), np.abs(candidates @ candidates.T)

    monkeypatch.setattr(pca_qd, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(pca_qd, "_cuda_similarity_pair", cuda_pair)
    monkeypatch.setattr(pca_qd, "_cpu_similarity_pair", cpu_pair)

    cross, internal = pca_qd._block_similarities(
        np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), "auto"
    )

    assert calls == ["cuda-cross", "cuda-internal", "empty", "cpu-cross", "cpu-internal"]
    np.testing.assert_array_equal(cross, np.eye(2))
    np.testing.assert_array_equal(internal, np.eye(2))


@pytest.mark.parametrize(
    "extreme",
    [
        np.array([-1e100, 0.0, 1e100, 2e100]),
        np.array([-1e-100, 0.0, 1e-100, 2e-100]),
    ],
)
def test_coarse_normalization_collapse_refines_with_legacy(extreme):
    incumbent = record("old", 0.03, extreme)
    candidate = record("new", 0.031, extreme * np.array([1.0, 1.0, 1.0, 1.000001]))
    legacy = PCAQDArchive(0.90, 0.002)
    batched = PCAQDArchive(0.90, 0.002)
    legacy.consider(incumbent)
    batched.consider(incumbent)

    expected = legacy.consider(candidate)
    actual = batched.consider_batch([candidate], backend="cpu")[0]

    assert _decision_tuple(actual) == _decision_tuple(expected)
    assert list(batched) == list(legacy)


@pytest.mark.parametrize(
    "bad", [np.array([1.0, 1.0, 1.0]), np.array([1.0, np.nan, 2.0])]
)
def test_invalid_coarse_and_legacy_signature_remains_incomparable(bad):
    archive = PCAQDArchive(0.90, 0.002)
    archive._records["bad"] = record("bad", 0.03, bad)
    candidate = record("candidate", 0.031, np.array([-1.0, 0.0, 1.0]))
    legacy = pickle.loads(pickle.dumps(archive))

    expected = legacy.consider(candidate)
    actual = archive.consider_batch([candidate], backend="cpu")[0]

    assert _decision_tuple(actual) == _decision_tuple(expected)


def test_cuda_precision_settings_are_restored_on_success_and_error(monkeypatch):
    import gp_factor_miner.pca_qd as pca_qd

    class Matmul:
        allow_tf32 = True
        allow_fp16_reduced_precision_reduction = True
        allow_bf16_reduced_precision_reduction = True

    class CudaBackend:
        matmul = Matmul()

    class Backends:
        cuda = CudaBackend()

    class FakeTorch:
        backends = Backends()
        precision = "high"

        @classmethod
        def get_float32_matmul_precision(cls):
            return cls.precision

        @classmethod
        def set_float32_matmul_precision(cls, value):
            cls.precision = value

    observed = []

    def operation():
        matmul = FakeTorch.backends.cuda.matmul
        observed.append(
            (
                matmul.allow_tf32,
                matmul.allow_fp16_reduced_precision_reduction,
                matmul.allow_bf16_reduced_precision_reduction,
                FakeTorch.precision,
            )
        )
        return "ok"

    assert pca_qd._with_strict_cuda_precision(FakeTorch, operation) == "ok"
    assert observed == [(False, False, False, "highest")]
    matmul = FakeTorch.backends.cuda.matmul
    assert (matmul.allow_tf32, matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction, FakeTorch.precision) == (True, True, True, "high")

    with pytest.raises(RuntimeError):
        pca_qd._with_strict_cuda_precision(
            FakeTorch, lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    assert (matmul.allow_tf32, matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction, FakeTorch.precision) == (True, True, True, "high")


def test_mixed_dimension_oom_recomputes_all_groups_on_cpu(monkeypatch):
    import gp_factor_miner.pca_qd as pca_qd

    calls = []

    class FakeCuda:
        class OutOfMemoryError(RuntimeError):
            pass

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls.append("empty")

    class FakeTorch:
        cuda = FakeCuda()

    def cuda_groups(groups, torch):
        calls.extend(["cuda-3", "cuda-4"])
        raise FakeCuda.OutOfMemoryError("group two")

    def cpu_pair(candidates, archive):
        calls.append(f"cpu-{candidates.shape[1]}")
        return np.abs(candidates @ archive.T), np.abs(candidates @ candidates.T)

    monkeypatch.setattr(pca_qd, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(pca_qd, "_cuda_similarity_groups", cuda_groups)
    monkeypatch.setattr(pca_qd, "_cpu_similarity_pair", cpu_pair)
    groups = [
        (np.ones((1, 3), np.float32), np.ones((1, 3), np.float32)),
        (np.ones((1, 4), np.float32), np.ones((1, 4), np.float32)),
    ]

    results = pca_qd._grouped_block_similarities(groups, "auto")

    assert len(results) == 2
    assert calls == ["cuda-3", "cuda-4", "empty", "cpu-3", "cpu-4"]


def test_batch_similarity_storage_is_dense_and_memory_bounded():
    rng = np.random.default_rng(4)
    archive = PCAQDArchive(0.9999, 1.0)
    for index in range(1000):
        signature = rng.normal(size=16)
        archive._records[str(index)] = record(str(index), 0.03, signature)
    candidates = [record(f"new-{index}", 0.03, rng.normal(size=16)) for index in range(128)]

    tracemalloc.start()
    archive.consider_batch(candidates, block_size=128, backend="cpu")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 20 * 1024 * 1024
