import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _load_torch():
    """Load optional PyTorch without making it a CPU installation dependency."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _normalized_matrix(signatures, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(signatures), dimension), dtype=np.float32)
    valid = np.zeros(len(signatures), dtype=bool)
    for index, signature in enumerate(signatures):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = np.asarray(signature, dtype=np.float32).ravel()
        if values.size != dimension or dimension < 2 or not np.isfinite(values).all():
            continue
        centered = values - values.mean(dtype=np.float32)
        norm = np.linalg.norm(centered)
        if not np.isfinite(norm) or norm == 0.0:
            continue
        matrix[index] = centered / norm
        valid[index] = True
    return matrix, valid


def _normalization_error_bounds(signatures, normalized, valid) -> np.ndarray:
    """Bound each float32 normalized row's L2 distance from legacy float64."""
    errors = np.full(len(signatures), np.inf, dtype=np.float64)
    for index, signature in enumerate(signatures):
        if not valid[index]:
            continue
        values = np.asarray(signature, dtype=np.float64).ravel()
        centered = values - values.mean()
        norm = np.linalg.norm(centered)
        if not np.isfinite(norm) or norm == 0.0:
            continue
        exact = centered / norm
        errors[index] = np.linalg.norm(
            np.asarray(normalized[index], dtype=np.float64) - exact
        )
    return errors


def _cpu_similarity_pair(candidates, archive):
    return (
        np.abs(candidates @ archive.T),
        np.abs(candidates @ candidates.T),
    )


def _cuda_similarity_pair(candidates, archive, torch):
    """Compute both matrices as one CUDA transaction."""
    with torch.inference_mode():
        candidate_tensor = torch.as_tensor(candidates, device="cuda")
        archive_tensor = torch.as_tensor(archive, device="cuda")
        cross = torch.abs(candidate_tensor @ archive_tensor.T)
        internal = torch.abs(candidate_tensor @ candidate_tensor.T)
        return cross.cpu().numpy(), internal.cpu().numpy()


def _with_strict_cuda_precision(torch, operation):
    """Disable TF32/reduced matmul and restore the caller's ambient settings."""
    matmul = getattr(getattr(torch, "backends", None), "cuda", None)
    matmul = getattr(matmul, "matmul", None)
    saved = {}
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if matmul is not None and hasattr(matmul, name):
            saved[name] = getattr(matmul, name)
            setattr(matmul, name, False)
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    prior_precision = get_precision() if callable(get_precision) else None
    if callable(set_precision):
        set_precision("highest")
    try:
        return operation()
    finally:
        if callable(set_precision) and prior_precision is not None:
            set_precision(prior_precision)
        for name, value in saved.items():
            setattr(matmul, name, value)


def _cuda_similarity_groups(groups, torch):
    return _with_strict_cuda_precision(
        torch,
        lambda: [
            _cuda_similarity_pair(candidates, archive, torch)
            for candidates, archive in groups
        ],
    )


def _grouped_block_similarities(groups, backend: str):
    """Compute all dimension groups as one atomic backend attempt."""
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("PCA-QD backend must be auto, cpu, or cuda")
    torch = _load_torch() if backend != "cpu" else None
    use_cuda = torch is not None and torch.cuda.is_available()
    if backend == "cuda" and not use_cuda:
        raise RuntimeError("CUDA PCA-QD backend requested but unavailable")
    if use_cuda:
        try:
            return _cuda_similarity_groups(groups, torch)
        except RuntimeError as error:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            if oom_type is None or not isinstance(error, oom_type):
                raise
            # Discard the entire CUDA attempt. Never mix a CUDA cross matrix
            # with a CPU internal matrix after a second-matmul OOM.
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
    return [_cpu_similarity_pair(candidates, archive) for candidates, archive in groups]


def _block_similarities(candidates, archive, backend: str):
    """Single-dimension compatibility wrapper."""
    return _grouped_block_similarities([(candidates, archive)], backend)[0]


def _matrix_similarity(left, right, backend: str) -> np.ndarray:
    """Compatibility helper used by the standalone benchmark."""
    return _block_similarities(left, right, backend)[0]


def _float32_dot_error_bound(dimension: int) -> float:
    """Dimension-derived ``gamma_n`` bound for one float32 dot product."""
    unit_roundoff = np.finfo(np.float32).eps
    scaled = dimension * unit_roundoff
    if scaled >= 1.0:
        return 1.0
    return scaled / (1.0 - scaled)


def _similarity_error_bound(left_error, right_error, dimension: int) -> float:
    """Rigorous coarse-vs-legacy bound from vector error plus dot rounding."""
    return min(
        1.0,
        left_error
        + right_error
        + left_error * right_error
        + _float32_dot_error_bound(dimension),
    )


def factor_signature(
    values: pd.DataFrame,
    sample_step: int = 20,
    power_iterations: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Return a deterministic, sign-fixed PCA signature for factor values."""
    sampled = values.iloc[::sample_step]
    ranked = sampled.rank(axis=1, method="average", pct=True).to_numpy(np.float64)
    ranked = np.nan_to_num(ranked, nan=0.5, posinf=0.5, neginf=0.5)
    ranked -= ranked.mean(axis=1, keepdims=True)
    if not np.any(ranked):
        raise ValueError("constant PCA-QD signature")

    rng = np.random.default_rng(seed)
    right = rng.normal(size=ranked.shape[1])
    right /= np.linalg.norm(right)
    for _ in range(power_iterations):
        left = ranked @ right
        left /= np.linalg.norm(left)
        right = ranked.T @ left
        right /= np.linalg.norm(right)

    signature = ranked @ right
    signature /= np.linalg.norm(signature)
    pivot = int(np.argmax(np.abs(signature)))
    if signature[pivot] < 0:
        signature *= -1
    return signature.astype(np.float32)


def signature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return absolute Pearson correlation for two aligned signatures."""
    left_array = np.asarray(left, dtype=np.float64).ravel()
    right_array = np.asarray(right, dtype=np.float64).ravel()
    if left_array.size != right_array.size or left_array.size < 2:
        return np.nan
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        return np.nan

    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    left_norm = np.linalg.norm(left_centered)
    right_norm = np.linalg.norm(right_centered)
    if left_norm == 0.0 or right_norm == 0.0:
        return np.nan
    return float(abs(np.dot(left_centered, right_centered) / (left_norm * right_norm)))


@dataclass(frozen=True)
class QDRecord:
    key: str
    expression: object
    abs_rank_ic: float
    nodes: int
    stage_depth: int
    signature: np.ndarray


@dataclass(frozen=True)
class QDDecision:
    action: str
    max_similarity: float
    closest_key: str | None = None
    replaced_key: str | None = None


class PCAQDArchive:
    """Retain behaviorally distinct records, preferring higher absolute RankIC."""

    def __init__(self, similarity_threshold: float, replacement_margin: float):
        self.similarity_threshold = similarity_threshold
        self.replacement_margin = replacement_margin
        self._records: dict[str, QDRecord] = {}

    def __iter__(self):
        return iter(sorted(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def consider(self, record: QDRecord) -> QDDecision:
        if not np.isfinite(record.abs_rank_ic):
            return QDDecision(action="reject", max_similarity=np.nan)

        if not self._records:
            self._records[record.key] = record
            return QDDecision(action="admit", max_similarity=0.0)

        similarities = [
            (signature_similarity(record.signature, stored.signature), key)
            for key, stored in self._records.items()
        ]
        finite_similarities = [item for item in similarities if np.isfinite(item[0])]

        incumbent = self._records.get(record.key)
        if incumbent is not None:
            max_similarity, closest_key = max(
                finite_similarities, key=lambda item: (item[0], item[1]), default=(np.nan, record.key)
            )
            if any(
                key != record.key and similarity >= self.similarity_threshold
                for similarity, key in finite_similarities
            ):
                return QDDecision(
                    action="reject",
                    max_similarity=max_similarity,
                    closest_key=closest_key,
                )
            if record.abs_rank_ic <= incumbent.abs_rank_ic + self.replacement_margin:
                return QDDecision(
                    action="reject",
                    max_similarity=max_similarity,
                    closest_key=record.key,
                )

            replacement = dict(self._records)
            replacement[record.key] = record
            self._records = replacement
            return QDDecision(
                action="replace",
                max_similarity=max_similarity,
                closest_key=record.key,
                replaced_key=record.key,
            )

        if not finite_similarities:
            self._records[record.key] = record
            return QDDecision(action="admit", max_similarity=np.nan)

        max_similarity, closest_key = max(finite_similarities, key=lambda item: (item[0], item[1]))
        if max_similarity < self.similarity_threshold:
            self._records[record.key] = record
            return QDDecision(
                action="admit",
                max_similarity=max_similarity,
                closest_key=closest_key,
            )

        closest_record = self._records[closest_key]
        if record.abs_rank_ic <= closest_record.abs_rank_ic + self.replacement_margin:
            return QDDecision(
                action="reject",
                max_similarity=max_similarity,
                closest_key=closest_key,
            )

        replacement = dict(self._records)
        del replacement[closest_key]
        replacement[record.key] = record
        self._records = replacement
        return QDDecision(
            action="replace",
            max_similarity=max_similarity,
            closest_key=closest_key,
            replaced_key=closest_key,
        )

    def consider_batch(
        self,
        records,
        *,
        block_size: int = 128,
        backend: str = "auto",
    ) -> list[QDDecision]:
        """Consider records in order while batching their similarity arithmetic.

        Admission remains sequential: records admitted or replaced earlier in a
        block are visible to every later record in that block.
        """
        pending = list(records)
        if block_size <= 0:
            raise ValueError("PCA-QD block size must be positive")
        decisions = []
        normalized_records = {}
        for key, stored in self._records.items():
            dimension = np.asarray(stored.signature).size
            matrix, valid = _normalized_matrix([stored.signature], dimension)
            errors = _normalization_error_bounds([stored.signature], matrix, valid)
            normalized_records[key] = (
                matrix[0].copy(), bool(valid[0]), dimension, float(errors[0])
            )
        for start in range(0, len(pending), block_size):
            block = pending[start : start + block_size]
            snapshot_keys = list(self._records)
            candidate_dimensions = [np.asarray(item.signature).size for item in block]
            grouped_candidates = {}
            for index, dimension in enumerate(candidate_dimensions):
                grouped_candidates.setdefault(dimension, []).append(index)
            grouped_archive = {}
            for key in snapshot_keys:
                grouped_archive.setdefault(normalized_records[key][2], []).append(key)

            candidate_normalized = {}
            group_storage = []
            group_inputs = []
            for dimension, indices in grouped_candidates.items():
                candidate_matrix, candidate_valid = _normalized_matrix(
                    [block[index].signature for index in indices], dimension
                )
                candidate_errors = _normalization_error_bounds(
                    [block[index].signature for index in indices],
                    candidate_matrix,
                    candidate_valid,
                )
                archive_keys = grouped_archive.get(dimension, [])
                archive_matrix = np.asarray(
                    [normalized_records[key][0] for key in archive_keys], dtype=np.float32
                ).reshape(len(archive_keys), dimension)
                archive_valid = np.asarray(
                    [normalized_records[key][1] for key in archive_keys], dtype=bool
                )
                group_inputs.append((candidate_matrix, archive_matrix))
                group_storage.append(
                    {
                        "dimension": dimension,
                        "indices": indices,
                        "row_by_candidate": {
                            candidate_index: row
                            for row, candidate_index in enumerate(indices)
                        },
                        "column_by_key": {
                            key: column for column, key in enumerate(archive_keys)
                        },
                        "candidate_valid": candidate_valid,
                        "archive_valid": archive_valid,
                    }
                )
                for row, index in enumerate(indices):
                    candidate_normalized[index] = (
                        candidate_matrix[row].copy(),
                        bool(candidate_valid[row]),
                        dimension,
                        float(candidate_errors[row]),
                    )

            group_results = _grouped_block_similarities(group_inputs, backend)
            group_by_dimension = {}
            for group, (cross, internal) in zip(group_storage, group_results):
                candidate_valid = group["candidate_valid"]
                archive_valid = group["archive_valid"]
                if cross.size:
                    cross[~candidate_valid, :] = np.nan
                    cross[:, ~archive_valid] = np.nan
                internal[~candidate_valid, :] = np.nan
                internal[:, ~candidate_valid] = np.nan
                group["cross"] = cross
                group["internal"] = internal
                group_by_dimension[group["dimension"]] = group

            # A current key points either at its snapshot column or at an earlier
            # candidate row. This makes replacements within the block exact.
            sources = {key: ("archive", key) for key in snapshot_keys}
            current_keys = list(snapshot_keys)
            for index, record in enumerate(block):
                if not np.isfinite(record.abs_rank_ic):
                    decisions.append(QDDecision(action="reject", max_similarity=np.nan))
                    continue

                finite = []
                group = group_by_dimension[candidate_dimensions[index]]
                row = group["row_by_candidate"][index]
                for key in current_keys:
                    source_type, source = sources[key]
                    if source_type == "archive":
                        column = group["column_by_key"].get(source)
                        similarity = (
                            group["cross"][row, column]
                            if column is not None
                            else np.nan
                        )
                    else:
                        other_row = group["row_by_candidate"].get(source)
                        similarity = (
                            group["internal"][row, other_row]
                            if other_row is not None
                            else np.nan
                        )
                    if np.isfinite(similarity):
                        bound = _similarity_error_bound(
                            candidate_normalized[index][3],
                            normalized_records[key][3],
                            candidate_dimensions[index],
                        )
                        finite.append((float(similarity), key, bound))
                    elif normalized_records[key][2] == candidate_dimensions[index]:
                        # Float32 can overflow/underflow even when the legacy
                        # float64 signatures are comparable. Never treat a
                        # coarse collapse as behavioral distinctness.
                        exact = signature_similarity(
                            record.signature, self._records[key].signature
                        )
                        if np.isfinite(exact):
                            finite.append((float(exact), key, 0.0))

                # Float32 is only a coarse search. Refine every possible exact
                # maximum/tie and every threshold-near value with the legacy
                # float64 implementation before mutating the archive.
                if finite:
                    maximum_lower_bound = max(
                        similarity - bound for similarity, _, bound in finite
                    )
                    refined = []
                    for similarity, key, bound in finite:
                        if (
                            similarity + bound >= maximum_lower_bound
                            or similarity - bound
                            <= self.similarity_threshold
                            <= similarity + bound
                        ):
                            similarity = signature_similarity(
                                record.signature, self._records[key].signature
                            )
                        if np.isfinite(similarity):
                            refined.append((float(similarity), key))
                    finite = refined

                incumbent = self._records.get(record.key)
                if incumbent is not None:
                    max_similarity, closest_key = max(
                        finite, key=lambda item: (item[0], item[1]), default=(np.nan, record.key)
                    )
                    if any(
                        key != record.key and similarity >= self.similarity_threshold
                        for similarity, key in finite
                    ):
                        decisions.append(QDDecision("reject", max_similarity, closest_key))
                        continue
                    if record.abs_rank_ic <= incumbent.abs_rank_ic + self.replacement_margin:
                        decisions.append(QDDecision("reject", max_similarity, record.key))
                        continue
                    replacement = dict(self._records)
                    replacement[record.key] = record
                    self._records = replacement
                    normalized_records[record.key] = candidate_normalized[index]
                    sources[record.key] = ("candidate", index)
                    decisions.append(
                        QDDecision("replace", max_similarity, record.key, record.key)
                    )
                    continue

                if not finite:
                    self._records[record.key] = record
                    normalized_records[record.key] = candidate_normalized[index]
                    current_keys.append(record.key)
                    sources[record.key] = ("candidate", index)
                    decisions.append(QDDecision("admit", np.nan if current_keys[:-1] else 0.0))
                    continue
                max_similarity, closest_key = max(finite, key=lambda item: (item[0], item[1]))
                if max_similarity < self.similarity_threshold:
                    self._records[record.key] = record
                    normalized_records[record.key] = candidate_normalized[index]
                    current_keys.append(record.key)
                    sources[record.key] = ("candidate", index)
                    decisions.append(QDDecision("admit", max_similarity, closest_key))
                    continue
                closest_record = self._records[closest_key]
                if record.abs_rank_ic <= closest_record.abs_rank_ic + self.replacement_margin:
                    decisions.append(QDDecision("reject", max_similarity, closest_key))
                    continue
                replacement = dict(self._records)
                del replacement[closest_key]
                replacement[record.key] = record
                self._records = replacement
                del normalized_records[closest_key]
                normalized_records[record.key] = candidate_normalized[index]
                current_keys.remove(closest_key)
                current_keys.append(record.key)
                del sources[closest_key]
                sources[record.key] = ("candidate", index)
                decisions.append(
                    QDDecision("replace", max_similarity, closest_key, closest_key)
                )
        return decisions
