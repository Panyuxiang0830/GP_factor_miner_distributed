#!/usr/bin/env python3
"""Benchmark legacy Pearson scans against the batched PCA-QD backend."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from gp_factor_miner.pca_qd import (
    PCAQDArchive,
    QDRecord,
    _matrix_similarity,
    _normalized_matrix,
    signature_similarity,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PCA-QD using a TRUSTED local pickle checkpoint.",
        epilog="SECURITY: pickle can execute code; never pass an untrusted checkpoint.",
    )
    parser.add_argument("checkpoint", type=Path, help="trusted run_state.pkl")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--archive", type=int, default=1_000)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--proxy-pairs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def adversarial_agreement(backend):
    length = 3554
    rng = np.random.default_rng(1)
    left = rng.normal(size=length)
    left -= left.mean()
    left /= np.linalg.norm(left)
    basis = rng.normal(size=length)
    basis -= basis.mean()
    basis -= left * np.dot(left, basis)
    basis /= np.linalg.norm(basis)
    correlation = 0.8999999758
    right = correlation * left + np.sqrt(1.0 - correlation**2) * basis
    incumbent = QDRecord("old", "old", 0.03, 1, 1, left)
    candidate = QDRecord("new", "new", 0.031, 1, 1, right)
    legacy = PCAQDArchive(0.90, 0.002)
    optimized = PCAQDArchive(0.90, 0.002)
    legacy.consider(incumbent)
    optimized.consider(incumbent)
    expected = legacy.consider(candidate)
    actual = optimized.consider_batch([candidate], backend=backend)[0]
    boundary = {
        "legacy_similarity": signature_similarity(left, right),
        "action_key_agreement": (
            expected.action, expected.closest_key, expected.replaced_key
        )
        == (actual.action, actual.closest_key, actual.replaced_key),
        "exact_similarity_agreement": expected.max_similarity == actual.max_similarity,
        "final_archive_key_agreement": set(legacy._records) == set(optimized._records),
    }
    rng = np.random.default_rng(7)
    probe = rng.normal(size=length)
    probe -= probe.mean()
    probe /= np.linalg.norm(probe)
    archived = []
    for seed, value in ((8, 0.8), (9, 0.800000003)):
        orthogonal = np.random.default_rng(seed).normal(size=length)
        orthogonal -= orthogonal.mean()
        orthogonal -= probe * np.dot(probe, orthogonal)
        orthogonal /= np.linalg.norm(orthogonal)
        archived.append(value * probe + np.sqrt(1.0 - value**2) * orthogonal)
    legacy = PCAQDArchive(0.79, 0.002)
    optimized = PCAQDArchive(0.79, 0.002)
    for key, signature in (("z", archived[0]), ("a", archived[1])):
        item = QDRecord(key, key, 0.03, 1, 1, signature)
        legacy._records[key] = item
        optimized._records[key] = item
    candidate = QDRecord("new", "new", 0.04, 1, 1, probe)
    expected = legacy.consider(candidate)
    actual = optimized.consider_batch([candidate], backend=backend)[0]
    tie = {
        "action_key_agreement": (
            expected.action, expected.closest_key, expected.replaced_key
        )
        == (actual.action, actual.closest_key, actual.replaced_key),
        "exact_similarity_agreement": expected.max_similarity == actual.max_similarity,
        "final_archive_key_agreement": set(legacy._records) == set(optimized._records),
    }
    return {"boundary": boundary, "near_tie": tie}


def main():
    args = parse_args()
    with args.checkpoint.open("rb") as handle:
        state = pickle.load(handle)
    records = list(state["archive"]._records.values())
    needed = args.candidates + args.archive
    if len(records) < needed:
        raise SystemExit(f"checkpoint has {len(records)} records, need {needed}")
    candidate_signatures = [np.asarray(item.signature) for item in records[: args.candidates]]
    archive_signatures = [np.asarray(item.signature) for item in records[-args.archive :]]
    dimension = candidate_signatures[0].size

    started = time.perf_counter()
    legacy = np.asarray(
        [
            [signature_similarity(candidate, archived) for archived in archive_signatures]
            for candidate in candidate_signatures
        ],
        dtype=np.float64,
    )
    legacy_seconds = time.perf_counter() - started

    candidates, candidate_valid = _normalized_matrix(candidate_signatures, dimension)
    archive, archive_valid = _normalized_matrix(archive_signatures, dimension)
    peak_gpu_mb = 0.0
    try:
        import torch

        if args.backend != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    started = time.perf_counter()
    optimized = _matrix_similarity(candidates, archive, args.backend)
    optimized_for_comparison = np.asarray(optimized, dtype=np.float64)
    optimized_seconds = time.perf_counter() - started
    if torch is not None and args.backend != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
    assert candidate_valid.all() and archive_valid.all()

    legacy_archive = PCAQDArchive(args.threshold, 0.002)
    optimized_archive = PCAQDArchive(args.threshold, 0.002)
    legacy_archive._records = {
        item.key: item for item in records[-args.archive :]
    }
    optimized_archive._records = dict(legacy_archive._records)
    started = time.perf_counter()
    legacy_decisions = [legacy_archive.consider(item) for item in records[: args.candidates]]
    legacy_decision_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized_decisions = optimized_archive.consider_batch(
        records[: args.candidates], block_size=128, backend=args.backend
    )
    optimized_decision_seconds = time.perf_counter() - started

    # Stored signatures used sample_step=20. Every fifth value is a transparent
    # time-sampling proxy for step=100; it does not pretend to refit PCA.
    rng = np.random.default_rng(args.seed)
    pair_count = min(args.proxy_pairs, len(records) * (len(records) - 1) // 2)
    left = rng.integers(0, len(records), size=pair_count)
    right = rng.integers(0, len(records), size=pair_count)
    unequal = left != right
    left, right = left[unequal], right[unequal]
    full = np.asarray(
        [signature_similarity(records[i].signature, records[j].signature) for i, j in zip(left, right)]
    )
    proxy = np.asarray(
        [
            signature_similarity(
                np.asarray(records[i].signature)[::5], np.asarray(records[j].signature)[::5]
            )
            for i, j in zip(left, right)
        ]
    )
    finite = np.isfinite(full) & np.isfinite(proxy)
    full, proxy = full[finite], proxy[finite]
    threshold_agreement = float(np.mean((full >= args.threshold) == (proxy >= args.threshold)))
    near = np.abs(full - args.threshold) <= 0.05

    comparisons = args.candidates * args.archive
    report = {
        "checkpoint": str(args.checkpoint),
        "records": len(records),
        "workload": {
            "candidates": args.candidates,
            "archive": args.archive,
            "signature_dimension": dimension,
            "comparisons": comparisons,
        },
        "legacy": {
            "seconds": legacy_seconds,
            "correlations_per_second": comparisons / legacy_seconds,
        },
        "optimized": {
            "requested_backend": args.backend,
            "seconds": optimized_seconds,
            "correlations_per_second": comparisons / optimized_seconds,
            "peak_gpu_memory_mb": peak_gpu_mb,
        },
        "speedup": legacy_seconds / optimized_seconds,
        "max_numerical_error": float(
            np.nanmax(np.abs(legacy - optimized_for_comparison))
        ),
        "threshold_decision_agreement": float(
            np.mean(
                (legacy >= args.threshold)
                == (optimized_for_comparison >= args.threshold)
            )
        ),
        "adversarial": adversarial_agreement(args.backend),
        "sequential_archive_decisions": {
            "legacy_seconds": legacy_decision_seconds,
            "optimized_seconds": optimized_decision_seconds,
            "speedup": legacy_decision_seconds / optimized_decision_seconds,
            "action_key_agreement": bool(
                [
                    (item.action, item.closest_key, item.replaced_key)
                    for item in legacy_decisions
                ]
                == [
                    (item.action, item.closest_key, item.replaced_key)
                    for item in optimized_decisions
                ]
            ),
            "final_archive_key_agreement": set(legacy_archive._records)
            == set(optimized_archive._records),
        },
        "step100_proxy": {
            "pairs": int(full.size),
            "correlation_between_similarities": float(np.corrcoef(full, proxy)[0, 1]),
            "mean_absolute_error": float(np.mean(np.abs(full - proxy))),
            "threshold_decision_agreement": threshold_agreement,
            "near_threshold_pairs": int(near.sum()),
            "near_threshold_decision_agreement": (
                float(np.mean((full[near] >= args.threshold) == (proxy[near] >= args.threshold)))
                if near.any()
                else None
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
