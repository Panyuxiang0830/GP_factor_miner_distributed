#!/usr/bin/env python3
"""Validate compiled results against Python on exactly the same input R."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
from reference import power_signature
from gp_factor_miner.pca_qd import PCAQDArchive, QDRecord, signature_similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--svd-count", type=int, default=0)
    parser.add_argument("--signature-tolerance", type=float, default=1e-6)
    parser.add_argument("--power-reference", type=Path,
                        help="previously verified same-input .npy baseline; avoids repeated large R reads")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    metrics = json.loads((args.run / "metrics.json").read_text())
    m, n, count = manifest["rows"], manifest["cols"], metrics["factors"]
    initial = np.fromfile(manifest["initial"], dtype="<f8")
    signatures = np.fromfile(args.run / "signatures.f32", dtype="<f4").reshape(count, m)
    records, expected_power, svd_errors, residuals = [], [], [], []
    if args.power_reference:
        reference = (np.load(args.power_reference) if args.power_reference.suffix == ".npy"
                     else np.fromfile(args.power_reference, dtype="<f4").reshape(count, m))
        if reference.shape != (count, m) or metrics["steps"] != 4:
            raise ValueError("cached reference must match the exact run shape and four steps")
        expected_power = list(reference)
    for i, row in enumerate(manifest["records"][:count]):
        if row["candidate"] and np.isfinite(signature_similarity(signatures[i], signatures[i])):
            records.append((i, QDRecord(row["key"], row["key"], row["quality"], 1, 1, signatures[i])))
        # Provided-signature replay doesn't need large R reads.
        if metrics.get("input_signatures", False):
            continue
        if (metrics["method"] == "power" and not args.power_reference) or i < args.svd_count:
            matrix = np.memmap(row["path"], dtype="<f4", mode="r",
                               offset=row["slot"] * m * n * 4, shape=(m, n)).astype(np.float64)
            if metrics.get("rank_input", 0):
                matrix = pd.DataFrame(matrix).rank(axis=1, method="average", pct=True).fillna(.5).to_numpy(np.float64)
                matrix -= matrix.mean(axis=1, keepdims=True)
        if metrics["method"] == "power" and not args.power_reference:
            expected_power.append(power_signature(matrix, initial, metrics["steps"]))
        if i < args.svd_count:
            # Exact SVD is validation only; production has no BLAS/LAPACK dependency.
            U, s, _ = np.linalg.svd(matrix, full_matrices=False)
            vector = signatures[i].astype(float)
            svd_errors.append(max(0., 1 - abs(np.dot(U[:, 0], vector) / np.linalg.norm(vector))))
            residuals.append(float(np.linalg.norm(matrix @ (matrix.T @ vector) - s[0]**2 * vector) /
                                   (s[0]**2 * np.linalg.norm(vector))))
    archive = PCAQDArchive(.90, .002)
    decisions = archive.consider_batch([record for _, record in records], backend="cpu")
    actual = json.loads((args.run / "decisions.json").read_text())
    assert len(actual) == len(decisions), "candidate filtering mismatch"
    mismatches = []
    similarity_error = 0.0
    for (index, record), expected, got in zip(records, decisions, actual):
        expected_keys = (expected.action, expected.closest_key, expected.replaced_key)
        got_keys = (got["action"], got["closest_key"], got["replaced_key"])
        if index != got["signature_index"] or record.key != got["key"] or expected_keys != got_keys:
            mismatches.append(dict(signature_index=index, expected=expected_keys, got=got_keys))
        actual_similarity = got["similarity"]
        if np.isfinite(expected.max_similarity):
            if actual_similarity is None or not np.isfinite(actual_similarity):
                raise AssertionError("finite similarity became nonfinite")
            similarity_error = max(similarity_error, abs(float(actual_similarity) - expected.max_similarity))
        elif actual_similarity is not None:
            raise AssertionError("nonfinite similarity mask differs")
    keys = json.loads((args.run / "archive_keys.json").read_text())
    key_agreement = keys == list(archive)
    max_error = 0
    if expected_power:
        expected = np.asarray(expected_power).astype(float)
        actual_sig = signatures.astype(float)
        norms = np.linalg.norm(expected, axis=1) * np.linalg.norm(actual_sig, axis=1)
        if not np.all(norms > 0):
            raise AssertionError("zero signature")
        max_error = float(np.max(np.maximum(0, 1 - np.abs(np.sum(expected * actual_sig, axis=1) / norms))))
    report = dict(candidates=len(records), decision_mismatches=mismatches,
                  archive_key_agreement=key_agreement,
                  max_one_minus_abs_cosine_vs_power=max_error,
                  svd_one_minus_abs_cosine=svd_errors, svd_relative_residual=residuals,
                  max_abs_similarity_error=similarity_error,
                  passed=not mismatches and key_agreement and max_error <= args.signature_tolerance and similarity_error <= 1e-12)
    (args.run / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
