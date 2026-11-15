#!/usr/bin/env python3
"""Validate completed weak/raw scaling: Python QD, numerical samples, repeats."""
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
sys.path.insert(0, str(ROOT / "python/src"))
from reference import power_signature
from gp_factor_miner.pca_qd import PCAQDArchive, QDRecord, signature_similarity


def load(path):
    return json.loads(path.read_text())


def oracle(records, signatures):
    selected = [(i, QDRecord(row["key"], row["key"], row["quality"], 1, 1, signature))
                for i, (row, signature) in enumerate(zip(records, signatures))
                if row["candidate"] and np.isfinite(signature_similarity(signature, signature))]
    archive = PCAQDArchive(.90, .002)
    decisions = archive.consider_batch([record for _, record in selected], backend="cpu")
    trace = [(i, record.key, d.action, d.closest_key, d.replaced_key, d.max_similarity)
             for (i, record), d in zip(selected, decisions)]
    return trace, list(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--kind", choices=("weak", "raw"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--numerical-count", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((ROOT / "pca/runs").resolve()):
        parser.error("output must be new and inside runs/")
    if args.numerical_count < 1:
        parser.error("at least one independent numerical reference is required")
    manifest = load(args.manifest)
    m, n = manifest["rows"], manifest["cols"]
    initial = np.fromfile(manifest["initial"], dtype="<f8")
    cache, numerical, reports, layout_signatures = {}, {}, [], {}
    for parent in args.runs:
        if not (parent / "SUCCESS").exists():
            raise ValueError(f"experiment not complete: {parent}")
        for run in sorted(parent.glob(f"{args.kind}_*_r1")):
            metrics = load(run / "metrics.json")
            count = metrics["factors"]
            assert metrics["rows"] == m and metrics["cols"] == n
            assert metrics["rank_input"] == int(args.kind == "raw")
            assert metrics["steps"] == 4 and metrics["method"] == "power"
            assert 0 < count <= len(manifest["records"])
            if args.kind == "weak":
                assert count == 32 * metrics["ranks"]
            signature_bytes = (run / "signatures.f32").read_bytes()
            layout_key = (count, metrics["layout"])
            if layout_key in layout_signatures:
                # Direct factor reads have no reduction-order variation.
                if metrics["layout"] in ("factor", "transpose"):
                    assert signature_bytes == layout_signatures[layout_key], run
            else:
                layout_signatures[layout_key] = signature_bytes
            signatures = np.frombuffer(signature_bytes, dtype="<f4").reshape(count, m)
            assert np.isfinite(signatures).all()
            if count not in cache:
                cache[count] = (*oracle(manifest["records"][:count], signatures), signatures.copy())
            expected, keys, reference = cache[count]
            a, b = signatures.astype(float), reference.astype(float)
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            assert np.all(norms > 0)
            signature_error = float(np.max(np.maximum(0, 1 - np.abs(np.sum(a*b, axis=1)/norms))))
            assert signature_error <= 1e-6, run
            # Always replay each run's own signatures: a threshold crossing must
            # be explained by its input, rather than hidden by a cached trace.
            if not np.array_equal(signatures, reference):
                expected, keys = oracle(manifest["records"][:count], signatures)
            actual = load(run / "decisions.json")
            assert len(actual) == len(expected)
            similarity_error = 0.
            for row, entry in zip(actual, expected):
                got = tuple(row[k] for k in ("signature_index", "key", "action", "closest_key", "replaced_key"))
                assert got == entry[:5], (run, row["signature_index"])
                if np.isfinite(entry[5]):
                    assert row["similarity"] is not None and np.isfinite(row["similarity"])
                    similarity_error = max(similarity_error, abs(row["similarity"]-entry[5]))
                else:
                    assert row["similarity"] is None
            assert similarity_error <= 1e-12, run
            assert load(run / "archive_keys.json") == keys
            sample_error = 0.
            for i, row in enumerate(manifest["records"][:min(count,args.numerical_count)]):
                if i not in numerical:
                    matrix = np.memmap(row["path"], dtype="<f4", mode="r", shape=(m,n),
                                       offset=row["slot"]*m*n*4).astype(float)
                    if args.kind == "raw":
                        matrix = pd.DataFrame(matrix).rank(axis=1, method="average", pct=True).fillna(.5).to_numpy(float)
                        matrix -= matrix.mean(axis=1, keepdims=True)
                    numerical[i] = power_signature(matrix, initial, 4).astype(float)
                x, y = numerical[i], a[i]
                sample_error = max(sample_error, max(0., 1-abs(float(x@y))/(np.linalg.norm(x)*np.linalg.norm(y))))
            assert sample_error <= 1e-6, (run,"independent numerical sample")
            for repeat in (2,3):
                other = run.with_name(run.name[:-1]+str(repeat))
                for artifact in ("signatures.f32", "decisions.json", "archive_keys.json"):
                    assert (run/artifact).read_bytes() == (other/artifact).read_bytes(), (other,artifact)
            reports.append(dict(run=str(run.resolve()),factors=count,ranks=metrics["ranks"],
                layout=metrics["layout"],python_qd_candidates=len(actual),
                max_abs_similarity_error=similarity_error,max_same_count_signature_error=signature_error,
                numerical_sample_count=min(count,args.numerical_count),max_numerical_sample_error=sample_error,
                repetitions_byte_identical=True,python_qd_passed=True))
            print(f"Verified {run.name}: {len(actual)} decisions",flush=True)
    if not reports:
        raise ValueError("no measured configurations found")
    for count in cache:
        assert all((count,layout) in layout_signatures for layout in ("date","factor","transpose"))
        assert layout_signatures[(count,"factor")] == layout_signatures[(count,"transpose")], count
    args.out.mkdir(parents=True)
    (args.out/"SUCCESS.json").write_text(json.dumps(dict(verified=True,kind=args.kind,
        manifest=str(args.manifest.resolve()),configurations=reports,
        scope="Full ordered Python QD and repeat checks; first numerical-count factors independently recomputed, not a full numerical oracle."),indent=2)+"\n")


if __name__ == "__main__":
    main()
