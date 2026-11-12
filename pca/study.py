#!/usr/bin/env python3
"""P2: C++ power/Lanczos vs exact SVD, seeds and QD decision trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/src"))
from gp_factor_miner.pca_qd import PCAQDArchive, QDRecord, signature_similarity

_LIB = None


def cpp_solve(library, R, initial, steps, method):
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(library))
        _LIB.pca_solve.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)]
        _LIB.pca_solve.restype = ctypes.c_int
    output, stats = np.zeros(R.shape[0], dtype=np.float32), np.zeros(5, dtype=np.float64)
    t = time.perf_counter()
    code = _LIB.pca_solve(R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), *R.shape,
                         initial.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), steps, method,
                         output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    elapsed = time.perf_counter() - t
    if code:
        raise RuntimeError(f"C++ solve status {code}")
    return output, stats, elapsed


def evaluate(task):
    index, row, m, n, library, initials, maxsteps = task
    R = np.memmap(row["path"], mode="r", dtype="<f4", shape=(m, n),
                  offset=row["slot"] * m * n * 4).astype(np.float64)
    t = time.perf_counter()
    U, singular, _ = np.linalg.svd(R, full_matrices=False)
    reference_seconds = time.perf_counter() - t
    signatures = np.zeros((len(initials), 2, maxsteps, m), dtype=np.float32)
    # Metrics: angle error, residual, steps, matvecs, elapsed (incl. diagnostics), sigma.
    metrics = np.zeros((len(initials), 2, maxsteps, 6), dtype=np.float64)
    for seed, initial in enumerate(initials):
        for method in range(2):
            for k in range(1, maxsteps + 1):
                output, stats, elapsed = cpp_solve(library, R, initial, k, method)
                signatures[seed, method, k-1] = output
                vector = output.astype(float)
                error = max(0., 1 - abs(np.dot(U[:, 0], vector) / np.linalg.norm(vector)))
                metrics[seed, method, k-1] = error, stats[2], stats[3], stats[4], elapsed, stats[0]
    gap = float(singular[1] / singular[0]) if len(singular) > 1 else 0.
    return index, signatures, metrics, gap, reference_seconds


def replay(rows, signatures):
    records = []
    for index, (row, signature) in enumerate(zip(rows, signatures)):
        if row["candidate"] and np.isfinite(signature_similarity(signature, signature)):
            records.append((index, QDRecord(row["key"], row["key"], row["quality"], 1, 1, signature)))
    archive = PCAQDArchive(.90, .002)
    decisions = archive.consider_batch([record for _, record in records], backend="cpu")
    trace = {i: (d.action, d.closest_key, d.replaced_key) for (i, _), d in zip(records, decisions)}
    return trace, list(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maxsteps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--numerical-only", action="store_true",
                        help="write the verified numerical checkpoint; replay QD separately with sensitivity.py")
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((Path(__file__).resolve().parent / "runs").resolve()):
        parser.error("--out must be new and inside runs/")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["records"][:args.count]
    m, n = manifest["rows"], manifest["cols"]
    if args.maxsteps < 4 or min(args.seeds, args.workers, len(rows)) < 1:
        parser.error("maxsteps >=4; count, seeds, workers must be positive")
    initials = [np.fromfile(manifest["initial"], dtype="<f8")]
    initials += [np.random.default_rng(seed).normal(size=n) for seed in range(1, args.seeds)]
    args.out.mkdir(parents=True)
    np.save(args.out / "initials.npy", np.asarray(initials))
    signatures = np.lib.format.open_memmap(args.out / "signatures.npy", mode="w+", dtype=np.float32,
                                          shape=(len(rows), args.seeds, 2, args.maxsteps, m))
    metrics = np.lib.format.open_memmap(args.out / "metrics.npy", mode="w+", dtype=np.float64,
                                       shape=(len(rows), args.seeds, 2, args.maxsteps, 6))
    ratios = np.zeros(len(rows)); svd_seconds = np.zeros(len(rows)); complete = np.zeros(len(rows), dtype=bool)
    tasks = [(i, row, m, n, str(args.library.resolve()), initials, args.maxsteps) for i, row in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            i, sig, met, gap, seconds = future.result()
            signatures[i] = sig; metrics[i] = met; ratios[i] = gap; svd_seconds[i] = seconds; complete[i] = True
            if completed % 20 == 0 or completed == len(rows):
                signatures.flush(); metrics.flush()
                np.save(args.out / "completed.npy", complete)
                print(f"SVD and C++ sweeps: {completed}/{len(rows)}", flush=True)
    np.save(args.out / "sigma2_over_sigma1.npy", ratios)
    np.save(args.out / "svd_seconds.npy", svd_seconds)
    numerical = dict(verified=bool(complete.all()), factors=len(rows), seeds=args.seeds,
                     methods=["power", "lanczos"], maxsteps=args.maxsteps,
                     scope="numerical checkpoint; QD replay has separate evidence")
    (args.out / "NUMERICAL_SUCCESS.json").write_text(json.dumps(numerical, indent=2) + "\n")
    if args.numerical_only:
        print("Numerical checkpoint complete; run sensitivity.py for QD trajectories", flush=True)
        return
    baseline, baseline_keys = replay(rows, signatures[:, 0, 0, 3])
    decisions = []
    for seed in range(args.seeds):
        for method in range(2):
            for k in range(args.maxsteps):
                trace, keys = replay(rows, signatures[:, seed, method, k])
                union = baseline.keys() | trace.keys()
                decisions.append(dict(seed=seed, method=("power", "lanczos")[method], k=k+1,
                                      changed_decisions=sum(baseline.get(i) != trace.get(i) for i in union),
                                      changed_actions=sum(baseline.get(i, ("skip",))[0] != trace.get(i, ("skip",))[0] for i in union),
                                      changed_closest_keys=sum(baseline.get(i, (None,None,None))[1] != trace.get(i, (None,None,None))[1] for i in union),
                                      changed_replaced_keys=sum(baseline.get(i, (None,None,None))[2] != trace.get(i, (None,None,None))[2] for i in union),
                                      final_archive_size=len(keys),
                                      archive_key_symmetric_difference=len(set(keys) ^ set(baseline_keys))))
    report = dict(factors=len(rows), seeds=args.seeds, maxsteps=args.maxsteps,
                  methods=["power", "lanczos"], metric_columns=["one_minus_abs_cosine", "relative_residual",
                  "steps", "matvecs", "seconds_including_two_residual_matvecs", "sigma1"],
                  ratio_quantiles={str(q): float(np.quantile(ratios, q)) for q in (0, .25, .5, .75, .9, .99, 1)},
                  fraction_ratio_above_09=float(np.mean(ratios > .9)),
                  fraction_ratio_above_099=float(np.mean(ratios > .99)),
                  decision_sensitivity=decisions, completed=bool(complete.all()),
                  note="R inputs are FP32 exports. Production solvers are C++; NumPy SVD is reference only. "
                       "Time includes residual diagnostics and should not replace pure-kernel performance tests.")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Study complete", flush=True)


if __name__ == "__main__":
    main()
