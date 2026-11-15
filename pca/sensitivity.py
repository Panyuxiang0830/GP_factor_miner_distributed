#!/usr/bin/env python3
"""Replay completed signature sweeps in parallel; never recompute SVD/solvers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from study import replay
from archive_io import fingerprint

ROWS = None
SIGNATURES = None


def initialize(manifest_path, study_path):
    global ROWS, SIGNATURES
    ROWS = json.loads(Path(manifest_path).read_text())["records"]
    SIGNATURES = np.load(Path(study_path) / "signatures.npy", mmap_mode="r")
    ROWS = ROWS[:SIGNATURES.shape[0]]


def run_case(case):
    seed, method, k = case
    start = time.perf_counter()
    trace, keys = replay(ROWS, SIGNATURES[:, seed, method, k])
    return case, trace, keys, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parent / "runs"
    if args.out.exists() or not args.out.resolve().is_relative_to(runs.resolve()):
        parser.error("--out must be new and inside runs/")
    completed = np.load(args.study / "completed.npy")
    signatures = np.load(args.study / "signatures.npy", mmap_mode="r")
    metrics = np.load(args.study / "metrics.npy", mmap_mode="r")
    if not completed.all() or len(completed) != signatures.shape[0] or not (args.study / "sigma2_over_sigma1.npy").exists():
        raise ValueError("the numerical sweep checkpoint is not complete")
    count, seeds, methods, steps, dimension = signatures.shape
    source=args.study/'INPUT.json'
    if source.exists():
        provenance=json.loads(source.read_text());manifest=json.loads(args.manifest.read_text())
        if provenance['manifest_hash_fnv1a64']!=fingerprint(args.manifest) or provenance['initial_hash_fnv1a64']!=fingerprint(manifest['initial']):
            raise ValueError('numerical checkpoint input provenance mismatch')
    if methods != 2 or steps < 4 or args.workers < 1 or metrics.shape != (count,seeds,methods,steps,6):
        raise ValueError("invalid checkpoint shape/workers")
    if not np.isfinite(metrics).all():raise ValueError('nonfinite numerical checkpoint')
    args.out.mkdir(parents=True)
    cases = [(s,m,k) for s in range(seeds) for m in range(2) for k in range(steps)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.manifest),str(args.study))) as pool:
        futures = [pool.submit(run_case, case) for case in cases]
        for done, future in enumerate(as_completed(futures), 1):
            case, trace, keys, seconds = future.result()
            s,m,k = case
            path = args.out / f"seed{s}_{('power','lanczos')[m]}_k{k+1:02d}.json"
            path.write_text(json.dumps(dict(seed=s,method=("power","lanczos")[m],k=k+1,
                                           trace=trace,archive_keys=keys,replay_seconds=seconds)) + "\n")
            results[case] = trace, keys
            (args.out / "progress.json").write_text(json.dumps(dict(completed=done,total=len(cases)))+"\n")
            print(f"QD trajectories: {done}/{len(cases)}",flush=True)
    baseline, baseline_keys = results[(0,0,3)]
    decisions = []
    for case in cases:
        s,m,k = case
        trace, keys = results[case]
        union = baseline.keys() | trace.keys()
        decisions.append(dict(seed=s,method=("power","lanczos")[m],k=k+1,
            changed_decisions=sum(baseline.get(i)!=trace.get(i) for i in union),
            changed_actions=sum(baseline.get(i,("skip",))[0]!=trace.get(i,("skip",))[0] for i in union),
            changed_closest_keys=sum(baseline.get(i,(None,None,None))[1]!=trace.get(i,(None,None,None))[1] for i in union),
            changed_replaced_keys=sum(baseline.get(i,(None,None,None))[2]!=trace.get(i,(None,None,None))[2] for i in union),
            final_archive_size=len(keys),archive_key_symmetric_difference=len(set(keys)^set(baseline_keys))))
    ratios = np.load(args.study / "sigma2_over_sigma1.npy")
    report = dict(factors=count,seeds=seeds,maxsteps=steps,methods=["power","lanczos"],
        metric_columns=["one_minus_abs_cosine","relative_residual","steps","matvecs",
                        "seconds_including_two_residual_matvecs","sigma1"],
        ratio_quantiles={str(q):float(np.quantile(ratios,q)) for q in (0,.25,.5,.75,.9,.99,1)},
        fraction_ratio_above_09=float(np.mean(ratios>.9)),fraction_ratio_above_099=float(np.mean(ratios>.99)),
        decision_sensitivity=decisions,completed=True,
        numerical_checkpoint=str(args.study.resolve()),
        note="All solver/SVD results reused from completed checkpoint; QD replay alone parallelized. "
             "Decision reference is Python consider_batch. Per-case traces retained for audit.")
    (args.out / "summary.json").write_text(json.dumps(report,indent=2)+"\n")
    # Keep numerical arrays in place: report accepts a separate sensitivity summary.
    print("All QD trajectories completed",flush=True)


if __name__=="__main__":
    main()
