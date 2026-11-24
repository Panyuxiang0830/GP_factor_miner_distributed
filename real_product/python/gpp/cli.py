"""End-to-end production run on exported panels.

    python -m gpp.cli --evolution-panels DIR --screening-panels DIR \
        --output-dir OUT --config configs/host_512g.json

Differences from gp_factor_miner.hierarchical_cli (all outputs keep the same
file names and columns):
  * no pandas EOD load in the main process: evolution and screening read the
    exported float32 panels (python/scripts/export_panels.py or gpp.export);
  * one in-process NativeBackend per run (no mpirun, no files per chunk);
  * depth stages 1..5 with per-stage node caps;
  * native OOS screening (gpp.screening), item-by-item identical decisions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gp_factor_miner.cli import _git_commit, _write_csv_atomic
from gp_factor_miner.hierarchical_cli import (
    _archive_payload,
    _candidate_payload,
    _package_versions,
    _progress_logger,
    _screening_candidates,
    _screening_progress_logger,
)
from gp_factor_miner.output import atomic_json
from gp_factor_miner.raw_features import profile_raw_panels

from .backend import NativeBackend
from .config import load_config
from .hier_engine import run_hierarchical_evolution
from .screening import screen_hierarchical_factors_native


def build_parser():
    p = argparse.ArgumentParser(description="Production hierarchical GP factor mining (gpp)")
    p.add_argument("--evolution-panels", type=Path, required=True)
    p.add_argument("--screening-panels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, help="JSON preset (configs/*.json)")
    p.add_argument("--smoke", action="store_true", help="tiny population, for wiring checks")
    p.add_argument("--threads", type=int)
    p.add_argument("--gpus", type=int)
    p.add_argument("--memory-budget-gb", type=float)
    p.add_argument("--stage-depths", type=int, nargs="+")
    p.add_argument("--population-size", type=int)
    p.add_argument("--generations-per-stage", type=int)
    p.add_argument("--correlation-sample-step", type=int,
                   help="final de-duplication sampling step (default from config: 20)")
    p.add_argument("--skip-screening", action="store_true")
    return p


def _panel_frames(panel_dir: Path):
    meta = json.loads((panel_dir / "meta.json").read_text())
    shape = (meta["T"], meta["N"])

    def frame(path):
        values = np.fromfile(path, dtype="<f4").reshape(shape)
        return pd.DataFrame(values, index=meta["dates"], columns=meta["assets"])

    return meta, frame


def _feature_profile(panel_dir: Path, setup_dir: Path):
    """Cached raw-field RankIC profile (gp_factor_miner.raw_features semantics)."""
    path = setup_dir / "evolution_profile.csv"
    meta, frame = _panel_frames(panel_dir)
    if path.exists():
        profile = pd.read_csv(path)
        if set(profile["feature"]) == set(meta["fields"]):
            return profile
        raise RuntimeError("cached evolution profile does not match the exported fields")
    target = frame(panel_dir / "target.f32")
    records = []
    for name in meta["fields"]:  # one field at a time: 34 MB each instead of 4.3 GB
        records.append(profile_raw_panels({name: frame(panel_dir / "fields" / f"{name}.f32")}, target))
    profile = pd.concat(records).sort_values("feature").reset_index(drop=True)
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(profile, path)
    return profile


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(
        args.config,
        engine_threads=args.threads,
        gpus=args.gpus,
        memory_budget_gb=args.memory_budget_gb,
        stage_depths=args.stage_depths,
        population_size=args.population_size,
        generations_per_stage=args.generations_per_stage,
        final_correlation_sample_step=args.correlation_sample_step,
    )
    if args.stage_depths:
        config = replace(config, stage_max_nodes=config.stage_max_nodes[: len(config.stage_depths)])
    if args.smoke:
        config = config.smoke()
    evolution_meta = json.loads((args.evolution_panels / "meta.json").read_text())
    screening_meta = json.loads((args.screening_panels / "meta.json").read_text())
    if set(evolution_meta["fields"]) != set(screening_meta["fields"]):
        raise ValueError("evolution and screening exports have different fields")
    if evolution_meta["assets"] != screening_meta["assets"]:
        raise ValueError("evolution and screening exports have different assets")

    eval_dir = args.output_dir / "eval"
    profile = _feature_profile(args.evolution_panels, eval_dir / "setup")
    # With a batch backend the evolution reads only the field names.
    field_keys = {name: None for name in evolution_meta["fields"]}

    results, identities = [], {}
    started = time.perf_counter()
    for run in range(config.independent_runs):
        run_number = run + 1
        seed = config.random_seed + run
        backend = NativeBackend(args.evolution_panels, config)
        identities[run_number] = backend.identity
        print(f"[run {run_number}/{config.independent_runs}] started seed={seed} "
              f"depths={config.stage_depths} population={config.population_size}", flush=True)
        try:
            result = run_hierarchical_evolution(
                field_keys, None, profile, config, seed=seed,
                progress_callback=_progress_logger(run_number, config.independent_runs, config),
                checkpoint_dir=eval_dir / f"run_{run_number:02d}",
                batch_backend=backend,
            )
        finally:
            backend.close()
        if tuple(stage.depth for stage in result.stages) != tuple(config.stage_depths):
            raise RuntimeError("final screening requires every configured depth stage")
        results.append(result)
        atomic_json(eval_dir / f"run_{run_number:02d}" / "backend_calls.json", backend.calls)
        print(f"[run {run_number}] completed archive={len(result.archive_records)} "
              f"evaluated={result.evaluated_count} cache_hits={result.cache_hits}", flush=True)
    evolution_seconds = time.perf_counter() - started

    gene_pools = {depth: [] for depth in config.stage_depths}
    archive_payload, histories = [], []
    for run_number, result in enumerate(results, start=1):
        for stage in result.stages:
            gene_pools[stage.depth].extend(
                _candidate_payload(c, run_number=run_number, stage_depth=stage.depth) for c in stage.gene_pool
            )
        archive_payload.extend(_archive_payload(r, run_number=run_number) for r in result.archive_records)
        histories.extend({"run": run_number, **row} for row in result.history)
    for depth, payload in gene_pools.items():
        atomic_json(args.output_dir / f"gene_pool_depth_{depth:02d}.json", payload)
    atomic_json(args.output_dir / "pca_qd_archive.json", archive_payload)
    _write_csv_atomic(pd.DataFrame(histories), args.output_dir / "gp_history.csv")

    screening_seconds = 0.0
    selected = ()
    if not args.skip_screening:
        candidates = _screening_candidates(results, None, None)
        print(f"[screening] candidates={len(candidates)} rankic_threshold={config.final_rankic_threshold:.2f} "
              f"sample_step={config.final_correlation_sample_step}", flush=True)
        started = time.perf_counter()
        selected = screen_hierarchical_factors_native(
            candidates, args.screening_panels, args.output_dir,
            rankic_threshold=config.final_rankic_threshold,
            corr_threshold=config.final_correlation_threshold,
            sample_step=config.final_correlation_sample_step,
            limit=config.final_factor_count,
            progress_callback=_screening_progress_logger(),
            engine_options=dict(
                threads=config.engine_threads, memory_budget_gb=config.memory_budget_gb,
                max_chunk=config.engine_max_chunk, gpus=config.gpus, streams_per_gpu=config.streams_per_gpu,
                block=config.screening_block, engine_batch=config.screening_engine_batch,
                score_begin=int(screening_meta.get("score_begin", 0)),
            ),
        )
        screening_seconds = time.perf_counter() - started
        atomic_json(args.output_dir / "selected_gp_factors.json", [asdict(item) for item in selected])

    atomic_json(args.output_dir / "run_config.json", {
        "evolution_panels": str(args.evolution_panels),
        "screening_panels": str(args.screening_panels),
        "evolution_T": evolution_meta["T"], "screening_T": screening_meta["T"], "N": evolution_meta["N"],
        "screening_score_begin": screening_meta.get("score_begin", 0),
        "feature_count": len(evolution_meta["fields"]),
        "random_seeds": [config.random_seed + r for r in range(config.independent_runs)],
        "completed_depths": list(config.stage_depths),
        "gp_config": asdict(config),
        "batch_backends": identities,
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
        "smoke": args.smoke,
        "seconds": {"evolution": evolution_seconds, "screening": screening_seconds},
        "selected": len(selected),
    })
    print(f"[done] outputs in {args.output_dir} evolution={evolution_seconds:.1f}s "
          f"screening={screening_seconds:.1f}s selected={len(selected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
