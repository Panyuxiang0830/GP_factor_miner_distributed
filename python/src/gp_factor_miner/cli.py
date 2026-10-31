from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess
import os

import pandas as pd

from .config import GPConfig
from .data import (
    compute_time_split,
    discover_schema,
    load_research_rows,
    split_research_frame,
)
from .gp_engine import run_evolution
from .output import write_outputs
from .raw_features import lock_raw_threshold, profile_raw_features
from .screening import screen_gp_factors, screen_raw_features


def _panels(frame, feature_names):
    names = list(feature_names)
    wide = frame.pivot(
        index="time_id", columns="asset_id", values=names
    ).sort_index()
    return {name: wide[name] for name in names}


def _target_panel(frame):
    return frame.pivot(index="time_id", columns="asset_id", values="target").sort_index()


def _first_time_ids(frame, count):
    selected = frame["time_id"].drop_duplicates().sort_values().iloc[:count]
    return frame[frame["time_id"].isin(selected)].copy()


def _progress_bar(completed, total, width=20):
    ratio = completed / total if total else 1.0
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio:6.1%}"


def _write_csv_atomic(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser():
    parser = argparse.ArgumentParser(description="Mine GP factors with Rank IC")
    parser.add_argument("--data-dir", type=Path, default=Path("/hy-tmp/data_pyx/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = GPConfig(data_dir=args.data_dir)
    if args.smoke:
        config = config.smoke()
    eval_dir = args.output_dir / "eval"
    setup_dir = eval_dir / "setup"
    print(f"[setup] discovering data in {args.data_dir}", flush=True)
    schema = discover_schema(args.data_dir)
    split = compute_time_split(args.data_dir)
    columns = ["time_id", "asset_id", "target", *schema.feature_names]
    research = load_research_rows(args.data_dir, split, columns)
    print(
        f"[setup] loaded research rows; features={len(schema.feature_names)}",
        flush=True,
    )
    evolution_frame, screening_frame = split_research_frame(research, split)
    if args.smoke:
        evolution_frame = _first_time_ids(
            evolution_frame, config.smoke_evolution_time_count
        )
        screening_frame = _first_time_ids(
            screening_frame, config.smoke_screening_time_count
        )
        research = pd.concat([evolution_frame, screening_frame], ignore_index=True)

    evolution_profile_path = setup_dir / "evolution_profile.csv"
    screening_profile_path = setup_dir / "screening_profile.csv"
    if evolution_profile_path.exists() and screening_profile_path.exists():
        print("[setup] loading cached raw feature profiles", flush=True)
        evolution_profile = pd.read_csv(evolution_profile_path)
        screening_profile = pd.read_csv(screening_profile_path)
        expected_features = set(schema.feature_names)
        if (
            set(evolution_profile["feature"]) != expected_features
            or set(screening_profile["feature"]) != expected_features
        ):
            raise RuntimeError("cached raw feature profiles do not match data schema")
    else:
        print("[setup] profiling raw features on evolution split", flush=True)
        evolution_profile = profile_raw_features(evolution_frame, schema.feature_names)
        print("[setup] profiling raw features on screening split", flush=True)
        screening_profile = profile_raw_features(screening_frame, schema.feature_names)
        _write_csv_atomic(evolution_profile, evolution_profile_path)
        _write_csv_atomic(screening_profile, screening_profile_path)
        print(f"[setup] cached raw feature profiles in {setup_dir}", flush=True)
    raw_threshold = lock_raw_threshold(
        evolution_profile,
        preferred=config.raw_threshold_preferred,
        fallback=config.raw_threshold_fallback,
    )
    selected_raw = screen_raw_features(
        evolution_profile, screening_profile, threshold=raw_threshold
    )

    evolution_data = _panels(evolution_frame, schema.feature_names)
    evolution_target = _target_panel(evolution_frame)
    all_research_data = _panels(research, schema.feature_names)
    screening_target = _target_panel(screening_frame)
    candidates = []
    histories = []
    for run in range(config.independent_runs):
        seed = config.random_seed + run
        run_number = run + 1
        print(
            f"[run {run_number}/{config.independent_runs}] started seed={seed}",
            flush=True,
        )

        def log_progress(event, run_number=run_number):
            kind = event["event"]
            generation = event["generation"]
            prefix = (
                f"[run {run_number}/{config.independent_runs}]"
                f"[generation {generation}/{config.generations}]"
            )
            if kind == "generation_started":
                print(
                    f"{prefix} started population={event['population']} "
                    f"pending={event['pending']} cached={event['cached']} "
                    f"chunks={event['chunks']}",
                    flush=True,
                )
            elif kind == "chunk_completed":
                print(
                    f"{prefix} chunk {event['chunk']}/{event['chunks']} "
                    f"{_progress_bar(event['completed'], event['total'])} "
                    f"evaluated={event['completed']}/{event['total']} "
                    f"valid={event['valid_candidates']}",
                    flush=True,
                )
            elif kind == "generation_completed":
                print(
                    f"{prefix} completed best_fitness={event['best_fitness']:.6f} "
                    f"best_abs_rank_ic={event['best_abs_rank_ic']:.6f} "
                    f"unique_evaluated={event['unique_evaluated']}",
                    flush=True,
                )
            elif kind == "island_generation_completed":
                print(
                    f"{prefix}[island {event['island']}] "
                    f"valid={event['valid_candidates']} "
                    f"best_fitness={event['best_fitness']:.6f} "
                    f"best_abs_rank_ic={event['best_abs_rank_ic']:.6f}",
                    flush=True,
                )
            elif kind == "migration_completed":
                print(
                    f"{prefix} migration completed migrants={event['count']}",
                    flush=True,
                )
            elif kind == "island_reproduction_completed":
                print(
                    f"{prefix}[island {event['island']}] parents={event['parent_pool']} "
                    f"feature_cap={event['parent_cap']:.0%} "
                    f"tournament={event['tournament']} lexicase={event['lexicase']}",
                    flush=True,
                )

        result = run_evolution(
            evolution_data,
            evolution_target,
            evolution_profile,
            config,
            seed=seed,
            progress_callback=log_progress,
            checkpoint_dir=eval_dir / f"run_{run_number:02d}",
        )
        candidates.extend(candidate for candidate in result.hall_of_fame if candidate.depth > 0)
        histories.extend({"run": run, "seed": seed, **row} for row in result.history)
        print(
            f"[run {run_number}/{config.independent_runs}] completed "
            f"hof={len(result.hall_of_fame)} evaluated={result.evaluated_count} "
            f"cache_hits={result.cache_hits}",
            flush=True,
        )

    selected_gp = screen_gp_factors(
        candidates,
        all_research_data,
        screening_target,
        threshold=config.gp_threshold,
        corr_threshold=config.dedup_correlation_threshold,
        limit=config.final_factor_count,
    )
    metrics = pd.DataFrame([asdict(item) for item in selected_gp])
    gp_payload = [asdict(item) for item in selected_gp]
    raw_payload = selected_raw.to_dict(orient="records")
    run_config = {
        "data_dir": str(args.data_dir),
        "source_files": sorted(path.name for path in args.data_dir.glob("*.parquet")),
        "n_time_ids": split.n_time_ids,
        "evolution_end": split.evolution_end,
        "screening_end": split.screening_end,
        "research_time_count": len(split.evolution_time_ids) + len(split.screening_time_ids),
        "processed_evolution_time_count": int(evolution_frame["time_id"].nunique()),
        "processed_screening_time_count": int(screening_frame["time_id"].nunique()),
        "raw_threshold": raw_threshold,
        "gp_threshold": config.gp_threshold,
        "random_seeds": [config.random_seed + run for run in range(config.independent_runs)],
        "gp_config": asdict(config),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "deap")
        },
        "git_commit": _git_commit(),
        "smoke": args.smoke,
    }
    write_outputs(
        args.output_dir,
        raw_features=raw_payload,
        gp_factors=gp_payload,
        metrics=metrics,
        history=pd.DataFrame(histories),
        run_config=run_config,
    )
    print(f"[done] outputs written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
