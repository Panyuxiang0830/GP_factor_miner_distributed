from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys

import pandas as pd

from .cli import (
    _git_commit,
    _progress_bar,
    _write_csv_atomic,
)
from .data import (
    load_eod_panel_dataset,
    split_eod_panels,
)
from .expressions import expression_key
from .hierarchical_config import HierarchicalGPConfig
from .hierarchical_engine import (
    run_hierarchical_evolution,
    validate_hierarchical_checkpoint_payload,
)
from .hierarchical_ops import expression_operators
from .hierarchical_screening import screen_hierarchical_factors
from .output import atomic_json
from .raw_features import profile_raw_panels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Mine GP factors with hierarchical PCA-QD evolution (V4)"
    )
    parser.add_argument(
        "--data-file",
        "--data-dir",
        dest="data_file",
        type=Path,
        default=Path("/hy-tmp/autodl-tmp/EOD_Columns/eod"),
        help="EOD pickle containing a field-to-DataFrame dictionary",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("/hy-tmp/data/factor/label_10.pkl"),
        help="wide forward-return label pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/hy-tmp/data_pyx/GP_output_eod/run_01"),
        help="dedicated output directory for this EOD GP run",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpp-panel-dir", type=Path, help="opt-in live C++ backend: exported evolution panels")
    parser.add_argument("--cpp-engine-binary", type=Path)
    parser.add_argument("--cpp-pca-binary", type=Path)
    parser.add_argument("--cpp-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpp-ranks", type=int, default=1)
    parser.add_argument("--cpp-precision", choices=("float", "double"), default="double")
    parser.add_argument("--cpp-mpiexec", default="mpirun")
    return parser


def _load_or_profile(
    setup_dir,
    evolution_data,
    evolution_target,
    screening_data,
    screening_target,
    feature_names,
):
    evolution_path = setup_dir / "evolution_profile.csv"
    screening_path = setup_dir / "screening_profile.csv"
    if evolution_path.exists() and screening_path.exists():
        print("[setup] loading cached raw feature profiles", flush=True)
        evolution_profile = pd.read_csv(evolution_path)
        screening_profile = pd.read_csv(screening_path)
        expected = set(feature_names)
        if (
            set(evolution_profile.get("feature", ())) != expected
            or set(screening_profile.get("feature", ())) != expected
        ):
            raise RuntimeError("cached raw feature profiles do not match data schema")
        return evolution_profile, screening_profile

    print("[setup] profiling raw features on evolution split", flush=True)
    evolution_profile = profile_raw_panels(evolution_data, evolution_target)
    print("[setup] profiling raw features on screening split", flush=True)
    screening_profile = profile_raw_panels(screening_data, screening_target)
    _write_csv_atomic(evolution_profile, evolution_path)
    _write_csv_atomic(screening_profile, screening_path)
    print(f"[setup] cached raw feature profiles in {setup_dir}", flush=True)
    return evolution_profile, screening_profile


def _format_frequencies(values) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _format_optional_float(value) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _progress_logger(run_number, run_count, config):
    run_prefix = f"[run {run_number}/{run_count}]"

    def log(event):
        kind = event["event"]
        stage = event.get("stage")
        depth = event.get("depth")
        generation = event.get("generation")
        prefix = f"{run_prefix}[stage {stage}/{len(config.stage_depths)} depth={depth}]"
        if kind == "stage_started":
            print(f"{prefix} started", flush=True)
        elif kind == "chunk_completed":
            print(
                f"{prefix}[{event['phase']} generation={generation}] "
                f"chunk {event['chunk']}/{event['chunks']} "
                f"{_progress_bar(event['completed'], event['total'])} "
                f"valid={event['valid_candidates']}",
                flush=True,
            )
        elif kind == "qd_batch_completed":
            print(
                f"{prefix}[{event['phase']} generation={generation}] "
                f"QD admitted={event['qd_admitted']} rejected={event['qd_rejected']} "
                f"replaced={event['qd_replaced']} archive={event['archive_size']} "
                f"max_similarity={_format_optional_float(event['max_similarity'])} "
                f"median_similarity={_format_optional_float(event['median_similarity'])}",
                flush=True,
            )
        elif kind == "family_replacement_completed":
            print(
                f"{prefix}[generation {generation}/{config.generations_per_stage}] "
                f"family replacement={event['replaced']}/{event['families']}",
                flush=True,
            )
        elif kind in {"warm_start_completed", "generation_completed"}:
            label = "warm start" if kind == "warm_start_completed" else f"generation {generation}"
            print(
                f"{prefix} {label} completed archive={event['archive_size']} "
                f"best_fitness={event['best_fitness']:.6f} "
                f"best_abs_rank_ic={event['best_abs_rank_ic']:.6f} "
                f"features={_format_frequencies(event['feature_frequencies'])} "
                f"operators={_format_frequencies(event['operator_frequencies'])}",
                flush=True,
            )
        elif kind == "generation_started":
            print(
                f"{prefix}[generation {generation}/{config.generations_per_stage}] started "
                f"population={event['population']}",
                flush=True,
            )
        elif kind == "stage_completed":
            print(
                f"{prefix} completed gene_pool={event['gene_pool_size']} "
                f"archive={event['archive_size']}",
                flush=True,
            )

    return log


def _screening_progress_logger():
    def log(event):
        kind = event["event"]
        if kind.startswith("validation_"):
            stage = "validation"
        elif kind.startswith("correlation_"):
            stage = "correlation"
        else:
            return
        status = "completed" if kind.endswith("_completed") else "progress"
        selected = (
            f" selected={event['selected']}" if stage == "correlation" else ""
        )
        print(
            f"[screening][{stage}] {status} "
            f"{event['completed']}/{event['total']}{selected}",
            flush=True,
        )

    return log


def _candidate_payload(candidate, *, run_number, stage_depth):
    return {
        **asdict(candidate),
        "key": expression_key(candidate.expression),
        "run": run_number,
        "source_stage": stage_depth,
        "operators": sorted(expression_operators(candidate.expression)),
    }


def _archive_payload(record, *, run_number):
    return {
        "key": record.key,
        "expression": record.expression,
        "abs_rank_ic": record.abs_rank_ic,
        "nodes": record.nodes,
        "stage_depth": record.stage_depth,
        "run": run_number,
    }


def _screening_candidates(results, evolution_data, evolution_target):
    del evolution_data, evolution_target
    candidates = {}
    for result in results:
        for stage in result.stages:
            for candidate in stage.gene_pool:
                candidates.setdefault(expression_key(candidate.expression), candidate)
        for candidate in result.archive_candidates:
            candidates[expression_key(candidate.expression)] = candidate
    return tuple(candidates[key] for key in sorted(candidates))


def _package_versions():
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "deap"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _validate_v4_output_dir(output_dir, config, backend_identities=None, defer_backend_identity=False):
    destination = Path(output_dir)
    legacy_outputs = (
        destination / "selected_raw_features.json",
        destination / "factor_metrics.csv",
    )
    if any(path.exists() for path in legacy_outputs):
        raise RuntimeError(
            "V4 requires a new output directory separate from V3 outputs"
        )

    for run_dir in sorted((destination / "eval").glob("run_*")):
        if not run_dir.is_dir():
            continue
        suffix = run_dir.name.removeprefix("run_")
        paths = (run_dir / "run_state.pkl", run_dir / "completed.pkl")
        if not any(path.exists() for path in paths):
            continue
        if not suffix.isdigit() or not 1 <= int(suffix) <= config.independent_runs:
            raise RuntimeError(
                "checkpoint configuration is incompatible; use a new output directory"
            )
        run_number = int(suffix)
        seed = config.random_seed + run_number - 1
        for path in paths:
            if not path.exists():
                continue
            try:
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception as error:
                raise RuntimeError(
                    "checkpoint format is incompatible; use a new output directory"
                ) from error
            validate_hierarchical_checkpoint_payload(
                payload,
                config,
                seed,
                completed=path.name == "completed.pkl",
                backend_identity=(payload.get("batch_backend_identity") if defer_backend_identity
                                  else (backend_identities or {}).get(run_number)),
            )

def main(argv=None):
    args = build_parser().parse_args(argv)
    cpp_enabled = args.cpp_panel_dir is not None
    if any(x is not None for x in (args.cpp_panel_dir, args.cpp_engine_binary, args.cpp_pca_binary)) and not all(
        x is not None for x in (args.cpp_panel_dir, args.cpp_engine_binary, args.cpp_pca_binary)
    ):
        raise ValueError("C++ backend requires panel directory, engine binary and PCA binary together")
    config = HierarchicalGPConfig(
        data_file=args.data_file,
        label_file=args.label_file,
    ).pilot()
    if args.smoke:
        config = config.smoke()

    eval_dir = args.output_dir / "eval"
    setup_dir = eval_dir / "setup"
    print(f"[setup] loading EOD panels from {args.data_file}", flush=True)
    dataset = load_eod_panel_dataset(args.data_file, args.label_file)
    split = dataset.split
    feature_names = dataset.feature_names
    if args.smoke:
        smoke_population = min(config.population_size, len(feature_names))
        config = replace(
            config,
            population_size=smoke_population,
            gene_pool_size=min(config.gene_pool_size, smoke_population),
            parent_pool_size=min(config.parent_pool_size, smoke_population),
            pca_sample_step=(
                1
                if len(split.evolution_time_ids) <= config.pca_sample_step
                else config.pca_sample_step
            ),
            progress_chunk_size=min(config.progress_chunk_size, smoke_population),
        )
    _validate_v4_output_dir(args.output_dir, config, defer_backend_identity=cpp_enabled)
    evolution_data, evolution_target, screening_data, screening_target = (
        split_eod_panels(dataset)
    )
    if args.smoke:
        evolution_ids = list(evolution_target.index[:256])
        screening_ids = list(screening_target.index[:64])
        asset_coverage = evolution_target.notna().sum(axis=0)
        smoke_assets = list(
            asset_coverage.sort_values(ascending=False, kind="stable").index[:512]
        )
        if cpp_enabled:
            # The export defines the exact smoke coordinates; never silently
            # compute on a different stock subset than the upstream engine.
            metadata = json.loads((args.cpp_panel_dir / "meta.json").read_text())
            dates = {str(x): x for x in evolution_target.index}
            assets = {str(x): x for x in evolution_target.columns}
            try:
                evolution_ids = [dates[x] for x in metadata["dates"][:256]]
                smoke_assets = [assets[x] for x in metadata["assets"][:512]]
            except KeyError as error:
                raise ValueError("C++ export calendar/assets do not match the evolution split") from error
        evolution_data = {
            name: panel.loc[evolution_ids, smoke_assets]
            for name, panel in evolution_data.items()
        }
        screening_data = {
            name: panel.loc[screening_ids, smoke_assets]
            for name, panel in screening_data.items()
        }
        evolution_target = evolution_target.loc[evolution_ids, smoke_assets]
        screening_target = screening_target.loc[screening_ids, smoke_assets]

    evolution_profile, _ = _load_or_profile(
        setup_dir,
        evolution_data,
        evolution_target,
        screening_data,
        screening_target,
        feature_names,
    )

    batch_backends = {}
    if cpp_enabled:
        bridge_dir = Path(__file__).resolve().parents[3] / "pca"
        if not (bridge_dir / "gp_bridge.py").exists():
            raise RuntimeError("C++ backend requires the repository pca/ module")
        sys.path.insert(0, str(bridge_dir))
        from gp_bridge import CppGPBackend
        for run in range(1, config.independent_runs + 1):
            implementation = CppGPBackend(
                args.cpp_panel_dir, args.cpp_engine_binary, args.cpp_pca_binary,
                eval_dir / f"cpp_backend_{run:02d}", dates=len(evolution_target.index),
                stocks=len(evolution_target.columns), ranks=args.cpp_ranks,
                device=args.cpp_device, precision=args.cpp_precision,
                mpiexec=args.cpp_mpiexec,
            )
            implementation.validate_inputs(evolution_data, evolution_target, config)
            batch_backends[run] = implementation
        _validate_v4_output_dir(args.output_dir, config, {run: impl.identity for run, impl in batch_backends.items()})

    results = []
    for run in range(config.independent_runs):
        run_number = run + 1
        seed = config.random_seed + run
        print(
            f"[run {run_number}/{config.independent_runs}] started seed={seed}",
            flush=True,
        )
        backend_options = {"batch_backend": batch_backends[run_number]} if cpp_enabled else {}
        result = run_hierarchical_evolution(
            evolution_data,
            evolution_target,
            evolution_profile,
            config,
            seed=seed,
            progress_callback=_progress_logger(
                run_number, config.independent_runs, config
            ),
            checkpoint_dir=eval_dir / f"run_{run_number:02d}",
            **backend_options,
        )
        completed_depths = tuple(stage.depth for stage in result.stages)
        if completed_depths != tuple(config.stage_depths):
            raise RuntimeError(
                "final screening requires completed depth stages 1, 2, and 3"
            )
        results.append(result)
        print(
            f"[run {run_number}/{config.independent_runs}] completed "
            f"archive={len(result.archive_records)} evaluated={result.evaluated_count} "
            f"cache_hits={result.cache_hits}",
            flush=True,
        )

    gene_pool_payloads = {depth: [] for depth in config.stage_depths}
    archive_payload = []
    histories = []
    for run_number, result in enumerate(results, start=1):
        for stage in result.stages:
            gene_pool_payloads[stage.depth].extend(
                _candidate_payload(
                    candidate, run_number=run_number, stage_depth=stage.depth
                )
                for candidate in stage.gene_pool
            )
        archive_payload.extend(
            _archive_payload(record, run_number=run_number)
            for record in result.archive_records
        )
        histories.extend({"run": run_number, **row} for row in result.history)

    for depth, payload in gene_pool_payloads.items():
        atomic_json(args.output_dir / f"gene_pool_depth_{depth:02d}.json", payload)
    atomic_json(args.output_dir / "pca_qd_archive.json", archive_payload)
    _write_csv_atomic(pd.DataFrame(histories), args.output_dir / "gp_history.csv")

    candidates = _screening_candidates(
        results, evolution_data=evolution_data, evolution_target=evolution_target
    )
    print(
        f"[screening] validation started candidates={len(candidates)} "
        f"rankic_threshold={config.final_rankic_threshold:.2f}",
        flush=True,
    )
    selected = screen_hierarchical_factors(
        candidates,
        screening_data,
        screening_target,
        args.output_dir,
        rankic_threshold=config.final_rankic_threshold,
        corr_threshold=config.final_correlation_threshold,
        sample_step=config.pca_sample_step,
        limit=config.final_factor_count,
        progress_callback=_screening_progress_logger(),
    )

    atomic_json(
        args.output_dir / "selected_gp_factors.json",
        [asdict(item) for item in selected],
    )
    run_config = {
        "data_file": str(args.data_file),
        "label_file": str(args.label_file),
        "source_files": [args.data_file.name, args.label_file.name],
        "source_field_count": dataset.source_field_count,
        "feature_count": len(feature_names),
        "excluded_fields": list(dataset.excluded_fields),
        "common_asset_count": dataset.common_asset_count,
        "n_time_ids": split.n_time_ids,
        "evolution_start": split.evolution_time_ids[0],
        "evolution_end": split.evolution_end,
        "screening_start": split.screening_time_ids[0],
        "screening_end": split.screening_end,
        "processed_evolution_time_count": len(evolution_target.index),
        "processed_screening_time_count": len(screening_target.index),
        "random_seeds": [
            config.random_seed + run for run in range(config.independent_runs)
        ],
        "completed_depths": list(config.stage_depths),
        "checkpoint_version": config.checkpoint_version,
        "pca_similarity_threshold": config.pca_similarity_threshold,
        "final_rankic_threshold": config.final_rankic_threshold,
        "final_correlation_threshold": config.final_correlation_threshold,
        "final_factor_count": config.final_factor_count,
        "gp_config": asdict(config),
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
        "smoke": args.smoke,
        "batch_backends": {run: impl.identity for run, impl in batch_backends.items()},
    }
    atomic_json(args.output_dir / "run_config.json", run_config)
    print(f"[done] V4 outputs written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
