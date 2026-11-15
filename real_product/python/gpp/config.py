"""Production configuration (one 128-core host, 4 x RTX 4090).

All fields of ``gp_factor_miner.hierarchical_config.HierarchicalGPConfig``
keep their names and meaning; defaults change for the 100k-population,
five-stage run, and the fields after ``checkpoint_version`` are new.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import json
from pathlib import Path

MAX_STAGE_DEPTH = 5


@dataclass(frozen=True)
class ProductionGPConfig:
    data_file: Path = Path("data/eod")
    label_file: Path = Path("data/label_10.pkl")
    stage_depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    population_size: int = 100_000
    generations_per_stage: int = 10
    independent_runs: int = 1
    warm_start_multiplier: int = 3
    # Gene pools and parent pools scale with the population (2,000 per 10,000).
    gene_pool_size: int = 20_000
    # Global node cap (evaluation filter); stage caps below are the binding ones.
    max_nodes: int = 21
    windows: tuple[int, ...] = field(default=(1, 5, 10, 15, 20, 30, 60))
    pca_sample_step: int = 5
    pca_power_iterations: int = 4
    pca_qd_block_size: int = 4096
    pca_qd_backend: str = "native"
    pca_similarity_threshold: float = 0.90
    qd_replacement_rankic_margin: float = 0.002
    final_rankic_threshold: float = 0.07
    final_correlation_threshold: float = 0.75
    final_factor_count: int | None = None
    random_immigrant_ratio: float = 0.10
    tournament_selection_probability: float = 0.50
    tournament_size: int = 3
    lexicase_epsilon_scale: float = 1.0
    parent_pool_size: int = 20_000
    parent_feature_cap_shares: tuple[float, ...] = (0.15, 0.20, 0.30)
    crossover_probability: float = 0.70
    feature_mutation_probability: float = 0.15
    operator_mutation_probability: float = 0.10
    window_mutation_probability: float = 0.10
    subtree_mutation_probability: float = 0.10
    stability_segments: int = 4
    stability_penalty: float = 0.50
    complexity_penalty: float = 0.0005
    minimum_same_sign_ratio: float = 0.75
    inconsistent_sign_multiplier: float = 0.25
    min_valid_time_ratio: float = 0.80
    max_missing_ratio: float = 0.20
    max_workers: int = 1
    # Expressions per native engine call (one DAG, or memory-bounded groups).
    progress_chunk_size: int = 50_000
    random_seed: int = 2026
    checkpoint_version: int = 4
    # ---- new fields -------------------------------------------------------
    # Node cap per stage depth (stage_depths[i] -> stage_max_nodes[i]): the
    # course cap of 11 at depth 3, plus 5 per further level (a binary op with
    # one promoted child adds about that much).
    stage_max_nodes: tuple[int, ...] = (3, 7, 11, 16, 21)
    # Seconds between in-generation checkpoints. The original saved after
    # every 1,000-expression chunk, which pickles the whole cache (millions
    # of candidates at this scale); generation and stage ends always save.
    checkpoint_interval_seconds: float = 900.0
    # Sampling step of the final correlation de-duplication. The course CLI
    # passed pca_sample_step (5) here; the screening function's own default
    # is 20 (used by the task specification). Both are supported.
    final_correlation_sample_step: int = 20
    # Native engine.
    engine_threads: int = 0              # 0: all cores (OMP_NUM_THREADS)
    memory_budget_gb: float = 0.0        # 0: unlimited; set from the host preset
    engine_max_chunk: int = 2000
    gpus: int = 0                        # CUDA scorer / similarity products
    streams_per_gpu: int = 4
    screening_block: int = 1024
    screening_engine_batch: int = 50_000
    # Prepended history rows in the screening export (0 = Python semantics:
    # windows restart on the first 2022 day).
    screening_warmup_days: int = 0

    def stage_nodes(self, depth: int) -> int:
        return self.stage_max_nodes[self.stage_depths.index(depth)]

    def smoke(self):
        return replace(
            self,
            population_size=48,
            generations_per_stage=2,
            warm_start_multiplier=2,
            gene_pool_size=16,
            parent_pool_size=16,
            progress_chunk_size=64,
            pca_sample_step=1,
            final_correlation_sample_step=2,
            pca_qd_block_size=16,
            checkpoint_interval_seconds=0.0,
        )

    def pilot(self):
        return replace(self)


def load_config(path: str | Path | None = None, **overrides) -> ProductionGPConfig:
    """Defaults <- JSON preset (configs/*.json, key "gp") <- keyword overrides."""
    values = {}
    if path is not None:
        payload = json.loads(Path(path).read_text())
        values.update(payload.get("gp", payload))
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name: f for f in fields(ProductionGPConfig)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for name in ("stage_depths", "windows", "parent_feature_cap_shares", "stage_max_nodes"):
        if name in values:
            values[name] = tuple(values[name])
    for name in ("data_file", "label_file"):
        if name in values:
            values[name] = Path(values[name])
    return ProductionGPConfig(**values)
