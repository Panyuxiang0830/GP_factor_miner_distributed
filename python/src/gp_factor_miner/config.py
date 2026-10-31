from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GPConfig:
    data_dir: Path = Path("/hy-tmp/data_pyx/train")
    population_size: int = 10_000
    generations: int = 30
    independent_runs: int = 10
    init_min_depth: int = 1
    init_max_depth: int = 2
    max_depth: int = 3
    max_nodes: int = 11
    crossover_probability: float = 0.70
    subtree_mutation_probability: float = 0.10
    elitism_ratio: float = 0.05
    tournament_size: int = 3
    uniform_feature_probability: float = 0.50
    rankic_weighted_probability: float = 0.50
    windows: tuple[int, ...] = field(default=(1, 5, 10, 15, 20, 30, 60))
    min_valid_time_ratio: float = 0.80
    max_missing_ratio: float = 0.20
    hall_of_fame_size: int = 500
    dedup_correlation_threshold: float = 0.90
    final_factor_count: int = 300
    raw_threshold_preferred: float = 0.03
    raw_threshold_fallback: float = 0.01
    gp_threshold: float = 0.03
    stability_segments: int = 4
    stability_penalty: float = 0.50
    complexity_penalty: float = 0.0005
    minimum_same_sign_ratio: float = 0.75
    inconsistent_sign_multiplier: float = 0.25
    random_seed: int = 2026
    max_workers: int = 32
    progress_chunk_size: int = 1_000
    island_size: int = 2_500
    island_preferred_operator_probability: float = 0.70
    migration_interval: int = 5
    migration_count: int = 50
    tournament_selection_probability: float = 0.50
    lexicase_epsilon_scale: float = 1.0
    parent_pool_size: int = 2_000
    parent_feature_cap_shares: tuple[float, ...] = (0.15, 0.20, 0.30)
    feature_mutation_probability: float = 0.15
    operator_mutation_probability: float = 0.10
    window_mutation_probability: float = 0.10
    random_immigrant_ratio: float = 0.10
    island_hall_of_fame_size: int = 125
    checkpoint_version: int = 2
    smoke_evolution_time_count: int = 256
    smoke_screening_time_count: int = 64

    def smoke(self) -> "GPConfig":
        values = dict(self.__dict__)
        values.update(
            population_size=48,
            island_size=12,
            parent_pool_size=10,
            migration_count=2,
            progress_chunk_size=12,
            generations=2,
            independent_runs=1,
        )
        return GPConfig(**values)
