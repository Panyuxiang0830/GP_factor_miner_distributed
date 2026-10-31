from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class HierarchicalGPConfig:
    data_file: Path = Path("/hy-tmp/autodl-tmp/EOD_Columns/eod")
    label_file: Path = Path("/hy-tmp/data/factor/label_10.pkl")
    stage_depths: tuple[int, ...] = (1, 2, 3)
    population_size: int = 10_000
    generations_per_stage: int = 10
    independent_runs: int = 1
    warm_start_multiplier: int = 3
    gene_pool_size: int = 2_000
    max_nodes: int = 11
    windows: tuple[int, ...] = field(default=(1, 5, 10, 15, 20, 30, 60))
    # EOD has roughly 1,000 evolution dates.  Step 5 retains about 200
    # observations in each behavioral signature; the old intraday data used 100.
    pca_sample_step: int = 5
    pca_power_iterations: int = 4
    pca_qd_block_size: int = 128
    pca_qd_backend: str = "auto"
    pca_similarity_threshold: float = 0.90
    qd_replacement_rankic_margin: float = 0.002
    final_rankic_threshold: float = 0.07
    final_correlation_threshold: float = 0.75
    # None keeps every factor that passes RankIC and correlation screening.
    final_factor_count: int | None = None
    random_immigrant_ratio: float = 0.10
    tournament_selection_probability: float = 0.50
    tournament_size: int = 3
    lexicase_epsilon_scale: float = 1.0
    parent_pool_size: int = 2_000
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
    max_workers: int = 32
    progress_chunk_size: int = 1_000
    random_seed: int = 2026
    checkpoint_version: int = 4

    def pilot(self):
        return replace(self)

    def smoke(self):
        return replace(
            self,
            population_size=24,
            generations_per_stage=2,
            warm_start_multiplier=2,
            gene_pool_size=12,
            parent_pool_size=12,
            progress_chunk_size=8,
            max_workers=1,
            pca_sample_step=1,
        )
