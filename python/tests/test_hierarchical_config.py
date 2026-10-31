from gp_factor_miner.hierarchical_config import HierarchicalGPConfig


def test_pilot_contract_matches_approved_design():
    config = HierarchicalGPConfig().pilot()
    assert config.stage_depths == (1, 2, 3)
    assert config.population_size == 10_000
    assert config.generations_per_stage == 10
    assert config.independent_runs == 1
    assert config.warm_start_multiplier == 3
    assert config.pca_sample_step == 5
    assert config.pca_qd_block_size == 128
    assert config.pca_qd_backend == "auto"
    assert config.pca_similarity_threshold == 0.90
    assert config.qd_replacement_rankic_margin == 0.002
    assert config.final_rankic_threshold == 0.07
    assert config.final_correlation_threshold == 0.75
    assert config.final_factor_count is None
    assert config.checkpoint_version == 4


def test_smoke_configuration_is_small_and_depth_complete():
    config = HierarchicalGPConfig().smoke()
    assert config.stage_depths == (1, 2, 3)
    assert config.population_size == 24
    assert config.generations_per_stage == 2
    assert config.warm_start_multiplier == 2
    assert config.max_workers == 1
