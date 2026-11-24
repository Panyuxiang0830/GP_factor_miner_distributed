"""Production hierarchical GP factor miner (one 128-core host, 4 x RTX 4090).

Modules
-------
config      ProductionGPConfig, host presets (configs/*.json)
hier_engine hierarchical evolution (copy of gp_factor_miner.hierarchical_engine,
            depth 1..5, per-stage node caps, throttled checkpoints)
selection   vectorized epsilon-lexicase with identical random draws
native      ctypes binding of libgpp_native
backend     in-process batch backend (evaluate + QD admission)
screening   native 2022 out-of-sample screening (same rules and CSVs)
export      panel export with optional screening warm-up rows
cli         end-to-end run
"""
