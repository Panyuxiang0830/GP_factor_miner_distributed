# PCA-QD delivery index

This delivery covers project component 3: PCA signatures, similarity, sequential QD admission, and the optional backend connecting them to the original hierarchical GP. The development branch is `pca-implementation`; it has not been merged into `main`.

## Reading order

1. [README.md](README.md): build, module execution, and basic interfaces.
2. [GP_INTEGRATION.md](GP_INTEGRATION.md): connect the original GP to the live MPI engine, PCA, QD, subsequent generations, and checkpoints.
3. [DELIVERY.md](DELIVERY.md): archive format, validation semantics, and private report packaging.
4. [EXPERIMENTS.md](EXPERIMENTS.md) / [REPRODUCE.md](REPRODUCE.md): experiment entry points and reproduction steps.

See [WALKTHROUGH.md](WALKTHROUGH.md) for algorithms and boundary semantics, and [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md) for the GPU percentile contract consumed by RankIC.

## Main report

The current server-side report directory is `pca/runs/final_submission_04/`, relative to the Git repository root. The body follows `report/main_zh.tex`: background and algorithms, data, platform, parallel design, validation, experiments, discussion, and conclusions. Experiments cover PCA signatures/similarity first, then power versus Lanczos, followed by actual GP integration. It retains 7 result/parallel-summary tables and 8 main figures, with introductory material and repeated illustrations reduced. Original values are unchanged; the first three report versions remain available as historical material.

- **Main report: `pca_paper.pdf`.**
- Editable body: `pca_paper.md`; required figures are in `figures/`.
- Manuscript integration fragment: `latex/pca_sections.tex`.
- Complete repository source: `source.tar.gz`, retaining the upstream layout needed for compilation.
- Acceptance summary: `DELIVERY_STATUS.json` and `evidence/`; file hashes are in `MANIFEST_SHA256.json`.

The report combines PCA experiments, interface acceptance, actual GP integration, and a reduced-evolution-budget pilot with all fields and full data dimensions. Handoff requirements keep data, expressions, and experiment outputs out of the public repository, so reports and result packages remain private. GitHub contains source, synthetic checks, required reproduction scripts, and this index.

## Source scope

| Component | Main files |
|---|---|
| CPU/OpenMP and Lanczos | core.cpp / core.hpp |
| CUDA ranking, power iteration, and similarity | gpu.cu / gpu.hpp |
| Sequential/MPI archives and snapshots | mpi_archive.hpp / archive_io.hpp / archive_io.py |
| Independent module entry points | main.cpp / prepare.py / validate.py |
| Live factor-engine bridge and GP backend | engine_bridge.cpp / gp_bridge.py / run_gp.py |
| Original project opt-in interfaces | ../python/src/gp_factor_miner/hierarchical_engine.py / hierarchical_cli.py |

Other C++, Python, and PBS files support independent references, numerical acceptance, and reproduction of report experiments. Upstream build and include/src dependencies are required source, not temporary files. Production targets do not link BLAS/cuBLAS/cuSOLVER; libraries serve only independent references.

## Acceptance status and limits

CPU/MPI, CUDA, historical-archive continuation, ordered admission, independent numerical references, complete snapshots, and negative cases have passed validation. Actual three-stage GP loops, interruption/resume, the original CLI, and synthetic out-of-sample flow have also been checked. A reduced-evolution-budget pilot used all real fields and complete data dimensions. Cross-node GPU validation and same-node dual-GPU performance are recorded separately.

The current scope is a Python+C++ workflow using existing evolution-period data; real 2022 out-of-sample evaluation is excluded. Production population/generation runs are planned, with CPU configuration comparison pending completion and no production-scale results yet. The original CLI's synthetic out-of-sample check remains interface evidence. GPU Lanczos is outside scope, and the complete team LaTeX manuscript was not compiled for this delivery.

## Included files

Keep source, required tests, reproducible experiment entry points, and documentation. Exclude run directories, historical reports, build products, caches, temporary files, account records, market data, and factor expressions. Historical outputs remain in private run directories for traceability. `resume_pending.py`, used during an earlier scheduler outage, is no longer part of the delivery.
