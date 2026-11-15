# Module interfaces and data conventions

R has sampled dates as rows and stocks as columns; a common shape is 293x5876. Existing R already has percentile ranking, NaN filling with 0.5, and row centering. Use `--rank-input 1` only for unranked sampled values.

## Files and APIs

| File | Purpose |
| --- | --- |
| core.hpp / core.cpp | CPU ranking, both solvers, handwritten Jacobi, similarity, and sequential archive |
| main.cpp | Independent batch driver, three MPI layouts, outputs, and stage timing |
| mpi_archive.hpp | Sharded archive signatures, local refinement, global MAXLOC |
| gpu.hpp / gpu.cu | Percentile ranking, batched power iteration, handwritten tiled similarity |
| c_api.cpp | C ABI used by Python numerical studies |
| reference.py | Same-input Python power reference, for validation only |
| validate.py | Per-candidate comparison with original Python admission and optional exact SVD samples |
| synthetic.py / adversarial.py | Synthetic inputs and threshold/tie boundary cases |

## Integration with the upstream engine

The upstream engine partitions original global trading dates, while the independent driver partitions already-sampled R rows. Their boundaries may differ. For integration, select owned dates with `t % sample_step == 0`, exclude halos, rank each selected stock row, and center it. The current upstream project has no legacy `gpe::signature_rows`; do not depend on that removed interface.

On CPU, put local sampled rows in `pca::Matrix` and use the same initial right vector on every rank. Wrap `MPI_Allreduce` as `pca::Sum` and pass it to `pca::solve`. Gather local signatures in global date order, then call `pca::finish` once; do not fix signs independently on each rank. The live `engine_bridge.cpp` uses upstream `keep_roots` to retain a bounded expression batch before sampling owned dates. See [GP_INTEGRATION.md](GP_INTEGRATION.md) for the optional Python backend, original CLI, and checkpoint contract.

GPU `gpu_rank_rows` fills missing values with 0.5 and centers rows for PCA. `gpu_percentile_rows` preserves NaN without filling or centering for RankIC callers. Sampling and pairwise-valid masks are the caller's responsibility. Input is contiguous FP32 `[row][stock]`; sorting currently supports at most 8192 columns. The `DeviceReduce` callback used by `gpu_power` receives a device pointer; confirm CUDA-aware MPI support before use.

The original engine's `rank_rows_pct` returns double. Use `gpu_percentile_rows_f64` when elementwise compatibility is needed. It reuses the sorting kernel, recovers exact average ranks on GPU, performs FP64 division, and returns host double arrays accepted by the original `daily_rank_ic` target-rank argument. See [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md). A callable API and consumer validation do not mean the main GP has automatically adopted it.

## Signatures and admission

Signature solving alternates `u=normalize(Rv)` and `v=normalize(R^T u)`, then computes one final `normalize(Rv)`. The result has a consistent sign and is converted to FP32. Lanczos approximates the leading direction using left/right bases and a small bidiagonal matrix. Exact NumPy SVD is for validation only.

Similarity separately recenters and normalizes signatures before taking absolute Pearson correlation. A nonzero R can still yield a signature constant across dates; such a signature is incomparable and skipped.

Admission preserves candidate order. Admit when maximum similarity is below 0.90; otherwise replace only if quality is strictly greater than the nearest record's quality plus 0.002. Quality is absolute RankIC. Similarity ties select the lexicographically larger key. FP32 similarity estimates and conservative error bounds identify comparisons requiring FP64 refinement to protect maximum/threshold decisions. Earlier admissions, replacements, and removals are immediately visible to later candidates.

By default, rank 0 maintains the archive; CUDA estimates similarity on the GPU. The MPI sharded archive is a separate path with replicated small metadata and global MAXLOC. It does not assume the archive remains unchanged within a block.

## Layouts and timing

`date` reduces across date partitions each iteration. `transpose` first exchanges complete matrices, then iterates locally. `factor` reads complete factors directly and excludes upstream redistribution. Fresh CPU ranking and its redistribution retain double precision; an existing FP32 R is redistributed as float.

Power in the date layout uses batched reductions; Lanczos currently reduces per factor, so fewer iterations do not necessarily mean fewer Allreduce calls. Numerical diagnostics add two matrix-vector products. Different ranks may provide different stage maxima; communication is nested within solve/redistribution timings, so adding stages does not recover total time.

Check validation markers before inspecting `metrics`, per-candidate `decisions`, and final `archive_keys`. A zero exit code or an existing directory alone does not establish correctness. All outputs stay in private runs and are excluded from the public repository.

To continue a historical batch, supply `--archive-input archive.json`. Centralized and sharded archives read the same keys, qualities, and signatures, and export a complete snapshot. The next batch must explicitly reference it and retain compatible sampled dates and algorithm conventions. Validation reads the actual initial archive from provenance rather than replaying a continued batch as if it started empty. See [DELIVERY.md](DELIVERY.md) for fields and final acceptance.
