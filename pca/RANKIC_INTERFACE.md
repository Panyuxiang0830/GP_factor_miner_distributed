# GPU percentile ranking shared with RankIC

When a caller needs agreement with the original engine's double-valued `gpe::rank_rows_pct`, use the FP64 entry point in `gpu.hpp`:

```cpp
pca::gpu_select(local_rank);
pca::GpuTiming timing;
std::vector<double> target_ranks =
    pca::gpu_percentile_rows_f64(target.buf, target.T, target.N, timing);
auto daily = gpe::daily_rank_ic(factor, target_ranks);
```

Here `target` and `factor` are upstream `gpe::Panel` objects with matching T and N. This example passes only target ranks to the existing consumer; factor ranking inside `daily_rank_ic` still uses the original implementation. The interface does not change the global GP call chain or establish a complete GPU RankIC speedup.

Input is a contiguous host FP32 array `[row][stock]`; output is a host FP64 array in the same order. Valid values use average rank divided by the row's non-NaN count. Ties share ranks; NaNs remain missing; positive and negative infinity participate in numeric order. Entirely missing rows remain all NaN, without 0.5 filling or centering. `stocks` must be in 1...8192, and `rows` may be zero. The caller handles sampling, date alignment, and pairwise-valid masks.

Three entry points share the same sorting kernel:

| Entry point | Return type | Purpose |
| --- | --- | --- |
| gpu_rank_rows | float | PCA preprocessing: rank, fill NaN with 0.5, center each row |
| gpu_percentile_rows | float | Raw average percentiles, preserving NaN |
| gpu_percentile_rows_f64 | double | Raw average percentiles matching the upstream double contract |

The FP64 entry point does not duplicate sorting. Average percentiles have the form q/(2V), where V is the non-NaN count and q is an integer. For V <= 8192, the FP32 percentile error multiplied by 2V stays below 1/2. GPU rounding therefore recovers exact q, after which FP64 division uses the exact half-integer rank. This reuses the FP32 sorting result instead of simply converting an already-rounded float to double.

The call includes host-to-device upload, two GPU kernels, and FP64 download. `GpuTiming` records upload, ranking, and download separately. It currently returns host arrays; a complete GPU RankIC pipeline needs a separate device-residency and consumer design.

`test_rankic_interface.cpp` compares every value against the original engine's `rank_rows_pct`, passes GPU output to the actual `daily_rank_ic`, and checks IC, valid dates, and missing statistics. Cases include 1/17/257/5876/8192 columns, ties, all-NaN rows, constants, +/- infinity, and signed zero; optional real raw panels can be added. `rankic_interface.pbs` also runs existing PCA GPU checks for compatibility and writes private `SUCCESS` only after all checks pass.

Build with `-DPCA_CUDA=ON -DPCA_ENGINE_BENCH=ON`; the experiment-engine target enables consumer validation. The interface and tests can be handed to the RankIC maintainer for integration, but adoption must be confirmed by that maintainer.
