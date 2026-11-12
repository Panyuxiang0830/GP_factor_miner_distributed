#pragma once
#include "core.hpp"
namespace pca {
using DeviceReduce = std::function<void(void*, int, bool)>;
struct GpuTiming {
    double upload = 0, download = 0, iteration = 0, gemv = 0, ranking = 0, similarity = 0;
    double bytes = 0, flops = 0;
    double nominal_memory_bytes_per_second = 0;
    int multiprocessors = 0, clock_khz = 0;
    std::string device;
};
std::vector<std::vector<float>> gpu_power(const std::vector<Matrix>& matrices,
    const std::vector<double>& initial, int steps, bool fp64, GpuTiming& timing,
    const DeviceReduce& device_sum = {});
// Input/output [factor][sampled day][stock]. NaNs excluded from average ranks.
std::vector<float> gpu_rank_rows(const std::vector<float>& panels, int rows, int stocks,
                                GpuTiming& timing);
// Reusable RankIC primitive: average percentile ranks, NaNs remain NaN,
// no fill and no row centering. Sampling/masking remain the caller's job.
std::vector<float> gpu_percentile_rows(const std::vector<float>& panels, int rows, int stocks,
                                      GpuTiming& timing);
// Center + normalize signatures on CPU; GPU performs only the tiled product.
std::vector<float> gpu_similarity(const std::vector<float>& left, int L,
                                const std::vector<float>& right, int A, int dimension,
                                GpuTiming& timing);
void gpu_select(int local_rank);
}  // namespace pca
