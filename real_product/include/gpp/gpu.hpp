#pragma once
// CUDA side (cuda/*.cu, built with GPP_ENABLE_CUDA=ON). FP32 storage, FP64
// accumulation where the CPU result must be reproduced (RankIC sums, power
// iteration), and no file I/O: panels go host -> device straight from the
// engine's memory.
//
// GpuScorer: replaces score_root. A worker thread copies a finished root panel
// into one of streams_per_gpu x gpus pinned staging slots (memcpy, ~3 ms per
// 34 MB panel), enqueues H2D + kernels + D2H on that slot's stream and returns,
// so the next operator overlaps the transfer and the GPU work. The slot's
// previous result is harvested (stream sync) when the slot is reused, and all
// slots at flush(). Per day, one thread block sorts the cross-section in shared
// memory (cub::BlockRadixSort, up to 8192 stocks), assigns average ranks with
// two block scans, and reduces the pairwise Pearson sums in FP64; sampled days
// also write the demeaned rank row into the device signature matrix R. Power
// iteration then runs on R in place (FP32 R, FP64 vectors). Only the daily IC
// (T doubles), two counters and the signature (m floats) come back.
//
// gpu_summaries / gpu_pair_sums: the archive and screening products with
// cuBLAS SGEMM (TF32 off) on column tiles spread over the GPUs, followed by
// per-row reduction kernels so that only summaries cross PCIe.

#include <memory>
#include <vector>

#include "gpe/panel.hpp"
#include "gpp/archive.hpp"
#include "gpp/scoring.hpp"
#include "gpp/screening.hpp"

namespace gpp {

class GpuScorer {
public:
    GpuScorer(int gpus, int streams_per_gpu, int T, int N, const std::vector<double>& target_rank,
              const ScoreOptions& options, const std::vector<double>& initial);
    ~GpuScorer();
    void set_initial(const std::vector<double>& initial);
    // Thread-safe. The panel may be released as soon as this returns.
    void submit(const gpe::Panel& panel, int depth, int nodes, RootScore* out);
    // Waits for every submitted root and fills the RootScore objects.
    void flush();
    double busy_seconds() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

void gpu_summaries(const SummaryInputs& in, std::vector<RowSummary>& out, int gpus);
void gpu_pair_sums(const std::vector<const float*>& left, const std::vector<const float*>& right, size_t length,
                   std::vector<PairSums>& out, int gpus);
int gpu_count();

}  // namespace gpp
