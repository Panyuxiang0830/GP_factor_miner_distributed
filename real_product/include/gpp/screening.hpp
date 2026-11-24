#pragma once
// 2022 out-of-sample screening, the C++ version of
// screen_hierarchical_factors (python/src/gp_factor_miner/hierarchical_screening.py).
//
// Python evaluates every candidate twice (validation, then again for the
// correlation vector), one at a time, with no subtree sharing. Here:
//   1. all candidates go through the shared-memory engine as one DAG (or as
//      memory-bounded groups); each root yields its OOS RankIC and, if
//      |RankIC| passes, its sampled float32 rank vector in the same pass;
//   2. eligible factors are sorted by quality exactly as in Python;
//   3. blocks of B eligible factors get their masked Pearson correlations
//      against the factors already selected (and against each other) from
//      one batched product per block: with m the valid mask and
//      x = (rank - 1/2) * m, the six pairwise sums n, Sx, Sy, Sxx, Syy, Sxy
//      are inner products of [m, x, x^2] rows (a GEMM, GPU-friendly);
//   4. a serial scan keeps the original greedy order and tie rules. Pairs
//      whose approximate correlation is within `band` of the threshold or of
//      the row maximum are recomputed with the two-pass FP64 formula of
//      _absolute_correlation, so decisions do not depend on rounding.

#include <functional>
#include <string>
#include <vector>

#include "gpe/json.hpp"
#include "gpp/engine.hpp"

namespace gpp {

struct ScreenOptions {
    double rankic_threshold = 0.07;
    double corr_threshold = 0.75;
    int sample_step = 20;
    long limit = -1;              // final_factor_count; -1: no limit
    int block = 1024;             // eligible factors per product block
    int engine_batch = 50000;     // expressions per engine call
    double band = 1e-8;           // refinement band (FP64 product)
};

struct ScreenInput {
    const gpe::Json* expression;
    double evolution_rank_ic;
    int nodes;
    std::string key;              // expression_key, for the final tie-break
};

struct ScreenTiming {
    double evaluate = 0, sort = 0, products = 0, scan = 0, total = 0;
    long long refined = 0, pairs = 0;
};

struct ScreenResult {
    // One entry per input (Python's validation_rankic.csv rows).
    std::vector<double> mean, valid_ratio, missing_ratio;
    std::vector<int> valid_times;
    std::vector<std::string> error;
    // Eligible inputs in quality order; the first `scanned` were processed
    // (correlation_decisions.csv rows) before the limit stopped the scan.
    std::vector<int> order;
    long scanned = 0;
    std::vector<char> selected;           // per scanned row
    std::vector<double> max_correlation;  // per scanned row
    std::vector<int> correlated_with;     // input index or -1
    ScreenTiming timing;
};

// Pairwise masked |Pearson| as _absolute_correlation (NaN if < 3 pairs or a
// zero norm). Two-pass FP64.
double masked_abs_correlation(const float* a, const float* b, size_t n);

// Optional replacement for the block product (CUDA): fills, for the pairs
// (i, j) of `left` x `right`, the six sums n, Sx, Sy, Sxx, Syy, Sxy.
struct PairSums { double n, sx, sy, sxx, syy, sxy; };
using PairSumBackend = std::function<void(const std::vector<const float*>& left, const std::vector<const float*>& right,
                                          size_t length, std::vector<PairSums>& out)>;
void cpu_pair_sums(const std::vector<const float*>& left, const std::vector<const float*>& right, size_t length,
                   std::vector<PairSums>& out);

// The engine must be opened on the screening panels with
// score.rank_vector_step = sample_step and score.signature_step = 0.
ScreenResult screen_factors(Engine& engine, const std::vector<ScreenInput>& inputs, const ScreenOptions& options,
                            const PairSumBackend& backend = {});

}  // namespace gpp
