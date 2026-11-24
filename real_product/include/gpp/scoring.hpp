#pragma once
// Fused per-root scoring: one cross-sectional sort per day feeds the daily
// RankIC, the PCA-QD signature rows and (out-of-sample screening) the sampled
// rank vector. The course engine sorted every day twice for the first two
// (daily_rank_ic, then pca::rank_rows) and the PCA bridge re-read the panels.

#include <cstdint>
#include <vector>

#include "gpe/metrics.hpp"
#include "gpe/panel.hpp"

namespace gpp {

// Percentile ranks of one day as pandas rank(method="average", pct=True):
// NaN stays NaN, ties get their average rank. Same arithmetic as gpe.
void rank_row_pct(const float* x, int N, double* out, std::vector<int>& idx);

// DataFrame.corrwith(axis=1) for one day over pairwise-complete entries,
// exactly as gpe::daily_rank_ic (constant series -> NaN, inf -> NaN).
double pearson_pairwise(const double* x, const double* y, int N);

struct ScoreOptions {
    gpe::FitnessConfig fitness;
    int begin = 0;                // first scored row (warm-up rows before it are skipped)
    int signature_step = 0;       // PCA-QD sampling step; 0 disables signatures
    int power_steps = 4;          // power iterations (Python: pca_power_iterations)
    int rank_vector_step = 0;     // OOS correlation sampling step; 0 disables
    double rank_vector_min_abs_ic = 0.0;  // drop the vector unless |mean IC| >= this (memory)
    bool signature_only_if_valid = true;  // skip the PCA when the candidate is rejected
};

// Result for one root.
struct RootScore {
    gpe::Candidate candidate;     // fitness statistics (valid=false: Python returns None)
    double mean_ic = 0.0;         // mean of finite daily ICs (NaN if none) - screening
    int valid_times = 0;          // finite daily ICs
    int scored_days = 0;          // days scored (len(target.index))
    double missing_ratio = 1.0;
    bool signature_valid = false;
    std::vector<float> signature;    // m = ceil(scored_days / signature_step)
    std::vector<float> rank_vector;  // rows(begin::rank_vector_step) x N, NaN kept
};

// Per-thread scratch reused across roots (no allocation in the hot loop).
struct ScoreScratch {
    std::vector<int> idx;
    std::vector<double> rank;
    std::vector<double> R;        // sampled rows x N
};

// Rows sampled by Python's values.iloc[::step] over `days` rows.
inline int sampled_rows(int days, int step) { return step > 0 ? (days + step - 1) / step : 0; }

// Shared, read-only inputs.
struct ScoreContext {
    const std::vector<double>* target_rank = nullptr;  // T x N, NaN where target is NaN
    const std::vector<double>* initial = nullptr;      // power-iteration start (N)
    int T = 0, N = 0;
    ScoreOptions options;
};

// Daily RankIC + candidate statistics + signature for one root panel. Serial
// (called by one thread per root under node-parallel scheduling).
void score_root(const gpe::Panel& panel, int depth, int nodes, const ScoreContext& ctx, ScoreScratch& scratch,
                RootScore& out);

// From an already computed daily IC series (the CUDA scorer returns these):
// fills candidate, mean_ic, valid_times and missing_ratio like score_root.
void finish_from_daily(const std::vector<double>& ic, long eligible, long valid, int depth, int nodes,
                       const gpe::FitnessConfig& fitness, RootScore& out);

// pca::solve(power) + pca::finish on a demeaned rank matrix; false if the
// signature is not comparable (Python: _signature_is_comparable).
bool power_signature(std::vector<double>& R, int rows, int N, const std::vector<double>& initial, int steps,
                     std::vector<float>& signature);
bool signature_comparable(const std::vector<float>& signature);

}  // namespace gpp
