#pragma once

#include <vector>

#include "gpe/panel.hpp"

namespace gpe {

// Thresholds and penalties of the GP fitness, with the defaults of
// HierarchicalGPConfig in the Python miner.
struct FitnessConfig {
    int max_depth = 3;
    int max_nodes = 11;
    double min_valid_time_ratio = 0.80;
    double max_missing_ratio = 0.20;
    int stability_segments = 4;
    double stability_penalty = 0.50;
    double complexity_penalty = 0.0005;
    double minimum_same_sign_ratio = 0.75;
    double inconsistent_sign_multiplier = 0.25;
};

// Percentile ranks within each day, as pandas rank(axis=1, method="average",
// pct=True): rank / number of non-NaN values, ties get their average rank,
// NaN stays NaN (inf is ranked like any other value). Rows 0..T-1, T*N values.
// The target is ranked once per data set and reused for every factor.
std::vector<double> rank_rows_pct(const Panel& p);

// Daily RankIC series (E6), as metrics.daily_rank_ic in Python.
struct DailyIC {
    std::vector<double> ic;     // one value per day; NaN if undefined
    double missing_ratio = 1.0; // 1 - #(factor and target valid) / #(target valid)
};

// Pearson correlation per day between the factor's percentile ranks and
// target_rank (from rank_rows_pct), over stocks where both are non-NaN.
// Infinite correlations become NaN.
DailyIC daily_rank_ic(const Panel& factor, const std::vector<double>& target_rank);

// GP candidate statistics and fitness (E7–E8), as _evaluate_with_values and
// _fitness_from_segments in Python.
struct Candidate {
    bool valid = false;          // false where Python returns no candidate
    double rank_ic = 0.0;        // mean of the finite daily ICs
    double valid_ratio = 0.0;    // finite days / all days
    double missing_ratio = 1.0;
    std::vector<double> segment_rank_ics;  // np.array_split into equal parts
    double segment_rank_ic_std = 0.0;      // population std of finite segments
    double same_sign_ratio = 0.0;
    double fitness = 0.0;
};

// depth and nodes are expression_depth() and expression_nodes() of the
// normalized expression (Node::depth and Node::tree_nodes of its root).
Candidate score_factor(const DailyIC& daily, int depth, int nodes, const FitnessConfig& config = {});

}  // namespace gpe
