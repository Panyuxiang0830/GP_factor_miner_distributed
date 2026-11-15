#include "gpp/scoring.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core.hpp"  // pca::solve / pca::finish from ../pca (reused unchanged)

namespace gpp {
namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

// Copied from src/core/metrics.cpp (anonymous namespace there), unchanged.
void rank_row_pct(const float* x, int N, double* out, std::vector<int>& idx) {
    idx.clear();
    for (int s = 0; s < N; ++s) {
        out[s] = kNaN;
        if (!std::isnan(x[s])) idx.push_back(s);
    }
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return x[i] < x[j]; });
    double n = static_cast<double>(idx.size());
    for (size_t i = 0; i < idx.size();) {
        size_t j = i;
        while (j + 1 < idx.size() && x[idx[j + 1]] == x[idx[i]]) ++j;
        double avg = 0.5 * ((i + 1) + (j + 1));
        for (size_t k = i; k <= j; ++k) out[idx[k]] = avg / n;
        i = j + 1;
    }
}

// Copied from src/core/metrics.cpp, unchanged.
double pearson_pairwise(const double* x, const double* y, int N) {
    double sx = 0.0, sy = 0.0;
    int n = 0;
    bool x_const = true, y_const = true;
    double x0 = 0.0, y0 = 0.0;
    for (int s = 0; s < N; ++s) {
        if (std::isnan(x[s]) || std::isnan(y[s])) continue;
        if (n == 0) { x0 = x[s]; y0 = y[s]; }
        x_const = x_const && x[s] == x0;
        y_const = y_const && y[s] == y0;
        sx += x[s]; sy += y[s]; ++n;
    }
    if (n < 2 || x_const || y_const) return kNaN;
    double mx = sx / n, my = sy / n, sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int s = 0; s < N; ++s) {
        if (std::isnan(x[s]) || std::isnan(y[s])) continue;
        double dx = x[s] - mx, dy = y[s] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    double dom = (n - 1) * std::sqrt(sxx / (n - 1)) * std::sqrt(syy / (n - 1));
    double r = sxy / dom;
    return std::isinf(r) ? kNaN : r;
}

bool signature_comparable(const std::vector<float>& s) {
    if (s.size() < 2) return false;
    double mean = 0.0;
    for (float v : s) {
        if (!std::isfinite(v)) return false;
        mean += v;
    }
    mean /= static_cast<double>(s.size());
    double ss = 0.0;
    for (float v : s) ss += (v - mean) * (v - mean);
    return std::sqrt(ss) > 0.0;
}

bool power_signature(std::vector<double>& R, int rows, int N, const std::vector<double>& initial, int steps,
                     std::vector<float>& signature) {
    signature.clear();
    // The matrix is swapped in and out to avoid a copy of rows x N doubles.
    pca::Matrix M;
    M.rows = rows;
    M.cols = N;
    M.data.swap(R);
    auto solution = pca::solve(M, initial, steps, "power", {}, false);
    M.data.swap(R);
    if (solution.local_signature.empty()) return false;
    signature = pca::finish(solution.local_signature);
    if (signature.empty()) return false;
    return signature_comparable(signature);
}

void finish_from_daily(const std::vector<double>& ic, long eligible, long valid, int depth, int nodes,
                       const gpe::FitnessConfig& fitness, RootScore& out) {
    gpe::DailyIC daily;
    daily.ic = ic;
    daily.eligible = eligible;
    daily.valid = valid;
    daily.missing_ratio = eligible == 0 ? 1.0 : 1.0 - static_cast<double>(valid) / static_cast<double>(eligible);
    out.candidate = gpe::score_factor(daily, depth, nodes, fitness);
    double sum = 0.0;
    int finite = 0;
    for (double v : ic)
        if (std::isfinite(v)) { sum += v; ++finite; }
    out.valid_times = finite;
    out.scored_days = static_cast<int>(ic.size());
    out.mean_ic = finite ? sum / finite : kNaN;
    out.missing_ratio = daily.missing_ratio;
}

void score_root(const gpe::Panel& panel, int depth, int nodes, const ScoreContext& ctx, ScoreScratch& scratch,
                RootScore& out) {
    const ScoreOptions& o = ctx.options;
    const int N = ctx.N, begin = o.begin, days = ctx.T - begin;
    if (panel.N != N || panel.T != ctx.T || days <= 0) throw std::invalid_argument("score_root: panel shape");
    const int m = sampled_rows(days, o.signature_step);
    const int mv = sampled_rows(days, o.rank_vector_step);
    scratch.rank.resize(N);
    if (m > 0) scratch.R.resize(static_cast<size_t>(m) * N);
    if (mv > 0) out.rank_vector.assign(static_cast<size_t>(mv) * N, std::numeric_limits<float>::quiet_NaN());
    else out.rank_vector.clear();

    std::vector<double> ic(days, kNaN);
    long eligible = 0, valid = 0;
    const std::vector<double>& target = *ctx.target_rank;
    for (int d = 0; d < days; ++d) {
        const int t = begin + d;
        const float* f = panel.row(t);
        const double* y = target.data() + static_cast<size_t>(t) * N;
        for (int s = 0; s < N; ++s) {
            if (std::isnan(y[s])) continue;
            ++eligible;
            if (!std::isnan(f[s])) ++valid;
        }
        double* r = scratch.rank.data();
        rank_row_pct(f, N, r, scratch.idx);
        ic[d] = pearson_pairwise(r, y, N);
        if (m > 0 && d % o.signature_step == 0) {
            // factor_signature: ranks, NaN -> 0.5, row demeaned (mean over all N).
            double* row = scratch.R.data() + static_cast<size_t>(d / o.signature_step) * N;
            double sum = 0.0;
            for (int s = 0; s < N; ++s) {
                row[s] = std::isnan(r[s]) ? 0.5 : r[s];
                sum += row[s];
            }
            // Same reduction order as pca::rank_rows (std::accumulate).
            const double mean = sum / N;
            for (int s = 0; s < N; ++s) row[s] -= mean;
        }
        if (mv > 0 && d % o.rank_vector_step == 0) {
            // _sampled_rank_vector: float32 ranks, NaN kept.
            float* row = out.rank_vector.data() + static_cast<size_t>(d / o.rank_vector_step) * N;
            for (int s = 0; s < N; ++s) row[s] = static_cast<float>(r[s]);
        }
    }
    finish_from_daily(ic, eligible, valid, depth, nodes, o.fitness, out);
    if (mv > 0 && !(std::fabs(out.mean_ic) >= o.rank_vector_min_abs_ic)) {
        out.rank_vector.clear();
        out.rank_vector.shrink_to_fit();
    }

    out.signature_valid = false;
    out.signature.clear();
    if (m > 0 && (out.candidate.valid || !o.signature_only_if_valid)) {
        out.signature_valid = power_signature(scratch.R, m, N, *ctx.initial, o.power_steps, out.signature);
        if (!out.signature_valid) out.signature.clear();
    }
}

}  // namespace gpp
