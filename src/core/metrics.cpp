#include "gpe/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gpe {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Percentile ranks of one day; out[s] is NaN where x[s] is NaN.
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

// DataFrame.corrwith(axis=1, method="pearson") for one day: both series are
// masked to pairwise-complete values, then
//   sum((x - mx)(y - my)) / ((n - 1) * std(x) * std(y)),  std with ddof = 1.
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
    // A constant series has zero variance, so pandas gets 0/0 = NaN. Test it
    // exactly: the rounded mean of equal values can differ from them by an
    // ulp, which would leave a tiny variance and a meaningless correlation.
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

}  // namespace

std::vector<double> rank_rows_pct(const Panel& p) {
    std::vector<double> out(static_cast<size_t>(p.T) * p.N);
    #pragma omp parallel
    {
        std::vector<int> idx;
        idx.reserve(p.N);
        #pragma omp for schedule(dynamic, 8)
        for (int t = 0; t < p.T; ++t) rank_row_pct(p.row(t), p.N, out.data() + static_cast<size_t>(t) * p.N, idx);
    }
    return out;
}

DailyIC daily_rank_ic(const Panel& factor, const std::vector<double>& target_rank) {
    const int T = factor.T, N = factor.N;
    DailyIC result;
    result.ic.assign(T, kNaN);
    long eligible = 0, valid = 0;
    #pragma omp parallel reduction(+ : eligible, valid)
    {
        std::vector<int> idx;
        std::vector<double> rank(N);
        idx.reserve(N);
        #pragma omp for schedule(dynamic, 8)
        for (int t = 0; t < T; ++t) {
            const float* f = factor.row(t);
            const double* y = target_rank.data() + static_cast<size_t>(t) * N;
            for (int s = 0; s < N; ++s) {
                if (std::isnan(y[s])) continue;
                ++eligible;
                if (!std::isnan(f[s])) ++valid;
            }
            rank_row_pct(f, N, rank.data(), idx);
            result.ic[t] = pearson_pairwise(rank.data(), y, N);
        }
    }
    result.eligible = eligible;
    result.valid = valid;
    result.missing_ratio = eligible == 0 ? 1.0 : 1.0 - static_cast<double>(valid) / eligible;
    return result;
}

void IcSummary::pack(double* out) const {
    const size_t k = seg_sum.size();
    for (size_t i = 0; i < k; ++i) {
        out[i] = seg_sum[i];
        out[k + i] = seg_count[i];
    }
    out[2 * k] = eligible;
    out[2 * k + 1] = valid;
}

IcSummary IcSummary::unpack(const double* in, int segments) {
    IcSummary s;
    s.seg_sum.assign(in, in + segments);
    s.seg_count.assign(in + segments, in + 2 * segments);
    s.eligible = in[2 * segments];
    s.valid = in[2 * segments + 1];
    return s;
}

IcSummary summarize_ic(const DailyIC& daily, int t0, int T_total, int segments) {
    IcSummary s;
    s.seg_sum.assign(segments, 0.0);
    s.seg_count.assign(segments, 0.0);
    s.eligible = static_cast<double>(daily.eligible);
    s.valid = static_cast<double>(daily.valid);
    // np.array_split: the first T_total % k segments get one extra day.
    const int base = T_total / segments, extra = T_total % segments;
    for (size_t i = 0; i < daily.ic.size(); ++i) {
        double v = daily.ic[i];
        if (!std::isfinite(v)) continue;
        int t = t0 + static_cast<int>(i);
        int seg = t < extra * (base + 1) ? t / (base + 1) : extra + (t - extra * (base + 1)) / base;
        s.seg_sum[seg] += v;
        s.seg_count[seg] += 1.0;
    }
    return s;
}

namespace {

// _evaluate_with_values filters and _fitness_from_segments.
Candidate score(const IcSummary& s, double missing_ratio, int T, int depth, int nodes, const FitnessConfig& config) {
    Candidate c;
    c.missing_ratio = missing_ratio;
    if (depth > config.max_depth || nodes > config.max_nodes) return c;

    double sum = 0.0, finite = 0.0;
    for (size_t i = 0; i < s.seg_sum.size(); ++i) {
        sum += s.seg_sum[i];
        finite += s.seg_count[i];
    }
    c.rank_ic = finite > 0 ? sum / finite : kNaN;
    c.valid_ratio = T ? finite / T : 0.0;
    if (!std::isfinite(c.rank_ic) || c.valid_ratio < config.min_valid_time_ratio ||
        c.missing_ratio > config.max_missing_ratio)
        return c;

    std::vector<double> finite_means;
    for (size_t i = 0; i < s.seg_sum.size(); ++i) {
        double mean = s.seg_count[i] > 0 ? s.seg_sum[i] / s.seg_count[i] : kNaN;
        c.segment_rank_ics.push_back(mean);
        if (s.seg_count[i] > 0) finite_means.push_back(mean);
    }
    if (finite_means.empty()) {
        c.fitness = -std::numeric_limits<double>::infinity();
        c.segment_rank_ic_std = std::numeric_limits<double>::infinity();
        c.valid = true;
        return c;
    }
    double m = 0.0;
    for (double v : finite_means) m += v;
    m /= finite_means.size();
    double ss = 0.0;
    int same = 0;
    double direction = c.rank_ic > 0 ? 1.0 : -1.0;
    for (double v : finite_means) {
        ss += (v - m) * (v - m);
        if (v * direction > 0) ++same;
    }
    c.segment_rank_ic_std = std::sqrt(ss / finite_means.size());
    c.same_sign_ratio = static_cast<double>(same) / finite_means.size();
    c.fitness = std::fabs(c.rank_ic) - config.stability_penalty * c.segment_rank_ic_std -
                config.complexity_penalty * std::max(nodes - 3, 0);
    if (c.same_sign_ratio < config.minimum_same_sign_ratio) c.fitness *= config.inconsistent_sign_multiplier;
    c.valid = true;
    return c;
}

}  // namespace

Candidate score_factor(const DailyIC& daily, int depth, int nodes, const FitnessConfig& config) {
    const int T = static_cast<int>(daily.ic.size());
    return score(summarize_ic(daily, 0, T, config.stability_segments), daily.missing_ratio, T, depth, nodes, config);
}

Candidate score_summary(const IcSummary& s, int T_total, int depth, int nodes, const FitnessConfig& config) {
    double missing = s.eligible == 0 ? 1.0 : 1.0 - s.valid / s.eligible;
    return score(s, missing, T_total, depth, nodes, config);
}

}  // namespace gpe
