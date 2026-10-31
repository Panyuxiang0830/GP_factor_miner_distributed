#include "gpe/operators.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gpe {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline double clean(double v) { return std::isinf(v) ? kNaN : v; }

// ---------------------------------------------------------------------------
// Rolling kernels on one stock's time series (pandas window_aggregations).
// x[i] is day lo+i; the window ending at i covers x[i-w+1 .. i].

// The kernels below port pandas 3.0.3 (_libs/window/aggregations.pyx) for a
// fixed window, so that rounding residues match: Kahan summation with separate
// compensations for additions and removals, removals before additions.

// Kahan-compensated running sum shared by roll_mean and roll_sum. It also
// tracks the run of identical observations: pandas returns the repeated value
// exactly when the last `nobs` observations are all equal.
struct KahanWindow {
    double sum = 0.0, comp_add = 0.0, comp_rem = 0.0, prev = 0.0;
    int nobs = 0, neg = 0, same_run = 0;

    void add(double v) {
        if (std::isnan(v)) return;
        ++nobs;
        double y = v - comp_add, t = sum + y;
        comp_add = t - sum - y;
        sum = t;
        if (std::signbit(v)) ++neg;
        same_run = (v == prev) ? same_run + 1 : 1;
        prev = v;
    }
    void remove(double v) {
        if (std::isnan(v)) return;
        --nobs;
        double y = -v - comp_rem, t = sum + y;
        comp_rem = t - sum - y;
        sum = t;
        if (std::signbit(v)) --neg;
    }
};

template <typename Result>
std::vector<double> kahan_roll(const std::vector<double>& x, int w, Result result) {
    std::vector<double> out(x.size(), kNaN);
    KahanWindow k;
    for (size_t i = 0; i < x.size(); ++i) {
        if (i == 0) {
            k.prev = x[0];
        } else if (i >= static_cast<size_t>(w)) {
            k.remove(x[i - w]);
        }
        k.add(x[i]);
        out[i] = result(k);
    }
    return out;
}

std::vector<double> roll_mean(const std::vector<double>& x, int w, int minp) {
    return kahan_roll(x, w, [minp](const KahanWindow& k) {
        if (k.nobs < minp || k.nobs == 0) return kNaN;
        if (k.same_run >= k.nobs) return k.prev;
        double r = k.sum / k.nobs;
        if (k.neg == 0 && r < 0) return 0.0;
        if (k.neg == k.nobs && r > 0) return 0.0;
        return r;
    });
}

std::vector<double> roll_sum(const std::vector<double>& x, int w, int minp) {
    return kahan_roll(x, w, [minp](const KahanWindow& k) {
        if (k.nobs == 0 && minp == 0) return 0.0;
        if (k.nobs < minp) return kNaN;
        if (k.same_run >= k.nobs) return k.prev * k.nobs;
        return k.sum;
    });
}

// Sample variance (ddof = 1) with pandas' Welford update. When an update
// loses most of the sum of squares to cancellation, pandas recomputes the
// window from scratch; that is reproduced too.
std::vector<double> roll_var(const std::vector<double>& x, int w, int minp) {
    constexpr double kInvCondTol = std::numeric_limits<double>::epsilon() * 1e3;
    std::vector<double> out(x.size(), kNaN);
    double mean = 0.0, ssq = 0.0, comp_add = 0.0, comp_rem = 0.0;
    double nobs = 0.0;
    bool unstable = false;

    auto add = [&](double v) {
        if (std::isnan(v)) return;
        double prev_m2 = ssq;
        nobs += 1;
        double prev_mean = mean - comp_add;
        double y = v - comp_add, t = y - mean;
        comp_add = t + mean - y;
        mean = nobs ? mean + t / nobs : 0.0;
        ssq += (v - prev_mean) * (v - mean);
        if (prev_m2 * kInvCondTol > ssq) unstable = true;
    };
    auto remove = [&](double v) {
        if (std::isnan(v)) return;
        double prev_m2 = ssq;
        nobs -= 1;
        if (nobs) {
            double prev_mean = mean - comp_rem;
            double y = v - comp_rem, t = y - mean;
            comp_rem = t + mean - y;
            mean -= t / nobs;
            ssq -= (v - prev_mean) * (v - mean);
            if (prev_m2 * kInvCondTol > ssq) unstable = true;
        } else {
            mean = ssq = 0.0;
            unstable = false;
        }
    };

    // Length of the run of equal observations ending at day i (NaN skipped).
    // When it covers the whole window the variance is exactly 0. The Welford
    // residue there depends on where the recursion started, so without this
    // rule a rank that starts at its halo would not reproduce the serial value.
    double run = 0.0, run_value = kNaN;

    int effective_minp = std::max(minp, 1);
    for (size_t i = 0; i < x.size(); ++i) {
        size_t s = i + 1 >= static_cast<size_t>(w) ? i + 1 - w : 0;
        if (!std::isnan(x[i])) {
            run = x[i] == run_value ? run + 1 : 1;
            run_value = x[i];
        }
        if (i > 0) {
            if (i >= static_cast<size_t>(w)) remove(x[i - w]);
            add(x[i]);
        }
        if (i == 0 || unstable) {
            mean = ssq = nobs = comp_add = comp_rem = 0.0;
            for (size_t j = s; j <= i; ++j) add(x[j]);
            unstable = false;
        }
        if (nobs >= effective_minp && nobs > 1) out[i] = run >= nobs ? 0.0 : ssq / (nobs - 1);
    }
    return out;
}

// pandas' zsqrt: negative variances from rounding become 0.
inline double zsqrt(double v) { return v < 0 ? 0.0 : std::sqrt(v); }

std::vector<double> roll_extreme(const std::vector<double>& x, int w, int minp, bool want_max) {
    std::vector<double> out(x.size(), kNaN);
    std::deque<size_t> dq;  // indices of non-NaN values, monotone
    int nobs = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double v = x[i];
        if (!std::isnan(v)) {
            ++nobs;
            while (!dq.empty() && (want_max ? x[dq.back()] <= v : x[dq.back()] >= v)) dq.pop_back();
            dq.push_back(i);
        }
        if (i >= static_cast<size_t>(w)) {
            size_t gone = i - w;
            if (!std::isnan(x[gone])) --nobs;
            while (!dq.empty() && dq.front() <= gone) dq.pop_front();
        }
        if (nobs >= minp && !dq.empty()) out[i] = x[dq.front()];
    }
    return out;
}

std::vector<double> roll_median(const std::vector<double>& x, int w, int minp) {
    std::vector<double> out(x.size(), kNaN);
    std::vector<double> sorted;  // non-NaN values of the window
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isnan(x[i])) sorted.insert(std::upper_bound(sorted.begin(), sorted.end(), x[i]), x[i]);
        if (i >= static_cast<size_t>(w) && !std::isnan(x[i - w]))
            sorted.erase(std::lower_bound(sorted.begin(), sorted.end(), x[i - w]));
        int n = static_cast<int>(sorted.size());
        if (n >= minp && n > 0)
            out[i] = (n % 2) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }
    return out;
}

// ewm(span, adjust=False, ignore_na=False).mean(), ported from pandas: the old
// weight decays by (1-alpha) every day, missing days included, and resets to 1
// after each observation. span 3 (com == 1) uses new_wt = 1 - old_wt.
// (y, old_wt) is the state before x[0]: (NaN, 1) at the start of the data,
// or the state handed over by the previous rank (distributed scan). With
// y = NaN the first row sets y = x[0], as pandas does.
std::vector<double> ewm_mean(const std::vector<double>& x, int span, double y = kNaN, double old_wt = 1.0) {
    std::vector<double> out(x.size(), kNaN);
    double com = (span - 1.0) / 2.0;
    double alpha = 1.0 / (1.0 + com);
    double old_wt_factor = 1.0 - alpha;
    double new_wt = alpha;
    for (size_t i = 0; i < x.size(); ++i) {
        double v = x[i];
        bool obs = !std::isnan(v);
        if (!std::isnan(y)) {
            old_wt *= old_wt_factor;
            if (obs) {
                if (y != v) {
                    if (com == 1.0) new_wt = 1.0 - old_wt;
                    y = (old_wt * y + new_wt * v) / (old_wt + new_wt);
                }
                old_wt = 1.0;
            }
        } else if (obs) {
            y = v;
        }
        out[i] = y;
    }
    return out;
}

// ---------------------------------------------------------------------------

// Rows from..T-1 of stock s.
void column(const Panel& p, int s, int from, std::vector<double>& col) {
    col.resize(p.T - from);
    for (int t = from; t < p.T; ++t) col[t - from] = p.at(t, s);
}

void store_column(const std::vector<double>& col, int src_lo, Panel& out, int s, int begin) {
    for (int t = begin; t < out.T; ++t) out.at(t, s) = static_cast<float>(col[t - src_lo]);
}

void check_shapes(const Panel& a, const Panel* b, const Panel& out, int begin) {
    if (a.T != out.T || a.N != out.N || a.lo > begin || begin < out.lo)
        throw std::invalid_argument("operator input does not cover the output rows");
    if (b && (b->T != a.T || b->N != a.N || b->lo > begin))
        throw std::invalid_argument("operator inputs have different shapes");
}

double binary(Op op, double x, double y) {
    switch (op) {
        case Op::Add: return clean(x + y);
        case Op::Sub: return clean(x - y);
        case Op::Mul: return clean(x * y);
        case Op::Div: return std::fabs(y) > 1e-12 ? clean(x / y) : kNaN;
        default: throw std::logic_error("not a binary operator");
    }
}

void elementwise(Op op, const Panel& a, const Panel* b, Panel& out, int begin) {
    #pragma omp parallel for schedule(static)
    for (int t = begin; t < out.T; ++t) {
        const float* ra = a.row(t);
        const float* rb = b ? b->row(t) : nullptr;
        float* ro = out.row(t);
        for (int s = 0; s < out.N; ++s) {
            double x = ra[s];
            double r;
            switch (op) {
                case Op::Log: r = x > 0 ? std::log(x) : kNaN; break;
                case Op::Abs: r = std::fabs(x); break;
                default: r = binary(op, x, rb[s]);
            }
            ro[s] = static_cast<float>(r);
        }
    }
}

void cross_sectional(Op op, const Panel& a, Panel& out, int begin) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (int t = begin; t < out.T; ++t) {
        const float* ra = a.row(t);
        float* ro = out.row(t);
        if (op == Op::Rank) {
            // Percentile rank among non-NaN values, ties get their average rank.
            std::vector<int> idx;
            idx.reserve(out.N);
            for (int s = 0; s < out.N; ++s) {
                ro[s] = std::numeric_limits<float>::quiet_NaN();
                if (!std::isnan(ra[s])) idx.push_back(s);
            }
            std::sort(idx.begin(), idx.end(), [&](int i, int j) { return ra[i] < ra[j]; });
            double n = static_cast<double>(idx.size());
            for (size_t i = 0; i < idx.size();) {
                size_t j = i;
                while (j + 1 < idx.size() && ra[idx[j + 1]] == ra[idx[i]]) ++j;
                double avg = 0.5 * ((i + 1) + (j + 1));
                for (size_t k = i; k <= j; ++k) ro[idx[k]] = static_cast<float>(avg / n);
                i = j + 1;
            }
        } else {  // Zscore: inf -> NaN, then (x - mean) / std with ddof = 1
            double sum = 0.0;
            int n = 0;
            for (int s = 0; s < out.N; ++s) {
                double x = clean(ra[s]);
                if (!std::isnan(x)) { sum += x; ++n; }
            }
            double mean = n ? sum / n : kNaN, ssq = 0.0;
            for (int s = 0; s < out.N; ++s) {
                double x = clean(ra[s]);
                if (!std::isnan(x)) ssq += (x - mean) * (x - mean);
            }
            double sd = n > 1 ? std::sqrt(ssq / (n - 1)) : kNaN;
            if (sd == 0.0) sd = kNaN;
            for (int s = 0; s < out.N; ++s) ro[s] = static_cast<float>((clean(ra[s]) - mean) / sd);
        }
    }
}

void rolling(Op op, int w, const Panel& a, const Panel* b, Panel& out, int begin, const EmaState* ema) {
    int minp = rolling_min_periods(op, w);
    // Windows read all available history (the inputs may carry different
    // halos); ema starts at `begin` from the handed-over state instead.
    int from = op == Op::Ema ? begin : std::max(a.lo, b ? b->lo : a.lo);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < out.N; ++s) {
        std::vector<double> x, y, r;
        column(a, s, from, x);
        switch (op) {
            case Op::Mean: r = roll_mean(x, w, minp); break;
            case Op::Sum: r = roll_sum(x, w, minp); break;
            case Op::Std:
                r = roll_var(x, w, minp);
                for (double& v : r) v = zsqrt(v);
                break;
            case Op::Max: r = roll_extreme(x, w, minp, true); break;
            case Op::Min: r = roll_extreme(x, w, minp, false); break;
            case Op::Median: r = roll_median(x, w, minp); break;
            case Op::Ema:
                r = ema ? ewm_mean(x, w, ema->y[s], ema->old_wt[s]) : ewm_mean(x, w);
                break;
            case Op::Lag: case Op::Delta: case Op::PctChange:
                r.assign(x.size(), kNaN);
                for (size_t i = w; i < x.size(); ++i) {
                    double cur = x[i], past = x[i - w];
                    // pct_change stays in float32 as in pandas, so it rounds the same way.
                    r[i] = op == Op::Lag ? past
                         : op == Op::Delta ? cur - past
                         : clean(static_cast<float>(cur) / static_cast<float>(past) - 1.0f);
                }
                break;
            case Op::Corr: case Op::Beta: {
                column(*b, s, from, y);
                // Pairwise-complete masking: X = a + 0*b, Y = b + 0*a.
                std::vector<double> mx(x.size()), my(x.size()), xy(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    mx[i] = x[i] + 0.0 * y[i];
                    my[i] = y[i] + 0.0 * x[i];
                    xy[i] = mx[i] * my[i];
                }
                std::vector<double> valid(x.size());
                for (size_t i = 0; i < x.size(); ++i) valid[i] = std::isnan(mx[i] + my[i]) ? 0.0 : 1.0;
                auto mean_xy = roll_mean(xy, w, minp);
                auto mean_x = roll_mean(mx, w, minp);
                auto mean_y = roll_mean(my, w, minp);
                auto count = roll_sum(valid, w, 0);
                r.assign(x.size(), kNaN);
                if (op == Op::Corr) {
                    auto var_x = roll_var(mx, w, minp);
                    auto var_y = roll_var(my, w, minp);
                    for (size_t i = 0; i < x.size(); ++i) {
                        double cov = (mean_xy[i] - mean_x[i] * mean_y[i]) * (count[i] / (count[i] - 1.0));
                        // A constant window has no correlation; its covariance is only a
                        // rounding residue, so do not let it become +-inf.
                        double den = var_x[i] * var_y[i];
                        r[i] = den == 0.0 ? kNaN : cov / std::pow(den, 0.5);  // NaN if negative, as numpy
                    }
                } else {
                    auto var_b = roll_var(y, w, minp);  // b alone, not masked by a
                    for (size_t i = 0; i < x.size(); ++i) {
                        double cov = (mean_xy[i] - mean_x[i] * mean_y[i]) * (count[i] / (count[i] - 1.0));
                        r[i] = std::fabs(var_b[i]) > 1e-12 ? clean(cov / var_b[i]) : kNaN;
                    }
                }
                break;
            }
            default: throw std::logic_error("not a rolling operator");
        }
        store_column(r, from, out, s, begin);
    }
}

}  // namespace

int rolling_min_periods(Op op, int window) {
    switch (op) {
        case Op::Mean: case Op::Sum: case Op::Median: return 1;
        case Op::Std: case Op::Max: case Op::Min: case Op::Corr: case Op::Beta:
            return std::min(3, window);
        default: return 1;
    }
}

void apply_op(Op op, int window, const Panel& a, const Panel* b, Panel& out, int begin, const EmaState* ema) {
    if (begin == kFromLo) begin = out.lo;
    check_shapes(a, b, out, begin);
    switch (op_info(op).kind) {
        case OpKind::Elementwise: elementwise(op, a, b, out, begin); break;
        case OpKind::CrossSectional: cross_sectional(op, a, out, begin); break;
        case OpKind::Rolling: case OpKind::PairRolling: rolling(op, window, a, b, out, begin, ema); break;
        default: throw std::logic_error("apply_op called on a leaf");
    }
}

}  // namespace gpe
