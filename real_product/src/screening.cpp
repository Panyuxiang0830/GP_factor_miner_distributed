#include "gpp/screening.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gpp {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

bool same_direction(double a, double b) { return std::isfinite(a) && std::isfinite(b) && a * b > 0; }

// Approximate |correlation| from the six sums; flags pairs that need the
// exact formula (too few pairs, tiny variance).
double from_sums(const PairSums& s, bool* exact_needed) {
    *exact_needed = false;
    if (s.n < 2.5) return kNaN;  // < 3 valid pairs: Python returns NaN
    double cov = s.sxy - s.sx * s.sy / s.n;
    double vx = s.sxx - s.sx * s.sx / s.n;
    double vy = s.syy - s.sy * s.sy / s.n;
    // Ranks are in (0, 1]; a genuine variance is far above 1e-9 * n.
    if (!(vx > 1e-9 * s.n) || !(vy > 1e-9 * s.n)) {
        *exact_needed = true;
        return 0.0;
    }
    return std::fabs(cov) / std::sqrt(vx * vy);
}

}  // namespace

double masked_abs_correlation(const float* a, const float* b, size_t n) {
    size_t count = 0;
    double ma = 0, mb = 0;
    for (size_t k = 0; k < n; ++k) {
        if (!std::isfinite(a[k]) || !std::isfinite(b[k])) continue;
        ma += a[k];
        mb += b[k];
        ++count;
    }
    if (count < 3) return kNaN;
    ma /= count;
    mb /= count;
    double cross = 0, aa = 0, bb = 0;
    for (size_t k = 0; k < n; ++k) {
        if (!std::isfinite(a[k]) || !std::isfinite(b[k])) continue;
        double x = a[k] - ma, y = b[k] - mb;
        cross += x * y;
        aa += x * x;
        bb += y * y;
    }
    double denominator = std::sqrt(aa) * std::sqrt(bb);
    if (denominator == 0.0) return kNaN;
    return std::fabs(cross / denominator);
}

void cpu_pair_sums(const std::vector<const float*>& left, const std::vector<const float*>& right, size_t length,
                   std::vector<PairSums>& out) {
    const size_t R = right.size();
    out.assign(left.size() * R, PairSums{0, 0, 0, 0, 0, 0});
    // Tiles of the vector length keep both rows in cache across the pairs.
    constexpr size_t kTile = 4096;
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (long i = 0; i < static_cast<long>(left.size()); ++i)
        for (long j = 0; j < static_cast<long>(R); ++j) {
            const float* a = left[i];
            const float* b = right[j];
            double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            for (size_t t0 = 0; t0 < length; t0 += kTile) {
                const size_t t1 = std::min(length, t0 + kTile);
                for (size_t k = t0; k < t1; ++k) {
                    const float x = a[k], y = b[k];
                    if (std::isnan(x) || std::isnan(y)) continue;
                    const double dx = double(x) - 0.5, dy = double(y) - 0.5;
                    n += 1;
                    sx += dx;
                    sy += dy;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
            }
            out[size_t(i) * R + j] = PairSums{n, sx, sy, sxx, syy, sxy};
        }
}

ScreenResult screen_factors(Engine& engine, const std::vector<ScreenInput>& inputs, const ScreenOptions& o,
                            const PairSumBackend& backend) {
    auto t_all = Clock::now();
    ScreenResult res;
    const size_t C = inputs.size();
    res.mean.assign(C, kNaN);
    res.valid_ratio.assign(C, 0.0);
    res.missing_ratio.assign(C, 1.0);
    res.valid_times.assign(C, 0);
    res.error.assign(C, std::string());
    std::vector<std::vector<float>> vectors(C);

    // 1. One engine pass per batch: RankIC and (eligible) rank vectors.
    auto t = Clock::now();
    for (size_t b0 = 0; b0 < C; b0 += static_cast<size_t>(o.engine_batch)) {
        const size_t b1 = std::min(C, b0 + static_cast<size_t>(o.engine_batch));
        std::vector<gpe::Json> batch;
        batch.reserve(b1 - b0);
        for (size_t k = b0; k < b1; ++k) batch.push_back(*inputs[k].expression);
        std::vector<std::string> errors;
        auto scores = engine.evaluate(batch, errors);
        for (size_t k = b0; k < b1; ++k) {
            const auto& s = scores[k - b0];
            if (!errors[k - b0].empty()) {
                res.error[k] = errors[k - b0];
                continue;
            }
            res.mean[k] = s.mean_ic;
            res.valid_times[k] = s.valid_times;
            res.valid_ratio[k] = s.scored_days ? double(s.valid_times) / s.scored_days : 0.0;
            res.missing_ratio[k] = s.missing_ratio;
            if (std::fabs(s.mean_ic) >= o.rankic_threshold && same_direction(inputs[k].evolution_rank_ic, s.mean_ic))
                vectors[k] = std::move(scores[k - b0].rank_vector);
        }
    }
    res.timing.evaluate = since(t);

    // 2. Quality order: (-|screen|, -|evolution|, nodes, key).
    t = Clock::now();
    for (size_t k = 0; k < C; ++k)
        if (res.error[k].empty() && std::fabs(res.mean[k]) >= o.rankic_threshold &&
            same_direction(inputs[k].evolution_rank_ic, res.mean[k]))
            res.order.push_back(static_cast<int>(k));
    std::sort(res.order.begin(), res.order.end(), [&](int a, int b) {
        double qa = -std::fabs(res.mean[a]), qb = -std::fabs(res.mean[b]);
        if (qa != qb) return qa < qb;
        double ea = -std::fabs(inputs[a].evolution_rank_ic), eb = -std::fabs(inputs[b].evolution_rank_ic);
        if (ea != eb) return ea < eb;
        if (inputs[a].nodes != inputs[b].nodes) return inputs[a].nodes < inputs[b].nodes;
        return inputs[a].key < inputs[b].key;
    });
    res.timing.sort = since(t);

    size_t length = 0;
    for (int k : res.order)
        if (!vectors[k].empty()) { length = vectors[k].size(); break; }
    for (int k : res.order)
        if (vectors[k].size() != length) throw std::runtime_error("screening: missing or mismatched rank vector");

    // 3-4. Blocked products + serial greedy scan.
    std::vector<int> selected;   // input indices, selection order
    const double thr = o.corr_threshold;
    auto exact = [&](int a, int b) {
        ++res.timing.refined;
        return masked_abs_correlation(vectors[a].data(), vectors[b].data(), length);
    };
    bool stop = false;
    for (size_t b0 = 0; b0 < res.order.size() && !stop; b0 += static_cast<size_t>(o.block)) {
        const size_t b1 = std::min(res.order.size(), b0 + static_cast<size_t>(o.block));
        const int B = static_cast<int>(b1 - b0);
        t = Clock::now();
        std::vector<const float*> left(B), right(selected.size());
        for (int i = 0; i < B; ++i) left[i] = vectors[res.order[b0 + i]].data();
        for (size_t j = 0; j < selected.size(); ++j) right[j] = vectors[selected[j]].data();
        std::vector<PairSums> cross, internal;
        if (backend) {
            backend(left, right, length, cross);
            backend(left, left, length, internal);
        } else {
            cpu_pair_sums(left, right, length, cross);
            cpu_pair_sums(left, left, length, internal);
        }
        res.timing.pairs += static_cast<long long>(cross.size() + internal.size());
        res.timing.products += since(t);

        t = Clock::now();
        const size_t S0 = selected.size();
        std::vector<int> block_selected;  // block positions selected in this block
        struct Pair { int with; double value; bool exact; };
        std::vector<Pair> row;
        for (int i = 0; i < B; ++i) {
            if (o.limit >= 0 && static_cast<long>(selected.size()) >= o.limit) { stop = true; break; }
            const int me = res.order[b0 + i];
            row.clear();
            for (size_t j = 0; j < S0; ++j) {
                bool need = false;
                double v = from_sums(cross[size_t(i) * S0 + j], &need);
                row.push_back({selected[j], v, false});
                if (need) { row.back().value = exact(me, selected[j]); row.back().exact = true; }
            }
            for (int k : block_selected) {
                bool need = false;
                double v = from_sums(internal[size_t(i) * B + k], &need);
                row.push_back({res.order[b0 + k], v, false});
                if (need) { row.back().value = exact(me, res.order[b0 + k]); row.back().exact = true; }
            }
            double approx_max = -1;
            for (const auto& p : row)
                if (std::isfinite(p.value)) approx_max = std::max(approx_max, p.value);
            for (auto& p : row) {
                if (p.exact || !std::isfinite(p.value)) continue;
                if (std::fabs(p.value - thr) <= o.band || p.value >= approx_max - 2 * o.band) {
                    p.value = exact(me, p.with);
                    p.exact = true;
                }
            }
            double max_corr = 0.0;
            int with = -1;
            bool has_finite = false, keep = true;
            for (const auto& p : row) {
                if (std::isfinite(p.value) && (!has_finite || p.value > max_corr)) {
                    max_corr = p.value;
                    with = p.with;
                    has_finite = true;
                }
                if (std::isfinite(p.value) && p.value >= thr) keep = false;
            }
            if (keep) {
                selected.push_back(me);
                block_selected.push_back(i);
            }
            res.selected.push_back(keep);
            res.max_correlation.push_back(max_corr);
            res.correlated_with.push_back(with);
            ++res.scanned;
        }
        res.timing.scan += since(t);
    }
    res.timing.total = since(t_all);
    return res;
}

}  // namespace gpp
