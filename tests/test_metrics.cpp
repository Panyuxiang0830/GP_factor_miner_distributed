// Hand-checked cases for RankIC and fitness (E6–E8). The comparison with the
// Python miner on real data is in test_ops_reference.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "gpe/metrics.hpp"

namespace {

constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

gpe::Panel panel(int T, int N, std::initializer_list<float> values) {
    gpe::Panel p(T, N);
    std::copy(values.begin(), values.end(), p.buf.begin());
    return p;
}

void test_rank() {
    gpe::Panel p = panel(1, 4, {3.0f, kNaNf, 1.0f, 3.0f});
    auto r = gpe::rank_rows_pct(p);
    check(near(r[0], 2.5 / 3) && std::isnan(r[1]) && near(r[2], 1.0 / 3) && near(r[3], 2.5 / 3),
          "average ranks of ties, divided by the non-NaN count");
}

void test_daily_ic() {
    // day 0: same order -> 1; day 1: reversed -> -1; day 2: constant factor -> NaN;
    // day 3: one pairwise-complete stock -> NaN.
    gpe::Panel target = panel(4, 3, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, kNaNf, kNaNf});
    gpe::Panel factor = panel(4, 3, {10, 20, 30, 30, 20, 10, 5, 5, 5, kNaNf, 1, 2});
    auto ic = gpe::daily_rank_ic(factor, gpe::rank_rows_pct(target));
    check(near(ic.ic[0], 1.0) && near(ic.ic[1], -1.0), "perfect positive and negative order");
    check(std::isnan(ic.ic[2]), "constant cross-section gives NaN");
    check(std::isnan(ic.ic[3]), "fewer than two complete pairs gives NaN");

    // 261 equal values: their rounded mean is off by an ulp, the variance must
    // still count as zero.
    gpe::Panel tc(1, 261), fc(1, 261);
    for (int s = 0; s < 261; ++s) { tc.buf[s] = static_cast<float>(s); fc.buf[s] = 0.0f; }
    check(std::isnan(gpe::daily_rank_ic(fc, gpe::rank_rows_pct(tc)).ic[0]), "large constant cross-section gives NaN");
    // target valid in 10 cells; factor also valid in 9 of them
    check(near(ic.missing_ratio, 1.0 - 9.0 / 10.0), "missing ratio counts target-valid cells");

    // Ranks are taken before the pairwise mask: stock 2 counts in the factor
    // ranks even though its target is missing.
    gpe::Panel t2 = panel(1, 4, {1, 2, kNaNf, 3});
    gpe::Panel f2 = panel(1, 4, {1, 3, 2, 4});
    auto ic2 = gpe::daily_rank_ic(f2, gpe::rank_rows_pct(t2));
    // factor ranks over 4 values: 0.25, 0.75, -, 1.0; target ranks: 1/3, 2/3, -, 1
    double x[3] = {0.25, 0.75, 1.0}, y[3] = {1.0 / 3, 2.0 / 3, 1.0};
    double mx = 2.0 / 3, my = 2.0 / 3, sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < 3; ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    check(near(ic2.ic[0], sxy / std::sqrt(sxx * syy)), "factor ranked before pairwise masking");
}

void test_fitness() {
    gpe::DailyIC d;
    d.ic = {0.1, 0.1, 0.2, 0.2, kNaN, 0.3, -0.1, -0.1};
    d.missing_ratio = 0.05;
    auto c = gpe::score_factor(d, /*depth=*/2, /*nodes=*/5);
    check(c.valid, "candidate accepted");
    check(near(c.rank_ic, 0.1) && near(c.valid_ratio, 7.0 / 8), "mean of finite days and valid ratio");
    check(c.segment_rank_ics.size() == 4 && near(c.segment_rank_ics[2], 0.3) && near(c.segment_rank_ics[3], -0.1),
          "four segments of two days");
    double sd = std::sqrt(0.0875 / 4);
    check(near(c.segment_rank_ic_std, sd) && near(c.same_sign_ratio, 0.75), "segment std and same-sign ratio");
    check(near(c.fitness, 0.1 - 0.5 * sd - 0.0005 * 2), "fitness without sign multiplier");

    // array_split with a remainder: 10 days -> 3, 3, 2, 2
    gpe::DailyIC e;
    e.ic = {2, 2, 2, -1, -1, -1, -1, -1, 1, 1};
    e.missing_ratio = 0.0;
    auto c2 = gpe::score_factor(e, 1, 3);
    check(near(c2.segment_rank_ics[0], 2) && near(c2.segment_rank_ics[1], -1) && near(c2.segment_rank_ics[2], -1),
          "first T % k segments get the extra day");
    // segment means 2, -1, -1, 1 around 0.25; only 2 of 4 share the sign of 0.3
    double sd2 = std::sqrt((1.75 * 1.75 + 2 * 1.25 * 1.25 + 0.75 * 0.75) / 4);
    check(near(c2.fitness, (0.3 - 0.5 * sd2) * 0.25), "sign multiplier when same-sign ratio < 0.75");

    gpe::DailyIC f = d;
    f.missing_ratio = 0.3;
    check(!gpe::score_factor(f, 2, 5).valid, "rejected above max missing ratio");
    check(!gpe::score_factor(d, 4, 5).valid, "rejected above max depth");
    check(!gpe::score_factor(d, 2, 12).valid, "rejected above max nodes");
}

}  // namespace

int main() {
    test_rank();
    test_daily_ic();
    test_fitness();
    if (failures) return EXIT_FAILURE;
    std::printf("all metrics tests passed\n");
    return EXIT_SUCCESS;
}
