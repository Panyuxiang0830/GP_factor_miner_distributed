// The ema block scan (ema_scan.hpp) reproduces the serial ema when the days
// are cut into blocks, without MPI: each block is summarised, the state is
// chained through the blocks and every block is recomputed from its state.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "gpe/ema_scan.hpp"
#include "gpe/operators.hpp"

namespace {

int failures = 0;

// Stocks with different missing-value patterns: none, scattered, a long
// leading gap (no value before several blocks), gaps longer than a block,
// all missing, and runs of equal values.
gpe::Panel make_series(int T, int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    gpe::Panel p(T, N);
    for (int s = 0; s < N; ++s) {
        int pattern = s % 6;
        double level = 10.0 + s;
        for (int t = 0; t < T; ++t) {
            level += noise(rng);
            float v = static_cast<float>(level);
            bool missing = (pattern == 1 && u(rng) < 0.3) || (pattern == 2 && t < T / 2) ||
                           (pattern == 3 && (t / 17) % 3 == 1) || pattern == 4;
            if (pattern == 5) v = static_cast<float>(std::floor(level / 5.0));
            p.at(t, s) = missing ? NAN : v;
        }
    }
    return p;
}

void check_blocks(const gpe::Panel& x, int span, const std::vector<int>& cuts) {
    const int T = x.T, N = x.N;
    gpe::Panel serial(T, N);
    gpe::apply_op(gpe::Op::Ema, span, x, nullptr, serial);

    std::vector<gpe::EmaCarry> carry(N);  // state before the first block: (NaN, 1)
    double worst = 0.0;
    long nan_mismatch = 0;
    for (size_t k = 0; k + 1 < cuts.size(); ++k) {
        int t0 = cuts[k], t1 = cuts[k + 1];
        gpe::Panel in(t1 - t0, N), out(t1 - t0, N);
        for (int t = t0; t < t1; ++t)
            for (int s = 0; s < N; ++s) in.at(t - t0, s) = x.at(t, s);

        std::vector<double> y(N), w(N);
        for (int s = 0; s < N; ++s) {
            y[s] = carry[s].y;
            w[s] = carry[s].old_wt;
        }
        gpe::EmaState state{y.data(), w.data()};
        gpe::apply_op(gpe::Op::Ema, span, in, nullptr, out, 0, k == 0 ? nullptr : &state);

        auto summary = gpe::ema_summarize_panel(in, span);
        for (int s = 0; s < N; ++s) carry[s] = gpe::ema_apply(summary[s], carry[s], span);

        for (int t = t0; t < t1; ++t)
            for (int s = 0; s < N; ++s) {
                double p = serial.at(t, s), q = out.at(t - t0, s);
                if (std::isnan(p) != std::isnan(q)) ++nan_mismatch;
                else if (!std::isnan(p)) worst = std::max(worst, std::fabs(p - q) / std::max(std::fabs(p), 1e-30));
            }
    }
    bool ok = worst <= 1e-6 && nan_mismatch == 0;
    if (!ok) ++failures;
    std::printf("span %2d, %zu blocks: max rel %.2e, nan mismatch %ld %s\n", span, cuts.size() - 1, worst,
                nan_mismatch, ok ? "ok" : "FAIL");
}

}  // namespace

int main() {
    const int T = 400, N = 24;
    gpe::Panel x = make_series(T, N, 7);
    for (int span : {3, 5, 20, 60}) {  // span 3 has com == 1, a special case in pandas
        check_blocks(x, span, {0, T});
        check_blocks(x, span, {0, 100, 200, 300, T});
        check_blocks(x, span, {0, 1, 2, 57, 250, 251, T});
    }
    if (failures) return EXIT_FAILURE;
    std::printf("all ema scan tests passed\n");
    return EXIT_SUCCESS;
}
