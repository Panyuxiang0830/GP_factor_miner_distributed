#include "gpe/ema_scan.hpp"

namespace gpe {
namespace {

// Same parameters as ewm_mean in operators.cpp.
struct EwmParams {
    double com, alpha, factor;
    explicit EwmParams(int span) : com((span - 1.0) / 2.0), alpha(1.0 / (1.0 + com)), factor(1.0 - alpha) {}
    double new_wt(double old_wt) const { return com == 1.0 ? 1.0 - old_wt : alpha; }
};

}  // namespace

EmaBlockSummary ema_summarize(const std::vector<double>& x, int span) {
    EwmParams p(span);
    EmaBlockSummary s;
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) s.decay_len *= p.factor;
    int j = 0;
    while (j < n && std::isnan(x[j])) ++j;
    if (j == n) return s;

    s.first_obs = j;
    s.v_first = x[j];
    for (int i = 0; i <= j; ++i) s.decay_first *= p.factor;
    double old_wt = 1.0;
    for (int i = j + 1; i < n; ++i) {
        old_wt *= p.factor;
        if (std::isnan(x[i])) continue;
        double nw = p.new_wt(old_wt);
        double scale = old_wt / (old_wt + nw);
        s.a *= scale;
        s.b = s.b * scale + nw * x[i] / (old_wt + nw);
        old_wt = 1.0;
    }
    s.old_wt_end = old_wt;
    return s;
}

EmaCarry ema_apply(const EmaBlockSummary& s, const EmaCarry& in, int span) {
    if (s.first_obs < 0) {
        // y only exists after an observation; old_wt decays only while it does
        return {in.y, std::isnan(in.y) ? in.old_wt : in.old_wt * s.decay_len};
    }
    double y_first = s.v_first;
    if (!std::isnan(in.y) && in.y != s.v_first) {
        EwmParams p(span);
        double o = in.old_wt * s.decay_first;
        double nw = p.new_wt(o);
        y_first = (o * in.y + nw * s.v_first) / (o + nw);
    }
    return {s.a * y_first + s.b, s.old_wt_end};
}

std::vector<EmaBlockSummary> ema_summarize_panel(const Panel& a, int span) {
    std::vector<EmaBlockSummary> out(a.N);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < a.N; ++s) {
        std::vector<double> x(a.T);
        for (int t = 0; t < a.T; ++t) x[t] = a.at(t, s);
        out[s] = ema_summarize(x, span);
    }
    return out;
}

}  // namespace gpe
