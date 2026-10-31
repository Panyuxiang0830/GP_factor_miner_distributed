// Compares the serial C++ operators (E5) and RankIC/fitness (E6–E8) with the
// outputs of the Python miner.
//
// Usage: test_ops_reference <reference_dir> [rtol]
// <reference_dir> is written by: export_reference.py values --out <dir> ...
// Market data stay outside the repository, so this test is not part of ctest.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/json.hpp"
#include "gpe/metrics.hpp"
#include "gpe/panel.hpp"

namespace {

// allclose-style criterion: |q - p| <= rtol*|p| + atol*rms(p).
// The absolute term absorbs float32 rounding in pandas itself (pct_change,
// zscore and rolling residues are computed in float32 or near zero).
constexpr double kAtol = 1e-5;

struct Comparison {
    double worst = 0.0;       // max |q-p| / (rtol|p| + atol*rms); passes if <= 1
    double max_rel = 0.0;     // max |q-p| / |p|, for information
    long nan_mismatch = 0;    // finite in one, NaN in the other
    long inf_vs_nan = 0;      // pandas inf where we give NaN or the reverse
};

Comparison compare(const gpe::Panel& ours, const gpe::Panel& ref, double rtol) {
    Comparison c;
    double sumsq = 0.0;
    long n = 0;
    for (float v : ref.buf)
        if (std::isfinite(v)) { sumsq += double(v) * v; ++n; }
    double scale = n ? std::sqrt(sumsq / n) : 0.0;

    for (size_t i = 0; i < ref.buf.size(); ++i) {
        double p = ref.buf[i], q = ours.buf[i];
        bool pf = std::isfinite(p), qf = std::isfinite(q);
        if (pf && qf) {
            double diff = std::fabs(q - p);
            double bound = rtol * std::fabs(p) + kAtol * scale;
            double e = bound > 0 ? diff / bound : (diff > 0 ? INFINITY : 0.0);
            if (e > c.worst) c.worst = e;
            if (p != 0 && diff / std::fabs(p) > c.max_rel) c.max_rel = diff / std::fabs(p);
        } else if (pf != qf) {
            if (std::isinf(p) || std::isinf(q)) ++c.inf_vs_nan;
            else ++c.nan_mismatch;
        } else if (std::isinf(p) != std::isinf(q) || (std::isinf(p) && (p > 0) != (q > 0))) {
            ++c.inf_vs_nan;
        }
    }
    return c;
}

// RankIC and fitness are checked in three ways:
//   ic_code   our RankIC of the pandas values rounded to float32, against pandas
//             on the same rounded values: tests the RankIC code alone;
//   fit_code  our fitness of the pandas daily ICs, against the Python candidate:
//             tests the fitness code alone;
//   e2e       our values -> our RankIC -> our fitness, against Python. Reported
//             only: float32 storage creates rank ties that pandas does not have.
constexpr double kCodeTol = 1e-12;

double num_or_nan(const gpe::Json& v) { return v.is_null() ? NAN : v.number; }

std::vector<double> json_series(const gpe::Json& v) {
    std::vector<double> out;
    for (const auto& x : v.array) out.push_back(num_or_nan(x));
    return out;
}

double diff_or_mismatch(double a, double b) {
    if (std::isfinite(a) && std::isfinite(b)) return std::fabs(a - b);
    return std::isfinite(a) == std::isfinite(b) ? 0.0 : INFINITY;
}

// max |a - b| over days, INFINITY if a day is finite in one series only
double series_err(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    for (size_t t = 0; t < a.size(); ++t) e = std::max(e, diff_or_mismatch(a[t], b[t]));
    return e;
}

// Largest difference between our candidate and the Python one; INFINITY if
// one accepts the expression and the other rejects it.
double candidate_err(const gpe::Candidate& c, const gpe::Json& ref) {
    if (c.valid != !ref.is_null()) return INFINITY;
    if (!c.valid) return 0.0;
    double e = std::max({diff_or_mismatch(c.rank_ic, ref.at("rank_ic").number),
                         diff_or_mismatch(c.fitness, ref.at("fitness").number),
                         diff_or_mismatch(c.segment_rank_ic_std, ref.at("segment_rank_ic_std").number),
                         std::fabs(c.valid_ratio - ref.at("valid_ratio").number),
                         std::fabs(c.same_sign_ratio - ref.at("same_sign_ratio").number)});
    std::vector<double> seg = json_series(ref.at("segment_rank_ics"));
    for (size_t i = 0; i < seg.size() && i < c.segment_rank_ics.size(); ++i)
        e = std::max(e, diff_or_mismatch(c.segment_rank_ics[i], seg[i]));
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <reference_dir> [rtol]\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    double rtol = argc > 2 ? std::stod(argv[2]) : 1e-4;
    gpe::Json ref = gpe::parse_json_file(dir + "/reference.json");
    gpe::PanelSet data = gpe::PanelSet::open(ref.at("panel_dir").string);
    int T = data.T, N = data.N;

    std::vector<double> target_rank = gpe::rank_rows_pct(data.load_target());
    int failures = 0, checked = 0, metric_failures = 0;
    std::vector<std::string> metric_lines;
    std::printf("%-4s %-9s %-9s %-9s %-8s %-6s %s\n", "id", "err", "max_rel", "nan_mism", "inf/nan", "status",
                "expression");
    for (const auto& c : ref.at("cases").array) {
        if (c.has("error")) continue;
        int id = static_cast<int>(c.at("id").number);

        gpe::Dag dag;
        std::string error;
        int root = dag.add_expression(c.at("input"), &error);
        if (root < 0) {
            std::printf("%-4d rejected: %s\n", id, error.c_str());
            ++failures;
            continue;
        }
        dag.finalize();
        auto out = gpe::evaluate_serial(dag, data);

        char name[32];
        std::snprintf(name, sizeof(name), "/values/%04d.f8", id);
        gpe::Panel expected = gpe::read_raw_panel(dir + name, T, N, /*float64=*/true);
        Comparison cmp = compare(out.at(root), expected, rtol);

        bool ok = cmp.worst <= 1.0 && cmp.nan_mismatch == 0;
        if (!ok) ++failures;
        ++checked;
        std::printf("%-4d %-9.2e %-9.2e %-9ld %-8ld %-6s %s\n", id, cmp.worst, cmp.max_rel, cmp.nan_mismatch,
                    cmp.inf_vs_nan, ok ? "ok" : "FAIL", c.at("input").dump().c_str());

        const gpe::Node& r = dag.nodes()[root];
        // `expected` holds the pandas values rounded to float32 on reading.
        gpe::DailyIC code_ic = gpe::daily_rank_ic(expected, target_rank);
        double ic_code = std::max(series_err(code_ic.ic, json_series(c.at("daily_ic_f32"))),
                                  std::fabs(code_ic.missing_ratio - c.at("missing_ratio").number));
        gpe::DailyIC ref_ic{json_series(c.at("daily_ic")), c.at("missing_ratio").number};
        double fit_code = candidate_err(gpe::score_factor(ref_ic, r.depth, r.tree_nodes), c.at("candidate"));

        gpe::DailyIC daily = gpe::daily_rank_ic(out.at(root), target_rank);
        gpe::Candidate cand = gpe::score_factor(daily, r.depth, r.tree_nodes);
        double e2e_ic = series_err(daily.ic, ref_ic.ic);
        double e2e_fit = candidate_err(cand, c.at("candidate"));

        bool mok = ic_code <= kCodeTol && fit_code <= kCodeTol;
        if (!mok) ++metric_failures;
        char line[160];
        std::snprintf(line, sizeof(line), "%-4d %-9.2e %-9.2e %-6s %-9.2e %-9.2e %s", id, ic_code, fit_code,
                      c.at("candidate").is_null() ? "no" : "yes", e2e_ic, e2e_fit, mok ? "ok" : "FAIL");
        metric_lines.push_back(line);
    }
    std::printf("%d cases checked, %d failed (rtol %.0e)\n\n", checked, failures, rtol);

    std::printf("%-4s %-9s %-9s %-6s %-9s %-9s %s\n", "id", "ic_code", "fit_code", "cand", "e2e_ic", "e2e_fit",
                "status");
    for (const auto& line : metric_lines) std::printf("%s\n", line.c_str());
    std::printf("RankIC/fitness code: %d cases checked, %d failed (abs tol %.0e)\n", checked, metric_failures,
                kCodeTol);
    return failures || metric_failures ? 1 : 0;
}
