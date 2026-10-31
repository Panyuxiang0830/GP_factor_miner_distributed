// Compares the serial C++ operators (E5) with pandas reference outputs.
//
// Usage: test_ops_reference <reference_dir> [rtol]
// <reference_dir> is written by: export_reference.py values --out <dir> ...
// Market data stay outside the repository, so this test is not part of ctest.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/json.hpp"
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

    int failures = 0, checked = 0;
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
    }
    std::printf("%d cases checked, %d failed (rtol %.0e)\n", checked, failures, rtol);
    return failures ? 1 : 0;
}
