// Engine vs the course serial engine (gpe::evaluate_serial + gpe metrics +
// the PCA bridge's signature path) on synthetic panels. Results must be
// bit-identical for any memory budget, chunk size or expression grouping.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "core.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/metrics.hpp"
#include "gpp/engine.hpp"

namespace {

int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++failures;                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);            \
            std::fprintf(stderr, "\n");                   \
        }                                                 \
    } while (0)

bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

const char* kOps1[] = {"log", "abs", "rank", "zscore"};
const char* kOps2[] = {"+", "-", "*", "/"};
const char* kRoll[] = {"mean", "std", "max", "min", "sum", "median", "ema", "lag", "delta", "pct_change"};
const char* kPair[] = {"correlation", "beta"};
const int kWindows[] = {1, 5, 10, 20, 60};

std::string random_expression(std::mt19937& rng, int depth, const std::vector<std::string>& genes) {
    std::uniform_int_distribution<int> u(0, 1 << 20);
    if (depth == 0 || u(rng) % 5 == 0) {
        if (!genes.empty() && u(rng) % 2) return genes[u(rng) % genes.size()];
        return "\"f" + std::to_string(u(rng) % 6) + "\"";
    }
    int kind = u(rng) % 4;
    auto child = [&]() { return random_expression(rng, depth - 1, genes); };
    std::string w = std::to_string(kWindows[u(rng) % 5]);
    if (kind == 0) return std::string("[\"") + kOps1[u(rng) % 4] + "\"," + child() + "]";
    if (kind == 1) return std::string("[\"") + kOps2[u(rng) % 4] + "\"," + child() + "," + child() + "]";
    if (kind == 2) return std::string("[\"") + kRoll[u(rng) % 10] + "\"," + child() + "," + w + "]";
    return std::string("[\"") + kPair[u(rng) % 2] + "\"," + child() + "," + child() + "," + w + "]";
}

struct Reference {
    gpe::Candidate candidate;
    bool signature_valid = false;
    std::vector<float> signature;
};

std::vector<Reference> reference(const std::vector<gpe::Json>& expressions, const gpe::PanelSet& data,
                                 const gpp::ScoreOptions& o, const std::vector<double>& initial,
                                 std::vector<bool>& ok) {
    std::vector<Reference> out(expressions.size());
    ok.assign(expressions.size(), false);
    gpe::Panel target = data.load_target();
    auto target_rank = gpe::rank_rows_pct(target);
    for (size_t i = 0; i < expressions.size(); ++i) {
        gpe::Dag dag;
        int root = dag.add_expression(expressions[i]);
        if (root < 0) continue;
        dag.finalize();
        ok[i] = true;
        gpe::Panel panel;
        if (dag.nodes()[root].op == gpe::Op::Leaf) panel = data.load_field(dag.fields()[0]);
        else panel = std::move(gpe::evaluate_serial(dag, data).at(root));
        auto daily = gpe::daily_rank_ic(panel, target_rank);
        const auto& node = dag.nodes()[root];
        out[i].candidate = gpe::score_factor(daily, node.depth, node.tree_nodes, o.fitness);
        if (!out[i].candidate.valid || o.signature_step <= 0) continue;
        // pca/engine_bridge.cpp path: sampled raw rows -> pca::rank_rows -> solve -> finish.
        int m = (data.T + o.signature_step - 1) / o.signature_step;
        pca::Matrix R{m, data.N, std::vector<double>(size_t(m) * data.N)};
        for (int r = 0; r < m; ++r)
            for (int s = 0; s < data.N; ++s) R.data[size_t(r) * data.N + s] = panel.at(r * o.signature_step, s);
        pca::rank_rows(R);
        auto sol = pca::solve(R, initial, o.power_steps, "power", {}, false);
        auto sig = pca::finish(sol.local_signature);
        out[i].signature_valid = !sig.empty() && gpp::signature_comparable(sig);
        if (out[i].signature_valid) out[i].signature = sig;
    }
    return out;
}

void compare(const char* label, const std::vector<gpp::RootScore>& got, const std::vector<std::string>& errors,
             const std::vector<Reference>& want, const std::vector<bool>& ok) {
    int checked = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(errors[i].empty() == ok[i], "%s: parse status of %zu", label, i);
        if (!ok[i]) continue;
        const auto& a = got[i].candidate;
        const auto& b = want[i].candidate;
        CHECK(a.valid == b.valid, "%s: valid %zu", label, i);
        CHECK(same(a.rank_ic, b.rank_ic) && same(a.fitness, b.fitness) && same(a.missing_ratio, b.missing_ratio) &&
                  same(a.valid_ratio, b.valid_ratio) && same(a.segment_rank_ic_std, b.segment_rank_ic_std) &&
                  same(a.same_sign_ratio, b.same_sign_ratio),
              "%s: stats %zu (%.17g vs %.17g)", label, i, a.rank_ic, b.rank_ic);
        CHECK(a.segment_rank_ics.size() == b.segment_rank_ics.size(), "%s: segments %zu", label, i);
        CHECK(got[i].signature_valid == want[i].signature_valid, "%s: signature validity %zu", label, i);
        if (got[i].signature_valid && want[i].signature_valid)
            CHECK(got[i].signature == want[i].signature, "%s: signature %zu", label, i);
        ++checked;
    }
    std::printf("%s: %d expressions compared\n", label, checked);
}

}  // namespace

int main() {
    const int T = 130, N = 96;
    gpe::PanelSet data = gpe::PanelSet::make_synthetic(T, N, 7);
    std::mt19937 rng(11);
    std::vector<std::string> genes, texts;
    // Hierarchical-looking batch: genes of depth 1..3 reused by depth 4..5 trees.
    for (int d = 1; d <= 3; ++d)
        for (int k = 0; k < 6; ++k) genes.push_back(random_expression(rng, d, {}));
    for (int k = 0; k < 6; ++k) texts.push_back("\"f" + std::to_string(k) + "\"");  // leaf roots
    for (int d = 1; d <= 5; ++d)
        for (int k = 0; k < 24; ++k) texts.push_back(random_expression(rng, d, genes));
    texts.push_back(texts[10]);                       // duplicate
    texts.push_back("[\"-\",\"f1\",\"f1\"]");          // degenerate, rejected
    texts.push_back("[\"nope\",\"f1\"]");              // unknown operator
    std::vector<gpe::Json> expressions;
    for (const auto& t : texts) expressions.push_back(gpe::parse_json(t));

    gpp::ScoreOptions so;
    so.signature_step = 5;
    so.power_steps = 4;
    so.fitness.max_depth = 5;
    so.fitness.max_nodes = 31;
    so.fitness.min_valid_time_ratio = 0.3;
    so.fitness.max_missing_ratio = 0.9;
    std::vector<double> initial(N);
    std::mt19937_64 r64(3);
    std::normal_distribution<double> nd;
    for (auto& v : initial) v = nd(r64);

    std::vector<bool> ok;
    auto want = reference(expressions, data, so, initial, ok);
    int valid = 0;
    for (const auto& w : want) valid += w.candidate.valid;
    std::printf("reference: %zu expressions, %d valid candidates\n", expressions.size(), valid);
    CHECK(valid > 20, "too few valid candidates for a meaningful test (%d)", valid);

    auto run = [&](const char* label, double budget_gb, int max_chunk, int min_chunk, int threads = 0) {
        gpp::EngineOptions eo;
        eo.threads = threads;
        eo.score = so;
        eo.memory_budget_bytes = budget_gb * 1073741824.0;
        eo.max_chunk = max_chunk;
        eo.min_chunk = min_chunk;
        gpp::Engine engine(data, eo);
        engine.set_initial(initial);
        std::vector<std::string> errors;
        auto got = engine.evaluate(expressions, errors);
        compare(label, got, errors, want, ok);
        const auto& st = engine.stats();
        std::printf("  groups=%ld dag_nodes=%ld chunks=%ld peak_panels=%ld\n", st.groups, st.dag_nodes, st.chunks,
                    st.peak_panels);
        return st;
    };
    auto unlimited = run("unlimited", 0.0, 2000, 0);
    CHECK(unlimited.groups == 1, "unlimited budget must use one group");
    // Budget a little above the largest single-expression peak: forces small
    // chunks and many groups.
    long single = 0;
    for (const auto& e : expressions) {
        gpe::Dag d;
        if (d.add_expression(e) < 0) continue;
        d.finalize();
        single = std::max(single, gpp::Engine::simulate_peak(d, -1, 1, 1 << 30));
    }
    const long slots = single + 2;
    std::printf("largest single-expression peak: %ld panels; budget %ld\n", single, slots);
    const double panel_gb = 4.0 * T * N / 1073741824.0;
    const double fixed_gb = (8.0 * T * N + 6 * 4.0 * T * N + 1 * (8.0 * 26 * N + 12.0 * N + 8.0 * T)) / 1073741824.0;
    auto tight = run("tight budget", fixed_gb + (slots + 0.5) * panel_gb, 2000, 1, 1);
    CHECK(tight.groups > 1, "tight budget should split the batch (groups=%ld)", tight.groups);
    CHECK(tight.peak_panels <= slots, "peak panels %ld exceed the budget %ld", tight.peak_panels, slots);
    run("chunk 3", 0.0, 3, 1);

    // Planner agrees with the executor on the peak.
    {
        gpe::Dag dag;
        for (const auto& e : expressions) dag.add_expression(e);
        dag.finalize();
        long chunks = 0;
        long peak = gpp::Engine::simulate_peak(dag, -1, 1, 2000, &chunks);
        CHECK(peak > 0 && peak <= static_cast<long>(dag.nodes().size()), "simulate_peak");
        CHECK(gpp::Engine::simulate_peak(dag, 0, 1, 2000) == -1, "zero budget must be infeasible");
    }
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
