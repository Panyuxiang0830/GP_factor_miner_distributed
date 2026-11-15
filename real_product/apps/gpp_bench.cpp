// Benchmark / planning tool for the shared-memory engine.
//
//   gpp_bench (--panels DIR | --synthetic T N) (--expressions FILE.json | --random COUNT)
//             [--depth D] [--genes G] [--threads n] [--budget-gb X] [--max-chunk C]
//             [--signature-step 5] [--repeat R] [--plan-only] [--gpus G]
//
// --random builds a hierarchical-looking batch: G promoted genes per depth
// below D, and COUNT depth-D trees whose children are genes 70% of the time
// (balanced_stage_proposals). Prints one JSON object per run.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gpp/engine.hpp"

namespace {

const char* kOps1[] = {"log", "abs", "rank", "zscore"};
const char* kOps2[] = {"+", "-", "*", "/"};
const char* kRoll[] = {"mean", "std", "max", "min", "sum", "median", "ema", "lag", "delta", "pct_change"};
const char* kPair[] = {"correlation", "beta"};
const int kWindows[] = {1, 5, 10, 15, 20, 30, 60};

struct Gen {
    std::mt19937_64 rng{2026};
    int fields = 132;
    std::vector<std::string> names;  // real field names (empty: synthetic f<k>)
    std::vector<std::vector<std::string>> genes;  // by depth
    int pick(int n) { return static_cast<int>(rng() % static_cast<unsigned long long>(n)); }
    std::string leaf() {
        if (!names.empty()) return "\"" + names[pick(static_cast<int>(names.size()))] + "\"";
        return "\"f" + std::to_string(pick(fields)) + "\"";
    }
    std::string child(int depth) {
        // a gene of depth < depth (70%) or a raw field
        if (depth > 1 && pick(10) < 7) {
            int d = 1 + pick(depth - 1);
            if (!genes[d].empty()) return genes[d][pick(static_cast<int>(genes[d].size()))];
        }
        return leaf();
    }
    std::string tree(int depth) {
        int kind = pick(20);
        std::string w = std::to_string(kWindows[pick(7)]);
        auto c = [&]() { return depth == 1 ? leaf() : child(depth); };
        if (kind < 4) return std::string("[\"") + kOps1[kind] + "\"," + c() + "]";
        if (kind < 8) return std::string("[\"") + kOps2[kind - 4] + "\"," + c() + "," + c() + "]";
        if (kind < 18) return std::string("[\"") + kRoll[kind - 8] + "\"," + c() + "," + w + "]";
        return std::string("[\"") + kPair[kind - 18] + "\"," + c() + "," + c() + "," + w + "]";
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string panels, expressions_file;
    int T = 0, N = 0, count = 0, depth = 3, gene_count = 2000, threads = 0, repeat = 1, max_chunk = 2000;
    int signature_step = 5, gpus = 0;
    double budget = 0;
    bool plan_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--panels") panels = next();
        else if (a == "--synthetic") { T = std::stoi(next()); N = std::stoi(next()); }
        else if (a == "--expressions") expressions_file = next();
        else if (a == "--random") count = std::stoi(next());
        else if (a == "--depth") depth = std::stoi(next());
        else if (a == "--genes") gene_count = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--budget-gb") budget = std::stod(next());
        else if (a == "--max-chunk") max_chunk = std::stoi(next());
        else if (a == "--signature-step") signature_step = std::stoi(next());
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--gpus") gpus = std::stoi(next());
        else if (a == "--plan-only") plan_only = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    gpe::PanelSet data = panels.empty() ? gpe::PanelSet::make_synthetic(T, N, 1) : gpe::PanelSet::open(panels);
    std::vector<gpe::Json> expressions;
    if (!expressions_file.empty()) {
        gpe::Json all = gpe::parse_json_file(expressions_file);
        for (auto& e : all.array) expressions.push_back(e.is_object() ? e.at("expression") : e);
    } else {
        Gen g;
        if (!data.synthetic) g.names = data.fields;
        g.genes.resize(depth + 1);
        for (int d = 1; d < depth; ++d)
            for (int k = 0; k < gene_count; ++k) g.genes[d].push_back(g.tree(d));
        for (int k = 0; k < count; ++k) expressions.push_back(gpe::parse_json(g.tree(depth)));
    }
    gpp::EngineOptions eo;
    eo.threads = threads;
    eo.memory_budget_bytes = budget * 1073741824.0;
    eo.max_chunk = max_chunk;
    eo.gpus = gpus;
    eo.score.signature_step = signature_step;
    eo.score.fitness.max_depth = 64;
    eo.score.fitness.max_nodes = 1 << 20;
    gpp::Engine engine(data, eo);
    std::vector<std::string> errors;
    if (plan_only) {
        auto groups = engine.plan(expressions, errors);
        long nodes = 0, peak = 0;
        for (const auto& g : groups) { nodes += g.dag_nodes; peak = std::max(peak, g.peak_panels); }
        std::printf("{\"expressions\":%zu,\"groups\":%zu,\"dag_nodes\":%ld,\"peak_panels\":%ld,\"budget_panels\":%ld,"
                    "\"panel_mb\":%.2f}\n",
                    expressions.size(), groups.size(), nodes, peak, engine.budget_panels(), engine.panel_bytes() / 1e6);
        return 0;
    }
    for (int r = 0; r < repeat; ++r) {
        auto t = std::chrono::steady_clock::now();
        auto scores = engine.evaluate(expressions, errors);
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        long valid = 0;
        for (const auto& s : scores) valid += s.candidate.valid;
        const auto& st = engine.stats();
        std::printf("{\"run\":%d,\"wall\":%.3f,\"expressions\":%zu,\"valid\":%ld,\"threads\":%d,\"groups\":%ld,"
                    "\"dag_nodes\":%ld,\"peak_panels\":%ld,\"peak_gb\":%.2f,\"load\":%.3f,\"compute\":%.3f}\n",
                    r, wall, expressions.size(), valid, engine.threads(), st.groups, st.dag_nodes, st.peak_panels,
                    st.peak_bytes / 1073741824.0, st.load, st.compute);
    }
    return 0;
}
