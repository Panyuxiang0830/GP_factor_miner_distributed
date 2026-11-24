// GPU paths vs the CPU paths (built only with GPP_ENABLE_CUDA=ON).
//   1. Engine with the CUDA scorer vs the CPU scorer: candidate statistics and
//      signatures must be bit-identical (same reductions, no FMA contraction).
//   2. QdArchive with gpu_summaries vs cpu_summaries: identical decisions.
//   3. gpu_pair_sums vs cpu_pair_sums: within 1e-12 relative.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "gpp/archive.hpp"
#include "gpp/engine.hpp"
#include "gpp/gpu.hpp"
#include "gpp/screening.hpp"

namespace {
int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            ++failures;                                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
        }                                                                  \
    } while (0)

bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

std::string random_expression(std::mt19937& rng, int depth) {
    static const char* unary[] = {"log", "abs", "rank", "zscore"};
    static const char* binary[] = {"+", "-", "*", "/"};
    static const char* rolling[] = {"mean", "std", "max", "min", "sum", "median", "ema", "lag", "delta", "pct_change"};
    static const int windows[] = {1, 5, 10, 20};
    std::uniform_int_distribution<int> u(0, 1 << 20);
    if (depth == 0 || u(rng) % 4 == 0) return "\"f" + std::to_string(u(rng) % 6) + "\"";
    auto child = [&]() { return random_expression(rng, depth - 1); };
    switch (u(rng) % 4) {
        case 0: return std::string("[\"") + unary[u(rng) % 4] + "\"," + child() + "]";
        case 1: return std::string("[\"") + binary[u(rng) % 4] + "\"," + child() + "," + child() + "]";
        case 2: return std::string("[\"") + rolling[u(rng) % 10] + "\"," + child() + "," + std::to_string(windows[u(rng) % 4]) + "]";
        default: return std::string("[\"correlation\",") + child() + "," + child() + ",5]";
    }
}

void engine_case(int T, int N, int begin) {
    gpe::PanelSet data = gpe::PanelSet::make_synthetic(T, N, 11);
    std::mt19937 rng(T * 31 + N);
    std::vector<gpe::Json> expressions;
    for (int i = 0; i < 300; ++i) expressions.push_back(gpe::parse_json(random_expression(rng, 1 + i % 4)));
    std::vector<double> initial(N);
    std::normal_distribution<double> nd;
    for (double& v : initial) v = nd(rng);
    auto run = [&](int gpus) {
        gpp::EngineOptions eo;
        eo.score.signature_step = 5;
        eo.score.power_steps = 4;
        eo.score.begin = begin;
        eo.score.fitness.min_valid_time_ratio = 0.0;
        eo.score.fitness.max_missing_ratio = 1.0;
        eo.score.fitness.max_depth = 8;
        eo.score.fitness.max_nodes = 64;
        eo.gpus = gpus;
        eo.streams_per_gpu = 3;
        gpp::Engine engine(data, eo);
        engine.set_initial(initial);
        std::vector<std::string> errors;
        return engine.evaluate(expressions, errors);
    };
    auto cpu = run(0), gpu = run(gpp::gpu_count());
    CHECK(cpu.size() == gpu.size(), "size");
    int sigs = 0, mismatches = 0;
    for (size_t i = 0; i < cpu.size() && i < gpu.size(); ++i) {
        const auto &a = cpu[i].candidate, &b = gpu[i].candidate;
        bool ok = a.valid == b.valid && same(a.rank_ic, b.rank_ic) && same(a.fitness, b.fitness) &&
                  same(a.valid_ratio, b.valid_ratio) && same(a.missing_ratio, b.missing_ratio) &&
                  same(a.segment_rank_ic_std, b.segment_rank_ic_std) && same(a.same_sign_ratio, b.same_sign_ratio) &&
                  a.segment_rank_ics.size() == b.segment_rank_ics.size() &&
                  cpu[i].signature_valid == gpu[i].signature_valid && cpu[i].signature == gpu[i].signature;
        for (size_t k = 0; ok && k < a.segment_rank_ics.size(); ++k) ok = same(a.segment_rank_ics[k], b.segment_rank_ics[k]);
        if (!ok && mismatches++ < 5)
            std::fprintf(stderr, "  T=%d N=%d expression %zu: rank_ic %.17g vs %.17g, signature %d/%d\n", T, N, i,
                         a.rank_ic, b.rank_ic, int(cpu[i].signature_valid), int(gpu[i].signature_valid));
        sigs += cpu[i].signature_valid;
    }
    CHECK(mismatches == 0, "T=%d N=%d begin=%d: %d of %zu roots differ between CPU and GPU", T, N, begin, mismatches,
          cpu.size());
    std::printf("engine T=%d N=%d begin=%d: %zu roots, %d signatures, %d mismatches\n", T, N, begin, cpu.size(), sigs,
                mismatches);
}

void archive_case() {
    const int dim = 60, count = 3000;
    std::mt19937_64 rng(3);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0, 1);
    std::vector<std::vector<double>> centres(40, std::vector<double>(dim));
    for (auto& c : centres)
        for (auto& v : c) v = nd(rng);
    std::vector<std::vector<float>> sig(count);
    std::vector<gpp::QdCandidate> in;
    for (int i = 0; i < count; ++i) {
        const auto& c = centres[static_cast<size_t>(ud(rng) * centres.size())];
        double noise = 0.15 + 0.6 * ud(rng);
        for (int k = 0; k < dim; ++k) sig[i].push_back(static_cast<float>(c[k] + noise * nd(rng)));
    }
    for (int i = 0; i < count; ++i) in.push_back({"k" + std::to_string(i % 2500), 0.01 + 0.1 * ud(rng), sig[i].data()});
    gpp::QdArchive cpu(dim, 0.9, 0.002), gpu(dim, 0.9, 0.002);
    const int gpus = gpp::gpu_count();
    gpu.backend = [gpus](const gpp::SummaryInputs& s, std::vector<gpp::RowSummary>& o) { gpp::gpu_summaries(s, o, gpus); };
    auto a = cpu.consider(in, 512), b = gpu.consider(in, 512);
    int diff = 0;
    for (int i = 0; i < count; ++i)
        diff += a[i].action != b[i].action || a[i].closest_slot != b[i].closest_slot ||
                a[i].replaced_slot != b[i].replaced_slot ||
                // unrefined entries report their coarse FP32 value (summation order differs)
                !(same(a[i].max_similarity, b[i].max_similarity) || std::fabs(a[i].max_similarity - b[i].max_similarity) < 1e-5);
    CHECK(diff == 0, "QD: %d of %d decisions differ", diff, count);
    std::printf("archive: %d decisions, %d differ, size %zu\n", count, diff, gpu.size());
}

void pair_sums_case() {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const size_t length = 20000;
    std::vector<std::vector<float>> rows(90, std::vector<float>(length));
    for (auto& r : rows)
        for (auto& v : r) v = u(rng) < 0.05f ? std::nanf("") : u(rng);
    std::vector<const float*> left, right;
    for (int i = 0; i < 40; ++i) left.push_back(rows[i].data());
    for (int i = 40; i < 90; ++i) right.push_back(rows[i].data());
    std::vector<gpp::PairSums> a, b;
    gpp::cpu_pair_sums(left, right, length, a);
    gpp::gpu_pair_sums(left, right, length, b, gpp::gpu_count());
    double worst = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        const double x[6] = {a[k].n, a[k].sx, a[k].sy, a[k].sxx, a[k].syy, a[k].sxy};
        const double y[6] = {b[k].n, b[k].sx, b[k].sy, b[k].sxx, b[k].syy, b[k].sxy};
        for (int s = 0; s < 6; ++s) worst = std::max(worst, std::fabs(x[s] - y[s]) / std::max(1.0, std::fabs(x[s])));
    }
    CHECK(worst < 1e-12, "pair sums differ by %.3g", worst);
    std::printf("pair sums: %zu pairs, worst relative difference %.3g\n", a.size(), worst);
}
}  // namespace

int main() {
    if (gpp::gpu_count() < 1) {
        std::printf("no CUDA device: skipped\n");
        return 0;
    }
    engine_case(120, 300, 0);
    engine_case(150, 3000, 7);   // 8 items per thread
    engine_case(60, 5876, 0);    // production N, 12 items per thread
    archive_case();
    pair_sums_case();
    std::printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
