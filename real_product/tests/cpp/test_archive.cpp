// QdArchive vs the course archive (pca::Archive, itself checked against the
// Python PCAQDArchive) on clustered random signatures with near-threshold
// pairs, repeated keys and NaN quality, for several block sizes.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core.hpp"
#include "gpp/archive.hpp"

namespace {
int failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++failures;                                               \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

const char* name(gpp::QdAction a) {
    return a == gpp::QdAction::Admit ? "admit" : a == gpp::QdAction::Replace ? "replace" : "reject";
}
}  // namespace

int main() {
    const int dim = 40, count = 900;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0, 1);
    // 30 cluster centres; members are centre + noise of varying size so that
    // similarities spread around the 0.90 threshold.
    std::vector<std::vector<double>> centres(30, std::vector<double>(dim));
    for (auto& c : centres)
        for (auto& v : c) v = nd(rng);
    std::vector<pca::Record> records;
    for (int i = 0; i < count; ++i) {
        pca::Record r;
        int c = static_cast<int>(ud(rng) * centres.size());
        double noise = 0.15 + 0.6 * ud(rng);
        for (int k = 0; k < dim; ++k) r.signature.push_back(static_cast<float>(centres[c][k] + noise * nd(rng)));
        r.key = "k" + std::to_string(i % 700);   // repeated keys exercise the incumbent path
        r.quality = 0.01 + 0.1 * ud(rng);
        if (i % 97 == 0) r.quality = std::nan("");
        if (i % 53 == 0 && i > 0) r.signature = records[i - 1].signature;   // exact duplicate signature
        records.push_back(r);
    }
    // Reference: one pca::Archive run.
    pca::Archive ref;
    auto want = ref.consider(records, 64);

    for (int block : {1, 7, 128, 4096}) {
        gpp::QdArchive a(dim, 0.90, 0.002);
        std::vector<gpp::QdCandidate> in;
        for (const auto& r : records) in.push_back({r.key, r.quality, r.signature.data()});
        // Split the stream into two calls to exercise the persistent state.
        std::vector<gpp::QdCandidate> first(in.begin(), in.begin() + 400), second(in.begin() + 400, in.end());
        auto got = a.consider(first, block);
        auto more = a.consider(second, block);
        got.insert(got.end(), more.begin(), more.end());
        int mismatches = 0;
        for (int i = 0; i < count; ++i) {
            const auto& g = got[i];
            const auto& w = want[i];
            std::string closest = g.closest_slot >= 0 ? a.slot_key(g.closest_slot) : "";
            std::string replaced = g.replaced_slot >= 0 ? a.slot_key(g.replaced_slot) : "";
            bool sim_ok = (std::isnan(g.max_similarity) && std::isnan(w.similarity)) ||
                          std::fabs(g.max_similarity - w.similarity) <= 1e-12;
            if (w.action != name(g.action) || closest != w.closest || replaced != w.replaced || !sim_ok) {
                if (mismatches++ < 5)
                    std::fprintf(stderr, "block %d record %d: got %s/%s/%s/%.17g want %s/%s/%s/%.17g\n", block, i,
                                 name(g.action), closest.c_str(), replaced.c_str(), g.max_similarity, w.action.c_str(),
                                 w.closest.c_str(), w.replaced.c_str(), w.similarity);
            }
        }
        CHECK(mismatches == 0, "block %d: %d mismatching decisions", block, mismatches);
        CHECK(a.size() == ref.records.size(), "block %d: archive size %zu vs %zu", block, a.size(), ref.records.size());
        std::printf("block %4d: %d decisions, archive %zu, refined %lld, fallbacks %lld\n", block, count, a.size(),
                    a.timing.refined, a.timing.fallbacks);
    }
    int admits = 0, replaces = 0;
    for (const auto& w : want) {
        admits += w.action == "admit";
        replaces += w.action == "replace";
    }
    std::printf("reference: %d admit, %d replace, %d reject\n", admits, replaces, count - admits - replaces);
    CHECK(admits > 20 && replaces > 5 && count - admits - replaces > 20, "test does not cover all actions");
    if (failures) return 1;
    std::printf("OK\n");
    return 0;
}
