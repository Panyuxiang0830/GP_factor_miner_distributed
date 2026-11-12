#pragma once
#include "core.hpp"
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pca {
// Signature storage is sharded; small key/quality metadata is replicated.
// Each candidate sees all earlier updates. MAXLOC tie positions are negative
// lexicographic indices so MPI's smallest-location tie rule matches Python.
class MpiArchive {
    int rank_, ranks_;
    MPI_Comm comm_;
    std::map<std::string, Record> local_;
    std::map<std::string, NormalizedSignature> normalized_;
    int owner(const std::string& key) const {
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char c : key) { hash ^= c; hash *= 1099511628211ULL; }
        return int(hash % ranks_);
    }
    void store(const Record& r) {
        qualities[r.key] = r.quality;
        if (owner(r.key) == rank_) { local_[r.key] = r; normalized_[r.key] = normalize_signature(r.signature); }
    }
public:
    std::map<std::string, double> qualities;
    double similarity_seconds = 0, admission_seconds = 0, communication_seconds = 0;
    long long coarse_pairs = 0, refined_pairs = 0;
    explicit MpiArchive(MPI_Comm comm) : comm_(comm) { MPI_Comm_rank(comm_, &rank_); MPI_Comm_size(comm_, &ranks_); }
    void seed(const Record& record) { store(record); }
    std::vector<Decision> consider(const std::vector<Record>& candidates) {
        using Clock = std::chrono::steady_clock;
        auto seconds = [](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
        std::vector<Decision> decisions;
        for (const auto& candidate : candidates) {
            if (!std::isfinite(candidate.quality)) {
                decisions.push_back({"reject", "", "", std::numeric_limits<double>::quiet_NaN()}); continue;
            }
            auto begin = Clock::now();
            std::vector<std::string> keys;
            for (const auto& q : qualities) keys.push_back(q.first);
            struct Entry { const Record* record; BoundedSimilarity pair; };
            std::vector<Entry> entries;
            double lower = -std::numeric_limits<double>::infinity();
            auto normalized_candidate = normalize_signature(candidate.signature);
            for (const auto& item : local_) {
                auto pair = bounded_similarity(normalized_candidate, normalized_.at(item.first)); ++coarse_pairs;
                if (!std::isfinite(pair.value)) { pair = {similarity(candidate.signature, item.second.signature), 0}; ++refined_pairs; }
                if (std::isfinite(pair.value)) { lower = std::max(lower, pair.value - pair.error); entries.push_back({&item.second, pair}); }
            }
            double maximum = -1; std::string closest;
            int other_similar = 0;
            for (const auto& e : entries) {
                double value = e.pair.value, bound = e.pair.error;
                if (value + bound >= lower || (value - bound <= .90 && value + bound >= .90)) {
                    value = similarity(candidate.signature, e.record->signature); ++refined_pairs;
                }
                if (e.record->key != candidate.key && value >= .90) other_similar = 1;
                if (value > maximum || (value == maximum && e.record->key > closest)) { maximum = value; closest = e.record->key; }
            }
            similarity_seconds += seconds(begin); begin = Clock::now();
            struct { double value; int position; } local{maximum, 0}, global;
            if (!closest.empty()) local.position = -int(std::lower_bound(keys.begin(), keys.end(), closest) - keys.begin());
            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm_);
            bool incumbent = qualities.count(candidate.key);
            if (incumbent) MPI_Allreduce(MPI_IN_PLACE, &other_similar, 1, MPI_INT, MPI_MAX, comm_);
            communication_seconds += seconds(begin); begin = Clock::now();
            maximum = global.value; closest = maximum < 0 ? "" : keys.at(-global.position);
            Decision d{"", closest, "", maximum < 0 ? (qualities.empty() ? 0 : std::numeric_limits<double>::quiet_NaN()) : maximum};
            if (incumbent) {
                if (other_similar) d.action = "reject";
                else if (candidate.quality <= qualities.at(candidate.key) + .002) { d.action = "reject"; d.closest = candidate.key; }
                else { d.action = "replace"; d.closest = d.replaced = candidate.key; store(candidate); }
            } else if (maximum < .90) { d.action = "admit"; store(candidate); }
            else if (candidate.quality <= qualities.at(closest) + .002) d.action = "reject";
            else {
                d.action = "replace"; d.replaced = closest;
                qualities.erase(closest); local_.erase(closest); normalized_.erase(closest); store(candidate);
            }
            decisions.push_back(d); admission_seconds += seconds(begin);
        }
        return decisions;
    }
};
}  // namespace pca
