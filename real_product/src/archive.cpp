#include "gpp/archive.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpp {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

double gamma_bound(int dim) {
    // _float32_dot_error_bound: np.finfo(np.float32).eps * n / (1 - ...)
    const double u = std::numeric_limits<float>::epsilon();
    double scaled = dim * u;
    return scaled >= 1.0 ? 1.0 : scaled / (1.0 - scaled);
}

float dot32(const float* a, const float* b, int dim) {
    float s = 0.0f;
    #pragma omp simd reduction(+ : s)
    for (int k = 0; k < dim; ++k) s += a[k] * b[k];
    return s;
}

// Keeps the K best (largest lower bound) entries, descending.
inline double lower_of(const SummaryEntry& e) { return double(e.value) - double(e.bound); }

void push_top(std::vector<SummaryEntry>& top, int K, const SummaryEntry& e) {
    const double lo = lower_of(e);
    if (static_cast<int>(top.size()) == K && lo <= lower_of(top.back())) return;
    auto it = std::find_if(top.begin(), top.end(), [&](const SummaryEntry& x) { return lower_of(x) < lo; });
    top.insert(it, e);
    if (static_cast<int>(top.size()) > K) top.pop_back();
}

}  // namespace

bool normalize_signature32(const float* s, int dim, float* out, double* error) {
    std::fill(out, out + dim, 0.0f);
    *error = std::numeric_limits<double>::infinity();
    if (dim < 2) return false;
    double mean = 0.0;
    for (int k = 0; k < dim; ++k) {
        if (!std::isfinite(s[k])) return false;
        mean += s[k];
    }
    mean /= dim;
    const float mean32 = static_cast<float>(mean);
    double ss32 = 0.0, ss = 0.0;
    for (int k = 0; k < dim; ++k) {
        float c = s[k] - mean32;
        ss32 += static_cast<double>(c) * c;
        double d = s[k] - mean;
        ss += d * d;
    }
    const float norm32 = static_cast<float>(std::sqrt(ss32));
    const double norm = std::sqrt(ss);
    if (!std::isfinite(norm32) || norm32 == 0.0f || !(norm > 0.0)) return false;
    double e2 = 0.0;
    for (int k = 0; k < dim; ++k) {
        out[k] = (s[k] - mean32) / norm32;
        double d = static_cast<double>(out[k]) - (s[k] - mean) / norm;
        e2 += d * d;
    }
    *error = std::sqrt(e2);
    return std::isfinite(*error);
}

double exact_similarity(const float* a, const float* b, int dim) {
    if (dim < 2) return kNaN;
    double ma = 0, mb = 0;
    for (int k = 0; k < dim; ++k) {
        if (!std::isfinite(a[k]) || !std::isfinite(b[k])) return kNaN;
        ma += a[k];
        mb += b[k];
    }
    ma /= dim;
    mb /= dim;
    double cross = 0, aa = 0, bb = 0;
    for (int k = 0; k < dim; ++k) {
        double x = a[k] - ma, y = b[k] - mb;
        cross += x * y;
        aa += x * x;
        bb += y * y;
    }
    if (!(aa > 0) || !(bb > 0)) return kNaN;
    return std::fabs(cross / (std::sqrt(aa) * std::sqrt(bb)));
}

double similarity_bound(double le, double re, int dim) {
    return std::min(1.0, le + re + le * re + gamma_bound(dim));
}

void SummaryBuilder::add(const SummaryInputs& in, int i, int c, float v, RowSummary& r) {
    const int K = in.top_k;
    const float b = static_cast<float>(similarity_bound(in.left_error[i], in.right_error[c], in.dim));
    // Rounding the bound to float may lose a little; widen by one ulp.
    const float bw = std::nextafter(b, 2.0f);
    const SummaryEntry e{c, v, bw};
    const double lo = static_cast<double>(v) - bw, hi = static_cast<double>(v) + bw;
    push_top(r.top, K, e);
    const double kth = static_cast<int>(r.top.size()) < K ? -1e30 : lower_of(r.top.back());
    if (hi >= kth) pool.push_back(e);
    if (lo <= in.threshold && in.threshold <= hi) r.band.push_back(e);
    if (lo > in.threshold) {
        if (static_cast<int>(r.above.size()) < in.above_cap) r.above.push_back(e);
        else r.above_overflow = true;
    }
    if (pool.size() > static_cast<size_t>(8 * K + 64)) {
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                  [&](const SummaryEntry& x) { return double(x.value) + double(x.bound) < kth; }),
                   pool.end());
    }
}

void SummaryBuilder::close(RowSummary& r, int K) {
    const double kth = static_cast<int>(r.top.size()) < K ? -1e30 : lower_of(r.top.back());
    for (const auto& e : pool)
        if (double(e.value) + double(e.bound) >= kth) r.pool.push_back(e);
    pool.clear();
}

void summaries_add_tile(const SummaryInputs& in, const float* values, int c0, int ncols,
                        std::vector<SummaryBuilder>& builders, std::vector<RowSummary>& out) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < in.rows; ++i) {
        if (!in.left_valid[i]) continue;
        const float* row = values + static_cast<size_t>(i) * ncols;
        for (int c = 0; c < ncols; ++c)
            if (in.right_alive[c0 + c]) builders[i].add(in, i, c0 + c, std::fabs(row[c]), out[i]);
    }
}

void cpu_summaries(const SummaryInputs& in, std::vector<RowSummary>& out) {
    out.assign(in.rows, RowSummary());
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < in.rows; ++i) {
        if (!in.left_valid[i]) continue;
        RowSummary& r = out[i];
        const float* a = in.left + static_cast<size_t>(i) * in.dim;
        SummaryBuilder builder;
        for (int c = 0; c < in.cols; ++c) {
            if (!in.right_alive[c]) continue;
            builder.add(in, i, c, std::fabs(dot32(a, in.right + static_cast<size_t>(c) * in.dim, in.dim)), r);
        }
        builder.close(r, in.top_k);
    }
}

int QdArchive::add_slot(const QdCandidate& c, const float* normalized, double error, bool valid) {
    int slot = static_cast<int>(slot_key_.size());
    slot_key_.push_back(c.key);
    Member m;
    m.quality = c.quality;
    m.signature.assign(c.signature, c.signature + dim_);
    m.error = error;
    m.valid = valid;
    m.col = static_cast<int>(col_slot_.size());
    members_.push_back(std::move(m));
    slot_alive_.push_back(1);
    by_key_[c.key] = slot;
    matrix_.insert(matrix_.end(), normalized, normalized + dim_);
    col_error_.push_back(error);
    col_alive_.push_back(valid ? 1 : 0);
    col_slot_.push_back(slot);
    return slot;
}

void QdArchive::remove_slot(int slot) {
    slot_alive_[slot] = 0;
    auto it = by_key_.find(slot_key_[slot]);
    if (it != by_key_.end() && it->second == slot) by_key_.erase(it);
    int col = members_[slot].col;
    if (col >= 0) {
        col_alive_[col] = 0;
        col_slot_[col] = -1;
        std::fill(matrix_.begin() + static_cast<size_t>(col) * dim_, matrix_.begin() + static_cast<size_t>(col + 1) * dim_,
                  0.0f);
        ++dead_cols_;
    }
    members_[slot].signature.clear();
    members_[slot].signature.shrink_to_fit();
}

void QdArchive::compact() {
    if (dead_cols_ * 4 < static_cast<long>(col_slot_.size())) return;
    std::vector<float> matrix;
    std::vector<double> error;
    std::vector<unsigned char> alive;
    std::vector<int> slots;
    for (size_t c = 0; c < col_slot_.size(); ++c) {
        int s = col_slot_[c];
        if (s < 0) continue;
        members_[s].col = static_cast<int>(slots.size());
        slots.push_back(s);
        matrix.insert(matrix.end(), matrix_.begin() + c * dim_, matrix_.begin() + (c + 1) * dim_);
        error.push_back(col_error_[c]);
        alive.push_back(col_alive_[c]);
    }
    matrix_.swap(matrix);
    col_error_.swap(error);
    col_alive_.swap(alive);
    col_slot_.swap(slots);
    dead_cols_ = 0;
}

void QdArchive::reset(const std::vector<QdCandidate>& records) {
    slot_key_.clear();
    members_.clear();
    slot_alive_.clear();
    by_key_.clear();
    matrix_.clear();
    col_error_.clear();
    col_alive_.clear();
    col_slot_.clear();
    dead_cols_ = 0;
    std::vector<float> norm(dim_);
    for (const auto& r : records) {
        double e;
        bool ok = normalize_signature32(r.signature, dim_, norm.data(), &e);
        auto it = by_key_.find(r.key);
        if (it != by_key_.end()) remove_slot(it->second);
        add_slot(r, norm.data(), e, ok);
    }
    compact();
}

std::vector<QdDecision> QdArchive::consider(const std::vector<QdCandidate>& records, int block_size) {
    if (block_size < 1) throw std::invalid_argument("block size must be positive");
    std::vector<QdDecision> decisions;
    decisions.reserve(records.size());
    const double thr = threshold_;
    for (size_t start = 0; start < records.size(); start += block_size) {
        const size_t stop = std::min(records.size(), start + static_cast<size_t>(block_size));
        const int B = static_cast<int>(stop - start);
        auto t = Clock::now();
        compact();
        const int cols = static_cast<int>(col_slot_.size());
        // Normalized candidates.
        std::vector<float> left(static_cast<size_t>(B) * dim_);
        std::vector<double> left_error(B);
        std::vector<unsigned char> left_valid(B);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < B; ++i) {
            double e;
            left_valid[i] = normalize_signature32(records[start + i].signature, dim_, left.data() + size_t(i) * dim_, &e);
            left_error[i] = e;
        }
        // Coarse snapshot summaries (GEMM pass) and the in-block matrix.
        SummaryInputs in;
        in.left = left.data();
        in.left_error = left_error.data();
        in.left_valid = left_valid.data();
        in.rows = B;
        in.right = matrix_.data();
        in.right_error = col_error_.data();
        in.right_alive = col_alive_.data();
        in.cols = cols;
        in.dim = dim_;
        in.threshold = thr;
        in.top_k = top_k;
        std::vector<RowSummary> summary;
        if (backend) backend(in, summary);
        else cpu_summaries(in, summary);
        std::vector<float> internal(static_cast<size_t>(B) * B, 0.0f);
        #pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < B; ++i)
            for (int k = 0; k < i; ++k)
                internal[size_t(i) * B + k] =
                    std::fabs(dot32(left.data() + size_t(i) * dim_, left.data() + size_t(k) * dim_, dim_));
        // Snapshot members that are alive but have no valid FP32 row.
        std::vector<int> invalid_cols;
        for (int c = 0; c < cols; ++c)
            if (col_slot_[c] >= 0 && !col_alive_[c]) invalid_cols.push_back(c);
        timing.similarity += since(t);
        t = Clock::now();

        std::vector<char> deleted(cols, 0);
        std::vector<int> block_slot(B, -1);    // slot of an in-block admitted candidate
        std::vector<int> members;              // in-block candidates currently in the archive
        struct Entry { int slot; double value; double bound; bool exact; };
        std::vector<Entry> entries;
        for (int i = 0; i < B; ++i) {
            const QdCandidate& rec = records[start + i];
            ++timing.candidates;
            QdDecision d;
            if (!std::isfinite(rec.quality)) {
                d.action = QdAction::Reject;
                d.max_similarity = kNaN;
                decisions.push_back(d);
                continue;
            }
            const bool archive_was_empty = by_key_.empty();
            auto inc_it = by_key_.find(rec.key);
            const int incumbent = inc_it == by_key_.end() ? -1 : inc_it->second;
            entries.clear();
            const RowSummary& sm = summary[i];
            bool full = !left_valid[i];
            double lower = -std::numeric_limits<double>::infinity();
            if (!full) {
                int first_alive = -1;
                for (size_t k = 0; k < sm.top.size(); ++k)
                    if (!deleted[sm.top[k].col]) { first_alive = static_cast<int>(k); break; }
                if (!sm.top.empty() && first_alive < 0) full = true;   // all K best deleted in this block
                if (!full && first_alive >= 0)
                    lower = lower_of(sm.top[first_alive]);
                if (!full && incumbent >= 0 && sm.above_overflow) {
                    // The 'any other >= threshold' test may need entries the cap dropped.
                    bool found = false;
                    for (const auto& e : sm.above)
                        if (!deleted[e.col] && col_slot_[e.col] != incumbent) { found = true; break; }
                    if (!found) full = true;
                }
            }
            if (full) {
                ++timing.fallbacks;
                // Exact similarity against every current member.
                for (int c = 0; c < cols; ++c) {
                    if (deleted[c] || col_slot_[c] < 0) continue;
                    int s = col_slot_[c];
                    double v = exact_similarity(rec.signature, members_[s].signature.data(), dim_);
                    ++timing.refined;
                    if (std::isfinite(v)) entries.push_back({s, v, 0.0, true});
                }
                for (int k : members) {
                    int s = block_slot[k];
                    double v = exact_similarity(rec.signature, members_[s].signature.data(), dim_);
                    ++timing.refined;
                    if (std::isfinite(v)) entries.push_back({s, v, 0.0, true});
                }
            } else {
                std::vector<int> seen;
                auto add_snapshot = [&](const SummaryEntry& e) {
                    if (deleted[e.col]) return;
                    if (std::find(seen.begin(), seen.end(), e.col) != seen.end()) return;
                    seen.push_back(e.col);
                    entries.push_back({col_slot_[e.col], double(e.value), double(e.bound), false});
                };
                for (const auto& e : sm.pool) add_snapshot(e);
                for (const auto& e : sm.band) add_snapshot(e);
                for (int c : invalid_cols) {
                    if (deleted[c]) continue;
                    int s = col_slot_[c];
                    double v = exact_similarity(rec.signature, members_[s].signature.data(), dim_);
                    ++timing.refined;
                    if (std::isfinite(v)) {
                        entries.push_back({s, v, 0.0, true});
                        lower = std::max(lower, v);
                    }
                }
                for (int k : members) {
                    int s = block_slot[k];
                    if (left_valid[k]) {
                        double v = internal[size_t(i) * B + k];
                        double b = similarity_bound(left_error[i], left_error[k], dim_);
                        entries.push_back({s, v, b, false});
                        lower = std::max(lower, v - b);
                    } else {
                        double v = exact_similarity(rec.signature, members_[s].signature.data(), dim_);
                        ++timing.refined;
                        if (std::isfinite(v)) {
                            entries.push_back({s, v, 0.0, true});
                            lower = std::max(lower, v);
                        }
                    }
                }
                // Refine exactly what Python refines.
                for (auto& e : entries) {
                    if (e.exact) continue;
                    if (e.value + e.bound >= lower || (e.value - e.bound <= thr && thr <= e.value + e.bound)) {
                        e.value = exact_similarity(rec.signature, members_[e.slot].signature.data(), dim_);
                        e.exact = true;
                        ++timing.refined;
                    }
                }
                entries.erase(std::remove_if(entries.begin(), entries.end(),
                                             [](const Entry& e) { return !std::isfinite(e.value); }),
                              entries.end());
            }
            // max by (similarity, key)
            int best = -1;
            for (size_t k = 0; k < entries.size(); ++k) {
                if (best < 0 || entries[k].value > entries[best].value ||
                    (entries[k].value == entries[best].value && slot_key_[entries[k].slot] > slot_key_[entries[best].slot]))
                    best = static_cast<int>(k);
            }
            auto admit_new = [&]() {
                std::vector<float> norm(left.begin() + size_t(i) * dim_, left.begin() + size_t(i + 1) * dim_);
                int s = static_cast<int>(slot_key_.size());
                slot_key_.push_back(rec.key);
                Member m;
                m.quality = rec.quality;
                m.signature.assign(rec.signature, rec.signature + dim_);
                m.error = left_error[i];
                m.valid = left_valid[i];
                m.col = -1;   // joins the matrix after this block
                members_.push_back(std::move(m));
                slot_alive_.push_back(1);
                by_key_[rec.key] = s;
                block_slot[i] = s;
                members.push_back(i);
                return s;
            };
            auto drop = [&](int slot) {
                int col = members_[slot].col;
                if (col >= 0) deleted[col] = 1;
                for (size_t k = 0; k < members.size(); ++k)
                    if (block_slot[members[k]] == slot) { members.erase(members.begin() + k); break; }
                remove_slot(slot);
            };
            if (incumbent >= 0) {
                double maxv = best >= 0 ? entries[best].value : kNaN;
                int closest = best >= 0 ? entries[best].slot : incumbent;
                bool other = false;
                if (!full) {
                    for (const auto& e : sm.above)
                        if (!deleted[e.col] && col_slot_[e.col] != incumbent) { other = true; break; }
                }
                for (const auto& e : entries)
                    if (e.slot != incumbent && e.value >= thr) { other = true; break; }
                d.max_similarity = maxv;
                if (other) {
                    d.action = QdAction::Reject;
                    d.closest_slot = closest;
                } else if (rec.quality <= members_[incumbent].quality + margin_) {
                    d.action = QdAction::Reject;
                    d.closest_slot = incumbent;
                } else {
                    d.action = QdAction::Replace;
                    drop(incumbent);
                    d.new_slot = admit_new();
                    // Python reports closest = replaced = record.key.
                    d.closest_slot = d.replaced_slot = incumbent;
                }
                decisions.push_back(d);
                continue;
            }
            if (best < 0) {
                d.action = QdAction::Admit;
                d.max_similarity = archive_was_empty ? 0.0 : kNaN;
                d.new_slot = admit_new();
                decisions.push_back(d);
                continue;
            }
            d.max_similarity = entries[best].value;
            d.closest_slot = entries[best].slot;
            if (entries[best].value < thr) {
                d.action = QdAction::Admit;
                d.new_slot = admit_new();
            } else if (rec.quality <= members_[d.closest_slot].quality + margin_) {
                d.action = QdAction::Reject;
            } else {
                d.action = QdAction::Replace;
                d.replaced_slot = d.closest_slot;
                drop(d.closest_slot);
                d.new_slot = admit_new();
            }
            decisions.push_back(d);
        }
        // In-block admissions join the dense matrix for the next block.
        for (int k : members) {
            int s = block_slot[k];
            members_[s].col = static_cast<int>(col_slot_.size());
            matrix_.insert(matrix_.end(), left.begin() + size_t(k) * dim_, left.begin() + size_t(k + 1) * dim_);
            col_error_.push_back(left_error[k]);
            col_alive_.push_back(left_valid[k]);
            col_slot_.push_back(s);
        }
        timing.admission += since(t);
    }
    return decisions;
}

}  // namespace gpp
