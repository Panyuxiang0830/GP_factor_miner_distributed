#pragma once
// PCA-QD archive with the decisions of PCAQDArchive.consider_batch
// (python/src/gp_factor_miner/pca_qd.py), sized for 10^5-10^6 members.
//
// Admission is sequential by definition: record i sees every admission and
// replacement of records 0..i-1. What is parallel is the coarse FP32
// similarity of a block of candidates against the archive snapshot taken at
// the start of the block (a GEMM). For every candidate the GEMM pass keeps a
// small summary instead of the whole row:
//   - the K best lower bounds (similarity - error bound), to know the
//     largest lower bound after in-block deletions,
//   - every entry whose upper bound reaches the K-th best lower bound (the
//     only entries the Python code may refine for the maximum),
//   - every entry whose bound interval contains the threshold,
//   - entries certainly above the threshold (capped; overflow -> fallback).
// The sequential scan then touches O(summary + block) entries per candidate
// and refines with FP64 exactly the comparisons Python refines. If in-block
// deletions invalidate a summary (all K best deleted), that candidate falls
// back to a full exact recomputation, so decisions never depend on K or on
// the block size.

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpp {

enum class QdAction : int { Admit = 0, Replace = 1, Reject = 2 };

struct QdDecision {
    QdAction action = QdAction::Reject;
    double max_similarity = 0.0;  // NaN where Python reports nan
    int closest_slot = -1;        // archive slot of closest_key (-1: None)
    int replaced_slot = -1;       // slot removed by a replacement (-1: None)
    int new_slot = -1;            // slot the candidate now occupies (admit/replace)
};

struct QdCandidate {
    std::string key;
    double quality = 0.0;          // abs RankIC (QDRecord.abs_rank_ic)
    const float* signature = nullptr;
};

struct SummaryEntry {
    int col;      // dense column of the snapshot
    float value;  // coarse |similarity|
    float bound;  // error bound of that value
};

struct RowSummary {
    std::vector<SummaryEntry> top;    // K best lower bounds, descending
    std::vector<SummaryEntry> pool;   // upper bound >= K-th best lower bound
    std::vector<SummaryEntry> band;   // lower <= threshold <= upper
    std::vector<SummaryEntry> above;  // lower > threshold (capped)
    bool above_overflow = false;
};

// Computes the summaries of `rows` candidates (normalized FP32 rows with
// their errors) against `cols` archive columns. A GPU implementation may
// replace the CPU one (gpu.hpp).
struct SummaryInputs {
    const float* left = nullptr;    // rows x dim
    const double* left_error = nullptr;
    const unsigned char* left_valid = nullptr;
    int rows = 0;
    const float* right = nullptr;   // cols x dim (dead columns are zero rows)
    const double* right_error = nullptr;
    const unsigned char* right_alive = nullptr;  // alive and valid
    int cols = 0;
    int dim = 0;
    double threshold = 0.9;
    int top_k = 4;
    int above_cap = 16;
};
using SummaryBackend = std::function<void(const SummaryInputs&, std::vector<RowSummary>&)>;
void cpu_summaries(const SummaryInputs& in, std::vector<RowSummary>& out);

// Builds one row's summary from coarse |similarities| fed in increasing
// column order; shared by cpu_summaries (values from dot32) and the GPU
// backend (values from SGEMM tiles). similarity_bound covers any summation
// order of the FP32 dot (and FMA), so both produce valid summaries.
struct SummaryBuilder {
    std::vector<SummaryEntry> pool;
    void add(const SummaryInputs& in, int row, int col, float value, RowSummary& out);
    void close(RowSummary& out, int top_k);
};
// Feeds one tile of coarse values (rows x ncols, row-major, columns
// c0..c0+ncols-1, not yet absolute) to every valid row, rows in parallel.
void summaries_add_tile(const SummaryInputs& in, const float* values, int c0, int ncols,
                        std::vector<SummaryBuilder>& builders, std::vector<RowSummary>& out);

// Float32 normalization (centered, unit L2) and its distance to the FP64
// normalization, as _normalized_matrix / _normalization_error_bounds.
bool normalize_signature32(const float* signature, int dim, float* out, double* error);
// signature_similarity(): |Pearson| in FP64 of two FP32 signatures (NaN if undefined).
double exact_similarity(const float* a, const float* b, int dim);
// _similarity_error_bound
double similarity_bound(double left_error, double right_error, int dim);

struct QdTiming {
    double similarity = 0, admission = 0;
    long long refined = 0, fallbacks = 0, candidates = 0;
};

class QdArchive {
public:
    QdArchive(int dim, double threshold, double margin) : dim_(dim), threshold_(threshold), margin_(margin) {}

    int dim() const { return dim_; }
    size_t size() const { return by_key_.size(); }
    int slots() const { return static_cast<int>(slot_key_.size()); }
    const std::string& slot_key(int slot) const { return slot_key_[slot]; }

    // Replaces the content (resume from a Python checkpoint); slots 0..n-1.
    void reset(const std::vector<QdCandidate>& records);
    // consider_batch with Python's semantics. block_size only changes speed.
    std::vector<QdDecision> consider(const std::vector<QdCandidate>& records, int block_size = 4096);

    SummaryBackend backend;   // empty: cpu_summaries
    int top_k = 4;
    QdTiming timing;

private:
    struct Member {
        double quality;
        std::vector<float> signature;
        double error;
        bool valid;
        int col;   // dense column, -1 if not in the matrix (added after the snapshot)
    };
    int add_slot(const QdCandidate& c, const float* normalized, double error, bool valid);
    void remove_slot(int slot);
    void compact();

    int dim_;
    double threshold_, margin_;
    std::vector<std::string> slot_key_;
    std::vector<Member> members_;            // by slot
    std::vector<char> slot_alive_;
    std::unordered_map<std::string, int> by_key_;
    // Dense snapshot matrix: column c holds slot col_slot_[c].
    std::vector<float> matrix_;
    std::vector<double> col_error_;
    std::vector<unsigned char> col_alive_;
    std::vector<int> col_slot_;
    long dead_cols_ = 0;
};

}  // namespace gpp
