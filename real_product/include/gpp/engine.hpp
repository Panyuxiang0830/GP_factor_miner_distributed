#pragma once
// Single-host, shared-memory batch evaluator (no MPI).
//
// One batch (a GP chunk of up to ~10^5 expressions) is merged into a DAG of
// distinct subexpressions and executed level by level; every non-leaf node of
// a level chunk is one OpenMP task (node-parallel, the best mode measured on
// Vanda). A root is scored by the thread that computed it, right after its
// operator, and its panel is released as soon as no parent needs it.
//
// Memory: leaf fields and the target ranks stay resident; intermediate panels
// are bounded by a byte budget. The chunk size of each level is chosen on the
// fly from the budget (largest chunk that fits, capped by max_chunk). When a
// batch cannot fit even with the smallest chunk, the batch is split into
// groups of expressions that share subtrees (sorted by child node) and each
// group is planned and executed on its own.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/json.hpp"
#include "gpe/panel.hpp"
#include "gpp/scoring.hpp"

namespace gpp {

class GpuScorer;  // gpu.hpp (only with GPP_HAVE_CUDA)

struct EngineOptions {
    int threads = 0;                   // 0: omp_get_max_threads()
    double memory_budget_bytes = 0.0;  // 0: unlimited (no grouping)
    int max_chunk = 2000;              // Vanda: 1000-2000 nodes per chunk were best
    int min_chunk = 0;                 // 0: number of threads
    int narrow_level = 0;              // levels with fewer tasks than this run operator-parallel; 0: threads/4
    ScoreOptions score;
    // Root scoring on the GPUs (CUDA build only): 0 = CPU.
    int gpus = 0;
    int streams_per_gpu = 4;
};

struct EngineStats {
    double load = 0, plan = 0, compute = 0, score_wait = 0, total = 0;
    long batches = 0, groups = 0, expressions = 0, dag_nodes = 0, roots = 0, chunks = 0;
    long peak_panels = 0;          // intermediate panels alive at once (max over batches)
    double peak_bytes = 0;         // resident + scratch + peak panels
    double resident_bytes = 0;     // leaves + target ranks
    double scratch_bytes = 0;      // per-thread scoring scratch
};

// Planner output for one expression group.
struct GroupPlan {
    std::vector<int> expressions;  // indices into the batch
    long dag_nodes = 0;
    long peak_panels = 0;
    long chunks = 0;
    bool feasible = true;
};

class Engine {
public:
    Engine(gpe::PanelSet data, EngineOptions options);
    ~Engine();

    void set_initial(std::vector<double> initial);   // power-iteration start, size N
    const gpe::PanelSet& data() const { return data_; }
    int sampled_signature_rows() const;

    // One RootScore per expression, in input order. errors[i] is non-empty if
    // expression i could not be parsed (unknown field/operator, x-x, ...).
    std::vector<RootScore> evaluate(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors);

    // Grouping and peak memory a batch would use, without evaluating it.
    std::vector<GroupPlan> plan(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors);

    const EngineStats& stats() const { return stats_; }
    double panel_bytes() const { return 4.0 * data_.T * data_.N; }
    long budget_panels() const;   // intermediate panels the budget allows (-1: unlimited)
    int threads() const { return threads_; }

    // Adaptive-chunk replay of the executor's allocation/release order.
    // Returns the peak number of intermediate panels, or -1 if some chunk
    // start finds fewer than min_chunk free slots (budget < 0: unlimited).
    static long simulate_peak(const gpe::Dag& dag, long budget, int min_chunk, int max_chunk, long* chunks = nullptr);

private:
    const gpe::Panel& field(const std::string& name);
    void ensure_fields(const gpe::Dag& dag);
    void run_group(const gpe::Dag& dag, std::vector<RootScore>& scores_by_root);
    std::vector<GroupPlan> make_groups(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors,
                                       std::vector<int>& valid);
    void refresh_memory_accounting();

    gpe::PanelSet data_;
    EngineOptions options_;
    int threads_ = 1;
    std::unordered_map<std::string, gpe::Panel> fields_;
    std::vector<double> target_rank_;
    std::vector<double> initial_;
    gpe::BufferPool pool_;
    std::vector<ScoreScratch> scratch_;
    EngineStats stats_;
    std::unique_ptr<GpuScorer> gpu_;
};

// Fields referenced by an expression (for validation before the DAG).
void collect_fields(const gpe::Json& expression, std::vector<std::string>& out);

}  // namespace gpp
