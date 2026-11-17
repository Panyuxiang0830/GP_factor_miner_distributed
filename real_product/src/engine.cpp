#include "gpp/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

#include "gpe/operators.hpp"
#ifdef GPP_HAVE_CUDA
#include "gpp/gpu.hpp"
#else
namespace gpp {
class GpuScorer {};  // CPU-only build: never instantiated
}
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gpp {
namespace {

using Clock = std::chrono::steady_clock;

// GPP_NUMA_INTERLEAVE=1: spread pages allocated from now on round-robin over
// all NUMA nodes (set_mempolicy(MPOL_INTERLEAVE)). Panels are written by one
// thread and read by threads on every socket, so first-touch placement piles
// them on one node and the other socket's cores stall on remote reads. The
// policy is per thread and inherited by threads created later (the OpenMP
// pool), so call it before the first parallel region.
void numa_interleave() {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    if (!std::getenv("GPP_NUMA_INTERLEAVE")) return;
    unsigned long mask[16] = {0};
    int nodes = 0;
    for (int n = 0; n < 1024; ++n) {
        const std::string path = "/sys/devices/system/node/node" + std::to_string(n);
        if (access(path.c_str(), F_OK) != 0) continue;
        mask[n / 64] |= 1UL << (n % 64);
        ++nodes;
    }
    const int MPOL_INTERLEAVE_ = 3;
    if (nodes > 1 && syscall(SYS_set_mempolicy, MPOL_INTERLEAVE_, mask, 1024 + 1) != 0)
        std::fprintf(stderr, "[gpp] set_mempolicy(MPOL_INTERLEAVE) failed\n");
#endif
}
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

std::vector<std::vector<int>> levels_of(const gpe::Dag& dag) {
    std::vector<std::vector<int>> levels(dag.max_level() + 1);
    const auto& nodes = dag.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) levels[nodes[i].level].push_back(static_cast<int>(i));
    return levels;
}

// Chunk size for the next chunk of a level: as many nodes as free slots,
// capped by max_chunk and the rest of the level.
long next_chunk(long remaining, long alive, long budget, int min_chunk, int max_chunk) {
    long want = std::min<long>(remaining, max_chunk);
    if (budget < 0) return want;
    long free_slots = budget - alive;
    if (free_slots >= want) return want;
    if (free_slots >= std::min<long>(remaining, min_chunk)) return std::max<long>(free_slots, 1);
    return -1;  // infeasible
}

}  // namespace

void collect_fields(const gpe::Json& e, std::vector<std::string>& out) {
    if (e.is_string()) {
        out.push_back(e.string);
        return;
    }
    if (!e.is_array()) return;
    for (size_t i = 1; i < e.array.size(); ++i) collect_fields(e.array[i], out);
}

Engine::Engine(gpe::PanelSet data, EngineOptions options) : data_(std::move(data)), options_(options) {
    numa_interleave();
    threads_ = options_.threads > 0 ? options_.threads : max_threads();
#ifdef _OPENMP
    omp_set_num_threads(threads_);
    omp_set_max_active_levels(1);  // node tasks run their operator serially
#endif
    if (options_.min_chunk <= 0) options_.min_chunk = threads_;
    if (options_.narrow_level <= 0) options_.narrow_level = std::max(2, threads_ / 4);
    if (options_.max_chunk <= 0) options_.max_chunk = 1 << 30;
    if (options_.score.begin < 0 || options_.score.begin >= data_.T) throw std::invalid_argument("score begin outside data");
    auto t = Clock::now();
    gpe::Panel target = data_.load_target();
    target_rank_ = gpe::rank_rows_pct(target);
    stats_.load += since(t);
    scratch_.resize(threads_);
    initial_.assign(data_.N, 1.0);
#ifdef GPP_HAVE_CUDA
    // The GPU scorer returns daily ICs and signatures; screening (sampled
    // rank vectors) keeps scoring on the CPU threads.
    if (options_.gpus > 0 && options_.score.rank_vector_step == 0)
        gpu_ = std::make_unique<GpuScorer>(options_.gpus, options_.streams_per_gpu, data_.T, data_.N, target_rank_,
                                           options_.score, initial_);
#else
    if (options_.gpus > 0) throw std::invalid_argument("this build has no CUDA scorer (GPP_ENABLE_CUDA=OFF)");
#endif
    refresh_memory_accounting();
}

Engine::~Engine() = default;

void Engine::set_initial(std::vector<double> initial) {
    if (static_cast<int>(initial.size()) != data_.N) throw std::invalid_argument("initial vector must have N values");
    initial_ = std::move(initial);
#ifdef GPP_HAVE_CUDA
    if (gpu_) gpu_->set_initial(initial_);
#endif
}

int Engine::sampled_signature_rows() const {
    return sampled_rows(data_.T - options_.score.begin, options_.score.signature_step);
}

void Engine::refresh_memory_accounting() {
    const double N = data_.N, days = data_.T - options_.score.begin;
    stats_.resident_bytes = 8.0 * data_.T * N + panel_bytes() * fields_.size();
    const int m = sampled_signature_rows();
    const int mv = sampled_rows(static_cast<int>(days), options_.score.rank_vector_step);
    // R (doubles), rank row, daily IC, rank vector being filled.
    stats_.scratch_bytes = threads_ * (8.0 * m * N + 12.0 * N + 8.0 * days + 4.0 * mv * N);
}

long Engine::budget_panels() const {
    if (options_.memory_budget_bytes <= 0) return -1;
    double free_bytes = options_.memory_budget_bytes - stats_.resident_bytes - stats_.scratch_bytes;
    // evaluate() loads every field of a batch before planning, so the
    // resident part is exact when the budget is used.
    return std::max<long>(0, static_cast<long>(free_bytes / panel_bytes()));
}

const gpe::Panel& Engine::field(const std::string& name) {
    auto it = fields_.find(name);
    if (it != fields_.end()) return it->second;
    auto t = Clock::now();
    gpe::Panel p = data_.load_field(name);
    stats_.load += since(t);
    auto& slot = fields_[name];
    slot = std::move(p);
    refresh_memory_accounting();
    return slot;
}

void Engine::ensure_fields(const gpe::Dag& dag) {
    for (const auto& name : dag.fields()) field(name);
}

long Engine::simulate_peak(const gpe::Dag& dag, long budget, int min_chunk, int max_chunk, long* chunks) {
    const auto& nodes = dag.nodes();
    std::set<int> roots(dag.roots().begin(), dag.roots().end());
    std::vector<int> pending(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) pending[i] = nodes[i].parents;
    long alive = 0, peak = 0, count = 0;
    for (const auto& level : levels_of(dag)) {
        size_t c0 = 0;
        while (c0 < level.size()) {
            long remaining = static_cast<long>(level.size() - c0);
            // Leaves are resident, so a leaf-only level needs no slot.
            bool leaves = nodes[level[c0]].op == gpe::Op::Leaf;
            long step = leaves ? remaining : next_chunk(remaining, alive, budget, min_chunk, max_chunk);
            if (step < 0) return -1;
            size_t c1 = c0 + static_cast<size_t>(step);
            ++count;
            for (size_t k = c0; k < c1; ++k)
                if (nodes[level[k]].op != gpe::Op::Leaf) ++alive;
            peak = std::max(peak, alive);
            for (size_t k = c0; k < c1; ++k) {
                int i = level[k];
                if (roots.count(i) && pending[i] == 0 && nodes[i].op != gpe::Op::Leaf) --alive;
            }
            for (size_t k = c0; k < c1; ++k)
                for (int c : {nodes[level[k]].a, nodes[level[k]].b})
                    if (c >= 0 && --pending[c] == 0 && nodes[c].op != gpe::Op::Leaf) --alive;
            c0 = c1;
        }
    }
    if (chunks) *chunks = count;
    return peak;
}

std::vector<GroupPlan> Engine::make_groups(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors,
                                           std::vector<int>& valid) {
    errors.assign(expressions.size(), std::string());
    valid.clear();
    // Validate every expression once and order the valid ones so that
    // expressions sharing children (promoted genes) are adjacent.
    gpe::Dag probe;
    std::vector<std::pair<std::pair<int, int>, int>> keyed;
    std::vector<std::string> names;
    for (size_t i = 0; i < expressions.size(); ++i) {
        names.clear();
        collect_fields(expressions[i], names);
        bool ok = true;
        for (const auto& n : names)
            if (!data_.has_field(n)) {
                errors[i] = "unknown terminal: " + n;
                ok = false;
                break;
            }
        if (!ok) continue;
        std::string error;
        int root = probe.add_expression(expressions[i], &error);
        if (root < 0) {
            errors[i] = error.empty() ? "invalid expression" : error;
            continue;
        }
        const auto& node = probe.nodes()[root];
        int a = node.op == gpe::Op::Leaf ? root : node.a;
        int b = node.op == gpe::Op::Leaf ? -1 : node.b;
        keyed.push_back({{a, b}, static_cast<int>(i)});
    }
    std::stable_sort(keyed.begin(), keyed.end(),
                     [](const auto& x, const auto& y) { return x.first < y.first; });
    for (const auto& k : keyed) valid.push_back(k.second);

    const long budget = budget_panels();
    std::vector<GroupPlan> groups;
    // Recursive halving until each group fits the budget.
    std::vector<std::pair<size_t, size_t>> todo{{0, valid.size()}};
    while (!todo.empty()) {
        auto [lo, hi] = todo.back();
        todo.pop_back();
        if (lo >= hi) continue;
        gpe::Dag dag;
        for (size_t k = lo; k < hi; ++k) dag.add_expression(expressions[valid[k]]);
        dag.finalize();
        long chunks = 0;
        long peak = simulate_peak(dag, budget, options_.min_chunk, options_.max_chunk, &chunks);
        if (peak < 0 && hi - lo > 1) {
            size_t mid = lo + (hi - lo) / 2;
            todo.push_back({mid, hi});  // processed after the first half (stack)
            todo.push_back({lo, mid});
            continue;
        }
        GroupPlan g;
        g.expressions.assign(valid.begin() + lo, valid.begin() + hi);
        g.dag_nodes = static_cast<long>(dag.nodes().size());
        g.peak_panels = peak;
        g.chunks = chunks;
        g.feasible = peak >= 0;
        groups.push_back(std::move(g));
    }
    return groups;
}

std::vector<GroupPlan> Engine::plan(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors) {
    std::vector<int> valid;
    return make_groups(expressions, errors, valid);
}

void Engine::run_group(const gpe::Dag& dag, std::vector<RootScore>& scores) {
    const auto& nodes = dag.nodes();
    const int T = data_.T, N = data_.N;
    const long budget = budget_panels();
    std::set<int> root_set(dag.roots().begin(), dag.roots().end());
    std::vector<char> is_root(nodes.size(), 0);
    for (int r : root_set) is_root[r] = 1;
    scores.assign(nodes.size(), RootScore());

    ScoreContext ctx;
    ctx.target_rank = &target_rank_;
    ctx.initial = &initial_;
    ctx.T = T;
    ctx.N = N;
    ctx.options = options_.score;

    std::vector<gpe::Panel> owned(nodes.size());
    std::vector<const gpe::Panel*> value(nodes.size(), nullptr);
    std::vector<int> pending(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) pending[i] = nodes[i].parents;
    long alive = 0;
    auto release = [&](int i) {
        if (nodes[i].op == gpe::Op::Leaf) return;
        pool_.give(std::move(owned[i].buf));
        owned[i] = gpe::Panel();
        value[i] = nullptr;
        --alive;
    };
    auto score = [&](int i) {
        RootScore& out = scores[i];
#ifdef GPP_HAVE_CUDA
        if (gpu_) {
            gpu_->submit(*value[i], nodes[i].depth, nodes[i].tree_nodes, &out);
            return;
        }
#endif
        score_root(*value[i], nodes[i].depth, nodes[i].tree_nodes, ctx, scratch_[thread_id()], out);
    };

    for (const auto& level : levels_of(dag)) {
        size_t c0 = 0;
        while (c0 < level.size()) {
            long remaining = static_cast<long>(level.size() - c0);
            const bool leaves = nodes[level[c0]].op == gpe::Op::Leaf;
            long step = leaves ? remaining : next_chunk(remaining, alive, budget, options_.min_chunk, options_.max_chunk);
            if (step < 0) throw std::runtime_error("memory budget too small for one chunk (planner bug)");
            const std::vector<int> chunk(level.begin() + c0, level.begin() + c0 + step);
            c0 += static_cast<size_t>(step);
            ++stats_.chunks;

            // Output buffers come from the pool before any task runs: the
            // pool is not thread-safe.
            std::vector<int> work, done;
            for (int i : chunk) {
                if (nodes[i].op == gpe::Op::Leaf) {
                    value[i] = &field(dag.fields()[nodes[i].field]);
                } else {
                    owned[i].buf = pool_.take(static_cast<size_t>(T) * N);
                    owned[i].reset(T, N, 0);
                    value[i] = &owned[i];
                    ++alive;
                    work.push_back(i);
                }
                if (is_root[i]) done.push_back(i);
            }
            stats_.peak_panels = std::max(stats_.peak_panels, alive);

            auto compute = [&](int i) {
                const gpe::Node& n = nodes[i];
                gpe::apply_op(n.op, n.window, *value[n.a], n.b >= 0 ? value[n.b] : nullptr, owned[i]);
            };
            auto t = Clock::now();
            if (static_cast<int>(work.size()) >= options_.narrow_level) {
                // Node-parallel: one task per node; a root is scored by the
                // same thread while its panel is still in that core's cache.
                std::vector<double> op_s(trace_ ? work.size() : 0), score_s(op_s.size());
                #pragma omp parallel for schedule(dynamic, 1)
                for (long k = 0; k < static_cast<long>(work.size()); ++k) {
                    auto t0 = Clock::now();
                    compute(work[k]);
                    auto t1 = Clock::now();
                    if (is_root[work[k]]) score(work[k]);
                    if (trace_) {
                        op_s[k] = std::chrono::duration<double>(t1 - t0).count();
                        score_s[k] = since(t1);
                    }
                }
                if (trace_) {
                    size_t worst = 0;
                    for (size_t k = 0; k < op_s.size(); ++k)
                        if (op_s[k] + score_s[k] > op_s[worst] + score_s[worst]) worst = k;
                    std::fprintf(stderr,
                                 "[gpp trace] chunk %ld: %zu tasks, %zu roots, wall %.2f s, op %.1f core-s, score %.1f "
                                 "core-s, slowest task %.2f s (%s window %d)\n",
                                 stats_.chunks, work.size(), done.size(), since(t),
                                 std::accumulate(op_s.begin(), op_s.end(), 0.0),
                                 std::accumulate(score_s.begin(), score_s.end(), 0.0),
                                 op_s.empty() ? 0.0 : op_s[worst] + score_s[worst],
                                 op_s.empty() ? "-" : gpe::op_info(nodes[work[worst]].op).name,
                                 op_s.empty() ? -1 : nodes[work[worst]].window);
                }
                std::vector<int> leaf_roots;
                for (int i : done)
                    if (nodes[i].op == gpe::Op::Leaf) leaf_roots.push_back(i);
                #pragma omp parallel for schedule(dynamic, 1)
                for (long k = 0; k < static_cast<long>(leaf_roots.size()); ++k) score(leaf_roots[k]);
            } else {
                // Narrow level: operators are parallel inside (their own
                // omp loops are active outside a parallel region), then the
                // roots are scored in parallel.
                for (int i : work) compute(i);
                #pragma omp parallel for schedule(dynamic, 1)
                for (long k = 0; k < static_cast<long>(done.size()); ++k) score(done[k]);
                if (trace_)
                    std::fprintf(stderr, "[gpp trace] chunk %ld: narrow, %zu tasks, %zu roots, wall %.2f s\n",
                                 stats_.chunks, work.size(), done.size(), since(t));
            }
            stats_.compute += since(t);

            for (int i : done)
                if (pending[i] == 0) release(i);
            for (int i : chunk)
                for (int c : {nodes[i].a, nodes[i].b})
                    if (c >= 0 && --pending[c] == 0) release(c);
        }
    }
#ifdef GPP_HAVE_CUDA
    if (gpu_) {
        auto t = Clock::now();
        gpu_->flush();
        stats_.score_wait += since(t);
    }
#endif
    for (size_t i = 0; i < nodes.size(); ++i)
        if (value[i] && nodes[i].op != gpe::Op::Leaf) release(static_cast<int>(i));
}

std::vector<RootScore> Engine::evaluate(const std::vector<gpe::Json>& expressions, std::vector<std::string>& errors) {
    auto t_start = Clock::now();
    ++stats_.batches;
    stats_.expressions += static_cast<long>(expressions.size());
    std::vector<RootScore> result(expressions.size());

    // Load every field the batch uses first so that the budget is exact.
    {
        std::vector<std::string> names;
        for (const auto& e : expressions) collect_fields(e, names);
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        for (const auto& n : names)
            if (data_.has_field(n)) field(n);
    }
    auto t = Clock::now();
    std::vector<int> valid;
    auto groups = make_groups(expressions, errors, valid);
    stats_.plan += since(t);

    std::vector<RootScore> by_root;
    for (const auto& g : groups) {
        if (!g.feasible) throw std::runtime_error("memory budget too small for a single expression");
        gpe::Dag dag;
        std::vector<int> root_of(g.expressions.size());
        for (size_t k = 0; k < g.expressions.size(); ++k) root_of[k] = dag.add_expression(expressions[g.expressions[k]]);
        dag.finalize();
        ensure_fields(dag);
        ++stats_.groups;
        stats_.dag_nodes += static_cast<long>(dag.nodes().size());
        stats_.roots += static_cast<long>(std::set<int>(dag.roots().begin(), dag.roots().end()).size());
        run_group(dag, by_root);
        for (size_t k = 0; k < g.expressions.size(); ++k) result[g.expressions[k]] = by_root[root_of[k]];
    }
    stats_.peak_bytes = std::max(stats_.peak_bytes,
                                 stats_.resident_bytes + stats_.scratch_bytes + panel_bytes() * stats_.peak_panels);
    stats_.total += since(t_start);
    return result;
}

}  // namespace gpp
