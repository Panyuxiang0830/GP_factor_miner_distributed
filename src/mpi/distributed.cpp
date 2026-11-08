#include "gpe/distributed.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "gpe/ema_scan.hpp"
#include "gpe/halo_plan.hpp"
#include "gpe/operators.hpp"

namespace gpe {

DateBlock date_block(int T, int ranks, int rank) {
    int base = T / ranks, extra = T % ranks;
    int t0 = rank * base + std::min(rank, extra);
    return {t0, t0 + base + (rank < extra ? 1 : 0)};
}

namespace {

// Halo exchange of the exchanged nodes of one level. Rank s needs, for node
// i, the global days [t0_s - need_i, t0_s); they may span several ranks to the
// left when blocks are shorter than the halo. Each rank packs, per partner,
// the rows of its own block that the partner needs (all nodes of the level in
// one message per partner) and exchanges them with non-blocking sends.
void exchange_halos(const std::vector<int>& ids, const std::vector<HaloPlan>& plan, std::vector<Panel>& values,
                    const DateBlock& me, int rank, int P, int T, int tag, MPI_Comm comm, MpiTiming& timing) {
    const int N = values[ids.front()].N;
    int max_need = 0;
    for (int i : ids) max_need = std::max(max_need, plan[i].need);
    // Global days of q's block that rank s needs for a node with history h.
    auto overlap = [&](int q, int s, int h) {
        DateBlock bq = date_block(T, P, q), bs = date_block(T, P, s);
        return std::make_pair(std::max(bq.t0, bs.t0 - h), std::min(bq.t1, bs.t0));
    };

    std::vector<std::vector<float>> send_buf, recv_buf;
    std::vector<int> recv_from;
    std::vector<MPI_Request> requests;
    for (int q = rank - 1; q >= 0 && date_block(T, P, q).t1 > me.t0 - max_need; --q) {
        size_t rows = 0;
        for (int i : ids) {
            auto [lo, hi] = overlap(q, rank, plan[i].need);
            rows += std::max(hi - lo, 0);
        }
        if (rows == 0) continue;
        recv_buf.emplace_back(rows * N);
        recv_from.push_back(q);
    }
    for (int s = rank + 1; s < P && date_block(T, P, s).t0 - max_need < me.t1; ++s) {
        std::vector<float> buf;
        for (int i : ids) {
            auto [lo, hi] = overlap(rank, s, plan[i].need);
            if (hi <= lo) continue;
            const float* src = values[i].row(lo - me.t0);
            buf.insert(buf.end(), src, src + static_cast<size_t>(hi - lo) * N);
        }
        if (buf.empty()) continue;
        send_buf.push_back(std::move(buf));
        requests.emplace_back();
        timing.halo_bytes += sizeof(float) * send_buf.back().size();
        ++timing.halo_messages;
        MPI_Isend(send_buf.back().data(), static_cast<int>(send_buf.back().size()), MPI_FLOAT, s, tag, comm,
                  &requests.back());
    }
    double t = MPI_Wtime();
    for (size_t k = 0; k < recv_buf.size(); ++k) {
        requests.emplace_back();
        MPI_Irecv(recv_buf[k].data(), static_cast<int>(recv_buf[k].size()), MPI_FLOAT, recv_from[k], tag, comm,
                  &requests.back());
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    timing.halo += MPI_Wtime() - t;

    for (size_t k = 0; k < recv_buf.size(); ++k) {
        size_t off = 0;
        for (int i : ids) {
            auto [lo, hi] = overlap(recv_from[k], rank, plan[i].need);
            if (hi <= lo) continue;
            std::memcpy(values[i].row(lo - me.t0), recv_buf[k].data() + off, sizeof(float) * (hi - lo) * N);
            off += static_cast<size_t>(hi - lo) * N;
        }
    }
}

// Ema state chain for all ema nodes of one level: rank r receives the state
// at day t0-1 from rank r-1, applies its block summaries and forwards the
// state at day t1-1. Returns the incoming state, 2*N doubles per node (y then
// old_wt).
std::vector<double> ema_chain(const std::vector<int>& ids, const std::vector<Node>& nodes,
                              const std::vector<Panel>& values, int left, int right, int tag, MPI_Comm comm,
                              MpiTiming& timing) {
    const int N = values[nodes[ids.front()].a].N;
    const size_t E = ids.size();
    double t = MPI_Wtime();
    std::vector<std::vector<EmaBlockSummary>> summaries(E);
    for (size_t e = 0; e < E; ++e) {
        const Node& n = nodes[ids[e]];
        summaries[e] = ema_summarize_panel(values[n.a], n.window);
    }
    timing.compute += MPI_Wtime() - t;

    t = MPI_Wtime();
    std::vector<double> in(2 * N * E);
    if (left == MPI_PROC_NULL) {
        for (size_t e = 0; e < E; ++e)
            for (int s = 0; s < N; ++s) {
                in[2 * N * e + s] = NAN;
                in[2 * N * e + N + s] = 1.0;
            }
    } else {
        MPI_Recv(in.data(), static_cast<int>(in.size()), MPI_DOUBLE, left, tag, comm, MPI_STATUS_IGNORE);
    }
    if (right != MPI_PROC_NULL) {
        std::vector<double> out(in.size());
        for (size_t e = 0; e < E; ++e) {
            int span = nodes[ids[e]].window;
            for (int s = 0; s < N; ++s) {
                EmaCarry c = ema_apply(summaries[e][s], {in[2 * N * e + s], in[2 * N * e + N + s]}, span);
                out[2 * N * e + s] = c.y;
                out[2 * N * e + N + s] = c.old_wt;
            }
        }
        MPI_Send(out.data(), static_cast<int>(out.size()), MPI_DOUBLE, right, tag, comm);
    }
    timing.scan += MPI_Wtime() - t;
    return in;
}

}  // namespace

MpiBatchResult evaluate_batch_mpi(const Dag& dag, const PanelSet& data, MPI_Comm comm, const MpiOptions& options) {
    const FitnessConfig& config = options.fitness;
    const bool keep_roots = options.keep_roots;
    const bool by_node = options.threads == ThreadMode::Node;
    double t_start = MPI_Wtime();
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    MpiBatchResult result;
    MpiTiming& timing = result.timing;
    const DateBlock block = date_block(data.T, P, rank);
    result.block = block;
    const int Tl = block.size(), N = data.N;
    const auto& nodes = dag.nodes();

    const std::vector<HaloPlan> plan = plan_halos(dag, options.H0);
    const int left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int right = rank < P - 1 ? rank + 1 : MPI_PROC_NULL;
    // History rows this rank holds for node i: none before the first day of
    // the data, so rank 0 holds none and redundant rows stop at day 0.
    auto rows_before = [&](int i) { return std::min(plan[i].need, block.t0); };

    std::vector<std::vector<int>> levels(dag.max_level() + 1);
    for (size_t i = 0; i < nodes.size(); ++i) levels[nodes[i].level].push_back(static_cast<int>(i));
    std::set<int> roots(dag.roots().begin(), dag.roots().end());
    std::vector<int> unique_roots(roots.begin(), roots.end());
    std::vector<int> slot(nodes.size(), -1);  // index of a root in unique_roots
    for (size_t r = 0; r < unique_roots.size(); ++r) slot[unique_roots[r]] = static_cast<int>(r);
    std::vector<Panel> values(nodes.size());
    BufferPool local_pool;
    BufferPool& pool = options.pool ? *options.pool : local_pool;
    std::vector<int> pending(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) pending[i] = nodes[i].parents;
    auto release = [&](int i) {
        pool.give(std::move(values[i].buf));
        values[i] = Panel();
    };

    // Target ranks for the RankIC of each root as soon as it is computed.
    double t = MPI_Wtime();
    Panel target = data.load_target_rows(block.t0, block.t1);
    timing.load += MPI_Wtime() - t;
    t = MPI_Wtime();
    std::vector<double> target_rank = rank_rows_pct(target);
    target = Panel();
    const int K = config.stability_segments, W = IcSummary::width(K);
    std::vector<double> stats(unique_roots.size() * W);
    timing.metrics += MPI_Wtime() - t;

    for (size_t L = 0; L < levels.size(); ++L) {
        // Separate tags for the ema chain and the halo of each level.
        const int ema_tag = 2 * static_cast<int>(L), halo_tag = ema_tag + 1;

        std::vector<int> ema_ids;
        for (int i : levels[L])
            if (nodes[i].op == Op::Ema) ema_ids.push_back(i);
        std::vector<double> ema_in;
        if (!ema_ids.empty()) ema_in = ema_chain(ema_ids, nodes, values, left, right, ema_tag, comm, timing);

        // Wide levels are processed in chunks of at most options.chunk nodes:
        // each chunk is computed, exchanged, scored and released before the
        // next one takes memory, which bounds the panels alive at once.
        const size_t width = levels[L].size();
        const size_t step = options.chunk > 0 ? static_cast<size_t>(options.chunk) : std::max<size_t>(width, 1);
        for (size_t c0 = 0; c0 < width; c0 += step) {
            const std::vector<int> chunk(levels[L].begin() + c0, levels[L].begin() + std::min(width, c0 + step));
            // Leaves are read and output panels taken from the pool before any
            // operator runs, so that the pool is never used by several threads.
            std::vector<int> work, exchanged;
            for (int i : chunk) {
                const Node& n = nodes[i];
                const int h = rows_before(i);
                if (n.op == Op::Leaf) {
                    t = MPI_Wtime();
                    values[i] = data.load_field_rows(dag.fields()[n.field], block.t0, block.t1, h);
                    timing.load += MPI_Wtime() - t;
                    continue;
                }
                values[i].buf = pool.take(static_cast<size_t>(Tl + h) * N);
                values[i].reset(Tl, N, -h);
                if (plan[i].redundant) timing.redundant_rows += h;
                if (plan[i].exchanged()) exchanged.push_back(i);
                work.push_back(i);
            }
            auto compute = [&](int i) {
                const Node& n = nodes[i];
                const Panel* b = n.b >= 0 ? &values[n.b] : nullptr;
                // Redundant nodes also compute their history rows; exchanged ones
                // compute their own days and receive the rest.
                const int begin = plan[i].redundant ? -rows_before(i) : 0;
                EmaState state;
                const EmaState* ema = nullptr;
                if (n.op == Op::Ema && left != MPI_PROC_NULL) {
                    size_t e = std::find(ema_ids.begin(), ema_ids.end(), i) - ema_ids.begin();
                    state.y = ema_in.data() + 2 * N * e;
                    state.old_wt = state.y + N;
                    ema = &state;
                }
                apply_op(n.op, n.window, values[n.a], b, values[i], begin, ema);
            };
            t = MPI_Wtime();
            if (by_node) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < work.size(); ++k) compute(work[k]);
            } else {
                for (int i : work) compute(i);
            }
            timing.compute += MPI_Wtime() - t;
            if (!exchanged.empty()) exchange_halos(exchanged, plan, values, block, rank, P, data.T, halo_tag, comm, timing);

            // RankIC of the roots of this level (own days only).
            std::vector<int> done;
            for (int i : chunk)
                if (slot[i] >= 0) done.push_back(i);
            std::vector<DailyIC> daily(done.size());
            auto score = [&](size_t k) {
                daily[k] = daily_rank_ic(values[done[k]], target_rank);
                summarize_ic(daily[k], block.t0, data.T, K).pack(stats.data() + slot[done[k]] * W);
            };
            t = MPI_Wtime();
            if (by_node) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < done.size(); ++k) score(k);
            } else {
                for (size_t k = 0; k < done.size(); ++k) score(k);
            }
            timing.metrics += MPI_Wtime() - t;

            if (keep_roots) {
                for (size_t k = 0; k < done.size(); ++k) result.daily[done[k]] = std::move(daily[k]);
            } else {
                for (int i : done)
                    if (pending[i] == 0) release(i);  // not a subtree of a later node
            }
            for (int i : chunk)
                for (int c : {nodes[i].a, nodes[i].b})
                    if (c >= 0 && --pending[c] == 0 && !(keep_roots && slot[c] >= 0)) release(c);

        }
    }

    // Separates waiting for the slowest rank from the reduction itself.
    t = MPI_Wtime();
    MPI_Barrier(comm);
    timing.wait += MPI_Wtime() - t;
    t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, stats.data(), static_cast<int>(stats.size()), MPI_DOUBLE, MPI_SUM, comm);
    timing.reduce += MPI_Wtime() - t;

    for (int root : dag.roots()) {
        const Node& n = nodes[root];
        result.candidates.push_back(
            score_summary(IcSummary::unpack(stats.data() + slot[root] * W, K), data.T, n.depth, n.tree_nodes, config));
    }
    if (keep_roots)
        for (int r : unique_roots) result.roots[r] = std::move(values[r]);
    for (Panel& v : values) pool.give(std::move(v.buf));
    timing.total = MPI_Wtime() - t_start;
    return result;
}

}  // namespace gpe
