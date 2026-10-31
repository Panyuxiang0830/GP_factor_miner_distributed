#include "gpe/distributed.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

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

// Packs the last `need` rows of each node into one buffer, sends it to the
// right neighbour and receives the left neighbour's rows into rows -need..-1.
void exchange_halos(const std::vector<int>& ids, const std::vector<HaloPlan>& plan, std::vector<Panel>& values,
                    int left, int right, int tag, MPI_Comm comm, MpiTiming& timing) {
    const int N = values[ids.front()].N;
    size_t send_rows = 0, recv_rows = 0;
    for (int i : ids) {
        if (right != MPI_PROC_NULL) send_rows += plan[i].need;
        if (left != MPI_PROC_NULL) recv_rows += plan[i].need;
    }
    std::vector<float> send(send_rows * N), recv(recv_rows * N);
    size_t off = 0;
    if (right != MPI_PROC_NULL) {
        for (int i : ids) {
            const Panel& p = values[i];
            int h = plan[i].need;
            std::memcpy(send.data() + off, p.row(p.T - h), sizeof(float) * h * N);
            off += static_cast<size_t>(h) * N;
        }
    }
    double t = MPI_Wtime();
    MPI_Sendrecv(send.data(), static_cast<int>(send.size()), MPI_FLOAT, right, tag, recv.data(),
                 static_cast<int>(recv.size()), MPI_FLOAT, left, tag, comm, MPI_STATUS_IGNORE);
    timing.halo += MPI_Wtime() - t;
    if (!send.empty()) {
        timing.halo_bytes += sizeof(float) * send.size();
        ++timing.halo_messages;
    }
    off = 0;
    if (left != MPI_PROC_NULL) {
        for (int i : ids) {
            Panel& p = values[i];
            int h = plan[i].need;
            std::memcpy(p.row(-h), recv.data() + off, sizeof(float) * h * N);
            off += static_cast<size_t>(h) * N;
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
    int max_exchange = 0;
    for (const HaloPlan& p : plan)
        if (p.exchanged()) max_exchange = std::max(max_exchange, p.need);
    int min_block = date_block(data.T, P, P - 1).size();
    if (P > 1 && min_block < max_exchange)
        throw std::runtime_error("date blocks of " + std::to_string(min_block) + " days are shorter than the halo of " +
                                 std::to_string(max_exchange) + " days; use fewer ranks or a larger H0");

    const int left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int right = rank < P - 1 ? rank + 1 : MPI_PROC_NULL;
    // History rows this rank holds for node i: none before the first day of
    // the data, so rank 0 holds none and redundant rows stop at day 0.
    auto rows_before = [&](int i) { return std::min(plan[i].need, block.t0); };

    std::vector<std::vector<int>> levels(dag.max_level() + 1);
    for (size_t i = 0; i < nodes.size(); ++i) levels[nodes[i].level].push_back(static_cast<int>(i));
    std::set<int> roots(dag.roots().begin(), dag.roots().end());
    std::vector<Panel> values(nodes.size());
    std::vector<int> pending(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) pending[i] = nodes[i].parents;

    for (size_t L = 0; L < levels.size(); ++L) {
        // Separate tags for the ema chain and the halo of each level.
        const int ema_tag = 2 * static_cast<int>(L), halo_tag = ema_tag + 1;

        std::vector<int> ema_ids;
        for (int i : levels[L])
            if (nodes[i].op == Op::Ema) ema_ids.push_back(i);
        std::vector<double> ema_in;
        if (!ema_ids.empty()) ema_in = ema_chain(ema_ids, nodes, values, left, right, ema_tag, comm, timing);

        std::vector<int> exchanged;
        for (int i : levels[L]) {
            const Node& n = nodes[i];
            if (n.op == Op::Leaf) {
                double t = MPI_Wtime();
                values[i] = data.load_field_rows(dag.fields()[n.field], block.t0, block.t1, rows_before(i));
                timing.load += MPI_Wtime() - t;
                continue;
            }
            double t = MPI_Wtime();
            const Panel& a = values[n.a];
            const Panel* b = n.b >= 0 ? &values[n.b] : nullptr;
            const int h = rows_before(i);
            values[i].reset(Tl, N, -h);
            // Redundant nodes also compute their history rows; exchanged ones
            // compute their own days and receive the rest.
            const int begin = plan[i].redundant ? -h : 0;
            if (plan[i].redundant) timing.redundant_rows += h;
            EmaState state;
            const EmaState* ema = nullptr;
            if (n.op == Op::Ema && left != MPI_PROC_NULL) {
                size_t e = std::find(ema_ids.begin(), ema_ids.end(), i) - ema_ids.begin();
                state.y = ema_in.data() + 2 * N * e;
                state.old_wt = state.y + N;
                ema = &state;
            }
            apply_op(n.op, n.window, a, b, values[i], begin, ema);
            timing.compute += MPI_Wtime() - t;
            if (plan[i].exchanged()) exchanged.push_back(i);
        }
        if (!exchanged.empty()) exchange_halos(exchanged, plan, values, left, right, halo_tag, comm, timing);

        for (int i : levels[L]) {
            for (int c : {nodes[i].a, nodes[i].b}) {
                if (c >= 0 && --pending[c] == 0 && !roots.count(c)) values[c] = Panel();
            }
        }
    }

    // Daily RankIC on own days, then one reduction for the whole batch.
    double t = MPI_Wtime();
    Panel target = data.load_target_rows(block.t0, block.t1);
    timing.load += MPI_Wtime() - t;
    t = MPI_Wtime();
    std::vector<double> target_rank = rank_rows_pct(target);
    const int K = config.stability_segments, W = IcSummary::width(K);
    std::vector<int> unique_roots(roots.begin(), roots.end());
    std::vector<double> stats(unique_roots.size() * W);
    for (size_t r = 0; r < unique_roots.size(); ++r) {
        DailyIC daily = daily_rank_ic(values[unique_roots[r]], target_rank);
        summarize_ic(daily, block.t0, data.T, K).pack(stats.data() + r * W);
        if (keep_roots) result.daily[unique_roots[r]] = std::move(daily);
    }
    timing.metrics += MPI_Wtime() - t;

    // Separates waiting for the slowest rank from the reduction itself.
    t = MPI_Wtime();
    MPI_Barrier(comm);
    timing.wait += MPI_Wtime() - t;
    t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, stats.data(), static_cast<int>(stats.size()), MPI_DOUBLE, MPI_SUM, comm);
    timing.reduce += MPI_Wtime() - t;

    for (int root : dag.roots()) {
        size_t r = std::lower_bound(unique_roots.begin(), unique_roots.end(), root) - unique_roots.begin();
        const Node& n = nodes[root];
        result.candidates.push_back(
            score_summary(IcSummary::unpack(stats.data() + r * W, K), data.T, n.depth, n.tree_nodes, config));
    }
    if (keep_roots)
        for (int r : unique_roots) result.roots[r] = std::move(values[r]);
    timing.total = MPI_Wtime() - t_start;
    return result;
}

}  // namespace gpe
