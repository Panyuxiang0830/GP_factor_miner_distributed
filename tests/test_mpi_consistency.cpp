// The distributed engine (strategy B) against the serial engine on exported
// data. Run with any number of ranks:
//
//   mpirun -np P test_mpi_consistency <panel_dir> <expressions.json> [H0] [max_expressions]
//
// <expressions.json> is a reference.json (cases[].input), a list of
// expressions, or a list of objects with an "expression" key. H0 is the
// redundant-computation budget (0 = strategy B, default). Rank 0 also runs
// the serial engine and compares, for every root:
//   - the panel, gathered from all ranks: bit-identical count, and the
//     allclose criterion of test_ops_reference, |q-p| <= 1e-4|p| + 1e-5 rms(p)
//     (blocks restart Kahan sums and the ema recursion at their halo, so the
//     last bits can differ);
//   - the candidate statistics and fitness;
// and checks that every rank obtained the same candidates.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/distributed.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/halo_plan.hpp"
#include "gpe/json.hpp"
#include "gpe/metrics.hpp"
#include "gpe/ops.hpp"

namespace {

std::vector<gpe::Json> read_expressions(const std::string& path, size_t max_count) {
    gpe::Json j = gpe::parse_json_file(path);
    std::vector<gpe::Json> out;
    if (j.is_object()) {
        for (const auto& c : j.at("cases").array)
            if (!c.has("error")) out.push_back(c.at("input"));
    } else {
        for (const auto& e : j.array) out.push_back(e.is_object() ? e.at("expression") : e);
    }
    if (out.size() > max_count) out.resize(max_count);
    return out;
}

constexpr double kRtol = 1e-4, kAtol = 1e-5;

template <class Floats>
double rms(const Floats& v) {
    double sum = 0.0;
    long n = 0;
    for (float x : v)
        if (std::isfinite(x)) { sum += double(x) * x; ++n; }
    return n ? std::sqrt(sum / n) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc < 3) {
        if (rank == 0) std::printf("usage: %s <panel_dir> <expressions.json> [H0] [max_expressions]\n", argv[0]);
        MPI_Finalize();
        return 2;
    }
    gpe::PanelSet data = gpe::PanelSet::open(argv[1]);
    gpe::MpiOptions options;
    options.H0 = argc > 3 ? std::stoi(argv[3]) : 0;
    options.keep_roots = true;
    size_t max_count = argc > 4 ? std::stoul(argv[4]) : 1000000;

    gpe::Dag dag;
    for (const auto& e : read_expressions(argv[2], max_count)) dag.add_expression(e);
    dag.finalize();

    MPI_Barrier(MPI_COMM_WORLD);
    gpe::MpiBatchResult dist = gpe::evaluate_batch_mpi(dag, data, MPI_COMM_WORLD, options);

    // Every rank must hold the same candidates (fitness compared bit for bit).
    const size_t R = dist.candidates.size();
    std::vector<double> fit(R), all_fit(R * P);
    for (size_t i = 0; i < R; ++i) fit[i] = dist.candidates[i].valid ? dist.candidates[i].fitness : -1e300;
    MPI_Gather(fit.data(), static_cast<int>(R), MPI_DOUBLE, all_fit.data(), static_cast<int>(R), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    // Timing: maximum over ranks of each phase.
    const gpe::MpiTiming& tm = dist.timing;
    double mine[10] = {tm.load,  tm.compute, tm.halo,       tm.scan, tm.metrics,
                       tm.reduce, tm.total,  tm.halo_bytes, tm.wait, static_cast<double>(tm.redundant_rows)};
    double worst[10];
    MPI_Reduce(mine, worst, 10, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::map<int, gpe::Panel> serial;
    std::vector<gpe::Candidate> serial_cands;
    double serial_time = 0.0;
    std::vector<double> target_rank;
    if (rank == 0) {
        double t = MPI_Wtime();
        serial = gpe::evaluate_serial(dag, data);
        target_rank = gpe::rank_rows_pct(data.load_target());
        for (int root : dag.roots()) {
            const gpe::Node& n = dag.nodes()[root];
            serial_cands.push_back(
                gpe::score_factor(gpe::daily_rank_ic(serial.at(root), target_rank), n.depth, n.tree_nodes));
        }
        serial_time = MPI_Wtime() - t;
    }

    // Gather each root panel (own days of every rank) on rank 0 and compare.
    std::vector<int> counts(P), displs(P);
    for (int r = 0; r < P; ++r) {
        gpe::DateBlock b = gpe::date_block(data.T, P, r);
        counts[r] = b.size() * data.N;
        displs[r] = b.t0 * data.N;
    }
    int failures = 0;
    long cells = 0, identical = 0, nan_mismatch = 0;
    double max_err = 0.0;  // max |q-p| / (rtol|p| + atol rms); <= 1 passes
    std::vector<float> full(rank == 0 ? static_cast<size_t>(data.T) * data.N : 0);
    for (const auto& [root, local] : dist.roots) {
        MPI_Gatherv(local.row(0), counts[rank], MPI_FLOAT, full.data(), counts.data(), displs.data(), MPI_FLOAT, 0,
                    MPI_COMM_WORLD);
        if (rank != 0) continue;
        const gpe::Panel& ref = serial.at(root);
        long root_nan = 0;
        double root_err = 0.0, scale = kAtol * rms(ref.buf);
        int first_bad = 0;
        bool seen_bad = false;
        for (size_t k = 0; k < full.size(); ++k) {
            float p = ref.buf[k], q = full[k];
            ++cells;
            if (std::memcmp(&p, &q, sizeof(float)) == 0 || (std::isnan(p) && std::isnan(q))) {
                ++identical;
                continue;
            }
            if (std::isnan(p) != std::isnan(q) || std::isinf(p) != std::isinf(q)) {
                ++root_nan;
            } else {
                double bound = kRtol * std::fabs(p) + scale;
                double e = bound > 0 ? std::fabs(double(q) - p) / bound : INFINITY;
                root_err = std::max(root_err, e);
                if (e <= 1.0) continue;
            }
            if (!seen_bad) { first_bad = static_cast<int>(k); seen_bad = true; }
        }
        max_err = std::max(max_err, root_err);
        nan_mismatch += root_nan;
        if (root_nan || root_err > 1.0) {
            const gpe::Node& n = dag.nodes()[root];
            std::printf("  root %d (%s, w=%d, level %d): err %.2e, nan/inf %ld, first at day %d stock %d: "
                        "serial %g vs distributed %g\n",
                        root, gpe::op_info(n.op).name, n.window, n.level, root_err, root_nan, first_bad / data.N,
                        first_bad % data.N, ref.buf[first_bad], full[first_bad]);
        }
    }

    if (rank == 0) {
        bool ranks_agree = true;
        for (int r = 1; r < P; ++r)
            if (std::memcmp(all_fit.data(), all_fit.data() + r * R, R * sizeof(double)) != 0) ranks_agree = false;

        int valid_mismatch = 0, accepted = 0;
        double fit_err = 0.0;
        for (size_t i = 0; i < R; ++i) {
            const auto& a = serial_cands[i];
            const auto& b = dist.candidates[i];
            if (a.valid != b.valid) { ++valid_mismatch; continue; }
            if (!a.valid) continue;
            ++accepted;
            fit_err = std::max({fit_err, std::fabs(a.fitness - b.fitness), std::fabs(a.rank_ic - b.rank_ic),
                                std::fabs(a.missing_ratio - b.missing_ratio)});
        }
        bool ok = ranks_agree && valid_mismatch == 0 && fit_err <= 1e-9 && nan_mismatch == 0 && max_err <= 1.0;
        if (!ok) failures = 1;

        int max_halo = 0;
        for (const auto& n : dag.nodes()) max_halo = std::max(max_halo, n.halo_rows);
        auto plan = gpe::plan_halos(dag, options.H0);
        int exchanged = 0, redundant = 0;
        for (size_t i = 0; i < plan.size(); ++i) {
            if (plan[i].exchanged()) ++exchanged;
            else if (plan[i].need > 0 && dag.nodes()[i].op != gpe::Op::Leaf) ++redundant;
        }
        std::printf("P=%d H0=%d  T=%d N=%d  expressions=%zu  DAG nodes=%zu levels=%d max halo=%d  "
                    "exchanged nodes=%d redundant nodes=%d\n",
                    P, options.H0, data.T, data.N, R, dag.nodes().size(), dag.max_level() + 1, max_halo, exchanged,
                    redundant);
        std::printf("panels:     %ld cells, %.4f%% bit-identical, max allclose err %.2e (<= 1 passes), "
                    "nan/inf mismatch %ld\n",
                    cells, 100.0 * identical / std::max(cells, 1L), max_err, nan_mismatch);
        std::printf("candidates: %d accepted, validity mismatch %d, max |diff| %.2e, ranks agree %s\n", accepted,
                    valid_mismatch, fit_err, ranks_agree ? "yes" : "NO");
        std::printf("time (max over ranks, s): total %.3f  load %.3f  compute %.3f  halo %.3f  scan %.3f  metrics %.3f"
                    "  wait %.3f  reduce %.4f;  serial %.3f\n",
                    worst[6], worst[0], worst[1], worst[2], worst[3], worst[4], worst[8], worst[5], serial_time);
        std::printf("halo bytes sent (max over ranks): %.1f MB;  redundant rows computed (max over ranks): %.0f\n",
                    worst[7] / 1e6, worst[9]);
        std::printf("%s\n", ok ? "CONSISTENT" : "INCONSISTENT");
    }
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures;
}
