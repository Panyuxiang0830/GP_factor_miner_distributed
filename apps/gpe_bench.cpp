// Timing driver for the distributed engine (no serial comparison).
//
//   mpirun -np P gpe_bench <panel_dir> <expressions.json> [options]
//     --H0 h           redundant-computation budget (default 0)
//     --dates T        use only the first T dates
//     --stocks N       use only the first N assets
//     --synthetic T    synthetic data with T dates (and --stocks N assets, default
//                      5876) instead of <panel_dir>, which is then ignored
//     --seed s         seed of the synthetic data (default 1)
//     --threads m      op (each operator splits days or stocks over the threads,
//                      default) or node (the nodes of a level run in parallel)
//     --exprs n        use only the first n expressions
//     --warmup w       untimed runs first (default 1)
//     --repeat r       timed repetitions (default 3)
//     --csv file       append one result row to this CSV file
//     --label text     free-text label stored in the CSV row
//     --profile file   append per-operator times (mean per repetition, max over
//                      ranks) to this CSV file and print them
//
// Each phase is the maximum over ranks within a repetition; the reported value
// is the median over repetitions.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/distributed.hpp"
#include "gpe/halo_plan.hpp"
#include "gpe/json.hpp"
#include "gpe/operators.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc < 3) {
        if (rank == 0) std::printf("usage: %s <panel_dir> <expressions.json> [options, see the source]\n", argv[0]);
        MPI_Finalize();
        return 2;
    }
    gpe::MpiOptions options;
    gpe::BufferPool pool;  // kept across repetitions, as across GP generations
    options.pool = &pool;
    int dates = 0, stocks = 0, synthetic = 0, warmup = 1, repeat = 3;
    unsigned long long seed = 1;
    size_t exprs = 1000000;
    std::string csv, label, profile, mode = "op";
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--H0") options.H0 = std::stoi(v);
        else if (k == "--dates") dates = std::stoi(v);
        else if (k == "--stocks") stocks = std::stoi(v);
        else if (k == "--synthetic") synthetic = std::stoi(v);
        else if (k == "--seed") seed = std::stoull(v);
        else if (k == "--threads") mode = v;
        else if (k == "--warmup") warmup = std::stoi(v);
        else if (k == "--exprs") exprs = std::stoul(v);
        else if (k == "--repeat") repeat = std::stoi(v);
        else if (k == "--csv") csv = v;
        else if (k == "--label") label = v;
        else if (k == "--profile") profile = v;
    }

    if (mode != "op" && mode != "node") {
        if (rank == 0) std::printf("--threads must be op or node\n");
        MPI_Finalize();
        return 2;
    }
    options.threads = mode == "node" ? gpe::ThreadMode::Node : gpe::ThreadMode::Operator;
#ifdef _OPENMP
    omp_set_max_active_levels(1);  // operators inside a parallel node loop run single-threaded
#endif

    gpe::PanelSet data = synthetic > 0 ? gpe::PanelSet::make_synthetic(synthetic, stocks > 0 ? stocks : 5876, seed)
                                       : gpe::PanelSet::open(argv[1]);
    if (dates > 0 && dates < data.T) {  // panels are date-major: the first T dates are a prefix
        data.T = dates;
        if (!data.dates.empty()) data.dates.resize(dates);
    }
    if (stocks > 0 && stocks < data.N) {
        data.N = stocks;
        if (!data.assets.empty()) data.assets.resize(stocks);
    }

    // Number of distinct hosts the ranks run on.
    MPI_Comm host_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &host_comm);
    int host_rank = 0, hosts = 0;
    MPI_Comm_rank(host_comm, &host_rank);
    int leader = host_rank == 0 ? 1 : 0;
    MPI_Allreduce(&leader, &hosts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_free(&host_comm);
    gpe::Dag dag;
    for (const auto& e : read_expressions(argv[2], exprs)) dag.add_expression(e);
    dag.finalize();

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    // Columns: phases (max over ranks), then counters.
    const char* names[] = {"total", "load", "compute", "halo", "scan", "metrics", "wait", "reduce"};
    const int K = 8;
    std::vector<std::vector<double>> samples(K);
    double halo_mb = 0.0, redundant = 0.0, halo_msgs = 0.0;
    for (int r = 1 - warmup; r <= repeat; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == 1) gpe::reset_op_stats();  // count the timed repetitions only (the warm-up fills the pool)
        gpe::MpiBatchResult res = gpe::evaluate_batch_mpi(dag, data, MPI_COMM_WORLD, options);
        const gpe::MpiTiming& t = res.timing;
        double mine[K + 3] = {t.total,  t.load,       t.compute, t.halo, t.scan, t.metrics, t.wait,
                              t.reduce, t.halo_bytes, static_cast<double>(t.redundant_rows),
                              static_cast<double>(t.halo_messages)};
        double worst[K + 3];
        MPI_Reduce(mine, worst, K + 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && r > 0) {  // r <= 0 are warm-ups (page cache, allocator, buffer pool)
            for (int k = 0; k < K; ++k) samples[k].push_back(worst[k]);
            halo_mb = worst[K] / 1e6;
            redundant = worst[K + 1];
            halo_msgs = worst[K + 2];
        }
    }

    if (rank == 0) {
        auto plan = gpe::plan_halos(dag, options.H0);
        int exchanged = 0;
        for (const auto& p : plan) exchanged += p.exchanged();
        std::vector<double> med(K);
        for (int k = 0; k < K; ++k) med[k] = median(samples[k]);
        double tmin = *std::min_element(samples[0].begin(), samples[0].end());
        double tmax = *std::max_element(samples[0].begin(), samples[0].end());
        std::printf("P=%d hosts=%d threads=%d(%s) H0=%d T=%d N=%d exprs=%zu nodes=%zu exchanged=%d |", P, hosts,
                    threads, mode.c_str(), options.H0, data.T, data.N, dag.roots().size(), dag.nodes().size(),
                    exchanged);
        for (int k = 0; k < K; ++k) std::printf(" %s %.3f", names[k], med[k]);
        std::printf(" | total min %.3f max %.3f | halo %.1f MB redundant rows %.0f\n", tmin, tmax, halo_mb, redundant);
        if (!csv.empty()) {
            std::ifstream probe(csv);
            bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
            probe.close();
            std::ofstream out(csv, std::ios::app);
            if (fresh) {
                out << "label,P,hosts,threads,mode,data,H0,T,N,exprs,nodes,exchanged,repeat";
                for (int k = 0; k < K; ++k) out << "," << names[k];
                out << ",total_min,total_max,halo_mb,halo_msgs,redundant_rows\n";
            }
            out << label << "," << P << "," << hosts << "," << threads << "," << mode << ","
                << (data.synthetic ? "synthetic" : "real") << "," << options.H0 << "," << data.T << "," << data.N
                << "," << dag.roots().size() << "," << dag.nodes().size() << "," << exchanged << "," << repeat;
            for (int k = 0; k < K; ++k) out << "," << med[k];
            out << "," << tmin << "," << tmax << "," << halo_mb << "," << halo_msgs << "," << redundant << "\n";
        }
    }
    if (!profile.empty()) {
        std::vector<double> mine(gpe::kNumOps), worst(gpe::kNumOps);
        for (int k = 0; k < gpe::kNumOps; ++k) mine[k] = gpe::op_stats()[k].seconds / repeat;
        MPI_Reduce(mine.data(), worst.data(), gpe::kNumOps, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::ifstream probe(profile);
            bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
            probe.close();
            std::ofstream out(profile, std::ios::app);
            if (fresh) out << "label,P,threads,op,calls,seconds,cells,ns_per_cell\n";
            std::printf("%-12s %6s %9s %10s\n", "op", "calls", "seconds", "ns/cell");
            for (int k = 0; k < gpe::kNumOps; ++k) {
                const gpe::OpStat& st = gpe::op_stats()[k];
                if (st.calls == 0) continue;
                const char* name = gpe::op_info(static_cast<gpe::Op>(k)).name;
                long calls = st.calls / repeat;
                double cells = st.cells / repeat, ns = cells > 0 ? 1e9 * worst[k] / cells : 0.0;
                std::printf("%-12s %6ld %9.3f %10.3f\n", name, calls, worst[k], ns);
                out << label << "," << P << "," << threads << "," << name << "," << calls << "," << worst[k] << ","
                    << cells << "," << ns << "\n";
            }
        }
    }
    MPI_Finalize();
    return 0;
}
