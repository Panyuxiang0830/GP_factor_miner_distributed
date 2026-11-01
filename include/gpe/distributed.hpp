#pragma once

#include <mpi.h>

#include <map>
#include <vector>

#include "gpe/dag.hpp"
#include "gpe/metrics.hpp"
#include "gpe/panel.hpp"

namespace gpe {

// Days t0..t1-1 owned by one rank. Blocks are contiguous; the first T % P
// ranks get one extra day.
struct DateBlock {
    int t0 = 0, t1 = 0;
    int size() const { return t1 - t0; }
};
DateBlock date_block(int T, int ranks, int rank);

// Seconds spent by this rank in each phase.
struct MpiTiming {
    double load = 0.0;     // reading leaf panels (own days + halo) and the target
    double compute = 0.0;  // operators, including ema block summaries
    double halo = 0.0;     // halo messages per level (includes waiting for the neighbours)
    double scan = 0.0;     // ema state chain
    double metrics = 0.0;  // daily RankIC and IC summaries
    double wait = 0.0;     // barrier before the reduction: load imbalance
    double reduce = 0.0;   // MPI_Allreduce of the IC summaries
    double total = 0.0;
    double halo_bytes = 0.0;  // bytes this rank sent in halo exchanges
    int halo_messages = 0;    // number of non-empty halo sends
    long redundant_rows = 0;  // history rows computed redundantly, summed over nodes (x N cells)
};

struct MpiBatchResult {
    DateBlock block;
    // One candidate per entry of dag.roots(), identical on every rank.
    std::vector<Candidate> candidates;
    // With keep_roots: local panel and local daily IC of each root node.
    std::map<int, Panel> roots;
    std::map<int, DailyIC> daily;
    MpiTiming timing;
};

struct MpiOptions {
    FitnessConfig fitness;
    int H0 = 0;               // redundant-computation budget, see halo_plan.hpp
    bool keep_roots = false;  // return root panels and daily ICs (tests)
    // Reuses panel memory across batches (one GP generation after another);
    // without it each batch starts from an empty pool.
    BufferPool* pool = nullptr;
};

// Distributed evaluation of a finalized DAG, decomposed by trading date.
//
// The DAG is processed level by level. How a node gets the history its
// parents read is set by plan_halos(dag, H0):
//   - redundant nodes are computed for their history rows as well (strategy A
//     when every node is redundant);
//   - exchanged nodes are computed for the rank's own days; after the level,
//     the rows each later rank needs are sent to it, all nodes of the level
//     packed into one message per partner (strategy B when H0 = 0). Usually
//     the partner is the right neighbour; with blocks shorter than the halo a
//     rank also receives from ranks further left;
//   - leaves read their history directly from the panel files;
//   - ema nodes use the scan in ema_scan.hpp.
// Daily RankIC is local; one MPI_Allreduce of IcSummary gives the fitness.
MpiBatchResult evaluate_batch_mpi(const Dag& dag, const PanelSet& data, MPI_Comm comm,
                                  const MpiOptions& options = {});

}  // namespace gpe
