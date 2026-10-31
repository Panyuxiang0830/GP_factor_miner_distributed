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
    double halo = 0.0;     // halo MPI_Sendrecv per level (includes waiting for the neighbour)
    double scan = 0.0;     // ema state chain
    double metrics = 0.0;  // daily RankIC and IC summaries
    double wait = 0.0;     // barrier before the reduction: load imbalance
    double reduce = 0.0;   // MPI_Allreduce of the IC summaries
    double total = 0.0;
    double halo_bytes = 0.0;  // bytes this rank sent in halo exchanges
    int halo_messages = 0;    // number of non-empty halo sends
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

// Distributed evaluation of a finalized DAG, decomposed by trading date.
//
// Strategy B (subtree halo exchange): each node is computed for the rank's own
// days only. The DAG is processed level by level; after a level, every node
// whose parents need history sends its last halo_rows rows to the next rank,
// all nodes of the level packed into one MPI_Sendrecv. Leaves read their halo
// directly from the panel files. ema nodes use the scan in ema_scan.hpp.
// Daily RankIC is local; one MPI_Allreduce of IcSummary gives the fitness.
//
// Requires every block to have at least max(halo_rows) days, so that a halo
// always comes from the left neighbour alone.
MpiBatchResult evaluate_batch_mpi(const Dag& dag, const PanelSet& data, MPI_Comm comm,
                                  const FitnessConfig& config = {}, bool keep_roots = false);

}  // namespace gpe
