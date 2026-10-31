#pragma once

#include <vector>

#include "gpe/dag.hpp"

namespace gpe {

// How each node obtains the history (rows before the rank's first day) that
// its parents read: the H0 knob of the date decomposition (E4).
//
// need is the number of history rows the node must hold. A node whose parents
// are computed for their own days only needs max own_lookback(parent), as in
// Node::halo_rows; a parent computed redundantly for R extra rows asks for
// R + own_lookback(parent).
//
//   need <= H0  the node is computed redundantly for those rows (no messages);
//   need >  H0  the node is computed for its own days and receives the rows
//               from the left neighbour (subtree halo exchange).
//
// Leaves always read their history from the panel files, and ema is never
// computed redundantly (a recursion started a few days early is not exact),
// so its history is always exchanged. H0 = 0 is strategy B; H0 >= the largest
// lookback is strategy A, where only ema outputs are exchanged.
struct HaloPlan {
    int need = 0;
    bool redundant = false;  // computed locally for the `need` rows (leaves: read from file)
    bool exchanged() const { return !redundant && need > 0; }
};

std::vector<HaloPlan> plan_halos(const Dag& dag, int H0);

}  // namespace gpe
