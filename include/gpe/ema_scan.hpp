#pragma once

#include <cmath>
#include <vector>

#include "gpe/panel.hpp"

namespace gpe {

// Distributed ema (ewm(span, adjust=False)) by a scan over date blocks.
//
// The ema is a recursion over all history, so no finite halo is exact. After
// k missing days the update is y = (o*y_prev + n*x) / (o + n) with
// o = (1-alpha)^(k+1), an affine map of y_prev. Each rank therefore
//   1. summarises its block without knowing the incoming state;
//   2. receives the state at the end of the previous block, applies its
//      summary and passes the result on (O(N) work per rank);
//   3. recomputes its rows from the received state (apply_op with EmaState).

// State after a day: last value (NaN before any observation) and the weight
// old_wt of that value in the next update.
struct EmaCarry {
    double y = NAN;
    double old_wt = 1.0;
};

// One stock's block, as a function of the incoming state.
struct EmaBlockSummary {
    int first_obs = -1;        // first non-NaN day in the block, -1 if none
    double v_first = 0.0;      // its value
    double decay_first = 1.0;  // (1-alpha)^(first_obs+1): decay of old_wt up to it
    double decay_len = 1.0;    // (1-alpha)^n: decay over a block without observations
    double a = 1.0, b = 0.0;   // y_end = a * y_first + b after the first observation
    double old_wt_end = 1.0;   // old_wt at the end of the block (independent of the input)
};

EmaBlockSummary ema_summarize(const std::vector<double>& x, int span);
EmaCarry ema_apply(const EmaBlockSummary& s, const EmaCarry& in, int span);

// Summaries of rows 0..T-1 of every stock.
std::vector<EmaBlockSummary> ema_summarize_panel(const Panel& a, int span);

}  // namespace gpe
