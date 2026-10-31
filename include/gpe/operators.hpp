#pragma once

#include "gpe/ops.hpp"
#include "gpe/panel.hpp"

namespace gpe {

// Serial reference implementation of one operator (E5), matching the pandas
// semantics of gp_factor_miner/operators.py:
//   - inf results of + - * / pct_change beta are turned into NaN;
//   - rolling windows count non-NaN observations, with min_periods = 1 for
//     mean/sum/median and min(3, w) for std/max/min/correlation/beta;
//   - correlation and cov use pairwise-complete observations, beta divides by
//     the variance of b alone (both as in pandas);
//   - ema is ewm(span=w, adjust=False): y = (o*y + alpha*x) / (o + alpha) with
//     o = (1-alpha)^(k+1) after k missing days, an affine map of y.
//
// `out` must already be sized. Rows out.lo..T-1 are written. Rolling operators
// read their input from a.lo, so a.lo is treated as the start of history.
void apply_op(Op op, int window, const Panel& a, const Panel* b, Panel& out);

// min_periods used by pandas for a rolling operator.
int rolling_min_periods(Op op, int window);

}  // namespace gpe
