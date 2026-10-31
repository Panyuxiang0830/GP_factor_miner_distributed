#pragma once

#include <string>

namespace gpe {

// Operators of the Python GP (gp_factor_miner/operators.py).
enum class Op {
    Leaf,
    // element-wise
    Add, Sub, Mul, Div, Log, Abs,
    // cross-sectional (one trading day at a time)
    Rank, Zscore,
    // rolling, one series
    Mean, Std, Sum, Max, Min, Median, Ema, Lag, Delta, PctChange,
    // rolling, two series
    Corr, Beta,
};

enum class OpKind { Leaf, Elementwise, CrossSectional, Rolling, PairRolling };

struct OpInfo {
    Op op;
    const char* name;
    OpKind kind;
    int series_args;     // number of panel operands
    bool commutative;    // a op b == b op a
};

const OpInfo& op_info(Op op);
// Returns false if the name is not an operator.
bool op_from_name(const std::string& name, Op* out);

// Days of history this operator needs before the current day, excluding its
// children's own needs: w-1 for rolling windows, w for lag/delta/pct_change.
// ema depends on the whole history; it reports 0 here and is handled by a scan.
int own_lookback(Op op, int window);
bool needs_scan(Op op);

}  // namespace gpe
