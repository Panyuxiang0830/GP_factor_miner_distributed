#include "gpe/ops.hpp"

#include <stdexcept>

namespace gpe {
namespace {

const OpInfo kOps[] = {
    {Op::Leaf, "<leaf>", OpKind::Leaf, 0, false},
    {Op::Add, "+", OpKind::Elementwise, 2, true},
    {Op::Sub, "-", OpKind::Elementwise, 2, false},
    {Op::Mul, "*", OpKind::Elementwise, 2, true},
    {Op::Div, "/", OpKind::Elementwise, 2, false},
    {Op::Log, "log", OpKind::Elementwise, 1, false},
    {Op::Abs, "abs", OpKind::Elementwise, 1, false},
    {Op::Rank, "rank", OpKind::CrossSectional, 1, false},
    {Op::Zscore, "zscore", OpKind::CrossSectional, 1, false},
    {Op::Mean, "mean", OpKind::Rolling, 1, false},
    {Op::Std, "std", OpKind::Rolling, 1, false},
    {Op::Sum, "sum", OpKind::Rolling, 1, false},
    {Op::Max, "max", OpKind::Rolling, 1, false},
    {Op::Min, "min", OpKind::Rolling, 1, false},
    {Op::Median, "median", OpKind::Rolling, 1, false},
    {Op::Ema, "ema", OpKind::Rolling, 1, false},
    {Op::Lag, "lag", OpKind::Rolling, 1, false},
    {Op::Delta, "delta", OpKind::Rolling, 1, false},
    {Op::PctChange, "pct_change", OpKind::Rolling, 1, false},
    {Op::Corr, "correlation", OpKind::PairRolling, 2, false},
    {Op::Beta, "beta", OpKind::PairRolling, 2, false},
};

}  // namespace

const OpInfo& op_info(Op op) {
    for (const auto& info : kOps)
        if (info.op == op) return info;
    throw std::logic_error("unknown op");
}

bool op_from_name(const std::string& name, Op* out) {
    for (const auto& info : kOps) {
        if (info.op != Op::Leaf && name == info.name) {
            *out = info.op;
            return true;
        }
    }
    return false;
}

int own_lookback(Op op, int window) {
    switch (op) {
        case Op::Mean: case Op::Std: case Op::Sum: case Op::Max: case Op::Min:
        case Op::Median: case Op::Corr: case Op::Beta:
            return window - 1;
        case Op::Lag: case Op::Delta: case Op::PctChange:
            return window;
        default:
            return 0;
    }
}

bool needs_scan(Op op) { return op == Op::Ema; }

}  // namespace gpe
