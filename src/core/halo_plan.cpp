#include "gpe/halo_plan.hpp"

#include <algorithm>

namespace gpe {

std::vector<HaloPlan> plan_halos(const Dag& dag, int H0) {
    const auto& nodes = dag.nodes();
    std::vector<HaloPlan> plan(nodes.size());
    // Parents have larger ids, so in reverse order a node's need is final
    // before it is decided and passed on to its children.
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        const Node& n = nodes[i];
        HaloPlan& p = plan[i];
        p.redundant = n.op == Op::Leaf || (n.op != Op::Ema && p.need <= H0);
        int extend = p.redundant ? p.need : 0;
        for (int c : {n.a, n.b})
            if (c >= 0) plan[c].need = std::max(plan[c].need, extend + own_lookback(n.op, n.window));
    }
    return plan;
}

}  // namespace gpe
