#include "gpe/evaluate.hpp"

#include <set>
#include <vector>

#include "gpe/operators.hpp"

namespace gpe {

std::map<int, Panel> evaluate_serial(const Dag& dag, const PanelSet& data) {
    const auto& nodes = dag.nodes();
    std::set<int> roots(dag.roots().begin(), dag.roots().end());
    std::vector<Panel> values(nodes.size());
    std::vector<int> pending(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) pending[i] = nodes[i].parents;

    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& n = nodes[i];
        if (n.op == Op::Leaf) {
            values[i] = data.load_field(dag.fields()[n.field]);
            continue;
        }
        const Panel& a = values[n.a];
        const Panel* b = n.b >= 0 ? &values[n.b] : nullptr;
        values[i].reset(a.T, a.N, a.lo);
        apply_op(n.op, n.window, a, b, values[i]);

        for (int c : {n.a, n.b}) {
            if (c < 0) continue;
            if (--pending[c] == 0 && !roots.count(c)) {
                values[c] = Panel();  // release memory
            }
        }
    }

    std::map<int, Panel> out;
    for (int r : roots) out[r] = std::move(values[r]);
    return out;
}

}  // namespace gpe
