#include "gpe/dag.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace gpe {

size_t Dag::KeyHash::operator()(const Key& k) const {
    size_t h = std::hash<int>()(static_cast<int>(k.op));
    auto mix = [&h](int v) { h ^= std::hash<int>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    mix(k.window);
    mix(k.a);
    mix(k.b);
    mix(k.field);
    return h;
}

int Dag::make_leaf(const std::string& name) {
    auto it = field_index_.find(name);
    int field;
    if (it == field_index_.end()) {
        field = static_cast<int>(fields_.size());
        fields_.push_back(name);
        field_index_[name] = field;
    } else {
        field = it->second;
    }
    Key key{Op::Leaf, 0, -1, -1, field};
    auto found = table_.find(key);
    if (found != table_.end()) return found->second;

    Node node;
    node.field = field;
    nodes_.push_back(node);
    int id = static_cast<int>(nodes_.size()) - 1;
    table_[key] = id;
    return id;
}

int Dag::make_op(Op op, int window, int a, int b) {
    const OpInfo& info = op_info(op);
    // Normalisation, mirroring normalize_expression() in Python.
    if (info.commutative && a > b) std::swap(a, b);
    if (op == Op::Rank && nodes_[a].op == Op::Rank) return a;               // rank(rank(x))
    if ((op == Op::Sub || op == Op::Div) && a == b)
        throw std::invalid_argument(std::string("degenerate expression: ") + info.name + "(x, x)");

    Key key{op, window, a, b, -1};
    auto found = table_.find(key);
    if (found != table_.end()) return found->second;                        // shared subtree

    Node node;
    node.op = op;
    node.window = window;
    node.a = a;
    node.b = b;
    const Node& ca = nodes_[a];
    const Node* cb = b >= 0 ? &nodes_[b] : nullptr;
    node.depth = 1 + std::max(ca.depth, cb ? cb->depth : 0);
    node.tree_nodes = 1 + ca.tree_nodes + (cb ? cb->tree_nodes : 0);
    node.level = 1 + std::max(ca.level, cb ? cb->level : 0);
    node.lookback = own_lookback(op, window) + std::max(ca.lookback, cb ? cb->lookback : 0);
    node.has_scan = needs_scan(op) || ca.has_scan || (cb && cb->has_scan);

    nodes_.push_back(node);
    int id = static_cast<int>(nodes_.size()) - 1;
    table_[key] = id;
    return id;
}

int Dag::build(const Json& e) {
    if (e.is_string()) return make_leaf(e.string);
    if (!e.is_array() || e.array.empty() || !e.array[0].is_string())
        throw std::invalid_argument("expression must be a field name or [op, ...]: " + e.dump());

    Op op;
    if (!op_from_name(e.array[0].string, &op))
        throw std::invalid_argument("unknown operator: " + e.array[0].string);
    const OpInfo& info = op_info(op);
    bool windowed = info.kind == OpKind::Rolling || info.kind == OpKind::PairRolling;
    size_t expected = 1 + info.series_args + (windowed ? 1 : 0);
    if (e.array.size() != expected)
        throw std::invalid_argument("wrong number of arguments for " + std::string(info.name) + ": " + e.dump());

    int window = 0;
    if (windowed) {
        const Json& w = e.array.back();
        if (!w.is_number() || w.number < 1 || w.number != std::floor(w.number))
            throw std::invalid_argument("window must be a positive integer: " + e.dump());
        window = static_cast<int>(w.number);
    }
    int a = build(e.array[1]);
    int b = info.series_args == 2 ? build(e.array[2]) : -1;
    return make_op(op, window, a, b);
}

int Dag::add_expression(const Json& expression, std::string* error) {
    // Roll back on failure so a rejected expression leaves no orphan nodes.
    size_t node_count = nodes_.size();
    size_t field_count = fields_.size();
    try {
        int root = build(expression);
        roots_.push_back(root);
        return root;
    } catch (const std::invalid_argument& ex) {
        for (auto it = table_.begin(); it != table_.end();)
            it = it->second >= static_cast<int>(node_count) ? table_.erase(it) : std::next(it);
        nodes_.resize(node_count);
        for (size_t f = field_count; f < fields_.size(); ++f) field_index_.erase(fields_[f]);
        fields_.resize(field_count);
        if (error) *error = ex.what();
        return -1;
    }
}

void Dag::finalize() {
    for (auto& n : nodes_) {
        n.halo_rows = 0;
        n.parents = 0;
    }
    for (const auto& p : nodes_) {
        int need = own_lookback(p.op, p.window);
        for (int c : {p.a, p.b}) {
            if (c < 0) continue;
            nodes_[c].parents += 1;
            nodes_[c].halo_rows = std::max(nodes_[c].halo_rows, need);
        }
    }
}

int Dag::max_level() const {
    int level = 0;
    for (const auto& n : nodes_) level = std::max(level, n.level);
    return level;
}

}  // namespace gpe
