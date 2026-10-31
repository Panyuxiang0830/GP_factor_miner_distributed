#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "gpe/json.hpp"
#include "gpe/ops.hpp"

namespace gpe {

// One distinct subexpression of a batch (E1–E3).
//
// Nodes are created children-first, so the node vector is already a valid
// execution (topological) order.
struct Node {
    Op op = Op::Leaf;
    int window = 0;        // rolling window / lag length; 0 if none
    int a = -1, b = -1;    // child node ids; -1 if absent
    int field = -1;        // leaves: index into Dag::fields()

    int depth = 0;         // operator depth, as expression_depth() in Python
    int tree_nodes = 1;    // subtree size, as expression_nodes() in Python
    int level = 0;         // execution level: leaves 0, else 1 + max(child levels)

    int lookback = 0;      // cumulative history days (strategy A / H0 decision)
    bool has_scan = false; // the subtree contains an ema
    int halo_rows = 0;     // rows parents need before day 0 (strategy B)
    int parents = 0;       // number of parent references (for refcounting)
};

class Dag {
public:
    // Adds one expression in the Python nested-list format, for example
    // ["mean", ["std", "close", 10], 5]. Returns the root node id, or -1 with
    // *error set if the expression is invalid or degenerate (x-x, x/x).
    int add_expression(const Json& expression, std::string* error = nullptr);

    // Computes halo_rows and parent counts. Call after the last add_expression.
    void finalize();

    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<std::string>& fields() const { return fields_; }
    // Root id of each successfully added expression, in insertion order.
    const std::vector<int>& roots() const { return roots_; }
    int max_level() const;

private:
    struct Key {
        Op op; int window; int a; int b; int field;
        bool operator==(const Key& o) const {
            return op == o.op && window == o.window && a == o.a && b == o.b && field == o.field;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const;
    };

    int build(const Json& expression);
    int make_leaf(const std::string& name);
    int make_op(Op op, int window, int a, int b);

    std::vector<Node> nodes_;
    std::vector<std::string> fields_;
    std::unordered_map<std::string, int> field_index_;
    std::unordered_map<Key, int, KeyHash> table_;
    std::vector<int> roots_;
};

}  // namespace gpe
