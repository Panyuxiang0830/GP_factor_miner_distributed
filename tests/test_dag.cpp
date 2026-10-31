// Tests for E1–E3: parsing, normalisation, DAG merging, lookback and halo rows.
//
// Usage: test_dag [normalize_cases.json]
// The fixture is produced by python/scripts/export_reference.py normalize.

#include <cstdio>
#include <map>
#include <string>

#include "gpe/dag.hpp"
#include "gpe/halo_plan.hpp"
#include "gpe/json.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

gpe::Json J(const char* text) { return gpe::parse_json(text); }

// mean(std(x,10),5), std(x,10)*y, correlation(std(x,10),y,20) share std(x,10).
void test_shared_subtree_example() {
    gpe::Dag dag;
    int r1 = dag.add_expression(J(R"(["mean", ["std", "x", 10], 5])"));
    int r2 = dag.add_expression(J(R"(["*", ["std", "x", 10], "y"])"));
    int r3 = dag.add_expression(J(R"(["correlation", ["std", "x", 10], "y", 20])"));
    dag.finalize();

    const auto& n = dag.nodes();
    CHECK(n.size() == 6);                 // x, std, mean, y, *, correlation
    int x = 0, sd = 1, y = 3;
    CHECK(n[sd].op == gpe::Op::Std && n[sd].a == x);
    CHECK(n[r2].a == sd || n[r2].b == sd);

    // Strategy B: each node keeps the rows its parents look back.
    CHECK(n[x].halo_rows == 9);           // std(.,10)
    CHECK(n[sd].halo_rows == 19);         // max(mean 5 -> 4, correlation 20 -> 19)
    CHECK(n[y].halo_rows == 19);          // correlation 20
    CHECK(n[r1].halo_rows == 0 && n[r2].halo_rows == 0 && n[r3].halo_rows == 0);

    // Strategy A: cumulative lookback of each root.
    CHECK(n[r1].lookback == 13);          // 4 + 9
    CHECK(n[r2].lookback == 9);
    CHECK(n[r3].lookback == 28);          // 19 + 9

    CHECK(n[sd].level == 1 && n[r1].level == 2 && n[r3].level == 2);
    CHECK(dag.max_level() == 2);
    CHECK(n[sd].parents == 3);
    CHECK(n[y].parents == 2);
}

// The H0 knob on the same example: H0 = 0 is strategy B, a large H0 computes
// std(x,10) redundantly so that x must provide 19 + 9 rows; ema is always
// exchanged.
void test_halo_plan() {
    gpe::Dag dag;
    dag.add_expression(J(R"(["mean", ["std", "x", 10], 5])"));
    dag.add_expression(J(R"(["*", ["std", "x", 10], "y"])"));
    dag.add_expression(J(R"(["correlation", ["std", "x", 10], "y", 20])"));
    dag.finalize();
    int x = 0, sd = 1, y = 3;

    auto b = gpe::plan_halos(dag, 0);
    CHECK(b[sd].need == 19 && b[sd].exchanged());
    CHECK(b[x].need == 9 && !b[x].exchanged());  // leaves read history from file
    CHECK(b[y].need == 19 && !b[y].exchanged());
    for (size_t i = 0; i < b.size(); ++i) CHECK(b[i].need == dag.nodes()[i].halo_rows);

    auto mid = gpe::plan_halos(dag, 10);         // 19 > 10: still exchanged
    CHECK(mid[sd].exchanged() && mid[x].need == 9);

    auto a = gpe::plan_halos(dag, 19);
    CHECK(a[sd].redundant && a[sd].need == 19);
    CHECK(a[x].need == 28);                      // = lookback of the correlation root
    for (const auto& p : a) CHECK(!p.exchanged());

    gpe::Dag e;
    int root = e.add_expression(J(R"(["mean", ["ema", ["mean", "x", 5], 3], 10])"));
    e.finalize();
    auto pe = gpe::plan_halos(e, 1000);
    int ema = e.nodes()[root].a, inner = e.nodes()[ema].a;
    CHECK(pe[ema].need == 9 && pe[ema].exchanged());
    CHECK(pe[inner].need == 0);                  // ema reads only the rank's own days
    CHECK(pe[e.nodes()[inner].a].need == 4);
}

void test_normalisation_and_rollback() {
    gpe::Dag dag;
    int ab = dag.add_expression(J(R"(["+", "a", "b"])"));
    int ba = dag.add_expression(J(R"(["+", "b", "a"])"));
    CHECK(ab == ba);

    int rank = dag.add_expression(J(R"(["rank", "a"])"));
    int rank_rank = dag.add_expression(J(R"(["rank", ["rank", "a"]])"));
    CHECK(rank == rank_rank);

    size_t before = dag.nodes().size();
    std::string error;
    CHECK(dag.add_expression(J(R"(["/", ["log", "c"], ["log", "c"]])"), &error) == -1);
    CHECK(!error.empty());
    CHECK(dag.nodes().size() == before);  // rolled back, no orphan nodes
    CHECK(dag.fields().size() == 2);      // "c" was rolled back too

    CHECK(dag.add_expression(J(R"(["mean", "a"])")) == -1);         // missing window
    CHECK(dag.add_expression(J(R"(["mean", "a", 0])")) == -1);      // bad window
    CHECK(dag.add_expression(J(R"(["unknown_op", "a"])")) == -1);

    int ema = dag.add_expression(J(R"(["ema", ["delta", "a", 5], 10])"));
    dag.finalize();
    CHECK(dag.nodes()[ema].has_scan);
    CHECK(dag.nodes()[ema].lookback == 5);  // ema itself is handled by a scan
}

// Every Python case: same rejection, same depth and node count, and two
// inputs share a root exactly when Python gives them the same key.
void test_python_fixture(const std::string& path) {
    gpe::Json cases = gpe::parse_json_file(path);
    gpe::Dag dag;
    std::map<std::string, int> root_of_key;
    std::map<int, std::string> key_of_root;
    int checked = 0;
    for (const auto& c : cases.array) {
        std::string error;
        int root = dag.add_expression(c.at("input"), &error);
        if (c.has("error")) {
            if (root != -1) std::printf("  expected rejection: %s\n", c.at("input").dump().c_str());
            CHECK(root == -1);
            continue;
        }
        if (root < 0) {
            std::printf("  unexpected rejection (%s): %s\n", error.c_str(), c.at("input").dump().c_str());
            CHECK(root >= 0);
            continue;
        }
        const auto& node = dag.nodes()[root];
        int depth = static_cast<int>(c.at("depth").number);
        int count = static_cast<int>(c.at("nodes").number);
        if (node.depth != depth || node.tree_nodes != count)
            std::printf("  depth/nodes %d/%d vs python %d/%d: %s\n", node.depth, node.tree_nodes,
                        depth, count, c.at("input").dump().c_str());
        CHECK(node.depth == depth);
        CHECK(node.tree_nodes == count);

        const std::string& key = c.at("key").string;
        auto k = root_of_key.find(key);
        if (k != root_of_key.end()) CHECK(k->second == root);
        auto r = key_of_root.find(root);
        if (r != key_of_root.end()) CHECK(r->second == key);
        root_of_key[key] = root;
        key_of_root[root] = key;
        ++checked;
    }
    dag.finalize();

    size_t tree_total = 0;
    for (int root : dag.roots()) tree_total += dag.nodes()[root].tree_nodes;
    std::printf("fixture %s: %d cases matched; %zu DAG nodes for %zu tree nodes\n", path.c_str(),
                checked, dag.nodes().size(), tree_total);
}

}  // namespace

int main(int argc, char** argv) {
    test_shared_subtree_example();
    test_halo_plan();
    test_normalisation_and_rollback();
    test_python_fixture(argc > 1 ? argv[1] : GPE_TEST_DATA_DIR "/normalize_cases.json");
    if (g_failures) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all DAG tests passed\n");
    return 0;
}
