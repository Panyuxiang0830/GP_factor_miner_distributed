// Factor values of the course C++ engine (gpe::evaluate_serial, unchanged),
// for value-level parity checks against the Python operators.
//   gpp_values PANEL_DIR EXPRESSIONS.json OUT_DIR
// EXPRESSIONS.json is a JSON list of expressions; OUT_DIR/<i>.f32 receives
// root i as float32 T x N (an invalid expression writes nothing).
#include <cstdio>
#include <fstream>
#include <string>

#include "gpe/dag.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/json.hpp"
#include "gpe/panel.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: gpp_values PANEL_DIR EXPRESSIONS.json OUT_DIR\n");
        return 2;
    }
    gpe::PanelSet data = gpe::PanelSet::open(argv[1]);
    gpe::Json list = gpe::parse_json_file(argv[2]);
    gpe::Dag dag;
    std::vector<int> root(list.array.size(), -1);
    for (size_t i = 0; i < list.array.size(); ++i) {
        std::string error;
        root[i] = dag.add_expression(list.array[i], &error);
        if (root[i] < 0) std::fprintf(stderr, "expression %zu: %s\n", i, error.c_str());
    }
    dag.finalize();
    auto values = gpe::evaluate_serial(dag, data);
    for (size_t i = 0; i < root.size(); ++i) {
        if (root[i] < 0) continue;
        const gpe::Panel& p = values.at(root[i]);
        std::ofstream out(std::string(argv[3]) + "/" + std::to_string(i) + ".f32", std::ios::binary);
        out.write(reinterpret_cast<const char*>(p.buf.data()), static_cast<std::streamsize>(p.buf.size() * sizeof(float)));
    }
    return 0;
}
