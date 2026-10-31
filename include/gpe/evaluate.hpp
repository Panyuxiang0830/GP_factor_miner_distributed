#pragma once

#include <map>

#include "gpe/dag.hpp"
#include "gpe/panel.hpp"

namespace gpe {

// Serial execution of a finalized DAG on full-history panels (no halo).
// Nodes run in array (topological) order. An intermediate panel is freed as
// soon as its last parent has run. Returns the panel of every root.
std::map<int, Panel> evaluate_serial(const Dag& dag, const PanelSet& data);

}  // namespace gpe
