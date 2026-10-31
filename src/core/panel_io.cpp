#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "gpe/json.hpp"
#include "gpe/panel.hpp"

namespace gpe {

Panel read_raw_panel(const std::string& path, int T, int N, bool float64) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Panel p(T, N);
    size_t count = static_cast<size_t>(T) * N;
    if (float64) {
        std::vector<double> tmp(count);
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(count * sizeof(double)));
        std::copy(tmp.begin(), tmp.end(), p.buf.begin());
    } else {
        in.read(reinterpret_cast<char*>(p.buf.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!in) throw std::runtime_error("short read: " + path);
    return p;
}

PanelSet PanelSet::open(const std::string& dir) {
    Json meta = parse_json_file(dir + "/meta.json");
    PanelSet set;
    set.dir = dir;
    set.T = static_cast<int>(meta.at("T").number);
    set.N = static_cast<int>(meta.at("N").number);
    for (const auto& v : meta.at("dates").array) set.dates.push_back(v.string);
    for (const auto& v : meta.at("assets").array) set.assets.push_back(v.string);
    for (const auto& v : meta.at("fields").array) set.fields.push_back(v.string);
    return set;
}

bool PanelSet::has_field(const std::string& name) const {
    return std::find(fields.begin(), fields.end(), name) != fields.end();
}

Panel PanelSet::load_field(const std::string& name) const {
    if (!has_field(name)) throw std::runtime_error("unknown field: " + name);
    return read_raw_panel(dir + "/fields/" + name + ".f32", T, N);
}

Panel PanelSet::load_target() const { return read_raw_panel(dir + "/target.f32", T, N); }

}  // namespace gpe
