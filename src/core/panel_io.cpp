#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "gpe/json.hpp"
#include "gpe/panel.hpp"

namespace gpe {

PanelBuffer BufferPool::take(size_t cells) {
    auto best = free_.end();
    for (auto it = free_.begin(); it != free_.end(); ++it)
        if (it->capacity() >= cells && (best == free_.end() || it->capacity() < best->capacity())) best = it;
    if (best == free_.end()) return {};
    PanelBuffer buf = std::move(*best);
    free_.erase(best);
    return buf;
}

void BufferPool::give(PanelBuffer&& buf) {
    if (buf.capacity() > 0) free_.push_back(std::move(buf));
}

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

namespace {

Panel read_rows(const std::string& path, int T_file, int N, int t0, int t1, int halo) {
    if (t0 - halo < 0 || t1 > T_file || t0 > t1) throw std::runtime_error("row range outside " + path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Panel p(t1 - t0, N, -halo);
    in.seekg(static_cast<std::streamoff>(t0 - halo) * N * sizeof(float));
    in.read(reinterpret_cast<char*>(p.buf.data()), static_cast<std::streamsize>(p.buf.size() * sizeof(float)));
    if (!in) throw std::runtime_error("short read: " + path);
    return p;
}

}  // namespace

Panel PanelSet::load_field_rows(const std::string& name, int t0, int t1, int halo) const {
    if (!has_field(name)) throw std::runtime_error("unknown field: " + name);
    return read_rows(dir + "/fields/" + name + ".f32", T, N, t0, t1, halo);
}

Panel PanelSet::load_target_rows(int t0, int t1) const { return read_rows(dir + "/target.f32", T, N, t0, t1, 0); }

}  // namespace gpe
