#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
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
    set.N_file = set.N;
    for (const auto& v : meta.at("dates").array) set.dates.push_back(v.string);
    for (const auto& v : meta.at("assets").array) set.assets.push_back(v.string);
    for (const auto& v : meta.at("fields").array) set.fields.push_back(v.string);
    return set;
}

PanelSet PanelSet::make_synthetic(int T, int N, unsigned long long seed) {
    PanelSet set;
    set.T = T;
    set.N = set.N_file = N;
    set.synthetic = true;
    set.seed = seed;
    return set;
}

bool PanelSet::has_field(const std::string& name) const {
    return synthetic || std::find(fields.begin(), fields.end(), name) != fields.end();
}

namespace {

uint64_t mix(uint64_t x) {  // splitmix64 finalizer
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double unit(uint64_t x) { return (static_cast<double>(x >> 11) + 0.5) * 0x1.0p-53; }  // in (0, 1)

uint64_t hash_name(const std::string& s) {  // FNV-1a
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
    return h;
}

// Rows t0-halo .. t1-1 of a synthetic field (name empty: the target).
Panel synthetic_rows(const PanelSet& set, const std::string& name, int t0, int t1, int halo) {
    const uint64_t key = mix(set.seed ^ (name.empty() ? 0 : hash_name(name)));
    const double u_kind = unit(mix(key + 1)), u_scale = unit(mix(key + 2));
    const bool target = name.empty(), lognormal = !target && u_kind < 0.6;
    const double mu = -2.0 + 8.0 * u_scale, sigma = target ? 0.02 : 0.01 * std::pow(100.0, u_scale);
    const int N = set.N;
    Panel p(t1 - t0, N, -halo);
    #pragma omp parallel for schedule(static)
    for (int t = -halo; t < p.T; ++t) {
        float* row = p.row(t);
        const uint64_t day = mix(key ^ (static_cast<uint64_t>(t0 + t) << 32));
        for (int s = 0; s < N; ++s) {
            uint64_t h = mix(day + static_cast<uint64_t>(s));
            if (unit(h) < 0.02) {
                row[s] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // Box-Muller from two further draws.
            double z = std::sqrt(-2.0 * std::log(unit(mix(h + 1)))) * std::cos(6.283185307179586 * unit(mix(h + 2)));
            row[s] = static_cast<float>(lognormal ? std::exp(mu + 0.5 * z) : sigma * z);
        }
    }
    return p;
}

Panel read_rows(const std::string& path, int T_file, int N_file, int N, int t0, int t1, int halo) {
    if (t0 - halo < 0 || t1 > T_file || t0 > t1 || N > N_file) throw std::runtime_error("row range outside " + path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Panel p(t1 - t0, N, -halo);
    in.seekg(static_cast<std::streamoff>(t0 - halo) * N_file * sizeof(float));
    if (N == N_file) {
        in.read(reinterpret_cast<char*>(p.buf.data()), static_cast<std::streamsize>(p.buf.size() * sizeof(float)));
    } else {  // first N assets of each row
        for (int t = -halo; t < p.T && in; ++t) {
            in.read(reinterpret_cast<char*>(p.row(t)), static_cast<std::streamsize>(N * sizeof(float)));
            in.seekg(static_cast<std::streamoff>(N_file - N) * sizeof(float), std::ios::cur);
        }
    }
    if (!in) throw std::runtime_error("short read: " + path);
    return p;
}

}  // namespace

Panel PanelSet::load_field(const std::string& name) const { return load_field_rows(name, 0, T, 0); }

Panel PanelSet::load_target() const { return load_target_rows(0, T); }

Panel PanelSet::load_field_rows(const std::string& name, int t0, int t1, int halo) const {
    if (!has_field(name)) throw std::runtime_error("unknown field: " + name);
    if (synthetic) return synthetic_rows(*this, name, t0, t1, halo);
    return read_rows(dir + "/fields/" + name + ".f32", T, N_file, N, t0, t1, halo);
}

Panel PanelSet::load_target_rows(int t0, int t1) const {
    if (synthetic) return synthetic_rows(*this, "", t0, t1, 0);
    return read_rows(dir + "/target.f32", T, N_file, N, t0, t1, 0);
}

}  // namespace gpe
