#pragma once

#include <string>
#include <vector>

namespace gpe {

// A date-major float32 panel: day t, asset s at row(t)[s].
//
// Rows lo..T-1 are stored, with lo <= 0. Rows below 0 hold history that
// precedes the rank's own block (halo or redundant computation). In the serial
// engine lo == 0 and row 0 is the first day of the data set.
struct Panel {
    int T = 0;
    int N = 0;
    int lo = 0;
    std::vector<float> buf;

    Panel() = default;
    Panel(int T_, int N_, int lo_ = 0) { reset(T_, N_, lo_); }

    void reset(int T_, int N_, int lo_ = 0) {
        T = T_;
        N = N_;
        lo = lo_;
        buf.assign(static_cast<size_t>(T - lo) * N, 0.0f);
    }
    int rows() const { return T - lo; }
    float* row(int t) { return buf.data() + static_cast<size_t>(t - lo) * N; }
    const float* row(int t) const { return buf.data() + static_cast<size_t>(t - lo) * N; }
    float& at(int t, int s) { return row(t)[s]; }
    float at(int t, int s) const { return row(t)[s]; }
};

// A directory written by python/scripts/export_panels.py.
struct PanelSet {
    int T = 0;
    int N = 0;
    std::vector<std::string> dates;
    std::vector<std::string> assets;
    std::vector<std::string> fields;   // all fields listed in meta.json
    std::string dir;

    static PanelSet open(const std::string& dir);
    bool has_field(const std::string& name) const;
    Panel load_field(const std::string& name) const;
    Panel load_target() const;

    // Global days t0-halo .. t1-1 as a local panel: local row 0 is day t0 and
    // the `halo` days before it are rows -halo..-1. Reads only those rows.
    Panel load_field_rows(const std::string& name, int t0, int t1, int halo = 0) const;
    Panel load_target_rows(int t0, int t1) const;
};

// Reads a raw little-endian array of T*N values (float32 or float64) into a panel.
Panel read_raw_panel(const std::string& path, int T, int N, bool float64 = false);

}  // namespace gpe
