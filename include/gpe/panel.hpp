#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gpe {

// Allocator whose value-initialization leaves elements uninitialized.
// Every panel row is written (by an operator, a file read or the halo
// exchange) before it is read, so zero-filling on allocation would only add a
// single-threaded pass over memory; the operators touch the pages in parallel.
template <class T>
struct NoInitAllocator : std::allocator<T> {
    template <class U>
    struct rebind { using other = NoInitAllocator<U>; };
    NoInitAllocator() = default;
    template <class U>
    NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
    template <class U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};

// A date-major float32 panel: day t, asset s at row(t)[s].
//
// Rows lo..T-1 are stored, with lo <= 0. Rows below 0 hold history that
// precedes the rank's own block (halo or redundant computation). In the serial
// engine lo == 0 and row 0 is the first day of the data set.
using PanelBuffer = std::vector<float, NoInitAllocator<float>>;

struct Panel {
    int T = 0;
    int N = 0;
    int lo = 0;
    PanelBuffer buf;  // uninitialized after reset(); keeps its capacity

    Panel() = default;
    Panel(int T_, int N_, int lo_ = 0) { reset(T_, N_, lo_); }

    void reset(int T_, int N_, int lo_ = 0) {
        T = T_;
        N = N_;
        lo = lo_;
        buf.clear();
        buf.resize(static_cast<size_t>(T - lo) * N);
    }
    int rows() const { return T - lo; }
    float* row(int t) { return buf.data() + static_cast<size_t>(t - lo) * N; }
    const float* row(int t) const { return buf.data() + static_cast<size_t>(t - lo) * N; }
    float& at(int t, int s) { return row(t)[s]; }
    float at(int t, int s) const { return row(t)[s]; }
};

// Buffers of released panels, handed out again so that later nodes reuse
// memory whose pages are already mapped. A fresh allocation is first touched
// by the operator; with many threads those page faults queue in the kernel.
class BufferPool {
public:
    // Smallest kept buffer that holds `cells`, or an empty one.
    PanelBuffer take(size_t cells);
    void give(PanelBuffer&& buf);

private:
    std::vector<PanelBuffer> free_;
};

// A directory written by python/scripts/export_panels.py, or synthetic data.
//
// T and N may be lowered after open() to use the first T dates and the first
// N assets (problem-size experiments); the files keep their full width.
//
// Synthetic panels (weak scaling) are generated on demand, so a rank only
// produces its own rows. Every cell is a function of (seed, field, global
// day, asset) alone, so results do not depend on the number of ranks:
//   - each field is either log-normal, exp(mu + 0.5 z), with mu in [-2, 6]
//     (prices, volumes, amounts), or normal, sigma z with sigma in [0.01, 1]
//     (returns, flows), chosen by a hash of the field name;
//   - the target is 0.02 z;
//   - z ~ N(0, 1) is drawn independently for every cell, and 2% of the cells
//     are missing (NaN), also independently.
// Any field name is accepted, so real expression batches run unchanged.
struct PanelSet {
    int T = 0;
    int N = 0;
    std::vector<std::string> dates;
    std::vector<std::string> assets;
    std::vector<std::string> fields;   // all fields listed in meta.json
    std::string dir;
    int N_file = 0;                    // assets per row in the files
    bool synthetic = false;
    unsigned long long seed = 0;

    static PanelSet open(const std::string& dir);
    static PanelSet make_synthetic(int T, int N, unsigned long long seed = 1);
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
