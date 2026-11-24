// CUDA root scorer: daily RankIC + PCA-QD signature of finished root panels.
//
// Bit-for-bit target: score_root() in src/scoring.cpp. Everything that is a
// floating-point reduction runs as the same sequential loop as the CPU code
// (one thread per sum; reduce_kernel runs all days' sums at once), and the build disables FMA contraction on both
// sides (nvcc --fmad=false, host -ffp-contract=off). What is parallel is the
// part with exact results: the per-day sort (cub::BlockRadixSort), average
// ranks (integer scans), element-wise row writes, and independent rows /
// columns of the power-iteration products. These reductions are short
// compared with the sort and the 34 MB host-to-device copy per panel.
#include "gpp/gpu.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include "core.hpp"  // pca::finish (host, unchanged)

namespace gpp {
namespace {

#define GPP_CUDA(x)                                                                                      \
    do {                                                                                                 \
        cudaError_t e_ = (x);                                                                            \
        if (e_ != cudaSuccess)                                                                           \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + " in " #x);   \
    } while (0)

constexpr int kThreads = 512;

struct MaxOp {
    __device__ int operator()(int a, int b) const { return a > b ? a : b; }
};
struct NotEqual {
    __device__ bool operator()(const float& a, const float& b) const { return a != b; }
};

struct DayArgs {
    const float* panel;       // T x N
    const double* target;     // T x N target ranks (NaN where the target is NaN)
    int N, begin, days;
    int sig_step, m;          // signature sampling; m = sampled rows (0: none)
    double* ic;               // days
    unsigned long long* valid;
    double* R;                // m x N  (row-major, for multiply_t)
    double* RT;               // N x m  (for multiply)
    double* ranks;            // days x N average ranks (NaN where the factor is NaN)
    double* means;            // m: mean of the sampled rank rows (NaN -> 0.5)
};

// One block per scored day.
template <int ITEMS>
__global__ void __launch_bounds__(kThreads) day_kernel(DayArgs a) {
    using Sort = cub::BlockRadixSort<float, kThreads, ITEMS, int>;
    using Disc = cub::BlockDiscontinuity<float, kThreads>;
    using Scan = cub::BlockScan<int, kThreads>;
    struct After {
        typename Disc::TempStorage disc;
        typename Scan::TempStorage scan;
        int run_end[kThreads * ITEMS];
    };
    __shared__ union {
        typename Sort::TempStorage sort;
        After after;
    } tmp;
    __shared__ int s_nvalid, s_pairs;
    double* rank = a.ranks + static_cast<size_t>(blockIdx.x) * a.N;  // global row of this day

    const int d = blockIdx.x, t = a.begin + d, N = a.N;
    const float* f = a.panel + static_cast<size_t>(t) * N;
    const double* y = a.target + static_cast<size_t>(t) * N;
    const float qnan = __int_as_float(0x7fc00000);  // sorts after +inf
    if (threadIdx.x == 0) { s_nvalid = 0; s_pairs = 0; }
    for (int s = threadIdx.x; s < N; s += kThreads) rank[s] = CUDART_NAN;
    __syncthreads();

    float keys[ITEMS];
    int vals[ITEMS];
    int nv = 0, np = 0;
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int idx = threadIdx.x * ITEMS + k;
        if (idx < N) {
            const float v = f[idx];
            const bool ok = !isnan(v);
            keys[k] = ok ? v : qnan;
            vals[k] = idx;
            nv += ok;
            np += ok && !isnan(y[idx]);
        } else {
            keys[k] = qnan;
            vals[k] = -1;
        }
    }
    if (nv) atomicAdd(&s_nvalid, nv);
    if (np) atomicAdd(&s_pairs, np);
    Sort(tmp.sort).Sort(keys, vals);  // blocked in, blocked out; -0/+0 adjacent
    __syncthreads();
    const int nvalid = s_nvalid;

    // Runs of equal values: start = max-scan of head positions, end written
    // at the start position by the run's tail.
    int heads[ITEMS], tails[ITEMS], start[ITEMS];
    Disc(tmp.after.disc).FlagHeads(heads, keys, NotEqual());
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) start[k] = heads[k] ? threadIdx.x * ITEMS + k : 0;
    Scan(tmp.after.scan).InclusiveScan(start, start, MaxOp());
    __syncthreads();
    Disc(tmp.after.disc).FlagTails(tails, keys, NotEqual());
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int pos = threadIdx.x * ITEMS + k;
        if (pos < nvalid && tails[k]) tmp.after.run_end[start[k]] = pos;
    }
    __syncthreads();
    const double n = static_cast<double>(nvalid);
    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int pos = threadIdx.x * ITEMS + k;
        if (pos < nvalid) {
            const int e = tmp.after.run_end[start[k]];
            // rank_row_pct: avg = 0.5 * ((i + 1) + (j + 1)); out = avg / n
            const double avg = 0.5 * static_cast<double>((start[k] + 1) + (e + 1));
            rank[vals[k]] = avg / n;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(a.valid, static_cast<unsigned long long>(s_pairs));
}

// The sequential reductions of one day, one thread per day: pearson_pairwise
// with the CPU's loops, and (sampled days) the sum for the row mean. Each sum
// keeps its own accumulator in the CPU's order, so the results are bitwise the
// CPU's; running all days' chains at once replaces a block per day in which
// only thread 0 worked.
__global__ void reduce_kernel(DayArgs a) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= a.days) return;
    const int N = a.N;
    const double* rank = a.ranks + static_cast<size_t>(d) * N;
    const double* y = a.target + static_cast<size_t>(a.begin + d) * N;
    const bool sampled = a.m > 0 && d % a.sig_step == 0;
    double sx = 0.0, sy = 0.0, x0 = 0.0, y0 = 0.0, sum = 0.0;
    int cnt = 0;
    bool xc = true, yc = true;
    for (int s = 0; s < N; ++s) {
        const double xv = rank[s], yv = y[s];
        if (sampled) sum += isnan(xv) ? 0.5 : xv;
        if (isnan(xv) || isnan(yv)) continue;
        if (cnt == 0) { x0 = xv; y0 = yv; }
        xc = xc && xv == x0;
        yc = yc && yv == y0;
        sx += xv; sy += yv; ++cnt;
    }
    if (sampled) a.means[d / a.sig_step] = sum / N;
    double r = CUDART_NAN;
    if (!(cnt < 2 || xc || yc)) {
        const double mx = sx / cnt, my = sy / cnt;
        double sxy = 0.0, sxx = 0.0, syy = 0.0;
        for (int s = 0; s < N; ++s) {
            const double xv = rank[s], yv = y[s];
            if (isnan(xv) || isnan(yv)) continue;
            const double dx = xv - mx, dy = yv - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        const double dom = (cnt - 1) * sqrt(sxx / (cnt - 1)) * sqrt(syy / (cnt - 1));
        r = sxy / dom;
        if (isinf(r)) r = CUDART_NAN;
    }
    a.ic[d] = r;
}

// Centred sampled rows for the power iteration, one block per sampled day.
__global__ void signature_rows_kernel(DayArgs a) {
    const int q = blockIdx.x, N = a.N;
    const double* rank = a.ranks + static_cast<size_t>(q) * a.sig_step * N;
    const double mean = a.means[q];
    for (int s = threadIdx.x; s < N; s += blockDim.x) {
        const double v = (isnan(rank[s]) ? 0.5 : rank[s]) - mean;
        a.R[static_cast<size_t>(q) * N + s] = v;
        a.RT[static_cast<size_t>(s) * a.m + q] = v;
    }
}

// pca::multiply: out[i] = sum_j R[i][j] v[j], j ascending (RT for coalescing).
__global__ void multiply_kernel(const double* RT, const double* v, double* out, int m, int N, const int* flag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || *flag) return;
    double s = 0.0;
    for (int j = 0; j < N; ++j) s += RT[static_cast<size_t>(j) * m + i] * v[j];
    out[i] = s;
}

// pca::multiply_t: out[j] = 0 + sum_i R[i][j] u[i], i ascending.
__global__ void multiply_t_kernel(const double* R, const double* u, double* out, int m, int N, const int* flag) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N || *flag) return;
    double r = 0.0;
    for (int i = 0; i < m; ++i) r += R[static_cast<size_t>(i) * N + j] * u[i];
    out[j] = r;
}

// pca::unit: sequential dot, sqrt, element-wise division; flag on failure.
__global__ void unit_kernel(double* a, int n, int* flag) {
    __shared__ double s_norm;
    __shared__ int s_ok;
    if (threadIdx.x == 0) {  // only thread 0 touches the flag: no early exits before the barrier
        s_ok = 0;
        if (!*flag) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += a[i] * a[i];
            const double nrm = sqrt(s);
            s_norm = nrm;
            s_ok = (nrm > 0) && isfinite(nrm);
            if (!s_ok) *flag = 1;
        }
    }
    __syncthreads();
    if (!s_ok) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] = a[i] / s_norm;
}

// Host copy of pca::unit (same arithmetic; compiled without contraction).
bool unit_host(std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    const double n = std::sqrt(s);
    if (!(n > 0) || !std::isfinite(n)) return false;
    for (double& x : a) x /= n;
    return true;
}

}  // namespace

struct GpuScorer::Impl {
    struct Device {
        int id = 0;
        double* target = nullptr;
        double* initial = nullptr;  // unit initial vector
    };
    struct Slot {
        std::mutex mu;
        int device = 0;
        cudaStream_t stream = nullptr;
        float* h_panel = nullptr;
        float* d_panel = nullptr;
        double *d_ranks = nullptr, *d_means = nullptr;
        double *d_R = nullptr, *d_RT = nullptr, *d_left = nullptr, *d_right = nullptr, *d_ic = nullptr;
        unsigned long long* d_valid = nullptr;
        int* d_flag = nullptr;
        double* h_ic = nullptr;
        double* h_sig = nullptr;
        unsigned long long* h_valid = nullptr;
        int* h_flag = nullptr;
        RootScore* pending = nullptr;
        int depth = 0, nodes = 0;
    };

    int T, N, days, m, items;
    ScoreOptions options;
    long eligible = 0;
    bool initial_ok = true;
    std::vector<Device> devices;
    std::vector<std::unique_ptr<Slot>> slots;
    std::atomic<unsigned> next{0};
    std::mutex time_mu;
    double busy = 0.0;

    void launch(Slot& s) {
        DayArgs a{s.d_panel, devices[s.device].target, N, options.begin, days, options.signature_step, m,
                  s.d_ic, s.d_valid, s.d_R, s.d_RT, s.d_ranks, s.d_means};
        if (items == 4) day_kernel<4><<<days, kThreads, 0, s.stream>>>(a);
        else if (items == 8) day_kernel<8><<<days, kThreads, 0, s.stream>>>(a);
        else day_kernel<12><<<days, kThreads, 0, s.stream>>>(a);
        reduce_kernel<<<(days + 63) / 64, 64, 0, s.stream>>>(a);
        if (m > 0) signature_rows_kernel<<<m, 256, 0, s.stream>>>(a);
        GPP_CUDA(cudaGetLastError());
        if (m > 0) {
            // pca::solve(power): right = unit(initial); steps x (left = unit(R right);
            // right = unit(R^T left)); local = unit(R right).
            const Device& dev = devices[s.device];
            GPP_CUDA(cudaMemcpyAsync(s.d_right, dev.initial, N * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
            const int bm = (m + 127) / 128, bn = (N + 127) / 128;
            for (int k = 0; k < options.power_steps; ++k) {
                multiply_kernel<<<bm, 128, 0, s.stream>>>(s.d_RT, s.d_right, s.d_left, m, N, s.d_flag);
                unit_kernel<<<1, 128, 0, s.stream>>>(s.d_left, m, s.d_flag);
                multiply_t_kernel<<<bn, 128, 0, s.stream>>>(s.d_R, s.d_left, s.d_right, m, N, s.d_flag);
                unit_kernel<<<1, 128, 0, s.stream>>>(s.d_right, N, s.d_flag);
            }
            multiply_kernel<<<bm, 128, 0, s.stream>>>(s.d_RT, s.d_right, s.d_left, m, N, s.d_flag);
            unit_kernel<<<1, 128, 0, s.stream>>>(s.d_left, m, s.d_flag);
            GPP_CUDA(cudaGetLastError());
            GPP_CUDA(cudaMemcpyAsync(s.h_sig, s.d_left, m * sizeof(double), cudaMemcpyDeviceToHost, s.stream));
            GPP_CUDA(cudaMemcpyAsync(s.h_flag, s.d_flag, sizeof(int), cudaMemcpyDeviceToHost, s.stream));
        }
        GPP_CUDA(cudaMemcpyAsync(s.h_ic, s.d_ic, days * sizeof(double), cudaMemcpyDeviceToHost, s.stream));
        GPP_CUDA(cudaMemcpyAsync(s.h_valid, s.d_valid, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s.stream));
    }

    // Slot lock held by the caller.
    void harvest(Slot& s) {
        if (!s.pending) return;
        auto t0 = std::chrono::steady_clock::now();
        GPP_CUDA(cudaSetDevice(devices[s.device].id));
        GPP_CUDA(cudaStreamSynchronize(s.stream));
        {
            std::lock_guard<std::mutex> lock(time_mu);
            busy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        RootScore& out = *s.pending;
        std::vector<double> ic(s.h_ic, s.h_ic + days);
        finish_from_daily(ic, eligible, static_cast<long>(*s.h_valid), s.depth, s.nodes, options.fitness, out);
        out.rank_vector.clear();
        out.signature_valid = false;
        out.signature.clear();
        if (m > 0 && (out.candidate.valid || !options.signature_only_if_valid) && initial_ok && *s.h_flag == 0) {
            // power_signature after solve(): pca::finish + comparability, on the host.
            out.signature = pca::finish(std::vector<double>(s.h_sig, s.h_sig + m));
            out.signature_valid = !out.signature.empty() && signature_comparable(out.signature);
            if (!out.signature_valid) out.signature.clear();
        }
        s.pending = nullptr;
    }
};

GpuScorer::GpuScorer(int gpus, int streams_per_gpu, int T, int N, const std::vector<double>& target_rank,
                     const ScoreOptions& options, const std::vector<double>& initial)
    : impl_(new Impl) {
    Impl& I = *impl_;
    const int available = gpu_count();
    if (gpus < 1 || available < 1) throw std::invalid_argument("GpuScorer: no CUDA device");
    gpus = std::min(gpus, available);
    if (options.rank_vector_step > 0) throw std::invalid_argument("GpuScorer: rank vectors are CPU-only");
    if (N > kThreads * 12) throw std::invalid_argument("GpuScorer: N > 6144 stocks is not supported");
    I.T = T;
    I.N = N;
    I.options = options;
    I.days = T - options.begin;
    if (I.days <= 0) throw std::invalid_argument("GpuScorer: nothing to score");
    I.m = sampled_rows(I.days, options.signature_step);
    I.items = N <= kThreads * 4 ? 4 : N <= kThreads * 8 ? 8 : 12;
    for (int t = options.begin; t < T; ++t)
        for (int s = 0; s < N; ++s)
            if (!std::isnan(target_rank[static_cast<size_t>(t) * N + s])) ++I.eligible;

    const size_t panel = static_cast<size_t>(T) * N;
    I.devices.resize(gpus);
    for (int g = 0; g < gpus; ++g) {
        Impl::Device& dev = I.devices[g];
        dev.id = g;
        GPP_CUDA(cudaSetDevice(g));
        GPP_CUDA(cudaMalloc(&dev.target, panel * sizeof(double)));
        GPP_CUDA(cudaMemcpy(dev.target, target_rank.data(), panel * sizeof(double), cudaMemcpyHostToDevice));
        GPP_CUDA(cudaMalloc(&dev.initial, N * sizeof(double)));
    }
    const int per = std::max(1, streams_per_gpu);
    for (int k = 0; k < gpus * per; ++k) {
        auto slot = std::make_unique<Impl::Slot>();
        Impl::Slot& s = *slot;
        s.device = k % gpus;
        GPP_CUDA(cudaSetDevice(I.devices[s.device].id));
        GPP_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        GPP_CUDA(cudaMallocHost(&s.h_panel, panel * sizeof(float)));
        GPP_CUDA(cudaMalloc(&s.d_panel, panel * sizeof(float)));
        const size_t rm = static_cast<size_t>(std::max(I.m, 1)) * N;
        GPP_CUDA(cudaMalloc(&s.d_R, rm * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_RT, rm * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_left, std::max(I.m, 1) * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_right, N * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_ic, I.days * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_ranks, static_cast<size_t>(I.days) * N * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_means, std::max(I.m, 1) * sizeof(double)));
        GPP_CUDA(cudaMalloc(&s.d_valid, sizeof(unsigned long long)));
        GPP_CUDA(cudaMalloc(&s.d_flag, sizeof(int)));
        GPP_CUDA(cudaMallocHost(&s.h_ic, I.days * sizeof(double)));
        GPP_CUDA(cudaMallocHost(&s.h_sig, std::max(I.m, 1) * sizeof(double)));
        GPP_CUDA(cudaMallocHost(&s.h_valid, sizeof(unsigned long long)));
        GPP_CUDA(cudaMallocHost(&s.h_flag, sizeof(int)));
        *s.h_flag = 0;
        I.slots.push_back(std::move(slot));
    }
    set_initial(initial);
}

GpuScorer::~GpuScorer() {
    if (!impl_) return;
    try { flush(); } catch (...) {}
    for (auto& p : impl_->slots) {
        Impl::Slot& s = *p;
        cudaSetDevice(impl_->devices[s.device].id);
        cudaStreamDestroy(s.stream);
        cudaFreeHost(s.h_panel); cudaFree(s.d_panel); cudaFree(s.d_R); cudaFree(s.d_RT);
        cudaFree(s.d_left); cudaFree(s.d_right); cudaFree(s.d_ic); cudaFree(s.d_ranks); cudaFree(s.d_means); cudaFree(s.d_valid); cudaFree(s.d_flag);
        cudaFreeHost(s.h_ic); cudaFreeHost(s.h_sig); cudaFreeHost(s.h_valid); cudaFreeHost(s.h_flag);
    }
    for (auto& dev : impl_->devices) {
        cudaSetDevice(dev.id);
        cudaFree(dev.target);
        cudaFree(dev.initial);
    }
}

void GpuScorer::set_initial(const std::vector<double>& initial) {
    Impl& I = *impl_;
    flush();
    if (static_cast<int>(initial.size()) != I.N) throw std::invalid_argument("GpuScorer: initial vector size");
    std::vector<double> right = initial;
    I.initial_ok = unit_host(right);
    for (auto& dev : I.devices) {
        GPP_CUDA(cudaSetDevice(dev.id));
        GPP_CUDA(cudaMemcpy(dev.initial, right.data(), I.N * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void GpuScorer::submit(const gpe::Panel& panel, int depth, int nodes, RootScore* out) {
    Impl& I = *impl_;
    if (panel.T != I.T || panel.N != I.N || panel.lo != 0) throw std::invalid_argument("GpuScorer: panel shape");
    Impl::Slot& s = *I.slots[I.next.fetch_add(1) % I.slots.size()];
    std::lock_guard<std::mutex> lock(s.mu);
    I.harvest(s);  // the slot's previous root (also frees the staging buffer)
    GPP_CUDA(cudaSetDevice(I.devices[s.device].id));
    std::memcpy(s.h_panel, panel.buf.data(), static_cast<size_t>(I.T) * I.N * sizeof(float));
    GPP_CUDA(cudaMemcpyAsync(s.d_panel, s.h_panel, static_cast<size_t>(I.T) * I.N * sizeof(float),
                             cudaMemcpyHostToDevice, s.stream));
    GPP_CUDA(cudaMemsetAsync(s.d_valid, 0, sizeof(unsigned long long), s.stream));
    GPP_CUDA(cudaMemsetAsync(s.d_flag, 0, sizeof(int), s.stream));
    *s.h_flag = 0;
    I.launch(s);
    s.pending = out;
    s.depth = depth;
    s.nodes = nodes;
}

void GpuScorer::flush() {
    for (auto& p : impl_->slots) {
        std::lock_guard<std::mutex> lock(p->mu);
        impl_->harvest(*p);
    }
}

double GpuScorer::busy_seconds() const { return impl_->busy; }

int gpu_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

}  // namespace gpp
