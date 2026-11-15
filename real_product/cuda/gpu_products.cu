// Similarity products on the GPUs.
//
// gpu_summaries (PCA-QD admission): coarse FP32 |similarity| tiles with
// cuBLAS SGEMM (pedantic math: no TF32), copied back per column tile and fed
// in column order to the same SummaryBuilder as the CPU path. The QD error
// bound covers any FP32 summation order, so decisions are those of the CPU
// path (exact FP64 refinement does the rest); only the coarse values differ.
//
// gpu_pair_sums (OOS screening): the six masked pair sums with FP64 DGEMM
// over chunks of the vector length, right-hand rows split across GPUs.
// Reordering error ~1e-13 relative, far inside the refinement band.
#include "gpp/gpu.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpp {
namespace {

#define GPP_CUDA(x)                                                                                      \
    do {                                                                                                 \
        cudaError_t e_ = (x);                                                                            \
        if (e_ != cudaSuccess)                                                                           \
            throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e_) + " in " #x);   \
    } while (0)
#define GPP_BLAS(x)                                                                                      \
    do {                                                                                                 \
        cublasStatus_t s_ = (x);                                                                         \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                 \
            throw std::runtime_error(std::string("cuBLAS error ") + std::to_string(int(s_)) + " in " #x);\
    } while (0)

template <class T>
struct DeviceBuffer {
    T* p = nullptr;
    size_t n = 0;
    void reserve(size_t count) {
        if (count <= n) return;
        if (p) cudaFree(p);
        GPP_CUDA(cudaMalloc(&p, count * sizeof(T)));
        n = count;
    }
    ~DeviceBuffer() { if (p) cudaFree(p); }
};

struct Context {
    int id = 0;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer<float> left, right, product;
    DeviceBuffer<double> l3, r3, sums;
    float* host = nullptr;  // pinned product tile
    size_t host_n = 0;

    explicit Context(int device) : id(device) {
        GPP_CUDA(cudaSetDevice(id));
        GPP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        GPP_BLAS(cublasCreate(&blas));
        GPP_BLAS(cublasSetStream(blas, stream));
        GPP_BLAS(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    }
    void host_reserve(size_t count) {
        if (count <= host_n) return;
        if (host) cudaFreeHost(host);
        GPP_CUDA(cudaMallocHost(&host, count * sizeof(float)));
        host_n = count;
    }
    ~Context() {
        cudaSetDevice(id);
        if (host) cudaFreeHost(host);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
};

// One context per device for the process (products are called serially).
std::vector<std::unique_ptr<Context>>& contexts(int gpus) {
    static std::vector<std::unique_ptr<Context>> all;
    const int want = std::min(gpus, gpu_count());
    if (want < 1) throw std::invalid_argument("no CUDA device");
    while (static_cast<int>(all.size()) < want) all.push_back(std::make_unique<Context>(static_cast<int>(all.size())));
    return all;
}

}  // namespace

void gpu_summaries(const SummaryInputs& in, std::vector<RowSummary>& out, int gpus) {
    out.assign(in.rows, RowSummary());
    if (in.rows == 0) return;
    std::vector<SummaryBuilder> builders(in.rows);
    if (in.cols > 0) {
        auto& ctx = contexts(gpus);
        const int G = std::min<int>(static_cast<int>(ctx.size()), (in.cols + 1023) / 1024);
        const int tile = 8192;
        const size_t rd = static_cast<size_t>(in.rows) * in.dim;
        for (int g = 0; g < G; ++g) {
            Context& c = *ctx[g];
            GPP_CUDA(cudaSetDevice(c.id));
            c.left.reserve(rd);
            c.right.reserve(static_cast<size_t>(tile) * in.dim);
            c.product.reserve(static_cast<size_t>(tile) * in.rows);
            c.host_reserve(static_cast<size_t>(tile) * in.rows);
            GPP_CUDA(cudaMemcpyAsync(c.left.p, in.left, rd * sizeof(float), cudaMemcpyHostToDevice, c.stream));
        }
        const float one = 1.0f, zero = 0.0f;
        for (int t0 = 0; t0 < in.cols; t0 += tile * G) {
            int c0s[16], ncs[16];
            for (int g = 0; g < G; ++g) {
                Context& c = *ctx[g];
                const int c0 = t0 + g * tile;
                c0s[g] = c0;
                ncs[g] = c0 < in.cols ? std::min(tile, in.cols - c0) : 0;
                if (!ncs[g]) continue;
                GPP_CUDA(cudaSetDevice(c.id));
                GPP_CUDA(cudaMemcpyAsync(c.right.p, in.right + static_cast<size_t>(c0) * in.dim,
                                         static_cast<size_t>(ncs[g]) * in.dim * sizeof(float), cudaMemcpyHostToDevice,
                                         c.stream));
                // product (nc x rows, column-major) = right_tile^T(nc x dim) * left(dim x rows):
                // product[c + i * nc] = <right[c0 + c], left[i]>, i.e. row i of the tile is contiguous.
                GPP_BLAS(cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, ncs[g], in.rows, in.dim, &one, c.right.p, in.dim,
                                     c.left.p, in.dim, &zero, c.product.p, ncs[g]));
                GPP_CUDA(cudaMemcpyAsync(c.host, c.product.p, static_cast<size_t>(ncs[g]) * in.rows * sizeof(float),
                                         cudaMemcpyDeviceToHost, c.stream));
            }
            for (int g = 0; g < G; ++g) {  // column order, as the CPU path
                if (!ncs[g]) continue;
                Context& c = *ctx[g];
                GPP_CUDA(cudaSetDevice(c.id));
                GPP_CUDA(cudaStreamSynchronize(c.stream));
                summaries_add_tile(in, c.host, c0s[g], ncs[g], builders, out);
            }
        }
    }
    for (int i = 0; i < in.rows; ++i)
        if (in.left_valid[i]) builders[i].close(out[i], in.top_k);
}

void gpu_pair_sums(const std::vector<const float*>& left, const std::vector<const float*>& right, size_t length,
                   std::vector<PairSums>& out, int gpus) {
    const size_t L = left.size(), R = right.size();
    out.assign(L * R, PairSums{0, 0, 0, 0, 0, 0});
    if (!L || !R || !length) return;
    auto& ctx = contexts(gpus);
    const int G = static_cast<int>(std::min<size_t>(ctx.size(), R));
    const size_t chunk = 8192;
    std::vector<size_t> r0(G + 1);
    for (int g = 0; g <= G; ++g) r0[g] = R * g / G;

    // Masked operands in FP64: X = x - 0.5 (0 if NaN), M = 1 (0 if NaN), XX = X * X.
    auto build = [&](const std::vector<const float*>& rows, size_t b, size_t e, size_t k0, size_t kc,
                     std::vector<double>& h) {
        const size_t n = e - b;
        h.assign(3 * n * kc, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = rows[b + i] + k0;
            double* X = h.data() + i * kc;
            double* M = h.data() + (n + i) * kc;
            double* XX = h.data() + (2 * n + i) * kc;
            for (size_t k = 0; k < kc; ++k) {
                if (std::isnan(v[k])) continue;
                const double x = double(v[k]) - 0.5;
                X[k] = x;
                M[k] = 1.0;
                XX[k] = x * x;
            }
        }
    };
    std::vector<double> hl, hr;
    for (int g = 0; g < G; ++g) {
        Context& c = *ctx[g];
        GPP_CUDA(cudaSetDevice(c.id));
        c.sums.reserve(6 * L * (r0[g + 1] - r0[g]));
        GPP_CUDA(cudaMemsetAsync(c.sums.p, 0, 6 * L * (r0[g + 1] - r0[g]) * sizeof(double), c.stream));
    }
    for (size_t k0 = 0; k0 < length; k0 += chunk) {
        const size_t kc = std::min(chunk, length - k0);
        build(left, 0, L, k0, kc, hl);
        for (int g = 0; g < G; ++g) {
            Context& c = *ctx[g];
            const size_t nr = r0[g + 1] - r0[g];
            build(right, r0[g], r0[g + 1], k0, kc, hr);
            GPP_CUDA(cudaSetDevice(c.id));
            c.l3.reserve(hl.size());
            c.r3.reserve(hr.size());
            GPP_CUDA(cudaMemcpyAsync(c.l3.p, hl.data(), hl.size() * sizeof(double), cudaMemcpyHostToDevice, c.stream));
            GPP_CUDA(cudaMemcpyAsync(c.r3.p, hr.data(), hr.size() * sizeof(double), cudaMemcpyHostToDevice, c.stream));
            const double* LX = c.l3.p;
            const double* LM = c.l3.p + L * kc;
            const double* LXX = c.l3.p + 2 * L * kc;
            const double* RX = c.r3.p;
            const double* RM = c.r3.p + nr * kc;
            const double* RXX = c.r3.p + 2 * nr * kc;
            const double one = 1.0;
            // sums[s] (nr x L, column-major): [j + i * nr] = sum_k Rop[j][k] * Lop[i][k]
            const double* ops[6][2] = {{RM, LM}, {RM, LX}, {RX, LM}, {RM, LXX}, {RXX, LM}, {RX, LX}};
            for (int s = 0; s < 6; ++s)
                GPP_BLAS(cublasDgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(nr), static_cast<int>(L),
                                     static_cast<int>(kc), &one, ops[s][0], static_cast<int>(kc), ops[s][1],
                                     static_cast<int>(kc), &one, c.sums.p + s * L * nr, static_cast<int>(nr)));
            GPP_CUDA(cudaStreamSynchronize(c.stream));  // host buffers are reused
        }
    }
    for (int g = 0; g < G; ++g) {
        Context& c = *ctx[g];
        const size_t nr = r0[g + 1] - r0[g];
        std::vector<double> h(6 * L * nr);
        GPP_CUDA(cudaSetDevice(c.id));
        GPP_CUDA(cudaMemcpy(h.data(), c.sums.p, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < L; ++i)
            for (size_t j = 0; j < nr; ++j) {
                const size_t a = j + i * nr;
                out[i * R + r0[g] + j] = PairSums{h[a], h[L * nr + a], h[2 * L * nr + a], h[3 * L * nr + a],
                                                  h[4 * L * nr + a], h[5 * L * nr + a]};
            }
    }
}

}  // namespace gpp
