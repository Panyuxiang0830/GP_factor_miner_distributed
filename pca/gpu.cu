#include "gpu.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pca {
namespace {
void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(size_t n) { if (n) check(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
};
struct Event {
    cudaEvent_t begin, end;
    Event() { check(cudaEventCreate(&begin)); check(cudaEventCreate(&end)); }
    ~Event() { cudaEventDestroy(begin); cudaEventDestroy(end); }
    void start() { check(cudaEventRecord(begin)); }
    double stop() {
        check(cudaGetLastError()); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        float ms; check(cudaEventElapsedTime(&ms, begin, end)); return ms / 1000.;
    }
};
template<class T> __device__ T sum_block(T value) {
    __shared__ T workspace[256];
    workspace[threadIdx.x] = value; __syncthreads();
    for (int distance = 128; distance; distance /= 2) {
        if (threadIdx.x < distance) workspace[threadIdx.x] += workspace[threadIdx.x + distance];
        __syncthreads();
    }
    return workspace[0];
}
template<class T> __global__ void rv(const T* R, const T* V, T* U, int m, int n) {
    int b = blockIdx.y, row = blockIdx.x;
    T value = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) value += R[(size_t(b) * m + row) * n + j] * V[size_t(b) * n + j];
    value = sum_block(value);
    if (threadIdx.x == 0) U[size_t(b) * m + row] = value;
}
template<class T> __global__ void rtu(const T* R, const T* U, T* V, int m, int n) {
    int b = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    T value = 0;
    for (int i = 0; i < m; ++i) value += R[(size_t(b) * m + i) * n + j] * U[size_t(b) * m + i];
    V[size_t(b) * n + j] = value;
}
template<class T> __global__ void normalize(T* vectors, int dimension) {
    T value = 0; size_t base = size_t(blockIdx.x) * dimension;
    for (int i = threadIdx.x; i < dimension; i += blockDim.x) value += vectors[base + i] * vectors[base + i];
    T length = sqrt(sum_block(value));
    for (int i = threadIdx.x; i < dimension; i += blockDim.x)
        vectors[base + i] = length > 0 ? vectors[base + i] / length : 0;
}
template<class T> __global__ void squared_norms(const T* vectors, int dimension, T* norms) {
    T value = 0; size_t base = size_t(blockIdx.x) * dimension;
    for (int i = threadIdx.x; i < dimension; i += blockDim.x) value += vectors[base + i] * vectors[base + i];
    value = sum_block(value);
    if (threadIdx.x == 0) norms[blockIdx.x] = value;
}
template<class T> __global__ void normalize_with_norms(T* vectors, int dimension, const T* norms) {
    T length = sqrt(norms[blockIdx.x]); size_t base = size_t(blockIdx.x) * dimension;
    for (int i = threadIdx.x; i < dimension; i += blockDim.x)
        vectors[base + i] = length > 0 ? vectors[base + i] / length : 0;
}
template<class T> std::vector<std::vector<float>> power_impl(const std::vector<Matrix>& matrices,
    const std::vector<double>& initial, int steps, GpuTiming& timing, const DeviceReduce& device_sum) {
    int B = int(matrices.size()), m = matrices[0].rows, n = matrices[0].cols;
    std::vector<T> hostR(size_t(B) * m * n), hostV(size_t(B) * n);
    for (int b = 0; b < B; ++b) {
        if (matrices[b].rows != m || matrices[b].cols != n) throw std::invalid_argument("GPU batch shape mismatch");
        std::copy(matrices[b].data.begin(), matrices[b].data.end(), hostR.begin() + size_t(b) * m * n);
        std::copy(initial.begin(), initial.end(), hostV.begin() + size_t(b) * n);
    }
    Device<T> R(hostR.size()), V(hostV.size()), U(size_t(B) * m), norms(B);
    Event event; event.start();
    check(cudaMemcpy(R.data, hostR.data(), hostR.size() * sizeof(T), cudaMemcpyHostToDevice));
    check(cudaMemcpy(V.data, hostV.data(), hostV.size() * sizeof(T), cudaMemcpyHostToDevice));
    timing.upload += event.stop();
    Event total; total.start();
    normalize<<<B, 256>>>(V.data, n);
    for (int k = 0; k <= steps; ++k) {
        event.start(); if (m) rv<<<dim3(m, B), 256>>>(R.data, V.data, U.data, m, n); timing.gemv += event.stop();
        if (device_sum) {
            squared_norms<<<B, 256>>>(U.data, m, norms.data);
            check(cudaDeviceSynchronize()); device_sum(norms.data, B, sizeof(T) == sizeof(double));
            normalize_with_norms<<<B, 256>>>(U.data, m, norms.data);
        } else normalize<<<B, 256>>>(U.data, m);
        if (k == steps) break;
        event.start(); rtu<<<dim3((n + 255) / 256, B), 256>>>(R.data, U.data, V.data, m, n); timing.gemv += event.stop();
        if (device_sum) { check(cudaDeviceSynchronize()); device_sum(V.data, B * n, sizeof(T) == sizeof(double)); }
        normalize<<<B, 256>>>(V.data, n);
    }
    timing.iteration += total.stop();
    std::vector<T> hostU(size_t(B) * m);
    event.start(); check(cudaMemcpy(hostU.data(), U.data, hostU.size() * sizeof(T), cudaMemcpyDeviceToHost));
    timing.download += event.stop();
    timing.bytes += double(2 * steps + 1) * B * m * n * sizeof(T);
    timing.flops += double(2 * steps + 1) * B * m * n * 2;
    std::vector<std::vector<float>> result(B);
    for (int b = 0; b < B; ++b) {
        std::vector<double> local(hostU.begin() + size_t(b) * m, hostU.begin() + size_t(b + 1) * m);
        result[b] = device_sum ? std::vector<float>(local.begin(), local.end()) : finish(local);
    }
    return result;
}

// One padded bitonic network per row. Tie order uses original indices;
// missing values and padding sort after all valid values, including +infinity.
__device__ bool less_pair(float a, int ia, float b, int ib) {
    bool na = isnan(a), nb = isnan(b);
    if (na != nb) return !na;
    if (!na && a != b) return a < b;
    return ia < ib;
}
__global__ void rank_rows_kernel(const float* input, float* output, int n, int padded, bool pca_preprocess) {
    extern __shared__ unsigned char memory[];
    float* value = reinterpret_cast<float*>(memory);
    int* index = reinterpret_cast<int*>(value + padded);
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        value[i] = i < n ? input[size_t(row) * n + i] : nanf(""); index[i] = i;
    }
    __syncthreads();
    for (int size = 2; size <= padded; size *= 2)
        for (int distance = size / 2; distance; distance /= 2) {
            for (int i = threadIdx.x; i < padded; i += blockDim.x) {
                int other = i ^ distance;
                if (other > i) {
                    bool swap = (i & size) == 0 ? less_pair(value[other], index[other], value[i], index[i]) :
                                                              less_pair(value[i], index[i], value[other], index[other]);
                    if (swap) { float v = value[i]; value[i] = value[other]; value[other] = v;
                        int j = index[i]; index[i] = index[other]; index[other] = j; }
                }
            }
            __syncthreads();
        }
    __shared__ int valid;
    if (threadIdx.x == 0) { valid = 0; while (valid < n && !isnan(value[valid])) ++valid; }
    __syncthreads();
    double localmean = 0;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        if (index[i] >= n) continue;
        double rank = pca_preprocess ? .5 : nan("");
        if (i < valid) {
            // Binary searches for equal-value interval, so all-tied rows stay O(n log n).
            int lo = 0, hi = valid;
            while (lo < hi) { int mid = (lo + hi) / 2; if (value[mid] < value[i]) lo = mid + 1; else hi = mid; }
            int first = lo; lo = 0; hi = valid;
            while (lo < hi) { int mid = (lo + hi) / 2; if (value[mid] <= value[i]) lo = mid + 1; else hi = mid; }
            rank = (first + lo + 1.) / (2 * valid);
        }
        output[size_t(row) * n + index[i]] = float(rank);
        if (pca_preprocess) localmean += rank;
    }
    if (pca_preprocess) {
        double mean = sum_block(localmean) / n;
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) output[size_t(row) * n + j] -= float(mean);
    }
}
__global__ void tiled_similarity(const float* left, const float* right, float* out, int L, int A, int d) {
    __shared__ float x[16][16], y[16][16];
    int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < d; base += 16) {
        int kx = base + threadIdx.x, ky = base + threadIdx.y;
        x[threadIdx.y][threadIdx.x] = row < L && kx < d ? left[size_t(row) * d + kx] : 0;
        y[threadIdx.y][threadIdx.x] = col < A && ky < d ? right[size_t(col) * d + ky] : 0;
        __syncthreads();
        for (int k = 0; k < 16; ++k) sum += x[threadIdx.y][k] * y[k][threadIdx.x];
        __syncthreads();
    }
    if (row < L && col < A) out[size_t(row) * A + col] = fabsf(sum);
}
}  // namespace

void gpu_select(int local_rank) {
    int devices; check(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("no CUDA devices");
    check(cudaSetDevice(local_rank % devices));
}
std::vector<std::vector<float>> gpu_power(const std::vector<Matrix>& matrices,
    const std::vector<double>& initial, int steps, bool fp64, GpuTiming& timing, const DeviceReduce& device_sum) {
    if (matrices.empty()) return {};
    int device; check(cudaGetDevice(&device)); cudaDeviceProp properties; check(cudaGetDeviceProperties(&properties, device));
    timing.device = properties.name;
    timing.nominal_memory_bytes_per_second = 2. * properties.memoryClockRate * 1000. * properties.memoryBusWidth / 8;
    timing.multiprocessors = properties.multiProcessorCount; timing.clock_khz = properties.clockRate;
    return fp64 ? power_impl<double>(matrices, initial, steps, timing, device_sum) : power_impl<float>(matrices, initial, steps, timing, device_sum);
}
static std::vector<float> rank_impl(const std::vector<float>& input, int rows, int stocks, GpuTiming& timing, bool pca_preprocess) {
    if (rows < 0 || stocks < 1 || stocks > 8192 || input.size() != size_t(rows) * stocks) throw std::invalid_argument("invalid GPU ranking shape");
    if (rows == 0) return {};
    int padded = 1; while (padded < stocks) padded *= 2;
    size_t shared = size_t(padded) * (sizeof(float) + sizeof(int));
    check(cudaFuncSetAttribute(rank_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    Device<float> in(input.size()), out(input.size()); Event event; event.start();
    check(cudaMemcpy(in.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    timing.upload += event.stop(); event.start();
    rank_rows_kernel<<<rows, 256, shared>>>(in.data, out.data, stocks, padded, pca_preprocess);
    timing.ranking += event.stop(); event.start();
    std::vector<float> result(input.size());
    check(cudaMemcpy(result.data(), out.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    timing.download += event.stop(); return result;
}
std::vector<float> gpu_rank_rows(const std::vector<float>& input, int rows, int stocks, GpuTiming& timing) {
    return rank_impl(input, rows, stocks, timing, true);
}
std::vector<float> gpu_percentile_rows(const std::vector<float>& input, int rows, int stocks, GpuTiming& timing) {
    return rank_impl(input, rows, stocks, timing, false);
}
std::vector<float> gpu_similarity(const std::vector<float>& left, int L,
    const std::vector<float>& right, int A, int d, GpuTiming& timing) {
    if (left.size() != size_t(L) * d || right.size() != size_t(A) * d) throw std::invalid_argument("GPU similarity shape");
    if (L == 0 || A == 0) return {};
    Device<float> x(left.size()), y(right.size()), output(size_t(L) * A);
    Event event; event.start();
    check(cudaMemcpy(x.data, left.data(), left.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(y.data, right.data(), right.size() * 4, cudaMemcpyHostToDevice));
    timing.upload += event.stop(); event.start();
    tiled_similarity<<<dim3((A + 15) / 16, (L + 15) / 16), dim3(16, 16)>>>(x.data, y.data, output.data, L, A, d);
    timing.similarity += event.stop(); event.start();
    std::vector<float> result(size_t(L) * A);
    check(cudaMemcpy(result.data(), output.data, result.size() * 4, cudaMemcpyDeviceToHost));
    timing.download += event.stop(); return result;
}
}  // namespace pca

// Optional RankIC interoperability entrypoint. Existing PCA kernels and their
// callers above remain unchanged. An average rank is q/(2*valid), with integer
// q. For valid<=8192, the FP32 rounding error multiplied by 2*valid is <1/2,
// so q is recovered exactly. Divide its exact half-integer value in FP64 to
// match the CPU rank_rows_pct arithmetic, without another sorting kernel.
namespace pca {
namespace {
__global__ void recover_rank64(const float* coarse,double* output,int n) {
    size_t base=size_t(blockIdx.x)*n;int count=0;
    for(int j=threadIdx.x;j<n;j+=blockDim.x)count+=!isnan(coarse[base+j]);
    int valid=sum_block(count);
    for(int j=threadIdx.x;j<n;j+=blockDim.x){
        float x=coarse[base+j];
        if(isnan(x))output[base+j]=nan("");
        else {double q=floor(double(x)*(2*valid)+.5);output[base+j]=__ddiv_rn(.5*q,double(valid));}
    }
}
}
std::vector<double> gpu_percentile_rows_f64(const std::vector<float>& input,int rows,int stocks,GpuTiming& timing) {
    if(rows<0||stocks<1||stocks>8192||input.size()!=size_t(rows)*stocks)throw std::invalid_argument("invalid GPU ranking shape");
    if(!rows)return {};
    int padded=1;while(padded<stocks)padded*=2;
    size_t shared=size_t(padded)*(sizeof(float)+sizeof(int));
    check(cudaFuncSetAttribute(rank_rows_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
    Device<float> in(input.size()),coarse(input.size());Device<double> out(input.size());Event event;event.start();
    check(cudaMemcpy(in.data,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));timing.upload+=event.stop();event.start();
    rank_rows_kernel<<<rows,256,shared>>>(in.data,coarse.data,stocks,padded,false);
    recover_rank64<<<rows,256>>>(coarse.data,out.data,stocks);timing.ranking+=event.stop();event.start();
    std::vector<double> result(input.size());check(cudaMemcpy(result.data(),out.data,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    timing.download+=event.stop();return result;
}
}
