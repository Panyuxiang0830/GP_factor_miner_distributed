// Isolated reference executable. Including the production implementation makes
// its exact internal kernel available here without changing the production API.
// This target does NOT link pca_gpu (the implementation is in this TU instead).
#include "gpu.cu"
#include "json_output.hpp"
#include <cublas_v2.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

namespace reference {
void check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(code)));
}
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
std::vector<float> signatures(int rows, int d, uint64_t offset) {
    std::vector<float> values(size_t(rows) * d);
    for (int row = 0; row < rows; ++row) {
        std::vector<double> x(d);
        for (int j = 0; j < d; ++j)
            x[j] = double(mix(offset + size_t(row) * d + j) >> 11) / 9007199254740992. - .5;
        double mean = std::accumulate(x.begin(), x.end(), 0.) / d, length = 0;
        for (double& v : x) { v -= mean; length += v*v; }
        length = std::sqrt(length);
        for (int j = 0; j < d; ++j) values[size_t(row)*d+j] = float(x[j]/length);
    }
    return values;
}
__global__ void absolute(float* x, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) x[i] = fabsf(x[i]);
}
gpe::Json number(double n) { gpe::Json x; x.type=gpe::Json::Type::Number; x.number=n; return x; }
gpe::Json label(const std::string& s) { gpe::Json x; x.type=gpe::Json::Type::String; x.string=s; return x; }
gpe::Json flag(bool yes) { gpe::Json x; x.type=gpe::Json::Type::Bool; x.boolean=yes; return x; }
}

int main(int argc, char** argv) {
    cublasHandle_t handle = nullptr;
    try {
        if (argc != 5) throw std::runtime_error("usage: pca_cublas_reference B A dimension new_output.json");
        int B=std::stoi(argv[1]), A=std::stoi(argv[2]), d=std::stoi(argv[3]);
        if (B<2 || A<2 || d<2 || B>1024 || A>100000 || d>8192)
            throw std::runtime_error("invalid reference benchmark shape");
        if (std::ifstream(argv[4]).good()) throw std::runtime_error("output already exists");
        pca::gpu_select(0);
        auto left=reference::signatures(B,d,20261111), right=reference::signatures(A,d,987654321);
        std::copy(left.begin(),left.begin()+d,right.begin());
        for (int j=0;j<d;++j) right[d+j]=-left[d+j];
        pca::Device<float> x(left.size()), y(right.size()), manual(size_t(B)*A), library(size_t(B)*A);
        pca::check(cudaMemcpy(x.data,left.data(),left.size()*sizeof(float),cudaMemcpyHostToDevice));
        pca::check(cudaMemcpy(y.data,right.data(),right.size()*sizeof(float),cudaMemcpyHostToDevice));
        reference::check(cublasCreate(&handle));
        reference::check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
        reference::check(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
        float alpha=1, beta=0;
        auto hand = [&] {
            pca::tiled_similarity<<<dim3((A+15)/16,(B+15)/16),dim3(16,16)>>>(x.data,y.data,manual.data,B,A,d);
            pca::check(cudaGetLastError());
        };
        auto gemm = [&] {
            // Row-major X[B,d] Y[A,d]^T == column-major Y^T[A,d] X[d,B].
            reference::check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,A,B,d,
                &alpha,y.data,CUDA_R_32F,d,x.data,CUDA_R_32F,d,&beta,
                library.data,CUDA_R_32F,A,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        };
        auto similarity = [&] {
            gemm();
            reference::absolute<<<(size_t(B)*A+255)/256,256>>>(library.data,size_t(B)*A);
            pca::check(cudaGetLastError());
        };
        hand(); similarity(); pca::check(cudaDeviceSynchronize());
        const int inner=50;
        auto measure = [&](auto operation) {
            pca::Event timer; timer.start();
            for (int i=0;i<inner;++i) operation();
            return timer.stop()/inner;
        };
        gpe::Json repetitions; repetitions.type=gpe::Json::Type::Array;
        for (int repeat=0;repeat<4;++repeat) {
            double h,g,s;
            if (repeat%2==0) { h=measure(hand); g=measure(gemm); s=measure(similarity); }
            else { g=measure(gemm); s=measure(similarity); h=measure(hand); }
            gpe::Json row; row.type=gpe::Json::Type::Object;
            row.object["repeat"]=reference::number(repeat);
            row.object["hand_similarity_seconds"]=reference::number(h);
            row.object["cublas_gemm_seconds"]=reference::number(g);
            row.object["cublas_similarity_seconds"]=reference::number(s);
            repetitions.array.push_back(row);
        }
        std::vector<float> a(size_t(B)*A), b(a.size());
        pca::check(cudaMemcpy(a.data(),manual.data,a.size()*sizeof(float),cudaMemcpyDeviceToHost));
        pca::check(cudaMemcpy(b.data(),library.data,b.size()*sizeof(float),cudaMemcpyDeviceToHost));
        double difference=0, hand_error=0, library_error=0;
        for (size_t i=0;i<a.size();++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite product");
            difference=std::max(difference,std::fabs(double(a[i])-b[i]));
        }
        for (int sample=0;sample<512;++sample) {
            size_t flat=sample==0?0:sample==1?size_t(A)+1:reference::mix(sample)%a.size();
            size_t i=flat/A,j=flat%A; double expected=0;
            for (int k=0;k<d;++k) expected+=double(left[i*d+k])*right[j*d+k];
            expected=std::fabs(expected);
            hand_error=std::max(hand_error,std::fabs(double(a[flat])-expected));
            library_error=std::max(library_error,std::fabs(double(b[flat])-expected));
        }
        if (difference>1e-5 || hand_error>1e-5 || library_error>1e-5)
            throw std::runtime_error("FP64 sampled reference or full product difference exceeds tolerance");
        int version=0,runtime=0,driver=0;cudaDeviceProp prop;
        reference::check(cublasGetVersion(handle,&version));
        pca::check(cudaGetDeviceProperties(&prop,0));
        pca::check(cudaRuntimeGetVersion(&runtime));pca::check(cudaDriverGetVersion(&driver));
        gpe::Json out;out.type=gpe::Json::Type::Object;
        auto n=[&](const std::string& key,double value){out.object[key]=reference::number(value);};
        n("B",B);n("A",A);n("dimension",d);n("inner_iterations",inner);n("warmup_repeats",1);n("measured_repeats",3);
        n("cublas_version",version);n("cuda_runtime_version",runtime);n("cuda_driver_version",driver);
        n("max_abs_hand_vs_cublas",difference);n("sampled_fp64_hand_error",hand_error);n("sampled_fp64_cublas_error",library_error);
        n("fp64_sample_pairs",512);n("flops_per_gemm",2.*B*A*d);
        out.object["device"]=reference::label(prop.name);
        out.object["compute_mode"]=reference::label("CUBLAS_COMPUTE_32F_PEDANTIC");
        out.object["math_mode"]=reference::label("CUBLAS_PEDANTIC_MATH");
        out.object["scope"]=reference::label("Isolated synthetic reference; exact production tiled kernel, identical GPU-resident inputs, no transfer/allocation/normalization in event timings; hand fuses abs, cuBLAS similarity includes a separate abs kernel. Host launch gaps may be included.");
        out.object["passed"]=reference::flag(true);
        out.object["production_uses_cublas"]=reference::flag(false);
        out.object["repetitions"]=repetitions;
        std::ofstream file(argv[4]);if(!file)throw std::runtime_error("cannot open output");
        pca::write_json(file,out);file<<'\n';file.close();
        reference::check(cublasDestroy(handle));handle=nullptr;
        std::cout<<"PASS: isolated cuBLAS reference B="<<B<<" A="<<A<<'\n';
    } catch(const std::exception& e) {
        if(handle)cublasDestroy(handle);
        std::cerr<<e.what()<<'\n';return 1;
    }
}
