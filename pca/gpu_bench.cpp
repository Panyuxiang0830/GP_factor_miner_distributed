#include "gpu.hpp"
#include "json_output.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
float value(uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;
    return float(double(x>>11)/9007199254740992.-.5);
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: pca_gpu_bench float|double B output.json");
        bool fp64=std::string(argv[1])=="double";int B=std::stoi(argv[2]),m=293,n=5876,A=16384,k=4;
        if(B<1||B>256) throw std::runtime_error("B must be 1..256");
        pca::gpu_select(0);pca::GpuTiming timing;
        std::vector<float> raw(size_t(B)*m*n);
        for(size_t i=0;i<raw.size();++i) raw[i]=(i%101==0)?std::numeric_limits<float>::quiet_NaN():value(i);
        auto R=pca::gpu_rank_rows(raw,B*m,n,timing);
        raw.clear();raw.shrink_to_fit();
        std::vector<pca::Matrix> matrices(B);
        for(int b=0;b<B;++b) {
            matrices[b].rows=m;matrices[b].cols=n;
            matrices[b].data.assign(R.begin()+size_t(b)*m*n,R.begin()+size_t(b+1)*m*n);
        }
        std::vector<double> initial(n);for(int j=0;j<n;++j) initial[j]=value(j+0x87654321ULL);
        auto signatures=pca::gpu_power(matrices,initial,k,fp64,timing);
        std::vector<float> candidates(size_t(B)*m),archive(size_t(A)*m);
        for(int b=0;b<B;++b) {
            auto normalized=pca::normalize_signature(signatures[b]);
            std::copy(normalized.coarse.begin(),normalized.coarse.end(),candidates.begin()+size_t(b)*m);
        }
        for(int a=0;a<A;++a) {
            std::vector<float> row(m);for(int j=0;j<m;++j) row[j]=value(size_t(a)*m+j+0xabcdefULL);
            auto normalized=pca::normalize_signature(row);
            std::copy(normalized.coarse.begin(),normalized.coarse.end(),archive.begin()+size_t(a)*m);
        }
        auto similarities=pca::gpu_similarity(candidates,B,archive,A,m,timing);
        double error=0;
        for(int b=0;b<std::min(B,4);++b) {
            auto expected=pca::finish(pca::solve(matrices[b],initial,k,"power",{},false).local_signature);
            for(int i=0;i<m;++i) error=std::max(error,std::fabs(signatures[b][i]-double(expected[i])));
        }
        if(error>(fp64?1e-6:1e-4)) throw std::runtime_error("benchmark signature verification failed");
        gpe::Json output;output.type=gpe::Json::Type::Object;
        auto number=[&](const std::string& name,double v) {gpe::Json x;x.type=gpe::Json::Type::Number;x.number=v;output.object[name]=x;};
        auto text=[&](const std::string& name,const std::string& v) {gpe::Json x;x.type=gpe::Json::Type::String;x.string=v;output.object[name]=x;};
        number("B",B);number("m",m);number("n",n);number("A",A);number("k",k);
        number("ranking_seconds",timing.ranking);number("gemv_seconds",timing.gemv);number("iteration_seconds",timing.iteration);
        number("similarity_seconds",timing.similarity);number("upload_seconds",timing.upload);number("download_seconds",timing.download);
        number("nominal_R_read_bytes",timing.bytes);number("gemv_flops",timing.flops);
        number("gemv_effective_bytes_per_second",timing.bytes/timing.gemv);
        number("gemv_flops_per_second",timing.flops/timing.gemv);number("gemv_arithmetic_intensity",2./(fp64?8:4));
        double product_flops=2.*B*A*m,product_bytes=4.*(double(B)*m+double(A)*m+double(B)*A);
        number("similarity_flops",product_flops);number("similarity_flops_per_second",product_flops/timing.similarity);
        number("similarity_minimum_io_bytes",product_bytes);number("similarity_arithmetic_intensity",product_flops/product_bytes);
        number("nominal_device_memory_bytes_per_second",timing.nominal_memory_bytes_per_second);
        number("device_multiprocessors",timing.multiprocessors);number("device_clock_khz",timing.clock_khz);
        number("signature_max_abs_error_first_four",error);text("device",timing.device);text("precision",fp64?"double":"float");
        text("input","synthetic uniform values with deterministic 1/101 missing entries, then GPU percentile ranking");
        text("roofline_note","GEMV traffic counts R reads only (lower bound); tiled-product IO is compulsory minimum, not measured DRAM traffic. Device bandwidth derived from reported memory clock/bus width.");
        std::ofstream file(argv[3]);pca::write_json(file,output);file<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
