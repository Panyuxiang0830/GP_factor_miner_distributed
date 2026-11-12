#include "benchmark_io.hpp"
#include "core.hpp"
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <omp.h>

using namespace pca::bench;
namespace fs=std::filesystem;
int main(int argc,char**argv){
    MPI_Init(&argc,&argv);int rank=0,P=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    try{
        auto args=options(argc,argv);auto plan=gpe::parse_json_file(args.at("--plan"));
        auto manifest=gpe::parse_json_file(plan.at("manifest").string);
        int m=int(plan.at("rows").number),n=int(plan.at("cols").number),B=int(plan.at("count").number);
        double target=plan.at("target").number;int maxk=int(plan.at("max_steps").number);
        if(P>m||B<1||B>int(manifest.at("records").array.size()))throw std::runtime_error("invalid shape/ranks");
        fs::path out=args.at("--out");if(rank==0){if(fs::exists(out))throw std::runtime_error("output exists");fs::create_directories(out);}MPI_Barrier(MPI_COMM_WORLD);
        int lo=m*rank/P,hi=m*(rank+1)/P,local=hi-lo;omp_set_num_threads(1);
        auto initial=read<double>(manifest.at("initial").string,0,n);
        // Cache exported FP32 data once; each timed solve uses a double copy.
        auto loading=Clock::now();std::vector<float> cache(size_t(B)*local*n);
        std::map<std::string,std::unique_ptr<std::ifstream>> streams;
        for(int b=0;b<B;++b){auto& e=manifest.at("records").array[b];auto path=e.at("path").string;
            if(!streams.count(path))streams[path]=std::make_unique<std::ifstream>(path,std::ios::binary);
            auto& f=*streams.at(path);f.clear();f.seekg(std::streamoff((size_t(e.at("slot").number)*m+lo)*n)*4);
            f.read(reinterpret_cast<char*>(cache.data()+size_t(b)*local*n),std::streamsize(size_t(local)*n*4));if(!f)throw std::runtime_error("short R read");
        }
        double load_seconds=seconds(loading),load_max=0;MPI_Reduce(&load_seconds,&load_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        std::vector<int> counts(P),offset(P);for(int r=0;r<P;++r){offset[r]=m*r/P;counts[r]=m*(r+1)/P-offset[r];}
        for(const std::string method:{"power","lanczos"}){
            const auto& spec=plan.at("methods").at(method);auto budgets=read<int32_t>(spec.at("steps").string,0,B);
            std::vector<float> reference;std::vector<double> serial_error;
            if(rank==0){reference=read<float>(spec.at("reference").string,0,size_t(B)*m);serial_error=read<double>(spec.at("svd_error").string,0,B);}
            for(int repeat=0;repeat<4;++repeat){
                auto all_start=Clock::now();Json records=array();std::vector<float> signatures;std::vector<double> full(rank==0?m:0);
                double conversion=0;MPI_Barrier(MPI_COMM_WORLD);
                for(int b=0;b<B;++b){
                    if(budgets[b]<1||budgets[b]>maxk)throw std::runtime_error("invalid step budget");
                    auto copy_start=Clock::now();auto begin=cache.begin()+size_t(b)*local*n;
                    pca::Matrix R{local,n,std::vector<double>(begin,begin+size_t(local)*n)};conversion+=seconds(copy_start);
                    long long calls=0,elements=0;double communication=0;
                    pca::Sum sum=[&](double* values,int count){auto t=Clock::now();MPI_Allreduce(MPI_IN_PLACE,values,count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);communication+=seconds(t);++calls;elements+=count;};
                    MPI_Barrier(MPI_COMM_WORLD); // isolate the solve from previous root-side recording
                    auto t=Clock::now();auto solution=pca::solve(R,initial,budgets[b],method,sum,false);double elapsed=seconds(t);
                    if(solution.local_signature.size()!=size_t(local))throw std::runtime_error("invalid solution");
                    double local_times[2]={elapsed,communication},max_times[2];MPI_Reduce(local_times,max_times,2,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                    MPI_Gatherv(solution.local_signature.data(),local,MPI_DOUBLE,full.data(),counts.data(),offset.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
                    if(rank==0){
                        auto s=pca::finish(full);double error=std::max(0.,1-cosine(s,reference.data()+size_t(b)*m));
                        bool identical=std::equal(s.begin(),s.end(),reference.data()+size_t(b)*m);
                        double theta=std::acos(std::clamp(1-serial_error[b],0.,1.));
                        double delta=identical?0.:std::acos(std::clamp(1-error-1e-12,0.,1.));
                        double upper=1-std::cos(std::min(std::acos(-1.)/2,theta+delta))+1e-12;
                        auto row=object();put(row,"index",b);put(row,"budget",budgets[b]);put(row,"steps",solution.steps);put(row,"matvecs",solution.matvecs);
                        put(row,"solve_seconds",max_times[0]);put(row,"communication_seconds",max_times[1]);put(row,"allreduce_calls",double(calls));
                        put(row,"logical_payload_bytes",double(elements)*8);put(row,"serial_svd_error",serial_error[b]);
                        put(row,"mpi_vs_serial_error",error);put(row,"svd_error_upper_bound",upper);records.array.push_back(row);
                        put(records.array.back(),"identical_to_serial",identical?1.:0.);
                        signatures.insert(signatures.end(),s.begin(),s.end());
                    }
                    if(rank==0&&(b+1)%256==0)std::cout<<method<<" r"<<repeat<<" "<<b+1<<"/"<<B<<std::endl;
                }
                double elapsed=seconds(all_start),wall=0;MPI_Reduce(&elapsed,&wall,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                double convert_max=0;MPI_Reduce(&conversion,&convert_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                if(rank==0){fs::path dir=out/(method+"_r"+std::to_string(repeat));fs::create_directory(dir);
                    auto metrics=object();put(metrics,"factors",B);put(metrics,"ranks",P);put(metrics,"rows",m);put(metrics,"cols",n);
                    put(metrics,"target",target);put(metrics,"max_steps",maxk);put(metrics,"total_seconds",wall);put(metrics,"input_cache_seconds",load_max);put(metrics,"conversion_seconds",convert_max);
                    put(metrics,"scope",std::string("Per-factor date reductions for both methods; offline SVD-selected budgets; no residual diagnostics; MPI bookkeeping/gather outside solve timings"));
                    write(dir/"metrics.json",metrics);write(dir/"factors.json",records);binary(dir/"signatures.f32",signatures);
                }
            }
        }
        if(rank==0){auto done=object();put(done,"factors",B);put(done,"executed",1.);write(out/"EXECUTED.json",done);}
        MPI_Finalize();return 0;
    }catch(const std::exception&e){std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
