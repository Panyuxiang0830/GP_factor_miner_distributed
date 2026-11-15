// Real MPI factor engine -> owned-date sampling -> PCA. No exported R input.
#include "benchmark_io.hpp"
#include "gpe/distributed.hpp"
#include "core.hpp"
#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <omp.h>
#ifdef PCA_CUDA
#include "gpu.hpp"
#include <mpi-ext.h>
#endif

using namespace pca::bench;
namespace fs=std::filesystem;
int main(int argc,char** argv){
    MPI_Init(&argc,&argv);int rank,P;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&P);
    try{
        auto args=options(argc,argv);auto request=gpe::parse_json_file(args.at("--request"));fs::path out=args.at("--out");
        if(!rank){if(fs::exists(out))throw std::runtime_error("output exists");fs::create_directories(out);}MPI_Barrier(MPI_COMM_WORLD);
        auto start=Clock::now();auto data=gpe::PanelSet::open(request.at("panel_dir").string);
        int T=int(request.at("dates").number),N=int(request.at("stocks").number),step=int(request.at("sample_step").number);
        int k=int(request.at("steps").number),chunk=int(request.at("engine_chunk").number);
        if(T<2||N<2||T>data.T||N>data.N||P>T||step<1||k<1||chunk<1)throw std::runtime_error("invalid bridge shape/budget");
        data.T=T;data.N=N;int m=(T+step-1)/step;auto initial=read<double>(request.at("initial").string,0,N);
        if(m<2||initial.size()!=size_t(N))throw std::runtime_error("invalid bridge sampling/initial");
        std::string method=request.at("method").string,device=request.at("device").string;
        if(method!="power"&&method!="lanczos")throw std::runtime_error("unknown solver");
        if(device!="cpu"&&device!="cuda")throw std::runtime_error("unknown device");
        if(device=="cuda"&&method!="power")throw std::runtime_error("GPU Lanczos is not supported");
#ifdef PCA_CUDA
        pca::GpuTiming gpu;
        if(device=="cuda"){
            if(!MPIX_Query_cuda_support())throw std::runtime_error("CUDA-aware MPI unavailable");
            MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);
            int r;MPI_Comm_rank(local,&r);pca::gpu_select(r);MPI_Comm_free(&local);
        }
#else
        if(device=="cuda")throw std::runtime_error("bridge needs a CUDA build");
#endif
        gpe::MpiOptions opt;opt.keep_roots=true;opt.chunk=int(request.at("dag_chunk").number);opt.H0=int(request.at("H0").number);
        const auto& f=request.at("fitness");
        opt.fitness.max_depth=int(f.at("max_depth").number);opt.fitness.max_nodes=int(f.at("max_nodes").number);
        opt.fitness.stability_segments=int(f.at("stability_segments").number);
        opt.fitness.min_valid_time_ratio=f.at("min_valid_time_ratio").number;opt.fitness.max_missing_ratio=f.at("max_missing_ratio").number;
        opt.fitness.stability_penalty=f.at("stability_penalty").number;opt.fitness.complexity_penalty=f.at("complexity_penalty").number;
        opt.fitness.minimum_same_sign_ratio=f.at("minimum_same_sign_ratio").number;opt.fitness.inconsistent_sign_multiplier=f.at("inconsistent_sign_multiplier").number;
        gpe::BufferPool pool;opt.pool=&pool;omp_set_max_active_levels(1);
        const auto& expressions=request.at("expressions").array;Json records=array();std::vector<float> signatures(expressions.size()*size_t(m),0);
        double engine_seconds=0,ranking_seconds=0,solve_seconds=0,gather_seconds=0,communication_seconds=0;
        auto block=gpe::date_block(T,P,rank);int first=((block.t0+step-1)/step)*step,rows=std::max(0,(block.t1-first+step-1)/step);
        std::vector<int> counts(P),offset(P);for(int r=0,pos=0;r<P;++r){auto b=gpe::date_block(T,P,r);int begin=((b.t0+step-1)/step)*step;counts[r]=std::max(0,(b.t1-begin+step-1)/step);offset[r]=pos;pos+=counts[r];}
        for(size_t begin=0;begin<expressions.size();begin+=chunk){
            size_t end=std::min(expressions.size(),begin+chunk);gpe::Dag dag;
            for(size_t i=begin;i<end;++i){std::string error;if(dag.add_expression(expressions[i].at("expression"),&error)<0)throw std::runtime_error("expression parse failed: "+error);}
            dag.finalize();for(const auto& field:dag.fields())if(!data.has_field(field))throw std::runtime_error("unknown field: "+field);
            auto t=Clock::now();auto evaluated=gpe::evaluate_batch_mpi(dag,data,MPI_COMM_WORLD,opt);engine_seconds+=seconds(t);
            std::vector<pca::Matrix> matrices;std::vector<size_t> selected;
            for(size_t i=begin;i<end;++i){
                const auto& c=evaluated.candidates[i-begin];auto row=object();put(row,"index",double(i));put(row,"key",expressions[i].at("key").string);put(row,"valid",c.valid?1.:0.);
                if(c.valid){
                    for(auto v:std::vector<std::pair<std::string,double>>{{"rank_ic",c.rank_ic},{"valid_ratio",c.valid_ratio},{"missing_ratio",c.missing_ratio},{"fitness",c.fitness},{"segment_rank_ic_std",c.segment_rank_ic_std},{"same_sign_ratio",c.same_sign_ratio}})put(row,v.first,v.second);
                    Json segments=array();for(double x:c.segment_rank_ics){Json v;v.type=Json::Type::Number;v.number=x;segments.array.push_back(v);}row.object["segment_rank_ics"]=segments;
                    const auto& node=dag.nodes()[dag.roots()[i-begin]];put(row,"depth",node.depth);put(row,"nodes",node.tree_nodes);
                    const auto& panel=evaluated.roots.at(dag.roots()[i-begin]);pca::Matrix R{rows,N,std::vector<double>(size_t(rows)*N)};
                    for(int r=0,day=first;day<block.t1;day+=step,++r)std::copy(panel.row(day-block.t0),panel.row(day-block.t0)+N,R.data.begin()+size_t(r)*N);
                    matrices.push_back(std::move(R));selected.push_back(i);
                }
                if(!rank)records.array.push_back(row);
            }
            t=Clock::now();
            if(!matrices.empty()){
#ifdef PCA_CUDA
                if(device=="cuda"&&rows){
                    std::vector<float> raw;for(const auto& R:matrices)raw.insert(raw.end(),R.data.begin(),R.data.end());
                    auto ranked=pca::gpu_rank_rows(raw,rows*int(matrices.size()),N,gpu);
                    for(size_t b=0;b<matrices.size();++b)std::copy(ranked.begin()+b*size_t(rows)*N,ranked.begin()+(b+1)*size_t(rows)*N,matrices[b].data.begin());
                }else
#endif
                for(auto& R:matrices)pca::rank_rows(R);
            }
            ranking_seconds+=seconds(t);t=Clock::now();std::vector<std::vector<double>> local;
#ifdef PCA_CUDA
            if(device=="cuda"&&!matrices.empty()){
                pca::DeviceReduce sum=[&](void* values,int count,bool fp64){auto t=Clock::now();MPI_Allreduce(MPI_IN_PLACE,values,count,fp64?MPI_DOUBLE:MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);communication_seconds+=seconds(t);};
                auto s=pca::gpu_power(matrices,initial,k,request.at("precision").string=="double",gpu,sum);for(const auto& v:s)local.emplace_back(v.begin(),v.end());
            }else
#endif
            {
                pca::Sum sum=[&](double* values,int count){auto t=Clock::now();MPI_Allreduce(MPI_IN_PLACE,values,count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);communication_seconds+=seconds(t);};
                for(const auto& R:matrices)local.push_back(pca::solve(R,initial,k,method,sum,false).local_signature);
            }
            solve_seconds+=seconds(t);t=Clock::now();
            for(size_t b=0;b<matrices.size();++b){
                std::vector<double> own(rows,0),full(rank?0:m);if(!local[b].empty())own=local[b];
                MPI_Gatherv(own.data(),rows,MPI_DOUBLE,full.data(),counts.data(),offset.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
                if(!rank){auto s=pca::finish(full);if(!s.empty())std::copy(s.begin(),s.end(),signatures.begin()+selected[b]*m);}
            }
            gather_seconds+=seconds(t);
        }
        double wall=seconds(start),maxwall;MPI_Reduce(&wall,&maxwall,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(!rank){
            binary(out/"signatures.f32",signatures);write(out/"candidates.json",records);auto metrics=object();put(metrics,"total_seconds",maxwall);put(metrics,"engine_seconds",engine_seconds);put(metrics,"ranking_seconds",ranking_seconds);put(metrics,"solve_seconds",solve_seconds);put(metrics,"gather_seconds",gather_seconds);put(metrics,"communication_seconds",communication_seconds);put(metrics,"ranks",P);put(metrics,"rows",m);put(metrics,"sample_step",step);put(metrics,"factors",double(expressions.size()));put(metrics,"device",device);
            put(metrics,"scope",std::string("Live MPI factor engine and RankIC, owned global-date sampling excludes halos, then PCA; launch and final serialization recorded by Python caller"));write(out/"metrics.json",metrics);
        }
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
