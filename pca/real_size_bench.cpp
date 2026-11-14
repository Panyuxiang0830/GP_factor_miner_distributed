#include "benchmark_io.hpp"
#include "core.hpp"
#include "gpe/dag.hpp"
#include "gpe/evaluate.hpp"
#include "gpe/panel.hpp"
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace pca::bench;
namespace fs=std::filesystem;

pca::Matrix sample(const gpe::Panel& panel,int step,int n) {
    pca::Matrix x; x.rows=(panel.T+step-1)/step;x.cols=n;x.data.resize(size_t(x.rows)*n);
    for(int i=0,t=0;t<panel.T;t+=step,++i)
        std::copy(panel.row(t),panel.row(t)+n,x.data.begin()+size_t(i)*n);
    return x;
}
int main(int argc,char**argv) try {
    auto args=options(argc,argv);auto source=gpe::parse_json_file(args.at("--source"));
    fs::path out=args.at("--out");if(fs::exists(out))throw std::runtime_error("output exists");fs::create_directories(out);
    auto data=gpe::PanelSet::open(source.at("panel_dir").string);
    if(args.count("--dates"))data.T=std::min(data.T,std::stoi(args.at("--dates")));
    if(args.count("--stocks"))data.N=std::min(data.N,std::stoi(args.at("--stocks")));
    const int want=args.count("--factors")?std::stoi(args.at("--factors")):128;
    if(data.T<20||data.N<6||want<2)throw std::runtime_error("invalid benchmark shape");
    auto candidates=gpe::parse_json_file(source.at("candidates").string);
    std::vector<gpe::Panel> panels;Json selected=array();auto engine_start=Clock::now();
    // All axes use the same prefix of genuinely evaluated full-universe panels.
    for(size_t begin=0;begin<candidates.array.size()&&int(panels.size())<want;begin+=8){
        gpe::Dag dag;std::vector<std::pair<int,size_t>> roots;
        for(size_t i=begin;i<std::min(begin+8,candidates.array.size());++i){
            std::string error;int r=dag.add_expression(candidates.array[i].at("expression"),&error);
            if(r>=0)roots.push_back({r,i});
        }
        dag.finalize();auto evaluated=gpe::evaluate_serial(dag,data);
        for(auto [r,i]:roots){
            if(int(panels.size())==want)break;
            auto& panel=evaluated.at(r);if(panel.buf.empty())continue;
            auto check=sample(panel,20,std::min(367,data.N));pca::rank_rows(check);
            if(!std::any_of(check.data.begin(),check.data.end(),[](double v){return v!=0;}))continue;
            panels.push_back(std::move(panel));auto row=object();
            row.object["index"]=candidates.array[i].at("index");row.object["key"]=candidates.array[i].at("key");
            row.object["quality"]=candidates.array[i].at("quality");selected.array.push_back(row);
        }
        std::cout<<"Prepared real panels "<<panels.size()<<"/"<<want<<std::endl;
    }
    if(int(panels.size())!=want)throw std::runtime_error("not enough stable nonconstant real panels");
    double engine_seconds=seconds(engine_start);
    std::vector<float> raw;for(int i=0;i<2;++i)raw.insert(raw.end(),panels[i].buf.begin(),panels[i].buf.end());
    binary(out/"raw_first2.f32",raw);raw.clear();raw.shrink_to_fit();
    auto metadata=object();put(metadata,"T",data.T);put(metadata,"N",data.N);put(metadata,"factors",want);
    put(metadata,"engine_evaluation_seconds",engine_seconds);metadata.object["records"]=selected;
    metadata.object["initial"]=source.at("initial");
    put(metadata,"scope",std::string("Full-history real factor panels fixed before PCA; each stock subset is reranked; engine time excluded from PCA timings"));
    write(out/"source.json",metadata);omp_set_num_threads(1);
    auto initial=read<double>(source.at("initial").string,0,data.N);
    struct Case {std::string tag;int b,step,n,k;};std::vector<Case> cases;
    for(int step:{1,2,5,10,20})cases.push_back({"m"+std::to_string(step),std::min(32,want),step,data.N,4});
    for(int n:{367,734,1469,2938})if(n<data.N)cases.push_back({"n"+std::to_string(n),std::min(32,want),5,n,4});
    for(int b:{8,16,32,64,128})if(b<=want)cases.push_back({"B"+std::to_string(b),b,5,data.N,4});
    for(int k:{1,2,4,8,10})cases.push_back({"k"+std::to_string(k),std::min(32,want),5,data.N,k});
    for(const auto& c:cases)for(int repeat=0;repeat<4;++repeat){
        fs::path dir=out/(c.tag+"_r"+std::to_string(repeat));fs::create_directory(dir);
        auto total=Clock::now(),t=total;std::vector<pca::Matrix> matrices;matrices.reserve(c.b);
        for(int b=0;b<c.b;++b)matrices.push_back(sample(panels[b],c.step,c.n));
        double prepare=seconds(t);t=Clock::now();for(auto& x:matrices)pca::rank_rows(x);double ranking=seconds(t);
        t=Clock::now();std::vector<float> signatures;std::vector<pca::Record> records;
        std::vector<double> v(initial.begin(),initial.begin()+c.n);
        for(int b=0;b<c.b;++b){
            auto solution=pca::solve(matrices[b],v,c.k,"power",{},false);auto s=pca::finish(solution.local_signature);
            if(s.size()!=size_t(matrices[b].rows))throw std::runtime_error("invalid signature");
            signatures.insert(signatures.end(),s.begin(),s.end());
            if(std::isfinite(pca::similarity(s,s)))
                records.push_back({selected.array[b].at("key").string,selected.array[b].at("quality").number,std::move(s)});
        }
        double solve=seconds(t);pca::Archive archive;t=Clock::now();auto decisions=archive.consider(records,128);
        double archive_time=seconds(t),elapsed=seconds(total);auto metrics=object();
        put(metrics,"factors",c.b);put(metrics,"sample_step",c.step);put(metrics,"rows",matrices[0].rows);
        put(metrics,"cols",c.n);put(metrics,"steps",c.k);put(metrics,"threads",1);put(metrics,"prepare_seconds",prepare);
        put(metrics,"candidates",double(records.size()));
        put(metrics,"ranking_seconds",ranking);put(metrics,"solve_seconds",solve);put(metrics,"archive_seconds",archive_time);
        put(metrics,"similarity_seconds",archive.similarity_seconds);put(metrics,"admission_seconds",archive.admission_seconds);
        put(metrics,"total_seconds",elapsed);write(dir/"metrics.json",metrics);binary(dir/"signatures.f32",signatures);
        auto trace=array();for(const auto& d:decisions){auto row=object();put(row,"action",d.action);
            row.object["closest_key"]=d.closest.empty()?Json():string(d.closest);row.object["replaced_key"]=d.replaced.empty()?Json():string(d.replaced);
            put(row,"similarity",d.similarity);trace.array.push_back(row);}
        write(dir/"decisions.json",trace);auto keys=array();for(const auto& kv:archive.records)keys.array.push_back(string(kv.first));write(dir/"archive_keys.json",keys);
        if(repeat==1){std::vector<double> reference;for(int b=0;b<2;++b)reference.insert(reference.end(),matrices[b].data.begin(),matrices[b].data.end());binary(dir/"R_first2.f64",reference);}
        std::cout<<c.tag<<" repeat "<<repeat<<" complete"<<std::endl;
    }
    auto done=object();put(done,"configurations",double(cases.size()));put(done,"executed",1.);write(out/"EXECUTED.json",done);
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
