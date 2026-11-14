#include "benchmark_io.hpp"
#include "gpu.hpp"
#include "gpe/metrics.hpp"
#include "gpe/panel.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace pca::bench;
bool equal(double a,double b){return (std::isnan(a)&&std::isnan(b))||a==b;}
void check(bool value,const char* what){if(!value)throw std::runtime_error(what);}
void verify(const std::vector<float>& values,int rows,int n){
    gpe::Panel panel(rows,n);std::copy(values.begin(),values.end(),panel.buf.begin());
    auto cpu=gpe::rank_rows_pct(panel);pca::GpuTiming timing;
    auto gpu=pca::gpu_percentile_rows_f64(values,rows,n,timing);
    check(cpu.size()==gpu.size(),"rank shape");
    for(size_t i=0;i<cpu.size();++i)check(equal(cpu[i],gpu[i]),"FP64 rank/NaN contract mismatch");
    // Exercise the actual upstream target-rank consumer, not a duplicate IC formula.
    auto expected=gpe::daily_rank_ic(panel,cpu),actual=gpe::daily_rank_ic(panel,gpu);
    for(size_t i=0;i<expected.ic.size();++i)check(equal(expected.ic[i],actual.ic[i]),"upstream daily RankIC mismatch");
    check(expected.eligible==actual.eligible&&expected.valid==actual.valid&&equal(expected.missing_ratio,actual.missing_ratio),"RankIC mask statistics");
}
int main(int argc,char** argv) try{
    auto args=options(argc,argv);pca::gpu_select(0);
    for(int n:{1,17,257,5876,8192}){
        std::vector<float> x(size_t(6)*n);
        for(int row=0;row<6;++row)for(int j=0;j<n;++j){
            float v=float((j*7+row)%29-14);
            if(row==0&&j%5==0)v=NAN;
            if(row==1)v=0;
            if(row==2)v=NAN;
            if(row==3&&j%3==0)v=INFINITY;
            if(row==4&&j%3==0)v=-INFINITY;
            if(row==5&&j%2==0)v=-0.f;
            x[size_t(row)*n+j]=v;
        }
        verify(x,6,n);
    }
    pca::GpuTiming t;check(pca::gpu_percentile_rows_f64({},0,257,t).empty(),"empty rank batch");
    int real_rows=0,real_cols=0;
    if(args.count("--raw")){real_rows=std::stoi(args.at("--rows"));real_cols=std::stoi(args.at("--cols"));
        verify(read<float>(args.at("--raw"),0,size_t(real_rows)*real_cols),real_rows,real_cols);}
    auto result=object();put(result,"verified",1.);put(result,"fp64_percentile_exact_to_engine",1.);
    put(result,"upstream_rankic_consumer_exact",1.);put(result,"synthetic_widths",5.);put(result,"real_rows",real_rows);put(result,"real_cols",real_cols);
    put(result,"scope",std::string("Rank arrays and actual target-rank consumer; automatic GP/RankIC backend switching remains a caller decision"));
    write(args.at("--out"),result);std::cout<<"PASS: shared GPU percentile interface matches engine FP64 ranks and RankIC consumer\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
