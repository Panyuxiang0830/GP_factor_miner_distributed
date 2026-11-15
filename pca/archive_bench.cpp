#include "mpi_archive.hpp"
#include "json_output.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

float sample(uint64_t i, uint64_t j) {
    uint64_t x = (i + 2026) * 0x9e3779b97f4a7c15ULL + j;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; x ^= x >> 31;
    return float(double(x >> 11) / 9007199254740992. - .5);
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (argc != 6) throw std::runtime_error("usage: pca_archive_bench A B dimension root|mpi output.json");
        int A=std::stoi(argv[1]), B=std::stoi(argv[2]), d=std::stoi(argv[3]);
        std::string mode=argv[4];
        if (A<1 || B<1 || d<2 || (mode!="root" && mode!="mpi")) throw std::runtime_error("invalid archive benchmark parameters");
        pca::Archive root; pca::MpiArchive distributed(MPI_COMM_WORLD);
        for (int i=0;i<A;++i) {
            pca::Record record; record.key="seed_"+std::to_string(i); record.quality=.03; record.signature.resize(d);
            for (int j=0;j<d;++j) record.signature[j]=sample(i,j);
            if (mode=="mpi") distributed.seed(record);
            else if (!rank) root.records[record.key]=record;
        }
        std::vector<pca::Record> candidates;
        for (int i=0;i<B;++i) {
            pca::Record record; record.key="candidate_"+std::to_string(i); record.quality=i%2?.04:.03;
            record.signature.resize(d);
            for (int j=0;j<d;++j) record.signature[j]=sample(i%3==0 ? i%A : A+i,j);
            candidates.push_back(record);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start=std::chrono::steady_clock::now();
        std::vector<pca::Decision> decisions;
        if (mode=="mpi") decisions=distributed.consider(candidates);
        else if (!rank) decisions=root.consider(candidates);
        double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(), maximum;
        MPI_Allreduce(&time,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        double similarity=mode=="mpi"?distributed.similarity_seconds:root.similarity_seconds;
        double admission=mode=="mpi"?distributed.admission_seconds:root.admission_seconds;
        double comm=mode=="mpi"?distributed.communication_seconds:0, sim_max, admission_max, comm_max;
        MPI_Allreduce(&similarity,&sim_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        MPI_Allreduce(&admission,&admission_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        MPI_Allreduce(&comm,&comm_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if (!rank) {
            gpe::Json output; output.type=gpe::Json::Type::Object;
            auto number=[&](const std::string& name,double value) { gpe::Json x; x.type=gpe::Json::Type::Number;x.number=value;output.object[name]=x; };
            number("archive_input_size",A);number("candidates",B);number("dimension",d);number("ranks",ranks);
            number("total_seconds",maximum);number("similarity_seconds",sim_max);number("admission_seconds",admission_max);number("communication_seconds",comm_max);
            number("archive_final_size",mode=="mpi"?distributed.qualities.size():root.records.size());
            int admit=0,replace=0,reject=0;
            for(const auto& decision:decisions) {admit+=decision.action=="admit";replace+=decision.action=="replace";reject+=decision.action=="reject";}
            number("admit",admit);number("replace",replace);number("reject",reject);
            gpe::Json label;label.type=gpe::Json::Type::String;label.string=mode;output.object["mode"]=label;
            std::ofstream file(argv[5]);pca::write_json(file,output);file<<'\n';
        }
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
