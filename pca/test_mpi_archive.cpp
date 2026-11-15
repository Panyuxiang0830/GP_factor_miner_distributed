#include "mpi_archive.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        using namespace pca;
        std::vector<float> a{1,1,-1,-1}, b{1,-1,1,-1}, c{2,0,0,-2};
        std::vector<Record> candidates{{"a",.03,a},{"z",.03,b},{"probe",.03,c},
                                       {"replacement",.04,a},{"reject",.041,a},{"replacement",.05,a}};
        Archive expected; MpiArchive actual(MPI_COMM_WORLD);
        auto serial = expected.consider(candidates), parallel = actual.consider(candidates);
        for (size_t i=0; i<serial.size(); ++i)
            if (serial[i].action != parallel[i].action || serial[i].closest != parallel[i].closest || serial[i].replaced != parallel[i].replaced)
                throw std::runtime_error("distributed archive decision mismatch");
        if (parallel[2].closest != "z") throw std::runtime_error("MAXLOC lexicographic tie");
        if (expected.records.size() != actual.qualities.size()) throw std::runtime_error("final archive size");
        for (const auto& item : expected.records) if (!actual.qualities.count(item.first)) throw std::runtime_error("final archive key");
        if (!rank) std::cout << "PASS: sharded signatures, MAXLOC tie, sequential replacements and duplicate keys\n";
        MPI_Finalize(); return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; MPI_Abort(MPI_COMM_WORLD, 1); return 1; }
}
