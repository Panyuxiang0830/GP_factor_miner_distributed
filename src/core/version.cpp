#include "gpe/version.hpp"

namespace gpe {

std::string build_info() {
    std::string info = "gpe core";
#ifdef GPE_HAVE_OPENMP
    info += " +openmp";
#endif
#ifdef GPE_HAVE_MPI
    info += " +mpi";
#endif
#ifdef GPE_HAVE_CUDA
    info += " +cuda";
#endif
    return info;
}

}  // namespace gpe
