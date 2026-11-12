#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
// Numerical C++ entrypoint for validation/experiment drivers, without spawning
// one process per factor. Python only supplies data and independent references.
extern "C" int pca_solve(const double* R, int m, int n, const double* initial,
                          int steps, int method, float* output, double* stats) {
    try {
        pca::Matrix matrix{m, n, std::vector<double>(R, R + size_t(m) * n)};
        auto solution = pca::solve(matrix, std::vector<double>(initial, initial + n),
                                   steps, method ? "lanczos" : "power", {}, true);
        auto signature = pca::finish(solution.local_signature);
        if (signature.size() != size_t(m)) return 2;
        std::copy(signature.begin(), signature.end(), output);
        stats[0] = solution.sigma; stats[1] = solution.second_sigma;
        stats[2] = solution.relative_residual; stats[3] = solution.steps; stats[4] = solution.matvecs;
        return 0;
    } catch (const std::exception&) { return 1; }
}
