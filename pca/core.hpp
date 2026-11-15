#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pca {
using Sum = std::function<void(double*, int)>;
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> data;
};
// In-place sampled raw values -> percentile ranks, NaN=0.5, row demeaned.
void rank_rows(Matrix& raw);
struct Solution {
    std::vector<double> local_signature;
    double sigma = 0, second_sigma = 0, relative_residual = 0;
    int matvecs = 0, steps = 0;
};
// Local rows of R, replicated right vector. sum only reduces distributed row sums.
Solution solve(const Matrix& R, const std::vector<double>& initial, int steps,
               const std::string& algorithm, const Sum& sum = {}, bool diagnostics = true);
std::vector<float> finish(std::vector<double> full_signature);
// Jacobi eigensolver for small symmetric matrices; no BLAS/LAPACK.
void jacobi(std::vector<double> A, int n, std::vector<double>& eigenvalues,
            std::vector<double>& eigenvectors);
struct Record {
    std::string key;
    double quality = 0;
    std::vector<float> signature;
};
struct Decision {
    std::string action, closest, replaced;
    double similarity = 0;
};
double similarity(const std::vector<float>& a, const std::vector<float>& b);
struct BoundedSimilarity { double value, error; };
struct NormalizedSignature {
    std::vector<float> coarse;
    double error = 0;
    bool valid = false;
};
NormalizedSignature normalize_signature(const std::vector<float>& signature);
BoundedSimilarity bounded_similarity(const NormalizedSignature& a, const NormalizedSignature& b);
BoundedSimilarity bounded_similarity(const std::vector<float>& a, const std::vector<float>& b);
// Coarse FP32 similarity with a conservative bound and selective FP64 refinement.
class Archive {
public:
    // Optional matrix-product backend; inputs are centered FP32 unit rows.
    using Product = std::function<std::vector<float>(const std::vector<float>&, int,
                                                    const std::vector<float>&, int, int)>;
    Product product;
    std::map<std::string, Record> records;
    long long refined_pairs = 0, coarse_pairs = 0;
    double similarity_seconds = 0, admission_seconds = 0;
    std::vector<Decision> consider(const std::vector<Record>& candidates, int block_size = 128);
};
}  // namespace pca
