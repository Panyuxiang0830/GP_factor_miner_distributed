#include "gpu.hpp"
#include "gpe/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
void check(bool yes, const char* what) { if (!yes) throw std::runtime_error(what); }
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: pca_gpu_test output.json");
        pca::gpu_select(0); pca::GpuTiming timing;
        const int m = 17, n = 5876, B = 8;
        std::vector<float> panels(size_t(m) * n);
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
            float value = i == 0 ? 1 : float((j * 137 + i * 43) % 123);
            if (i == 1 || (i > 1 && j % 17 == 0)) value = std::numeric_limits<float>::quiet_NaN();
            if (i == 2 && j == 1) value = std::numeric_limits<float>::infinity();
            if (i == 2 && j == 2) value = -std::numeric_limits<float>::infinity();
            panels[size_t(i) * n + j] = value;
        }
        auto ranked = pca::gpu_rank_rows(panels, m, n, timing);
        auto percentiles = pca::gpu_percentile_rows(panels, m, n, timing);
        double ranking_error = 0, percentile_error = 0;
        for (int i = 0; i < m; ++i) {
            std::vector<int> order;
            for (int j = 0; j < n; ++j) if (!std::isnan(panels[size_t(i) * n + j])) order.push_back(j);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return panels[size_t(i) * n + a] < panels[size_t(i) * n + b]; });
            std::vector<double> reference(n, .5);
            for (size_t begin = 0; begin < order.size();) {
                size_t end = begin + 1;
                while (end < order.size() && panels[size_t(i) * n + order[end]] == panels[size_t(i) * n + order[begin]]) ++end;
                double value = (begin + end + 1.) / (2 * order.size());
                for (size_t j = begin; j < end; ++j) reference[order[j]] = value;
                begin = end;
            }
            double mean = std::accumulate(reference.begin(), reference.end(), 0.) / n;
            for (int j = 0; j < n; ++j) {
                ranking_error = std::max(ranking_error, std::fabs(ranked[size_t(i) * n + j] - (reference[j] - mean)));
                if (std::isnan(panels[size_t(i) * n + j]))
                    check(std::isnan(percentiles[size_t(i) * n + j]), "RankIC percentile primitive preserves NaN mask");
                else percentile_error = std::max(percentile_error,std::fabs(percentiles[size_t(i) * n + j]-reference[j]));
            }
        }
        check(ranking_error < 2e-7, "GPU average ranks, missing, ties, infinities");
        check(percentile_error < 1e-7, "GPU raw percentiles without fill/centering");
        check(pca::gpu_percentile_rows({},0,n,timing).empty(),"empty date block");
        auto singleton=pca::gpu_percentile_rows({3,std::numeric_limits<float>::quiet_NaN()},2,1,timing);
        check(singleton[0]==1 && std::isnan(singleton[1]),"singleton stock and all-missing row");
        std::vector<pca::Matrix> matrices(B);
        std::vector<double> init(n);
        for (int j = 0; j < n; ++j) init[j] = std::sin(.133 * j) + std::cos(.217 * j);
        for (int b = 0; b < B; ++b) {
            matrices[b].rows = m; matrices[b].cols = n; matrices[b].data.resize(size_t(m) * n);
            for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
                matrices[b].data[size_t(i) * n + j] = std::sin(.07 * (i + b) * j) + .1 * std::cos(.3 * j + i);
        }
        auto double_signatures = pca::gpu_power(matrices, init, 4, true, timing);
        auto float_signatures = pca::gpu_power(matrices, init, 4, false, timing);
        double double_error = 0, float_error = 0;
        for (int b = 0; b < B; ++b) {
            auto expected = pca::finish(pca::solve(matrices[b], init, 4, "power", {}, false).local_signature);
            for (int i = 0; i < m; ++i) {
                double_error = std::max(double_error, std::fabs(double_signatures[b][i] - double(expected[i])));
                float_error = std::max(float_error, std::fabs(float_signatures[b][i] - double(expected[i])));
            }
        }
        check(double_error < 1e-6 && float_error < 1e-4, "GPU power precision");
        const int L = 17, A = 19, dimension = 293;
        std::vector<float> x(size_t(L) * dimension), y(size_t(A) * dimension);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(std::sin(.17 * i) / std::sqrt(dimension));
        for (size_t i = 0; i < y.size(); ++i) y[i] = float(std::cos(.23 * i) / std::sqrt(dimension));
        auto product = pca::gpu_similarity(x, L, y, A, dimension, timing);
        double similarity_error = 0;
        for (int i = 0; i < L; ++i) for (int j = 0; j < A; ++j) {
            double expected = 0;
            for (int k = 0; k < dimension; ++k) expected += double(x[size_t(i) * dimension + k]) * y[size_t(j) * dimension + k];
            similarity_error = std::max(similarity_error, std::fabs(product[size_t(i) * A + j] - std::fabs(expected)));
        }
        check(similarity_error < 1e-6, "GPU tiled similarity, nonmultiple dimensions");
        std::ofstream output(argv[1]);
        output << "{\"passed\":true,\"device\":\"" << timing.device << "\",\"ranking_error\":" << ranking_error
               << ",\"percentile_error\":" << percentile_error << ",\"raw_percentile_mask_preserved\":true"
               << ",\"fp64_signature_max_abs_error\":" << double_error
               << ",\"fp32_signature_max_abs_error\":" << float_error
               << ",\"similarity_error\":" << similarity_error
               << ",\"ranking_seconds\":" << timing.ranking
               << ",\"gemv_seconds\":" << timing.gemv
               << ",\"similarity_seconds\":" << timing.similarity << "}\n";
        std::cout << "PASS: CUDA ranking, FP32/FP64 power, tiled similarity\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
