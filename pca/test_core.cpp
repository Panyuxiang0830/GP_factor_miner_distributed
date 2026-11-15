#include "core.hpp"
#include "json_output.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool value, const char* name) { if (!value) throw std::runtime_error(name); }
int main() {
    using namespace pca;
    gpe::Json escaped; escaped.type = gpe::Json::Type::String; escaped.string = "[\"field\",\"a\\b\"]\n";
    std::ostringstream encoded; write_json(encoded, escaped);
    check(gpe::parse_json(encoded.str()).string == escaped.string, "JSON expression-key escaping");
    std::vector<double> values, vectors;
    jacobi({2, 1, 1, 2}, 2, values, vectors);
    check(std::fabs(std::max(values[0], values[1]) - 3) < 1e-12, "Jacobi eigenvalues");
    Matrix R{5, 4, {3, 1, 0, 2, 0, 1, 4, 0, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0}};
    auto lanczos = solve(R, {1, .2, .3, .4}, 4, "lanczos");
    auto power = solve(R, {1, .2, .3, .4}, 100, "power");
    check(lanczos.relative_residual < 1e-11, "full Krylov residual");
    check(std::fabs(lanczos.sigma - power.sigma) < 1e-10, "Lanczos singular value");
    auto l = finish(lanczos.local_signature), p = finish(power.local_signature);
    double cosine = 0; for (size_t i = 0; i < l.size(); ++i) cosine += double(l[i]) * p[i];
    check(std::fabs(cosine - 1) < 1e-6, "Lanczos principal left vector");
    Matrix rankone{3, 2, {1, 2, 2, 4, 3, 6}};
    auto early = solve(rankone, {1, 1}, 10, "lanczos");
    check(early.steps == 1 && early.relative_residual < 1e-12, "rank-one breakdown");
    check(solve({2, 2, {0, 0, 0, 0}}, {1, 1}, 4, "power").local_signature.empty(), "zero R");
    Matrix raw{2, 4, {3, NAN, 1, 3, NAN, NAN, NAN, NAN}};
    rank_rows(raw);
    check(std::fabs(raw.data[0] - (5./6 - .625)) < 1e-14 &&
          std::fabs(raw.data[1] - (.5 - .625)) < 1e-14 && raw.data[4] == 0, "CPU ranking ties, missing, centering");
    Archive archive;
    std::vector<float> a{1, 1, -1, -1}, b{1, -1, 1, -1};
    auto decisions = archive.consider({{"a", .03, a}, {"b", .04, a}, {"c", .041, a}, {"d", .03, b}}, 128);
    check(decisions[0].action == "admit" && decisions[1].action == "replace" &&
          decisions[1].replaced == "a" && decisions[2].action == "reject" &&
          decisions[3].action == "admit", "within-block archive changes");
    auto samekey = archive.consider({{"b", .045, a}});
    check(samekey[0].action == "replace" && samekey[0].replaced == "b", "same-key update");
    Archive ties; ties.records["a"] = {"a", .03, a}; ties.records["z"] = {"z", .03, a};
    check(ties.consider({{"x", .04, a}})[0].closest == "z", "lexicographic tie");
    Archive accelerated;
    accelerated.product = [](const std::vector<float>& x, int L, const std::vector<float>& y, int A, int d) {
        std::vector<float> output(size_t(L) * A);
        for (int i = 0; i < L; ++i) for (int j = 0; j < A; ++j) {
            float value = 0; for (int k = 0; k < d; ++k) value += x[size_t(i) * d + k] * y[size_t(j) * d + k];
            output[size_t(i) * A + j] = std::fabs(value);
        }
        return output;
    };
    auto ad = accelerated.consider({{"a", .03, a}, {"b", .04, a}, {"c", .041, a}, {"d", .03, b}}, 128);
    for (size_t i = 0; i < ad.size(); ++i)
        check(ad[i].action == decisions[i].action && ad[i].closest == decisions[i].closest && ad[i].replaced == decisions[i].replaced,
              "accelerated product preserves archive semantics");
    std::cout << "PASS: Jacobi, Lanczos/power, residual, rank breakdown, sequential archive semantics\n";
}
