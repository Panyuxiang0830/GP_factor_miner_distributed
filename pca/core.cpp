#include "core.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pca {
namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
void reduce(const Sum& sum, double* p, int n) { if (sum) sum(p, n); }
double norm(const std::vector<double>& a, const Sum& sum = {}) {
    double s = dot(a, a); reduce(sum, &s, 1); return std::sqrt(s);
}
bool unit(std::vector<double>& a, const Sum& sum = {}) {
    double n = norm(a, sum);
    if (!(n > 0) || !std::isfinite(n)) return false;
    for (double& x : a) x /= n;
    return true;
}
std::vector<double> multiply(const Matrix& r, const std::vector<double>& v) {
    std::vector<double> result(r.rows);
    #pragma omp parallel for if(r.rows * r.cols > 100000)
    for (int i = 0; i < r.rows; ++i) {
        double s = 0;
        for (int j = 0; j < r.cols; ++j) s += r.data[size_t(i) * r.cols + j] * v[j];
        result[i] = s;
    }
    return result;
}
std::vector<double> multiply_t(const Matrix& r, const std::vector<double>& u, const Sum& sum) {
    std::vector<double> result(r.cols);
    // Row-major contiguous traversal, each thread owns a tile of columns.
    #pragma omp parallel for if(r.rows * r.cols > 100000)
    for (int tile = 0; tile < (r.cols + 63) / 64; ++tile) {
        for (int i = 0; i < r.rows; ++i)
            for (int j = tile * 64; j < std::min(r.cols, (tile + 1) * 64); ++j)
                result[j] += r.data[size_t(i) * r.cols + j] * u[i];
    }
    reduce(sum, result.data(), r.cols);
    return result;
}
void reorthogonalize(std::vector<double>& v, const std::vector<std::vector<double>>& basis,
                     const Sum& sum = {}) {
    // Two passes mitigate loss of orthogonality for near-rank-deficient inputs.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<double> coefficients(basis.size());
        for (size_t j = 0; j < basis.size(); ++j) coefficients[j] = dot(v, basis[j]);
        reduce(sum, coefficients.data(), int(coefficients.size()));
        for (size_t j = 0; j < basis.size(); ++j)
            for (size_t i = 0; i < v.size(); ++i) v[i] -= coefficients[j] * basis[j][i];
    }
}
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
using Normalized = NormalizedSignature;
Normalized normalize(const std::vector<float>& signature) {
    Normalized result;
    int n = int(signature.size());
    if (n < 2) return result;
    double mean = 0;
    float mean32 = 0;
    for (float x : signature) {
        if (!std::isfinite(x)) return result;
        mean += x; mean32 += x;
    }
    mean /= n; mean32 /= n;
    double square = 0;
    float square32 = 0;
    for (float x : signature) {
        square += (x - mean) * (x - mean);
        float z = x - mean32; square32 += z * z;
    }
    if (!(square > 0) || !(square32 > 0) || !std::isfinite(square32)) return result;
    double length = std::sqrt(square);
    float length32 = std::sqrt(square32);
    double error2 = 0;
    for (float x : signature) {
        float z = (x - mean32) / length32;
        result.coarse.push_back(z);
        double d = z - (x - mean) / length; error2 += d * d;
    }
    result.error = std::sqrt(error2);
    result.valid = true;
    return result;
}
struct Pair { double value, bound; };
Pair coarse(const Normalized& a, const Normalized& b) {
    if (!a.valid || !b.valid || a.coarse.size() != b.coarse.size())
        return {std::numeric_limits<double>::quiet_NaN(), 1};
    float value = 0;
    for (size_t i = 0; i < a.coarse.size(); ++i) value += a.coarse[i] * b.coarse[i];
    double nu = (2 * a.coarse.size() + 8) * std::numeric_limits<float>::epsilon();
    double gamma = nu < 1 ? nu / (1 - nu) : 1;
    return {std::fabs(double(value)), a.error + b.error + a.error * b.error + gamma};
}
Pair product_pair(const Normalized& a, const Normalized& b, float value) {
    if (!a.valid || !b.valid || a.coarse.size() != b.coarse.size())
        return {std::numeric_limits<double>::quiet_NaN(), 1};
    double nu = (2 * a.coarse.size() + 8) * std::numeric_limits<float>::epsilon();
    double gamma = nu < 1 ? nu / (1 - nu) : 1;
    return {std::fabs(double(value)), a.error + b.error + a.error * b.error + gamma};
}
}  // namespace

void rank_rows(Matrix& raw) {
    #pragma omp parallel
    {
        std::vector<int> order;
        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < raw.rows; ++i) {
            double* row = raw.data.data() + size_t(i) * raw.cols;
            order.clear();
            for (int j = 0; j < raw.cols; ++j) if (!std::isnan(row[j])) order.push_back(j);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return row[a] < row[b]; });
            std::vector<double> ranked(raw.cols, .5);
            for (size_t first = 0; first < order.size();) {
                size_t last = first + 1;
                while (last < order.size() && row[order[last]] == row[order[first]]) ++last;
                double rank = (first + last + 1.) / (2 * order.size());
                for (size_t j = first; j < last; ++j) ranked[order[j]] = rank;
                first = last;
            }
            double mean = std::accumulate(ranked.begin(), ranked.end(), 0.) / raw.cols;
            for (int j = 0; j < raw.cols; ++j) row[j] = ranked[j] - mean;
        }
    }
}

void jacobi(std::vector<double> A, int n, std::vector<double>& values,
            std::vector<double>& vectors) {
    vectors.assign(size_t(n) * n, 0);
    for (int i = 0; i < n; ++i) vectors[size_t(i) * n + i] = 1;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, scale = 0;
        for (int i = 0; i < n; ++i) scale = std::max(scale, std::fabs(A[size_t(i) * n + i]));
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
            double apq = A[size_t(p) * n + q]; off = std::max(off, std::fabs(apq));
            if (std::fabs(apq) <= 1e-15 * scale || apq == 0) continue;
            double tau = (A[size_t(q) * n + q] - A[size_t(p) * n + p]) / (2 * apq);
            double t = std::copysign(1.0, tau) / (std::fabs(tau) + std::hypot(1.0, tau));
            double c = 1 / std::sqrt(1 + t * t), s = t * c;
            A[size_t(p) * n + p] -= t * apq;
            A[size_t(q) * n + q] += t * apq;
            A[size_t(p) * n + q] = A[size_t(q) * n + p] = 0;
            for (int j = 0; j < n; ++j) if (j != p && j != q) {
                double x = A[size_t(j) * n + p], y = A[size_t(j) * n + q];
                A[size_t(j) * n + p] = A[size_t(p) * n + j] = c * x - s * y;
                A[size_t(j) * n + q] = A[size_t(q) * n + j] = s * x + c * y;
            }
            for (int j = 0; j < n; ++j) {
                double x = vectors[size_t(j) * n + p], y = vectors[size_t(j) * n + q];
                vectors[size_t(j) * n + p] = c * x - s * y;
                vectors[size_t(j) * n + q] = s * x + c * y;
            }
        }
        if (off <= 1e-14 * scale) break;
        if (sweep == 99) throw std::runtime_error("Jacobi did not converge");
    }
    values.resize(n);
    for (int i = 0; i < n; ++i) values[i] = A[size_t(i) * n + i];
}

Solution solve(const Matrix& R, const std::vector<double>& initial, int steps,
               const std::string& algorithm, const Sum& sum, bool diagnostics) {
    if (steps < 1 || int(initial.size()) != R.cols || R.data.size() != size_t(R.rows) * R.cols)
        throw std::invalid_argument("invalid solver shape or step count");
    Solution result;
    std::vector<double> right = initial;
    if (!unit(right)) return result;
    if (algorithm == "power") {
        for (int i = 0; i < steps; ++i) {
            auto left = multiply(R, right); ++result.matvecs;
            if (!unit(left, sum)) return result;
            right = multiply_t(R, left, sum); ++result.matvecs;
            if (!unit(right)) return result;
            ++result.steps;
        }
        result.local_signature = multiply(R, right); ++result.matvecs;
        if (!unit(result.local_signature, sum)) { result.local_signature.clear(); return result; }
    } else if (algorithm == "lanczos") {
        std::vector<std::vector<double>> U, V;
        std::vector<double> alpha, beta;
        V.push_back(right);
        auto left = multiply(R, right); ++result.matvecs;
        double a = norm(left, sum);
        if (!(a > 0)) return result;
        for (double& x : left) x /= a;
        U.push_back(left); alpha.push_back(a);
        for (int j = 0; j < steps - 1; ++j) {
            right = multiply_t(R, U.back(), sum); ++result.matvecs;
            for (int i = 0; i < R.cols; ++i) right[i] -= alpha.back() * V.back()[i];
            reorthogonalize(right, V);
            double b = norm(right);
            if (b <= 1e-13 * a) break;
            for (double& x : right) x /= b;
            left = multiply(R, right); ++result.matvecs;
            for (int i = 0; i < R.rows; ++i) left[i] -= b * U.back()[i];
            reorthogonalize(left, U, sum);
            a = norm(left, sum);
            // Exact rank breakdown: the current subspace already contains the answer.
            if (a <= 1e-13 * b) {
                // Rectangular projected bidiagonal has an extra last column beta.
                beta.push_back(b);
                break;
            }
            for (double& x : left) x /= a;
            beta.push_back(b); alpha.push_back(a); U.push_back(left); V.push_back(right);
        }
        int k = int(U.size());
        std::vector<double> gram(size_t(k) * k, 0), values, vectors;
        for (int i = 0; i < k; ++i) {
            gram[size_t(i) * k + i] = alpha[i] * alpha[i] +
                (i < int(beta.size()) ? beta[i] * beta[i] : 0);
            if (i + 1 < k)
                gram[size_t(i) * k + i + 1] = gram[size_t(i + 1) * k + i] = beta[i] * alpha[i + 1];
        }
        jacobi(gram, k, values, vectors);
        std::vector<int> order(k); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int i, int j) { return values[i] > values[j]; });
        result.local_signature.assign(R.rows, 0);
        for (int j = 0; j < k; ++j) for (int i = 0; i < R.rows; ++i)
            result.local_signature[i] += U[j][i] * vectors[size_t(j) * k + order[0]];
        unit(result.local_signature, sum);
        result.steps = k;
        if (k > 1) result.second_sigma = std::sqrt(std::max(0.0, values[order[1]]));
    } else throw std::invalid_argument("algorithm must be power or lanczos");
    if (!diagnostics) return result;
    // A posteriori relative normal-equation residual (two extra matvecs).
    auto rt = multiply_t(R, result.local_signature, sum);
    auto rr = multiply(R, rt);
    double lambda = dot(result.local_signature, rr); reduce(sum, &lambda, 1);
    result.sigma = std::sqrt(std::max(0.0, lambda));
    for (int i = 0; i < R.rows; ++i) rr[i] -= lambda * result.local_signature[i];
    result.relative_residual = norm(rr, sum) / std::max(lambda, std::numeric_limits<double>::min());
    return result;
}

std::vector<float> finish(std::vector<double> s) {
    if (s.empty() || !unit(s)) return {};
    auto pivot = std::max_element(s.begin(), s.end(), [](double a, double b) { return std::fabs(a) < std::fabs(b); });
    if (*pivot < 0) for (double& x : s) x = -x;
    return {s.begin(), s.end()};
}

double similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::quiet_NaN();
        ma += a[i]; mb += b[i];
    }
    ma /= a.size(); mb /= a.size();
    double cross = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i] - ma, y = b[i] - mb;
        cross += x * y; aa += x * x; bb += y * y;
    }
    if (!(aa > 0) || !(bb > 0)) return std::numeric_limits<double>::quiet_NaN();
    return std::fabs(cross / (std::sqrt(aa) * std::sqrt(bb)));
}
BoundedSimilarity bounded_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    auto pair = coarse(normalize(a), normalize(b));
    return {pair.value, pair.bound};
}
NormalizedSignature normalize_signature(const std::vector<float>& signature) { return normalize(signature); }
BoundedSimilarity bounded_similarity(const NormalizedSignature& a, const NormalizedSignature& b) {
    auto pair = coarse(a, b); return {pair.value, pair.bound};
}

std::vector<Decision> Archive::consider(const std::vector<Record>& candidates, int block_size) {
    if (block_size < 1) throw std::invalid_argument("block size must be positive");
    std::vector<Decision> decisions;
    for (size_t start = 0; start < candidates.size(); start += block_size) {
        size_t stop = std::min(candidates.size(), start + block_size), count = stop - start;
        auto t = Clock::now();
        std::vector<Record> snapshot;
        for (const auto& entry : records) snapshot.push_back(entry.second);
        std::vector<Normalized> candidate_norm, archive_norm;
        for (size_t i = start; i < stop; ++i) candidate_norm.push_back(normalize(candidates[i].signature));
        for (const auto& record : snapshot) archive_norm.push_back(normalize(record.signature));
        std::vector<Pair> cross(count * snapshot.size()), internal(count * count);
        std::vector<float> accelerated_cross, accelerated_internal;
        if (product && count) {
            int dimension = int(candidates[start].signature.size());
            std::vector<float> left(count * dimension, 0), right(snapshot.size() * dimension, 0);
            for (size_t i = 0; i < count; ++i) {
                if (int(candidates[start + i].signature.size()) != dimension) throw std::invalid_argument("mixed signature dimensions in product backend");
                if (candidate_norm[i].valid) std::copy(candidate_norm[i].coarse.begin(), candidate_norm[i].coarse.end(), left.begin() + i * dimension);
            }
            for (size_t i = 0; i < snapshot.size(); ++i) {
                if (int(snapshot[i].signature.size()) != dimension) throw std::invalid_argument("mixed archive dimensions in product backend");
                if (archive_norm[i].valid) std::copy(archive_norm[i].coarse.begin(), archive_norm[i].coarse.end(), right.begin() + i * dimension);
            }
            accelerated_cross = product(left, int(count), right, int(snapshot.size()), dimension);
            accelerated_internal = product(left, int(count), left, int(count), dimension);
        }
        #pragma omp parallel for
        for (int i = 0; i < int(count); ++i) {
            for (size_t j = 0; j < snapshot.size(); ++j) cross[size_t(i) * snapshot.size() + j] = product ?
                product_pair(candidate_norm[i], archive_norm[j], accelerated_cross[size_t(i) * snapshot.size() + j]) : coarse(candidate_norm[i], archive_norm[j]);
            for (size_t j = 0; j < count; ++j) internal[size_t(i) * count + j] = product ?
                product_pair(candidate_norm[i], candidate_norm[j], accelerated_internal[size_t(i) * count + j]) : coarse(candidate_norm[i], candidate_norm[j]);
        }
        coarse_pairs += cross.size() + internal.size();
        similarity_seconds += elapsed(t); t = Clock::now();
        // Sources track removals/replacements within this very block.
        std::map<std::string, std::pair<bool, size_t>> sources;
        for (size_t j = 0; j < snapshot.size(); ++j) sources[snapshot[j].key] = {false, j};
        for (size_t i = 0; i < count; ++i) {
            const auto& candidate = candidates[start + i];
            if (!std::isfinite(candidate.quality)) {
                decisions.push_back({"reject", "", "", std::numeric_limits<double>::quiet_NaN()}); continue;
            }
            struct Entry { std::string key; Pair pair; };
            std::vector<Entry> finite;
            double lower = -std::numeric_limits<double>::infinity();
            for (const auto& source : sources) {
                auto [is_candidate, j] = source.second;
                Pair pair = is_candidate ? internal[i * count + j] : cross[i * snapshot.size() + j];
                if (!std::isfinite(pair.value)) {
                    pair = {similarity(candidate.signature, records.at(source.first).signature), 0};
                    ++refined_pairs;
                }
                if (std::isfinite(pair.value)) {
                    lower = std::max(lower, pair.value - pair.bound);
                    finite.push_back({source.first, pair});
                }
            }
            double maximum = -1; std::string closest;
            bool other_similar = false;
            for (auto& entry : finite) {
                auto pair = entry.pair;
                if (pair.value + pair.bound >= lower ||
                    (pair.value - pair.bound <= .90 && pair.value + pair.bound >= .90)) {
                    pair.value = similarity(candidate.signature, records.at(entry.key).signature);
                    ++refined_pairs;
                }
                if (!std::isfinite(pair.value)) continue;
                if (entry.key != candidate.key && pair.value >= .90) other_similar = true;
                if (pair.value > maximum || (pair.value == maximum && entry.key > closest)) {
                    maximum = pair.value; closest = entry.key;
                }
            }
            auto incumbent = records.find(candidate.key);
            Decision decision;
            decision.closest = closest;
            decision.similarity = maximum < 0 ? (records.empty() ? 0 : std::numeric_limits<double>::quiet_NaN()) : maximum;
            if (incumbent != records.end()) {
                if (other_similar) decision.action = "reject";
                else if (candidate.quality <= incumbent->second.quality + .002) {
                    decision.action = "reject"; decision.closest = candidate.key;
                } else {
                    decision.action = "replace"; decision.closest = decision.replaced = candidate.key;
                    records[candidate.key] = candidate; sources[candidate.key] = {true, i};
                }
            } else if (maximum < .90) {
                decision.action = "admit"; records[candidate.key] = candidate; sources[candidate.key] = {true, i};
            } else if (candidate.quality <= records.at(closest).quality + .002) decision.action = "reject";
            else {
                decision.action = "replace"; decision.replaced = closest;
                records.erase(closest); sources.erase(closest);
                records[candidate.key] = candidate; sources[candidate.key] = {true, i};
            }
            decisions.push_back(decision);
        }
        admission_seconds += elapsed(t);
    }
    return decisions;
}
}  // namespace pca
