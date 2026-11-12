#include "core.hpp"
#include "gpe/json.hpp"
#include "json_output.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <omp.h>
#ifdef PCA_CUDA
#include "gpu.hpp"
#endif
#ifdef PCA_MPI
#include <mpi.h>
#include "mpi_archive.hpp"
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#endif

namespace {
using gpe::Json;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
Json number(double v) { Json j; if (std::isfinite(v)) { j.type = Json::Type::Number; j.number = v; } return j; }
Json string(const std::string& s) { Json j; j.type = Json::Type::String; j.string = s; return j; }
Json array() { Json j; j.type = Json::Type::Array; return j; }
Json object() { Json j; j.type = Json::Type::Object; return j; }
void put(Json& j, const std::string& key, double v) { j.object[key] = number(v); }
void put(Json& j, const std::string& key, const std::string& v) { j.object[key] = string(v); }
void write(const std::string& path, const Json& value) {
    std::ofstream out(path); pca::write_json(out, value); out << '\n';
    if (!out) throw std::runtime_error("cannot write " + path);
}
template<class T> std::vector<T> read_binary(const std::string& path, size_t offset, size_t count) {
    // Reading many factors from the same chunks should not open the same file
    // millions of times during date-decomposed scaling runs.
    static std::map<std::string, std::unique_ptr<std::ifstream>> streams;
    if (!streams.count(path)) streams[path] = std::make_unique<std::ifstream>(path, std::ios::binary);
    auto& input = *streams.at(path);
    if (!input.is_open()) throw std::runtime_error("cannot open " + path);
    input.clear();
    input.seekg(offset * sizeof(T));
    std::vector<T> data(count);
    input.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    if (!input) throw std::runtime_error("short read " + path);
    return data;
}
struct Block { int begin, count; };
Block block(int n, int P, int rank) {
    int size = n / P, extra = n % P;
    return {rank * size + std::min(rank, extra), size + (rank < extra)};
}
int rank = 0, ranks = 1;
double communication = 0;
long long collective_calls = 0, collective_elements = 0;
void sum(double* data, int count) {
#ifdef PCA_MPI
    auto t = Clock::now();
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    communication += seconds(t); ++collective_calls; collective_elements += count;
#endif
}
double max_rank(double v) {
#ifdef PCA_MPI
    double answer; MPI_Allreduce(&v, &answer, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return answer;
#else
    return v;
#endif
}

// Batched row decomposition: one B-scalar and one B*N-vector reduction per step.
std::vector<std::vector<double>> batch_power(const std::vector<pca::Matrix>& matrices,
                                            const std::vector<double>& initial, int k) {
    int B = int(matrices.size()), n = int(initial.size()), m = B ? matrices[0].rows : 0;
    std::vector<double> V(size_t(B) * n), U(size_t(B) * m), norms(B);
    double initnorm = 0; for (double v : initial) initnorm += v * v;
    initnorm = std::sqrt(initnorm);
    if (!(initnorm > 0)) throw std::runtime_error("zero initial vector");
    for (int b = 0; b < B; ++b) for (int j = 0; j < n; ++j) V[size_t(b) * n + j] = initial[j] / initnorm;
    for (int step = 0; step <= k; ++step) {
        #pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            norms[b] = 0;
            for (int i = 0; i < m; ++i) {
                double value = 0;
                for (int j = 0; j < n; ++j) value += matrices[b].data[size_t(i) * n + j] * V[size_t(b) * n + j];
                U[size_t(b) * m + i] = value; norms[b] += value * value;
            }
        }
        sum(norms.data(), B);
        for (int b = 0; b < B; ++b) {
            double length = std::sqrt(norms[b]);
            for (int i = 0; i < m; ++i) U[size_t(b) * m + i] = length > 0 ? U[size_t(b) * m + i] / length : 0;
        }
        if (step == k) break;
        std::fill(V.begin(), V.end(), 0);
        #pragma omp parallel for
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    V[size_t(b) * n + j] += matrices[b].data[size_t(i) * n + j] * U[size_t(b) * m + i];
        sum(V.data(), int(V.size()));
        #pragma omp parallel for
        for (int b = 0; b < B; ++b) {
            double length = 0;
            for (int j = 0; j < n; ++j) length += V[size_t(b) * n + j] * V[size_t(b) * n + j];
            length = std::sqrt(length);
            for (int j = 0; j < n; ++j) if (length > 0) V[size_t(b) * n + j] /= length;
        }
    }
    std::vector<std::vector<double>> output(B);
    for (int b = 0; b < B; ++b) output[b] = {U.begin() + size_t(b) * m, U.begin() + size_t(b + 1) * m};
    return output;
}

template<class Scalar>
std::vector<pca::Matrix> transpose_to_factors(const std::vector<pca::Matrix>& local,
                                            int m, int n, int count) {
    Block own = block(count, ranks, rank), dates = block(m, ranks, rank);
    std::vector<int> sendcounts(ranks), recvcounts(ranks), sd(ranks), rd(ranks);
    size_t sendsize = 0, recvsize = 0;
    for (int r = 0; r < ranks; ++r) {
        size_t sc = size_t(block(count, ranks, r).count) * dates.count * n;
        size_t rc = size_t(own.count) * block(m, ranks, r).count * n;
        if (sc > size_t(std::numeric_limits<int>::max()) || rc > size_t(std::numeric_limits<int>::max()) ||
            sendsize + sc > size_t(std::numeric_limits<int>::max()) || recvsize + rc > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("MPI count overflow: use a smaller --chunk");
        sendcounts[r] = int(sc); recvcounts[r] = int(rc);
        sd[r] = int(sendsize); rd[r] = int(recvsize); sendsize += sc; recvsize += rc;
    }
    std::vector<Scalar> sendbuf(sendsize), recvbuf(recvsize);
    for (int r = 0; r < ranks; ++r) {
        Block factors = block(count, ranks, r); size_t pos = sd[r];
        for (int b = factors.begin; b < factors.begin + factors.count; ++b)
            for (double x : local[b].data) sendbuf[pos++] = Scalar(x);
    }
#ifdef PCA_MPI
    auto t = Clock::now();
    auto datatype = sizeof(Scalar) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sd.data(), datatype,
                  recvbuf.data(), recvcounts.data(), rd.data(), datatype, MPI_COMM_WORLD);
    communication += seconds(t); ++collective_calls; collective_elements += sendsize;
#else
    recvbuf = sendbuf;
#endif
    std::vector<pca::Matrix> output(own.count);
    for (auto& matrix : output) { matrix.rows = m; matrix.cols = n; matrix.data.resize(size_t(m) * n); }
    for (int r = 0; r < ranks; ++r) {
        Block d = block(m, ranks, r); size_t pos = rd[r];
        for (int b = 0; b < own.count; ++b)
            for (int i = 0; i < d.count; ++i) for (int j = 0; j < n; ++j)
                output[b].data[size_t(d.begin + i) * n + j] = recvbuf[pos++];
    }
    return output;
}

int run(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("options require values");
        args[argv[i]] = argv[i + 1];
    }
    auto option = [&](const std::string& key, const std::string& fallback) {
        return args.count(key) ? args.at(key) : fallback;
    };
    std::string manifest = args.at("--manifest"), out = args.at("--out");
    std::string method = option("--method", "power"), layout = option("--layout", "factor");
    std::string archive_mode = option("--archive", "root");
    if (archive_mode != "root" && archive_mode != "mpi") throw std::invalid_argument("unknown archive mode");
#ifndef PCA_MPI
    if (archive_mode == "mpi") throw std::invalid_argument("MPI archive needs an MPI build");
#endif
    if (layout != "date" && layout != "factor" && layout != "transpose") throw std::invalid_argument("unknown layout");
    if (method != "power" && method != "lanczos") throw std::invalid_argument("unknown method");
    int k = std::stoi(option("--steps", "4")), chunk = std::stoi(option("--chunk", "128"));
    bool rank_input = option("--rank-input", "0") == "1";
    if (k < 1 || k > 100 || chunk < 1) throw std::invalid_argument("invalid steps/chunk");
    Json input = gpe::parse_json_file(manifest);
    int m = int(input.at("rows").number), n = int(input.at("cols").number);
    const auto& entries = input.at("records").array;
    int count = std::min(int(entries.size()), std::stoi(option("--count", std::to_string(entries.size()))));
    if (m < 2 || n < 1 || count < 1) throw std::invalid_argument("invalid shape/count");
    auto initial = read_binary<double>(option("--init", input.at("initial").string), 0, n);
    std::string supplied = option("--signature-input", "");
    std::string device = option("--device", "cpu"), precision = option("--precision", "double");
    if (device != "cpu" && device != "cuda") throw std::invalid_argument("unknown device");
    if (precision != "float" && precision != "double") throw std::invalid_argument("unknown precision");
    if (device == "cuda" && method != "power") throw std::invalid_argument("CUDA currently supports power");
    bool cuda_aware = option("--cuda-aware", "0") == "1";
#if defined(PCA_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA)
    if (cuda_aware && !MPIX_Query_cuda_support()) throw std::runtime_error("MPI runtime has no CUDA device-buffer support");
#endif
    if (device == "cuda" && layout == "date" && !cuda_aware)
        throw std::invalid_argument("CUDA date layout requires --cuda-aware 1 and a verified CUDA-aware MPI build");
#ifdef PCA_CUDA
    pca::GpuTiming gpu;
    if (device == "cuda") {
        int local_rank = 0;
#ifdef PCA_MPI
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
#endif
        pca::gpu_select(local_rank);
    }
#else
    if (device == "cuda") throw std::runtime_error("rebuild with -DPCA_CUDA=ON");
#endif
    if (rank == 0) {
        if (std::filesystem::exists(out)) throw std::runtime_error("refusing to overwrite " + out);
        std::filesystem::create_directories(out);
    }
#ifdef PCA_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto total_start = Clock::now();
    double load_time = 0, solve_time = 0, gather_time = 0, transpose_time = 0, ranking_time = 0;
    Json diagnostics = array(), decisions_json = array();
    std::vector<pca::Record> candidates;
    pca::Archive archive;
#ifdef PCA_MPI
    pca::MpiArchive distributed_archive(MPI_COMM_WORLD);
#endif
#ifdef PCA_CUDA
    if (device == "cuda") archive.product = [&](const std::vector<float>& x, int L,
        const std::vector<float>& y, int A, int d) { return pca::gpu_similarity(x, L, y, A, d, gpu); };
#endif
    std::ofstream sigfile;
    if (rank == 0) sigfile.open(out + "/signatures.f32", std::ios::binary);
    for (int start = 0; start < count; start += chunk) {
        int B = std::min(chunk, count - start);
        Block own = block(B, ranks, rank), dates = block(m, ranks, rank);
        auto t = Clock::now();
        std::vector<pca::Matrix> matrices;
        if (supplied.empty()) {
            int first = layout == "factor" ? own.begin : 0;
            int last = layout == "factor" ? own.begin + own.count : B;
            int row0 = layout == "factor" ? 0 : dates.begin;
            int rows = layout == "factor" ? m : dates.count;
            for (int b = first; b < last; ++b) {
                const auto& e = entries[start + b];
                size_t offset = size_t(e.at("slot").number) * m * n + size_t(row0) * n;
                auto values = read_binary<float>(e.at("path").string, offset, size_t(rows) * n);
                matrices.push_back({rows, n, {values.begin(), values.end()}});
            }
        }
        load_time += seconds(t);
        if (rank_input && supplied.empty()) {
            auto rank_start = Clock::now();
            for (auto& matrix : matrices) {
#ifdef PCA_CUDA
                if (device == "cuda") {
                    if (matrix.rows) {
                        std::vector<float> raw(matrix.data.begin(), matrix.data.end());
                        auto values = pca::gpu_rank_rows(raw, matrix.rows, n, gpu);
                        matrix.data.assign(values.begin(), values.end());
                    }
                } else
#endif
                pca::rank_rows(matrix);
            }
            ranking_time += seconds(rank_start);
        }
        if (layout == "transpose" && supplied.empty()) {
            t = Clock::now();
            // Pre-exported R is FP32. Fresh CPU ranks are FP64 and must not
            // lose precision solely because the layout is changed.
            matrices = rank_input ? transpose_to_factors<double>(matrices, m, n, B)
                                  : transpose_to_factors<float>(matrices, m, n, B);
            transpose_time += seconds(t);
        }
        t = Clock::now();
        std::vector<float> full(size_t(B) * m, 0);
        if (!supplied.empty()) {
            if (rank == 0) full = read_binary<float>(supplied, size_t(start) * m, size_t(B) * m);
        } else if (layout == "date") {
            std::vector<std::vector<double>> local;
#ifdef PCA_CUDA
            if (device == "cuda") {
                pca::DeviceReduce device_sum = [&](void* buffer, int elements, bool fp64) {
#ifdef PCA_MPI
                    auto begin = Clock::now();
                    MPI_Allreduce(MPI_IN_PLACE, buffer, elements, fp64 ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    communication += seconds(begin); ++collective_calls; collective_elements += elements;
#else
                    (void)buffer; (void)elements; (void)fp64;
#endif
                };
                auto signatures = pca::gpu_power(matrices, initial, k, precision == "double", gpu, device_sum);
                for (auto& signature : signatures) local.emplace_back(signature.begin(), signature.end());
            } else
#endif
            if (method == "power") local = batch_power(matrices, initial, k);
            else for (auto& matrix : matrices) local.push_back(pca::solve(matrix, initial, k, method, sum, false).local_signature);
            solve_time += seconds(t); t = Clock::now();
            std::vector<double> packed(size_t(B) * dates.count, 0);
            for (int b = 0; b < B; ++b) if (!local[b].empty())
                std::copy(local[b].begin(), local[b].end(), packed.begin() + size_t(b) * dates.count);
            std::vector<int> counts(ranks), displacements(ranks);
            int position = 0;
            for (int r = 0; r < ranks; ++r) { counts[r] = B * block(m, ranks, r).count; displacements[r] = position; position += counts[r]; }
            std::vector<double> received(rank == 0 ? size_t(B) * m : 0);
#ifdef PCA_MPI
            MPI_Gatherv(packed.data(), int(packed.size()), MPI_DOUBLE, received.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
            received = packed;
#endif
            if (rank == 0) for (int b = 0; b < B; ++b) {
                std::vector<double> signature(m);
                for (int r = 0; r < ranks; ++r) {
                    Block d = block(m, ranks, r);
                    for (int i = 0; i < d.count; ++i)
                        signature[d.begin + i] = received[displacements[r] + b * d.count + i];
                }
                auto normalized = pca::finish(signature);
                if (!normalized.empty()) std::copy(normalized.begin(), normalized.end(), full.begin() + size_t(b) * m);
            }
            gather_time += seconds(t);
        } else {
            std::vector<pca::Solution> solutions(own.count);
            std::vector<std::vector<float>> gpu_signatures;
#ifdef PCA_CUDA
            if (device == "cuda") gpu_signatures = pca::gpu_power(matrices, initial, k, precision == "double", gpu);
#endif
            // Each factor is independent; suppress nested operator teams.
            #pragma omp parallel for schedule(dynamic)
            for (int b = 0; b < own.count; ++b)
                if (device == "cpu") solutions[b] = pca::solve(matrices[b], initial, k, method, {}, false);
            std::vector<float> local(size_t(own.count) * m, 0);
            for (int b = 0; b < own.count; ++b) {
                auto s = device == "cuda" ? gpu_signatures[b] : pca::finish(solutions[b].local_signature);
                if (!s.empty()) std::copy(s.begin(), s.end(), local.begin() + size_t(b) * m);
            }
            solve_time += seconds(t); t = Clock::now();
            std::vector<int> counts(ranks), displacements(ranks);
            for (int r = 0; r < ranks; ++r) { auto b = block(B, ranks, r); counts[r] = b.count * m; displacements[r] = b.begin * m; }
#ifdef PCA_MPI
            MPI_Gatherv(local.data(), int(local.size()), MPI_FLOAT, full.data(), counts.data(), displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
            full = local;
#endif
            gather_time += seconds(t);
        }
        if (rank == 0) sigfile.write(reinterpret_cast<const char*>(full.data()), full.size() * sizeof(float));
#ifdef PCA_MPI
        if (archive_mode == "mpi") {
            auto broadcast_start = Clock::now();
            MPI_Bcast(full.data(), int(full.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
            communication += seconds(broadcast_start);
        }
#endif
        if (rank == 0 || archive_mode == "mpi") {
            candidates.clear(); std::vector<int> candidate_indices;
            for (int b = 0; b < B; ++b) {
                const auto& e = entries[start + b];
                std::vector<float> signature(full.begin() + size_t(b) * m, full.begin() + size_t(b + 1) * m);
                if (!e.at("candidate").boolean || !std::isfinite(pca::similarity(signature, signature))) continue;
                candidates.push_back({e.at("key").string, e.at("quality").number, signature});
                candidate_indices.push_back(start + b);
            }
            std::vector<pca::Decision> decisions;
#ifdef PCA_MPI
            if (archive_mode == "mpi") decisions = distributed_archive.consider(candidates);
            else
#endif
            decisions = archive.consider(candidates, 128);
            if (rank == 0) {
            for (size_t i = 0; i < decisions.size(); ++i) {
                auto j = object(); const auto& d = decisions[i];
                put(j, "signature_index", candidate_indices[i]); put(j, "key", candidates[i].key);
                put(j, "action", d.action); put(j, "similarity", d.similarity);
                j.object["closest_key"] = d.closest.empty() ? Json() : string(d.closest);
                j.object["replaced_key"] = d.replaced.empty() ? Json() : string(d.replaced);
                decisions_json.array.push_back(j);
            }
            std::cout << "completed " << start + B << "/" << count << std::endl;
            }
        }
    }
    double total = max_rank(seconds(total_start));
    Json metrics = object();
    for (auto phase : std::vector<std::pair<std::string,double>>{
             {"load_seconds", load_time}, {"ranking_seconds", ranking_time}, {"solve_seconds", solve_time}, {"transpose_seconds", transpose_time},
             {"communication_seconds", communication}, {"gather_seconds", gather_time}})
        put(metrics, phase.first, max_rank(phase.second));
    put(metrics, "total_seconds", total); put(metrics, "ranks", ranks); put(metrics, "threads", omp_get_max_threads());
    put(metrics, "factors", count); put(metrics, "rows", m); put(metrics, "cols", n); put(metrics, "steps", k);
    put(metrics, "method", method); put(metrics, "layout", layout); put(metrics, "chunk", chunk);
    put(metrics, "archive_mode", archive_mode);
    put(metrics, "rank_input", rank_input ? 1 : 0);
#ifdef PCA_MPI
    if (archive_mode == "mpi") {
        archive.similarity_seconds = max_rank(distributed_archive.similarity_seconds);
        archive.admission_seconds = max_rank(distributed_archive.admission_seconds);
        put(metrics, "archive_communication_seconds", max_rank(distributed_archive.communication_seconds));
        archive.coarse_pairs = distributed_archive.coarse_pairs;
        archive.refined_pairs = distributed_archive.refined_pairs;
    }
#endif
    put(metrics, "device", device); put(metrics, "precision", precision);
    put(metrics, "cuda_aware", cuda_aware ? 1 : 0);
#ifdef PCA_CUDA
    for (auto phase : std::vector<std::pair<std::string, double>>{
        {"gpu_upload_seconds", gpu.upload}, {"gpu_download_seconds", gpu.download},
        {"gpu_iteration_seconds", gpu.iteration}, {"gpu_gemv_seconds", gpu.gemv},
        {"gpu_similarity_seconds", gpu.similarity}, {"gpu_ranking_seconds", gpu.ranking}})
        put(metrics, phase.first, max_rank(phase.second));
    put(metrics, "gpu_matrix_bytes", gpu.bytes); put(metrics, "gpu_matvec_flops", gpu.flops);
    put(metrics, "gpu_name", gpu.device);
#endif
    Json replay; replay.type = Json::Type::Bool; replay.boolean = !supplied.empty();
    metrics.object["input_signatures"] = replay;
    put(metrics, "collective_calls", double(collective_calls)); put(metrics, "collective_elements", double(collective_elements));
    if (rank == 0) {
        if (!sigfile) throw std::runtime_error("signature output failed");
        put(metrics, "coarse_pairs", double(archive.coarse_pairs)); put(metrics, "refined_pairs", double(archive.refined_pairs));
        put(metrics, "similarity_seconds", archive.similarity_seconds); put(metrics, "admission_seconds", archive.admission_seconds);
        size_t archive_size = archive.records.size();
        auto keys = array(); for (const auto& entry : archive.records) keys.array.push_back(string(entry.first));
#ifdef PCA_MPI
        if (archive_mode == "mpi") {
            keys.array.clear(); archive_size = distributed_archive.qualities.size();
            for (const auto& entry : distributed_archive.qualities) keys.array.push_back(string(entry.first));
        }
#endif
        put(metrics, "archive_size", double(archive_size));
        write(out + "/metrics.json", metrics); write(out + "/decisions.json", decisions_json); write(out + "/archive_keys.json", keys);
    }
    return 0;
}
}  // namespace
int main(int argc, char** argv) {
#ifdef PCA_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif
    omp_set_max_active_levels(1);
    try {
        int result = run(argc, argv);
#ifdef PCA_MPI
        MPI_Finalize();
#endif
        return result;
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << std::endl;
#ifdef PCA_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
