#include "gpp/c_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gpp/archive.hpp"
#include "gpp/engine.hpp"
#include "gpp/screening.hpp"
#ifdef GPP_HAVE_CUDA
#include "gpp/gpu.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace gpp;

namespace {

void set_error(char* err, int errlen, const std::string& message) {
    if (!err || errlen <= 0) return;
    std::snprintf(err, static_cast<size_t>(errlen), "%s", message.c_str());
}

double number(const gpe::Json& o, const char* key, double fallback) {
    return o.has(key) && o.at(key).is_number() ? o.at(key).number : fallback;
}

gpe::FitnessConfig fitness_from(const gpe::Json& o) {
    gpe::FitnessConfig f;
    if (!o.has("fitness")) return f;
    const auto& j = o.at("fitness");
    f.max_depth = static_cast<int>(number(j, "max_depth", f.max_depth));
    f.max_nodes = static_cast<int>(number(j, "max_nodes", f.max_nodes));
    f.min_valid_time_ratio = number(j, "min_valid_time_ratio", f.min_valid_time_ratio);
    f.max_missing_ratio = number(j, "max_missing_ratio", f.max_missing_ratio);
    f.stability_segments = static_cast<int>(number(j, "stability_segments", f.stability_segments));
    f.stability_penalty = number(j, "stability_penalty", f.stability_penalty);
    f.complexity_penalty = number(j, "complexity_penalty", f.complexity_penalty);
    f.minimum_same_sign_ratio = number(j, "minimum_same_sign_ratio", f.minimum_same_sign_ratio);
    f.inconsistent_sign_multiplier = number(j, "inconsistent_sign_multiplier", f.inconsistent_sign_multiplier);
    return f;
}

EngineOptions engine_options(const gpe::Json& o) {
    EngineOptions e;
    e.threads = static_cast<int>(number(o, "threads", 0));
    e.memory_budget_bytes = number(o, "memory_budget_gb", 0) * 1073741824.0;
    e.max_chunk = static_cast<int>(number(o, "max_chunk", e.max_chunk));
    e.min_chunk = static_cast<int>(number(o, "min_chunk", 0));
    e.narrow_level = static_cast<int>(number(o, "narrow_level", 0));
    e.gpus = static_cast<int>(number(o, "gpus", 0));
    e.streams_per_gpu = static_cast<int>(number(o, "streams_per_gpu", e.streams_per_gpu));
    e.score.fitness = fitness_from(o);
    e.score.begin = static_cast<int>(number(o, "score_begin", 0));
    e.score.signature_step = static_cast<int>(number(o, "signature_step", 0));
    e.score.power_steps = static_cast<int>(number(o, "power_steps", 4));
    return e;
}

gpe::PanelSet open_panels(const char* dir, int T, int N) {
    gpe::PanelSet data = gpe::PanelSet::open(dir);
    if (T > 0) {
        if (T > data.T) throw std::invalid_argument("T exceeds the exported dates");
        data.T = T;
    }
    if (N > 0) {
        if (N > data.N) throw std::invalid_argument("N exceeds the exported assets");
        data.N = N;
    }
    return data;
}

std::vector<std::string> split_keys(const char* keys, int count) {
    std::vector<std::string> out;
    out.reserve(count);
    const char* p = keys;
    for (int i = 0; i < count; ++i) {
        out.emplace_back(p);
        p += out.back().size() + 1;
    }
    return out;
}

std::vector<gpe::Json> parse_expressions(const char* text, int count) {
    gpe::Json all = gpe::parse_json(text);
    if (!all.is_array() || static_cast<int>(all.array.size()) != count)
        throw std::invalid_argument("expressions_json must be an array of `count` expressions");
    return std::move(all.array);
}

void write_out(char* out, int outlen, const std::string& text) {
    if (out && outlen > 0) std::snprintf(out, static_cast<size_t>(outlen), "%s", text.c_str());
}

std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream s;
    s.precision(17);
    s << v;
    return s.str();
}

struct ArchiveHandle {
    QdArchive archive;
    int gpus = 0;
    ArchiveHandle(int dim, double thr, double margin) : archive(dim, thr, margin) {}
};

}  // namespace

extern "C" {

int gpp_has_cuda(void) {
#ifdef GPP_HAVE_CUDA
    return 1;
#else
    return 0;
#endif
}

int gpp_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void* gpp_engine_open(const char* panel_dir, int T, int N, const char* options_json, char* err, int errlen) {
    try {
        gpe::Json o = gpe::parse_json(options_json ? options_json : "{}");
        return new Engine(open_panels(panel_dir, T, N), engine_options(o));
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return nullptr;
    }
}

int gpp_engine_signature_rows(void* h) { return static_cast<Engine*>(h)->sampled_signature_rows(); }

int gpp_engine_set_initial(void* h, const double* initial, int n, char* err, int errlen) {
    try {
        static_cast<Engine*>(h)->set_initial(std::vector<double>(initial, initial + n));
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

int gpp_engine_evaluate(void* h, const char* expressions_json, int count, int segments, double* stats,
                        float* signatures, int* flags, char* err, int errlen) {
    try {
        Engine& engine = *static_cast<Engine*>(h);
        auto expressions = parse_expressions(expressions_json, count);
        std::vector<std::string> errors;
        auto scores = engine.evaluate(expressions, errors);
        const int W = 8 + segments, m = engine.sampled_signature_rows();
        for (int i = 0; i < count; ++i) {
            const RootScore& s = scores[i];
            const auto& c = s.candidate;
            double* row = stats + static_cast<size_t>(i) * W;
            for (int k = 0; k < W; ++k) row[k] = std::nan("");
            int f = 0;
            if (!errors[i].empty()) {
                flags[i] = 1;
                continue;
            }
            if (c.valid) {
                f |= 2;
                row[0] = c.rank_ic;
                row[1] = c.valid_ratio;
                row[2] = c.missing_ratio;
                row[3] = c.fitness;
                row[4] = c.segment_rank_ic_std;
                row[5] = c.same_sign_ratio;
                for (int k = 0; k < segments && k < static_cast<int>(c.segment_rank_ics.size()); ++k)
                    row[8 + k] = c.segment_rank_ics[k];
            }
            if (signatures) {
                float* out = signatures + static_cast<size_t>(i) * m;
                if (s.signature_valid && static_cast<int>(s.signature.size()) == m) {
                    std::memcpy(out, s.signature.data(), sizeof(float) * m);
                    f |= 4;
                } else {
                    std::fill(out, out + m, 0.0f);
                }
            }
            flags[i] = f;
        }
        // depth/nodes from the DAG are not needed by Python (it recomputes them).
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

int gpp_engine_plan(void* h, const char* expressions_json, char* out, int outlen, char* err, int errlen) {
    try {
        Engine& engine = *static_cast<Engine*>(h);
        gpe::Json all = gpe::parse_json(expressions_json);
        std::vector<std::string> errors;
        auto groups = engine.plan(all.array, errors);
        std::ostringstream s;
        s << "{\"panel_bytes\":" << json_number(engine.panel_bytes()) << ",\"budget_panels\":" << engine.budget_panels()
          << ",\"resident_bytes\":" << json_number(engine.stats().resident_bytes)
          << ",\"scratch_bytes\":" << json_number(engine.stats().scratch_bytes) << ",\"groups\":[";
        for (size_t g = 0; g < groups.size(); ++g)
            s << (g ? "," : "") << "{\"expressions\":" << groups[g].expressions.size()
              << ",\"dag_nodes\":" << groups[g].dag_nodes << ",\"peak_panels\":" << groups[g].peak_panels
              << ",\"chunks\":" << groups[g].chunks << ",\"feasible\":" << (groups[g].feasible ? "true" : "false")
              << "}";
        s << "]}";
        write_out(out, outlen, s.str());
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

int gpp_engine_stats(void* h, char* out, int outlen) {
    const auto& e = static_cast<Engine*>(h)->stats();
    std::ostringstream s;
    s << "{\"load\":" << json_number(e.load) << ",\"plan\":" << json_number(e.plan)
      << ",\"compute\":" << json_number(e.compute) << ",\"score_wait\":" << json_number(e.score_wait)
      << ",\"total\":" << json_number(e.total) << ",\"batches\":" << e.batches << ",\"groups\":" << e.groups
      << ",\"expressions\":" << e.expressions << ",\"dag_nodes\":" << e.dag_nodes << ",\"roots\":" << e.roots
      << ",\"chunks\":" << e.chunks << ",\"peak_panels\":" << e.peak_panels
      << ",\"peak_bytes\":" << json_number(e.peak_bytes) << ",\"resident_bytes\":" << json_number(e.resident_bytes)
      << ",\"scratch_bytes\":" << json_number(e.scratch_bytes)
      << ",\"threads\":" << static_cast<Engine*>(h)->threads() << "}";
    write_out(out, outlen, s.str());
    return 0;
}

void gpp_engine_close(void* h) { delete static_cast<Engine*>(h); }

void* gpp_archive_new(int dim, double threshold, double margin, const char* options_json, char* err, int errlen) {
    try {
        gpe::Json o = gpe::parse_json(options_json ? options_json : "{}");
        auto* h = new ArchiveHandle(dim, threshold, margin);
        h->archive.top_k = static_cast<int>(number(o, "top_k", 4));
        h->gpus = static_cast<int>(number(o, "gpus", 0));
#ifdef GPP_HAVE_CUDA
        if (h->gpus > 0) {
            int gpus = h->gpus;
            h->archive.backend = [gpus](const SummaryInputs& in, std::vector<RowSummary>& out) {
                gpu_summaries(in, out, gpus);
            };
        }
#else
        if (h->gpus > 0) {
            delete h;
            throw std::invalid_argument("this build has no CUDA (GPP_ENABLE_CUDA=OFF)");
        }
#endif
        return h;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return nullptr;
    }
}

int gpp_archive_reset(void* h, int count, const char* keys, const double* quality, const float* signatures,
                      char* err, int errlen) {
    try {
        auto& a = static_cast<ArchiveHandle*>(h)->archive;
        auto names = split_keys(keys, count);
        std::vector<QdCandidate> records(count);
        for (int i = 0; i < count; ++i)
            records[i] = {names[i], quality[i], signatures + static_cast<size_t>(i) * a.dim()};
        a.reset(records);
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

int gpp_archive_consider(void* h, int count, const char* keys, const double* quality, const float* signatures,
                         int block, int* action, double* similarity, int* closest, int* replaced, int* new_slot,
                         char* err, int errlen) {
    try {
        auto& a = static_cast<ArchiveHandle*>(h)->archive;
        auto names = split_keys(keys, count);
        std::vector<QdCandidate> records(count);
        for (int i = 0; i < count; ++i)
            records[i] = {names[i], quality[i], signatures + static_cast<size_t>(i) * a.dim()};
        auto decisions = a.consider(records, block);
        for (int i = 0; i < count; ++i) {
            action[i] = static_cast<int>(decisions[i].action);
            similarity[i] = decisions[i].max_similarity;
            closest[i] = decisions[i].closest_slot;
            replaced[i] = decisions[i].replaced_slot;
            new_slot[i] = decisions[i].new_slot;
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

int gpp_archive_size(void* h) { return static_cast<int>(static_cast<ArchiveHandle*>(h)->archive.size()); }

int gpp_archive_stats(void* h, char* out, int outlen) {
    const auto& t = static_cast<ArchiveHandle*>(h)->archive.timing;
    std::ostringstream s;
    s << "{\"similarity\":" << json_number(t.similarity) << ",\"admission\":" << json_number(t.admission)
      << ",\"refined\":" << t.refined << ",\"fallbacks\":" << t.fallbacks << ",\"candidates\":" << t.candidates
      << ",\"size\":" << static_cast<ArchiveHandle*>(h)->archive.size() << "}";
    write_out(out, outlen, s.str());
    return 0;
}

void gpp_archive_close(void* h) { delete static_cast<ArchiveHandle*>(h); }

int gpp_screen(const char* panel_dir, int T, int N, const char* options_json, const char* expressions_json, int count,
               const double* evolution_rank_ic, const int* nodes, const char* keys, double* mean, double* valid_ratio,
               double* missing_ratio, int* valid_times, int* status, char* errors, int error_len, int* order,
               int* n_order, unsigned char* selected, double* max_corr, int* corr_with, long* scanned,
               char* stats_json, int stats_len, char* err, int errlen) {
    try {
        gpe::Json o = gpe::parse_json(options_json ? options_json : "{}");
        ScreenOptions so;
        so.rankic_threshold = number(o, "rankic_threshold", so.rankic_threshold);
        so.corr_threshold = number(o, "corr_threshold", so.corr_threshold);
        so.sample_step = static_cast<int>(number(o, "sample_step", so.sample_step));
        so.limit = static_cast<long>(number(o, "limit", -1));
        so.block = static_cast<int>(number(o, "block", so.block));
        so.engine_batch = static_cast<int>(number(o, "engine_batch", so.engine_batch));
        so.band = number(o, "band", so.band);
        EngineOptions eo = engine_options(o);
        eo.score.signature_step = 0;
        eo.score.rank_vector_step = so.sample_step;
        eo.score.rank_vector_min_abs_ic = so.rankic_threshold;
        // Screening scores every candidate: no fitness filter applies.
        eo.score.fitness.max_depth = 1 << 20;
        eo.score.fitness.max_nodes = 1 << 20;
        Engine engine(open_panels(panel_dir, T, N), eo);

        auto expressions = parse_expressions(expressions_json, count);
        auto names = split_keys(keys, count);
        std::vector<ScreenInput> inputs(count);
        for (int i = 0; i < count; ++i) inputs[i] = {&expressions[i], evolution_rank_ic[i], nodes[i], names[i]};
        PairSumBackend backend;
#ifdef GPP_HAVE_CUDA
        if (eo.gpus > 0 && number(o, "gpu_products", 0) != 0) {
            // FP64 DGEMM tiles: reordering error ~1e-13, far inside the band.
            int gpus = eo.gpus;
            so.band = std::max(so.band, number(o, "gpu_band", 1e-6));
            backend = [gpus](const std::vector<const float*>& l, const std::vector<const float*>& r, size_t n,
                             std::vector<PairSums>& out) { gpu_pair_sums(l, r, n, out, gpus); };
        }
#endif
        ScreenResult r = screen_factors(engine, inputs, so, backend);
        for (int i = 0; i < count; ++i) {
            mean[i] = r.mean[i];
            valid_ratio[i] = r.valid_ratio[i];
            missing_ratio[i] = r.missing_ratio[i];
            valid_times[i] = r.valid_times[i];
            status[i] = r.error[i].empty() ? 0 : 1;
            if (errors && error_len > 0) std::snprintf(errors + static_cast<size_t>(i) * error_len, error_len, "%s",
                                                       r.error[i].c_str());
        }
        *n_order = static_cast<int>(r.order.size());
        for (size_t k = 0; k < r.order.size(); ++k) order[k] = r.order[k];
        *scanned = r.scanned;
        for (long k = 0; k < r.scanned; ++k) {
            selected[k] = r.selected[k];
            max_corr[k] = r.max_correlation[k];
            corr_with[k] = r.correlated_with[k];
        }
        const auto& t = r.timing;
        const auto& es = engine.stats();
        std::ostringstream s;
        s << "{\"evaluate\":" << json_number(t.evaluate) << ",\"sort\":" << json_number(t.sort)
          << ",\"products\":" << json_number(t.products) << ",\"scan\":" << json_number(t.scan)
          << ",\"total\":" << json_number(t.total) << ",\"refined\":" << t.refined << ",\"pairs\":" << t.pairs
          << ",\"dag_nodes\":" << es.dag_nodes << ",\"groups\":" << es.groups << ",\"roots\":" << es.roots
          << ",\"peak_bytes\":" << json_number(es.peak_bytes) << "}";
        write_out(stats_json, stats_len, s.str());
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
        return 1;
    }
}

}  // extern "C"
