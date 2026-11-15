/* C interface of libgpp_prod, loaded by python/gpp/native.py with ctypes.
 * Every function returns 0 on success; on failure it returns non-zero and
 * writes a message into err (errlen bytes). Strings lists ("keys") are
 * concatenations of NUL-terminated UTF-8 strings.
 */
#ifndef GPP_C_API_H
#define GPP_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- evaluation engine (one per panel set, kept for the whole run) ---- */
/* options_json: {"threads", "memory_budget_gb", "max_chunk", "min_chunk",
 *   "signature_step", "power_steps", "fitness": {...}, "gpus", "streams_per_gpu"} */
void* gpp_engine_open(const char* panel_dir, int T, int N, const char* options_json, char* err, int errlen);
int gpp_engine_signature_rows(void* engine);
int gpp_engine_set_initial(void* engine, const double* initial, int n, char* err, int errlen);
/* expressions_json: JSON array of expressions in the Python nested-list form.
 * stats: count x (8 + segments) doubles:
 *   rank_ic, valid_ratio, missing_ratio, fitness, segment_std, same_sign_ratio, depth, nodes, segments...
 * signatures: count x signature_rows floats (may be NULL).
 * flags: bit 0 parse error, bit 1 valid candidate, bit 2 comparable signature. */
int gpp_engine_evaluate(void* engine, const char* expressions_json, int count, int segments, double* stats,
                        float* signatures, int* flags, char* err, int errlen);
/* Memory plan of a batch without evaluating it: groups, dag nodes, peak panels (JSON). */
int gpp_engine_plan(void* engine, const char* expressions_json, char* out, int outlen, char* err, int errlen);
int gpp_engine_stats(void* engine, char* out, int outlen);
void gpp_engine_close(void* engine);

/* ---- PCA-QD archive ---- */
void* gpp_archive_new(int dim, double threshold, double margin, const char* options_json, char* err, int errlen);
int gpp_archive_reset(void* archive, int count, const char* keys, const double* quality, const float* signatures,
                      char* err, int errlen);
/* action: 0 admit, 1 replace, 2 reject. closest/replaced/new_slot: archive slots, -1 = None.
 * Slots are numbered 0.. in reset order, then one new slot per admit/replace. */
int gpp_archive_consider(void* archive, int count, const char* keys, const double* quality, const float* signatures,
                         int block, int* action, double* similarity, int* closest, int* replaced, int* new_slot,
                         char* err, int errlen);
int gpp_archive_size(void* archive);
int gpp_archive_stats(void* archive, char* out, int outlen);
void gpp_archive_close(void* archive);

/* ---- 2022 out-of-sample screening ---- */
/* options_json: engine options plus {"rankic_threshold", "corr_threshold", "sample_step",
 *   "limit", "block", "engine_batch", "band"}.
 * Per input: mean, valid_ratio, missing_ratio, valid_times, status (0 ok, 1 error; message in errors).
 * order[0..n_order) eligible input indices in quality order; selected/max_corr/corr_with for the
 * first *scanned of them. errors: count x error_len bytes (may be NULL). */
int gpp_screen(const char* panel_dir, int T, int N, const char* options_json, const char* expressions_json, int count,
               const double* evolution_rank_ic, const int* nodes, const char* keys, double* mean, double* valid_ratio,
               double* missing_ratio, int* valid_times, int* status, char* errors, int error_len, int* order,
               int* n_order, unsigned char* selected, double* max_corr, int* corr_with, long* scanned,
               char* stats_json, int stats_len, char* err, int errlen);

int gpp_has_cuda(void);
int gpp_threads(void);

#ifdef __cplusplus
}
#endif
#endif
