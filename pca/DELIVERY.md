# PCA-QD delivery interface

See [SUBMISSION.md](SUBMISSION.md) for the current delivery index and main report location.

This version computes signatures from raw sampled values or an existing ranked R, updates QD in the supplied candidate order, accepts an empty or existing archive, and exports a complete snapshot for the next batch. The original Python GP enables live engine integration through an explicit `batch_backend` or the EOD CLI's `--cpp-*` options; see [GP_INTEGRATION.md](GP_INTEGRATION.md). GPU Lanczos is outside the current scope.

## 1. Existing archives and consecutive batches

```bash
mpirun -np 4 pca/build/pca_run \
  --manifest pca/runs/batch_01/manifest.json --out pca/runs/result_01 \
  --layout date --archive mpi --archive-input pca/runs/previous/archive.json
mpirun -np 4 pca/build/pca_run \
  --manifest pca/runs/batch_02/manifest.json --out pca/runs/result_02 \
  --layout date --archive mpi --archive-input pca/runs/result_01/archive.json
```

Without `--archive-input`, a run starts with an empty archive. All chunks within a run share the same archive. CPU centralized archives, GPU similarity, and MPI sharded archives use the same snapshot format. MPI export gathers signature shards to rank 0.

`archive.json` contains `version=1`, `dimension`, a relative `signatures` path, and a `records` array. Each record has a nonempty unique `key`, finite `quality`, and unique `slot`. The binary array is little-endian FP32 with layout `[slot][sampled_date]`. Signatures must be finite and nonconstant; `dimension` must equal the number of rows in this batch's R. Batches must also use the same global sampled dates, stock definition, and signature algorithm. Equal dimensions alone do not establish date alignment.

Default outputs are `archive.json`, `archive_signatures.f32`, `archive_keys.json`, `signatures.f32`, `decisions.json`, `metrics.json`, and `provenance.json`. Complete snapshot serialization happens after the main timer stops; this does not redefine the scope of historical performance measurements. Input snapshots are read-only, and existing output directories are not overwritten.

## 2. Validation fields

```bash
python3 pca/validate.py --manifest pca/runs/batch_01/manifest.json \
  --run pca/runs/result_01
```

`provenance.json` identifies the initial archive and actual initial vector automatically for new runs. You can also supply the same `--archive-input` explicitly. Validation records:

- `numerical_checked` / `numerical_passed`: agreement with the Python power or Golub-Kahan reference at the same budget, or exact replay of supplied signatures. An unchecked result is `null`, not zero error.
- `independent_python_qd_agreement`: candidate filtering, per-candidate actions/nearest keys/replaced keys, and final membership against independent reference signatures.
- `archive_key_agreement`: admission semantics isolated using the actual output signatures.
- `snapshot_checked` / `snapshot_passed`: complete agreement of final keys, qualities, and FP32 signatures.
- `svd_checked` / `svd_passed`: convergence against exact SVD. This is a pass condition only when `--svd-tolerance` is supplied. For example, `--svd-count 4 --svd-tolerance 1e-6` checks controlled cases with a sufficient budget.

A low-budget approximation need not converge. Default validation therefore checks agreement at the same budget rather than requiring equality with exact SVD. Computed NaN outputs cannot pass; supplied-signature replay retains and skips incomparable records according to the original policy.

`provenance.json` binds the manifest, actual initial vector, and initial archive contents. Cached `--power-reference` is supported only for four-step power iteration and checks matching inputs and settings. A historical cache without metadata requires explicit `--allow-legacy-reference` after manual evidence review. New studies write `INPUT.json`, which `prepare_completion.py` checks; an audited historical study requires `--allow-legacy-study`. The fingerprint uses noncryptographic FNV-1a and does not rehash tens of GB of R. Underlying R files must remain unchanged and read-only.

## 3. Final small-scale acceptance

```bash
qsub -v PCA_RUN="$PWD/pca/runs/delivery_cpu_01" pca/delivery_cpu.pbs
qsub -v PCA_RUN="$PWD/pca/runs/delivery_gpu_01" pca/delivery_gpu.pbs
```

These scripts use small synthetic inputs and do not replace full performance experiments. `delivery_check.py` checks an existing archive, ordered replacement, duplicate keys, ties, and thresholds. It compares one complete batch with two consecutive batches, validates same-budget numerics and sufficient-budget SVD cases, and checks rejection of invalid inputs. `SUCCESS` is written only after all checks pass. GPU acceptance also covers similarity against a nonempty archive and FP32/FP64 power iteration.

`cross_node_gpu.pbs` is a separate two-node, one-GPU-per-node entry point. It records MPI hosts and rejects placement on a single node. By default it uses small inputs, one warmup, and three measurements, checking one-/two-GPU decisions against Python. A ready script is not evidence of execution; delivery status must follow actual `SUCCESS.json` records. Existing same-node dual-GPU results are not cross-node measurements. Select resources according to the requested experiment and site quota.

## 4. Private report and source package

`package_delivery.py` does not rerun numerical experiments. It takes a complete report, CPU/GPU acceptance directories, and optional cross-node evidence, then writes a new child of `pca/runs`:

```bash
python3 pca/package_delivery.py --report /path/to/private/report_completed \
  --cpu-check pca/runs/delivery_cpu_01 --gpu-check pca/runs/delivery_gpu_01 \
  --out pca/runs/final_delivery_01
```

It integrates the real-size and full-batch cost sections rather than appending potentially inconsistent supplements. It retains tables, figures, validation summaries, an upstream full LaTeX source copy, and a complete repository source archive. `--pandoc /path/to/pandoc` generates a LaTeX body fragment; install Markdown and WeasyPrint and use `--pdf` for a reading PDF. Actual compilation of the full LaTeX manuscript, cross-node GPU validation, and automatic GP integration have separate status fields; successful packaging does not establish any of them.

For the integrated delivery, also supply `--integration-cpu`, `--integration-gpu`, and `--live-gp`, pointing to CPU/GPU `bridge_check.py` acceptance and a real-data `run_gp.py` run. All three are required together. `--live-gp-gpu` adds real GPU evolution evidence; `--cli-check` adds original CLI, out-of-sample screening, and resume evidence from `cli_bridge_check.py`; `--cross-node` adds cross-node GPU validation. Commit source before packaging so the archive matches its recorded commit. Outputs remain in ignored private directories.

`--preflight /path/to/completed/preflight` incorporates a CPU/GPU evolution pilot using the complete data dimensions, checks final status and stable evidence hashes, and records the source commit actually tested. It does not establish completion of the production population/generation budget or real out-of-sample evaluation. `--lean` keeps the report, figures referenced in its body, acceptance summaries, source, and optional LaTeX fragment. It omits intermediate HTML, unreferenced plot data, and a separate upstream manuscript copy. The current final delivery uses both options.

Use `--manuscript /path/to/private/manuscript.md` to package an edited Markdown body after the same evidence checks. Status records the manuscript hash. Figures can use bare filenames or paths relative to `figures/`. `--evolution-only` places real out-of-sample evaluation outside the delivery scope. `--production-planned` records a planned, unmeasured production run. These flags record scope and status; they submit no jobs. Check original table values and figures when editing, and retain previous reports.

Display equations can use `$$` delimiters on separate lines, restricted to the LaTeX subset supported by Matplotlib mathtext. Markdown and LaTeX fragments retain formula source; the reading PDF embeds SVG rendered with mathtext and therefore also requires Matplotlib. This typesetting dependency is separate from GP numerical computation. Manuscripts without display equations do not require it.

While GPU runtime verification is waiting for resources, a candidate package requires explicit `--pending-gpu --cuda-build /path/to/verified/build`. It checks a separate `COMPILE_CHECK.json` and leaves `current_gpu_interface_verified` and `independent_module_verified` false. Such a candidate does not replace final acceptance after current GPU regression checks pass.

Market data, expressions, logs, experiment results, reports, and delivery packages stay in private runs. The public branch contains source, job scripts, and documentation. Old reports and PDFs are retained. Final paper assembly still requires the team to agree on authors, numbering, cross-references, and overall conclusions.
