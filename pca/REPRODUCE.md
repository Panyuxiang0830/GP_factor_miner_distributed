# Reproduce the study from private inputs

Run stages from the repository root. Confirm the previous stage's outputs and success markers before submitting dependent jobs; PBS `afterok` dependencies can connect stages automatically. This document contains no market data, expressions, outputs, or fixed job IDs. Every output must use a new child of `pca/runs`; reruns need a different set of directory names.

## 1. Inputs and baseline validation

Use `prepare.py` as described in README to map the supplied R, scores, signatures, and initial vector into `pca/runs/real_input/manifest.json`. Large matrices remain read-only in their original location.

```bash
PCA_RUNS="$PWD/pca/runs"
PCA_REAL="$PCA_RUNS/real_input/manifest.json"
mkdir -p "$PCA_RUNS"
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/cpu_validation_01" pca/validate_cpu.pbs
```

Without private input, run synthetic checks with `smoke.pbs` / `smoke_gpu.pbs` first. They do not replace the full-batch study. P0 requires the reference initial vector, scores, and supplied signatures.

## 2. Full-batch numerical study and QD replay

`study.pbs` defaults to `PCA_COUNT=32` for development. For the handed-off batch of 9388 nonconstant R matrices, explicitly set the following parameters. A different dataset size also requires updating dataset-specific checks in strict report assembly.

```bash
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/study_full_01",PCA_COUNT=9388,PCA_WORKERS=8,PCA_NUMERICAL_ONLY=1 pca/study.pbs
```

This stage writes only numerical checkpoints: exact SVD, both C++ solvers, three initial vectors, and k=1...10. `NUMERICAL_SUCCESS.json` marks numerical completion, not admission-experiment completion. After this job succeeds, run:

```bash
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_STUDY="$PCA_RUNS/study_full_01",PCA_RUN="$PCA_RUNS/sensitivity_full_01" pca/sensitivity.pbs
```

The complete 60 admission trajectories and `summary.json` are in `sensitivity_full_01`. Supply both `--study` and `--sensitivity` to the basic `report.py`. Without `PCA_NUMERICAL_ONLY`, `study.py` retains its original numerical-study-then-sequential-replay behavior.

Run the controlled-spectrum experiment separately with `PCA_SYNTHETIC=1`, `PCA_COUNT=24`, and `PCA_RUN=study_synthetic_01`. Still supply `PCA_MANIFEST` for the PBS entry point; the script generates and switches to its own synthetic manifest. This input differs from weak-scaling input.

New studies also write `INPUT.json` to bind the manifest and initial vector. Reusing a historical study without this file requires explicit `--allow-legacy-study` in `prepare_completion.py` after manual input verification. Historical results are not relabeled as newly validated results.

## 3. CPU size, strong, hybrid, and weak scaling

`serial_scale.pbs` -> `serial_sizes_01`; `archive_scale.pbs` -> `archive_sizes_01`; `iteration_cost.pbs` uses real `PCA_REAL` -> `iteration_cost_01`. The last is a representative matched-accuracy experiment on the first 8 real factors with 8 ranks.

Submit `scaling.pbs` for 1/2/4 nodes, producing `scaling_1node_01`, `scaling_2node_01`, and `scaling_4node_01`. Explicitly request `ncpus=72`, `mpiprocs=72`, and `mem=180gb` per node. Four nodes exceed the free pool limit; use a project routing queue, for example:

```bash
qsub -q auto -l select=4:ncpus=72:mpiprocs=72:mem=180gb \
  -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/scaling_4node_01" pca/scaling.pbs
```

The same script also runs the hybrid layout with 4 ranks x 18 threads per node. Set the corresponding allocation and output name for 1/2-node runs.

`weak_input.pbs` -> `weak_input_01` generates 9216 synthetic R matrices of shape 293x5876. After completion, use `weak_input_01/manifest.json` with 1/2/4-node `weak.pbs` runs -> `weak_1/2/4node_01`. Each rank has 32 complete-matrix-equivalent inputs; every scale reads a prefix of the same sequence.

`raw_scaling.pbs` uses the same complete synthetic manifest but treats the values as unranked sampled input, ranks them, and produces `raw_scaling_1/2/4node_01`. Ranking changes the spectrum; these are neither real market inputs nor preserved controlled-spectrum matrices. All 288-core jobs must use a project routing queue within site limits.

## 4. GPU, precision, and supplied signatures

After CPU strong scaling succeeds, a serial signature output on the same input can serve as the baseline:

```bash
PCA_BASELINE="$PCA_RUNS/scaling_1node_01/strong_date_p1_t1_r1/signatures.f32"
```

| Job | Additional input | Expected output name |
| --- | --- | --- |
| validate_gpu_mpi.pbs | PCA_MANIFEST=PCA_REAL | gpu_two_01 |
| gpu_roofline.pbs | No market input | gpu_roofline_01 |
| precision_full.pbs | PCA_MANIFEST=PCA_REAL, PCA_BASELINE | precision_full_03 |
| gpu_api_test.pbs | PCA_MANIFEST=PCA_REAL, PCA_BASELINE | gpu_api_01 |
| adversarial.pbs | No market input | adversarial_02 |
| cublas_reference.pbs | No market input | cublas_reference_01 |

Set the corresponding `PCA_RUN` for every job. Run `precision_full.pbs` for the complete precision experiment; `precision_finish.pbs` is only for recovery that reuses completed GPU numerical outputs, not a mandatory stage. Route GPU jobs through `auto_free` or `auto`; do not submit directly to an execution queue marked `from_route_only`. Single-/dual-GPU scaling and the 288-core CPU study are separate experiments.

## 5. Real-size and full-batch matched-accuracy supplements

This stage depends on complete numerical checkpoints from Section 2. Preparation reads the existing manifest, SVD study, and original scores to generate private candidate and budget plans. Market data and exported R remain read-only in their original location. Replace the two placeholder paths below with the supplied `scores.jsonl` and raw field directory containing `meta.json` and field files.

```bash
python3 pca/prepare_completion.py --manifest "$PCA_REAL" \
  --study "$PCA_RUNS/study_full_01" \
  --scores /path/to/export/scores.jsonl --panel-dir /path/to/panels \
  --out "$PCA_RUNS/completion_plan_01" --count 9388 --target 1e-6
qsub -v PCA_SOURCE="$PCA_RUNS/completion_plan_01/real_source.json",PCA_RUN="$PCA_RUNS/real_sizes_01" pca/real_size.pbs
qsub -v PCA_PLAN="$PCA_RUNS/completion_plan_01/accuracy_plan.json",PCA_RUN="$PCA_RUNS/matched_full_01" pca/matched_full.pbs
```

The two jobs are independent. The real-size experiment computes 128 fixed factor panels with the engine, then varies B, sampled dates m, stock count n, and steps k over 19 configurations. Each configuration reranks its stock subset. Factor values are fixed after evaluation on the complete stock universe; cross-sectional expressions are not reevaluated on subsets. Admission uses original scores without recomputing quality. Eight-thread factor generation is timed separately; PCA uses one thread. Each configuration checks full sequential QD and three repeats; the first two raw panels also receive pandas ranking and independent-signature checks.

The full-batch cost job executes every input. Each solver uses its first target-reaching budget from the serial study. Inputs missing the target within ten steps still execute at ten steps and are reported separately. MPI results are compared with same-budget serial signatures and certified through an error bound. Cost comparison uses only the identical set certified for both solvers. This is an offline-budget experiment: both methods reduce per factor and omit residual diagnostics, unlike production power's batched reductions.

`matched_full.pbs` reserves 16 CPU cores but launches 8 single-threaded ranks. The full FP32 cache needs about 64.7 GB. Vanda may rewrite memory according to core count, so the job records the actual allocation and exits if memory is insufficient. Report reserved cores and active ranks separately. Real-size and full-batch jobs write `SUCCESS` only after independent validation.

After real-size validation succeeds, its raw panels can also validate the shared RankIC interface:

```bash
qsub -v PCA_REAL_SIZE="$PCA_RUNS/real_sizes_01",PCA_RUN="$PCA_RUNS/rankic_interface_01" pca/rankic_interface.pbs
```

Without `PCA_REAL_SIZE`, the job runs synthetic boundary checks. It verifies agreement of FP64 GPU percentiles with original engine ranking and the actual `daily_rank_ic` consumer; it does not automatically change the main GP backend.

## 6. Independent acceptance and reporting

| Acceptance | Prerequisites | Output name |
| --- | --- | --- |
| validate_full.pbs | PCA_REAL, PCA_BASELINE, PCA_CPU_RUN=scaling_2node_01, PCA_GPU_RUN=gpu_two_01 | full_validation_01 |
| validate_scaling.pbs | scaling_1/4node_01, default sensitivity_full_01 trajectory | scaling_validation_local_01 |
| validate_synthetic_runs.pbs | Three weak and three raw runs, PCA_MANIFEST=weak_input_01/manifest.json, PCA_KIND=both | synthetic_validation_01 |

Every job also needs `PCA_RUN`. Synthetic acceptance checks full sequential QD, three repeats, and layout agreement. Independent numerical recomputation covers the first 8 factors in each group; do not present this sample as exhaustive numerical validation.

After all prerequisites pass, invoke `final_report.pbs` with a new `PCA_RUN`. It checks prescribed directories, timing repeats, configuration counts, and validation markers, then writes the complete report, figures, and `EVIDENCE.json`. Missing evidence is an error. The code directory contains no actual results; acceptance requires checking server-side runs and reports, not only GitHub files.

`final_report.pbs` retains the baseline experiment entry point. For a strict report including both supplements, load scientific-computing and matplotlib modules on a compute node and run:

```bash
python3 pca/assemble_report.py --runs "$PCA_RUNS" \
  --out "$PCA_RUNS/report_completed_01" \
  --scaling-validation scaling_validation_local_01 \
  --synthetic-validation synthetic_validation_01 \
  --real-size "$PCA_RUNS/real_sizes_01" \
  --matched-full "$PCA_RUNS/matched_full_01" --require-completion
```

These gates require the full handoff dimensions, 9388 factors, 8 MPI ranks, 19 real-size configurations, and success markers for both supplements. They apply to this specific dataset. The report includes `completion_sections.md`, two added plots, and CSVs, and marks `completion_supplement_verified` only when supplementary validation is complete. For migration, `--runs` can point to an existing private evidence root. The underlying `report.py` also accepts an explicit `--runs-root`; output must still be a new child of that root.

## 7. Current interface acceptance and delivery

In addition to baseline experiments, use `delivery_cpu.pbs` / `delivery_gpu.pbs` to validate existing archives, consecutive batches, and negative cases, then package current source and report following [DELIVERY.md](DELIVERY.md). Run `cross_node_gpu.pbs` separately and record actual execution status; same-node dual-GPU results do not establish cross-node validation. See [GP_INTEGRATION.md](GP_INTEGRATION.md) for automatic GP integration, the original EOD CLI, and full-loop acceptance. GPU Lanczos is outside this delivery's scope.
