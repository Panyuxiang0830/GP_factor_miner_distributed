# PCA experiment entry points

The task is defined in [docs/pca/task.md](../docs/pca/task.md). Submit PBS commands from the repository root. `PCA_RUN` must point to a new private output directory; scripts using real input also need `PCA_MANIFEST`. Jobs run on CPU/GPU compute nodes. Adjust queues and resources to the site configuration.

See [REPRODUCE.md](REPRODUCE.md) for stages, inputs, and environment variables. `study.pbs` defaults to only 32 factors; full-batch studies require explicit `PCA_COUNT`. Set `PCA_NUMERICAL_ONLY=1` to produce numerical checkpoints first, then replay admission in parallel with `sensitivity.pbs`.

| Scope | Script | Checks or measurements |
| --- | --- | --- |
| Module checks without private data | smoke.pbs / smoke_gpu.pbs | New output directory, Python reference, CPU/MPI/GPU, separate cuBLAS reference link dependencies |
| Current interface acceptance | delivery_cpu.pbs / delivery_gpu.pbs | Initial archive, batch continuation, complete snapshot, independent numerical reference, negative cases |
| Original GP full loop | bridge_cpu.pbs / bridge_gpu.pbs | Live MPI engine -> PCA -> QD -> original Python next generation, no legacy numerical fallback, checkpoint resume |
| Original EOD CLI pipeline | cli_bridge_check.py | Optional C++ backend, three depths, original out-of-sample screening and output |
| GP on real fields | live_gp.pbs | Actual full loop at specified field, population, and generation settings; distinguish its size from the production budget |
| P0 real-input subset | validate_cpu.pbs / validate_gpu.pbs | Same-input signatures, Python admission, repeatability, sampled exact SVD checks |
| P0 boundary semantics | adversarial.pbs | Near 0.90, ties, duplicate keys, strict quality margin, invalid signatures |
| Supplied-signature replay | gpu_api_test.pbs | Raw percentile GPU API and CPU/CUDA/MPI admission replay |
| P1 serial size | serial_scale.pbs | Independently vary B, m, n, k on synthetic input; not reranking a fixed real batch |
| P1 real-input size | real_size.pbs | Fixed real factor panels; vary sampled dates, stock prefixes, B, k, and rerank each configuration |
| P2 numerical study | study.pbs / sensitivity.pbs | Exact SVD, both solvers, multiple starting vectors, budgets, and sequential QD trajectories |
| P2 matched-accuracy cost | iteration_cost.pbs | Matrix-vector products, logical communication payload, reduction calls, and time |
| P2 full-batch matched-accuracy cost | matched_full.pbs | Execute every real R; report censored budgets separately and compare cost on the common certified set |
| Shared RankIC interface | rankic_interface.pbs | FP64 percentiles match the original engine elementwise; validate the actual daily_rank_ic consumer |
| P3 strong/hybrid scaling | scaling.pbs | date/transpose/factor on 1/2/4 nodes, each with warmup and three timings |
| Scaling including ranking | raw_scaling.pbs | Rank synthetic sampled values first; report ranking, solve, and redistribution separately |
| P4 weak scaling | weak_input.pbs / weak.pbs | Fixed complete-matrix-equivalent work per rank, using prefixes of the same input sequence |
| P5 GPU kernels | gpu_roofline.pbs | Ranking, batched GEMV, tiled similarity, and nominal roofline |
| P5 one/two GPUs | validate_gpu_mpi.pbs | CUDA-aware MPI device reductions, warmup, and repeated timing |
| P5 cross-node GPU | cross_node_gpu.pbs | One GPU on each of two distinct hosts, Python reference, repeated validation; run separately |
| P5 precision | precision_full.pbs / precision_finish.pbs | FP32/FP64, same-input Python admission, CPU double ranking and redistribution |
| Independent library reference | cublas_reference.pbs | Same device input, strict FP32, separate GEMM and GEMM+abs timings |
| P6 archive size | archive_scale.pbs | Centralized versus MPI-sharded archives; similarity, admission, and communication measured separately |
| Full-batch/scaling acceptance | validate_full.pbs / validate_scaling.pbs / validate_synthetic_runs.pbs | Full sequential QD, repeated outputs, layout agreement; numerical sample scope recorded separately |

`prepare.py` creates a manifest for the original export without copying large matrices. Do not rank an existing R again; use `--rank-input 1` only for raw sampled values. Controlled-spectrum synthetic input, weak-scaling synthetic input, and real-data proposals are distinct datasets, not interchangeable evidence of actual GP evolution.

The original synthetic dimension experiment in `serial_scale.pbs` and the first-eight-real-factor experiment in `iteration_cost.pbs` are retained. The added real-size study fixes factor values computed on the complete stock universe, then reranks subsets without recomputing cross-sectional expressions or RankIC. The full-batch matched-accuracy study selects budgets offline using exact SVD; factors missing the target within the limit still execute and are reported separately. Compare speed only on the same set certified for both solvers. Both methods reduce per factor in this study, unlike production power's batched reductions. GPU implements power iteration; Lanczos runs on CPU/MPI.

GPU percentiles reusable by RankIC offer FP32 and upstream-compatible FP64 output. See [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md) for the contract and actual-consumer validation. Automatic upstream integration and adoption by the RankIC maintainer remain integration tasks.

`report.py` can plot partial completed studies. `assemble_report.py` strictly assembles the complete experiment suite, using the prescribed directory/configuration names and requiring validation markers, repeat counts, and full dimensions. It is not a generic reporting tool for arbitrary smoke outputs. Its CLI help lists the default directory names.

```bash
python3 pca/assemble_report.py --runs pca/runs \
  --out pca/runs/report_01 \
  --scaling-validation scaling_validation_local_01 \
  --synthetic-validation synthetic_validation_01 \
  --real-size "$PWD/pca/runs/real_sizes_01" \
  --matched-full "$PWD/pca/runs/matched_full_01" --require-completion
```

Run assembly only when evidence is complete. `paper_sections.py` uses the same evidence to organize manuscript sections following `report/main_zh.tex`. Reports and figures remain in private runs; the source repository contains neither experiment outputs nor manuscript supplements.

Baseline performance experiments use one warmup and three measurements, reporting medians. Correctness and performance measurements have separate scopes. GPU roofline byte counts are nominal lower bounds, not profiler-measured DRAM traffic. Kernel speedup is not full GP speedup. New GP full-loop runs primarily validate interfaces, control flow, and recovery; a single small-run walltime is not a repeated benchmark of the production budget.

Current validators record same-budget numerics, SVD convergence, admission, final snapshots, and input identity separately. Retain existing results as historical evidence, but do not claim an updated validator produced them without rerunning it. See [DELIVERY.md](DELIVERY.md) for the complete interface and private delivery package.
