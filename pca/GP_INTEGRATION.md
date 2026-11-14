# Integrating the original GP with C++/MPI/CUDA

This entry point calls the original `run_hierarchical_evolution`; it does not replace selection, crossover, or mutation. The original engine accepts an optional `batch_backend`. With the default `None`, it uses its original numerical path. With `CppGPBackend`, the existing MPI engine evaluates factors, RankIC, and fitness during evolution, while the PCA module computes signatures and handles admission. Results return to the original Python loop for the next generation.

## 1. Data flow

1. Python proposes expressions using the original hierarchical strategy, cache, parent selection, crossover, and mutation.
2. Each GP progress chunk becomes one MPI request. C++ divides it into bounded `engine_chunk` batches, builds a DAG, and calls `gpe::evaluate_batch_mpi`. It reuses the upstream halo, ema, RankIC, and fitness implementations rather than reimplementing factor operators.
3. The engine temporarily retains the current batch's root panels. Each rank samples only its owned dates using global `t % sample_step == 0`, excluding negative-index halo rows. Sampled rows form R. CPU ranking is local; CUDA ranks the batch's factor rows together. MPI reduces the power-iteration norms and right vectors; CUDA uses device pointers directly. Signatures are gathered in global date order and assigned a consistent sign.
4. The backend returns `Candidate` records and FP32 signatures in request order. The original Python code determines admission order using its fitness sorting and family/parent rules; MPI completion order must not reorder candidates.
5. C++ QD reads the current Python archive snapshot, processes candidates in the specified order, and returns per-candidate decisions plus a complete new archive. Python checks every key, quality, and signature before updating the original archive object and generating the next population.

There is no manual R export or full-generation factor-panel dump. The engine retains only a bounded expression batch's root panels. Its memory use still depends on `engine_chunk`, `dag_chunk`, stock count, and the date partition.

## 2. Build

```bash
cmake -S pca -B pca/build_live -DPCA_MPI=ON -DPCA_ENGINE_BENCH=ON
cmake --build pca/build_live -j 4 --target pca_engine_bridge pca_run
```

For CUDA, use a separate directory and add `-DPCA_CUDA=ON`. Production targets do not link BLAS, cuBLAS, or cuSOLVER. NumPy supports the original Python GP's control/preparation, data-consistency checks, and independent validation. GPU Lanczos is outside the current scope; the CPU backend supports Lanczos.

## 3. Use the existing EOD CLI

The original CLI accepts the following optional arguments. Supply all three backend paths together:

```bash
python3 -m gp_factor_miner.hierarchical_cli \
  --data-file /path/to/eod --label-file /path/to/label.pkl \
  --output-dir /path/to/new/private/output \
  --cpp-panel-dir /path/to/exported/evolution/panels \
  --cpp-engine-binary "$PWD/pca/build_live/pca_engine_bridge" \
  --cpp-pca-binary "$PWD/pca/build_live/pca_run" \
  --cpp-ranks 4 --cpp-device cpu
```

The Python environment must be able to import `python/src/gp_factor_miner`. `--cpp-device cuda` requires a GPU allocation and CUDA binaries; use one MPI rank per GPU. `--cpp-mpiexec` sets an absolute MPI launcher path, and `--cpp-precision` selects float/double. The original out-of-sample RankIC screening, correlation filtering, and outputs retain their existing implementation. This backend accelerates evolution; it does not move the entire downstream research pipeline to the GPU.

With `--smoke`, the C++ export's dates and stock order define a small evolution subset of at most 256 dates and 512 stocks; screening keeps the same stocks. Startup checks coordinates and values against the Python data, rejecting inputs with matching dimensions but different dates or stocks. Both inputs must come from the same cleaning pipeline.

Without `--cpp-*` arguments, the CLI uses its original Python backend. Keep the complete repository layout; copying `gp_bridge.py` alone is insufficient.

## 4. Run directly from exported binary fields

An existing field export can run the same three-stage GP without the original EOD pickle:

```bash
python3 pca/run_gp.py --panel-dir /path/to/evolution/panels \
  --engine-binary pca/build_live/pca_engine_bridge --pca-binary pca/build_live/pca_run \
  --out pca/runs/live_gp_01 --ranks 4 --device cpu \
  --population 24 --generations 1 --gene-pool 12 --windows 1 2 3
```

This entry point defaults to a small configuration. It runs depths 1, 2, and 3 for the requested number of generations, writing original GP checkpoints, progress, the complete QD archive, call timing, and `SUCCESS`. It uses only the supplied evolution-period fields and does not generate out-of-sample data. `--resume` requires unchanged parameters, data, and backend identity.

### Production configuration

Population 10000, 10 generations per depth, gene pool 2000, warm-start multiplier 3, progress chunk 1000, and the window set come from the original [HierarchicalGPConfig](../python/src/gp_factor_miner/hierarchical_config.py). The stage-generation section of [main_zh.tex](../report/main_zh.tex) also specifies the first three settings. The new entry point defaults to a small run for validation; pass the following settings explicitly for the production budget.

After building as described in Section 2, run this CPU command from the repository root on allocated compute nodes. Replace `--panel-dir` with the existing evolution-period field directory. This example uses 16 single-threaded MPI processes; request the corresponding CPUs and enough memory, or adjust `--ranks` to the actual allocation. The process count is an example, not a measured optimum for production.

```bash
python3 pca/run_gp.py \
  --panel-dir /path/to/evolution/panels \
  --engine-binary "$PWD/pca/build_live/pca_engine_bridge" \
  --pca-binary "$PWD/pca/build_live/pca_run" \
  --out pca/runs/gp_production_01 \
  --device cpu --ranks 16 --threads 1 \
  --population 10000 --generations 10 --gene-pool 2000 \
  --warm-start-multiplier 3 --progress-chunk 1000 \
  --windows 1 5 10 15 20 30 60 \
  --sample-step 5 --steps 4 --precision double --method power \
  --engine-chunk 16 --dag-chunk 32 --H0 0 --seed 2026
```

The command runs depths 1, 2, and 3 sequentially, with 10 generations per depth and gene/parent pool limits of 2000. It uses all exported fields, dates, and stocks, and runs evolution only. The output must be a new child of `pca/runs`. After completion, check `completed`, `stages`, and the recorded dimensions in `SUCCESS.json`. To resume, use the same command and output directory with `--resume`; keep the data, binaries, and parameters unchanged.

For GPUs, build the CUDA version from Section 2, replace both binary paths, set `--device cuda`, and match MPI ranks to the allocated GPU count. MPI must support CUDA-aware communication. The small full-loop run has been validated; the production budget above has not completed measurement, so runtime and resources still need to be determined.

`live_gp.pbs` is a small cluster entry point with default windows 1/2/3; it is not the production configuration above. For production, place the complete command in the receiving researcher's PBS script and set resources and walltime for the chosen queue. Vanda multi-node runs need a consistent MPI environment and `PRTE_MCA_prte_launch_agent`; see `live_gp.pbs` and `scripts/vanda/prted.sh`. Start the parent Python process and its MPI batches on compute nodes, not login nodes. `OMP_NUM_THREADS` sets threads per rank; allocate the corresponding PE when binding ranks.

## 5. Checkpoints and acceptance

Original GP V4 checkpoints also record `batch_backend_identity`: input export, matrix coordinates, file sizes and timestamps, executable hashes, MPI rank count, precision, method, and chunking. Legacy checkpoints without the backend remain compatible with the original path; they cannot be treated as intermediate states of the new backend. Resuming with different data or a different backend raises an error instead of silently falling back to Python computation.

`bridge_check.py` checks fixed-expression Python/C++ scores and signatures, admission into a nonempty archive, complete three-stage evolution, interruption/resume, and rejection of a mismatched backend identity. During the full-loop test, original Python evaluation/PCA/QD entry points raise if called, ruling out hidden fallback. `bridge_cpu.pbs` and `bridge_gpu.pbs` run these checks on compute nodes.

`cli_bridge_check.py` checks the original EOD CLI, three-depth GP, original out-of-sample screening, final factor outputs, and resumption of a completed run. `cross_node_gpu.pbs` separately verifies device-memory communication between different hosts.

## 6. Interpreting performance

One MPI subprocess handles a GP progress chunk, not a single factor. Evaluation and QD currently use separate executable calls to establish reliable integration. For small batches, process startup, input preparation, and checkpoint I/O can dominate. Call walltime in `calls.json` includes startup and final output; full GP walltime also includes the original Python control loop and checkpointing. Independent PCA timing cannot directly predict full GP runtime.

Bridge `total_seconds` is the maximum across ranks. Stage diagnostics such as `engine_seconds` and `ranking_seconds` currently come from rank 0 and must not be presented as maximum-rank stage times. Configurations, raw outputs, and timing stay in private run directories. MPI rank count and precision can introduce small score differences that evolution amplifies through ordering. Fixed-input, fixed-order QD semantics and equality of complete evolution runs across configurations are separate claims.
