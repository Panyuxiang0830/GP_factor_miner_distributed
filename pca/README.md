# PCA signatures, similarity, and sequential QD admission

This directory implements project component 3; see [docs/pca/task.md](../docs/pca/task.md). Production numerical computation uses handwritten C++17, OpenMP, MPI, and CUDA. Python handles input preparation, independent validation, and experiment analysis.

See [SUBMISSION.md](SUBMISSION.md) for the delivery contents, reading order, main report location, and current scope.

```text
Sampled factor values (optional ranking) or an existing ranked matrix R
  -> Power iteration / Golub-Kahan Lanczos
  -> PCA signature
  -> FP32 similarity estimates + FP64 refinement
  -> Update the QD archive in candidate order
```

The module runs independently or through the original GP's optional `batch_backend`. See [GP_INTEGRATION.md](GP_INTEGRATION.md) for the live C++ engine, PCA, archive admission, and subsequent generations. The original EOD CLI enables this backend through `--cpp-*` options; its default remains the original Python backend. This directory does not depend on the legacy `pca_workflow/`, `pca_step1/`, or removed `gpe_export` target.

## Build

Run these commands from the repository root. Build and run on cluster compute nodes.

```bash
cmake -S pca -B pca/build -DPCA_MPI=ON
cmake --build pca/build -j 4
OMP_NUM_THREADS=2 ctest --test-dir pca/build --output-on-failure
```

Requirements: a C++17 compiler, CMake >= 3.18, OpenMP, and MPI. Set `-DPCA_MPI=OFF` for a serial build. Use a separate build directory and `-DPCA_CUDA=ON` for GPU support; the default CUDA architecture is 86. The PBS scripts show the Vanda module configuration.

Production targets do not link BLAS, LAPACK, cuBLAS, or cuSOLVER. `-DPCA_CUBLAS_REFERENCE=ON` adds a separate `pca_cublas_reference` executable to a CUDA build without changing production link dependencies.

Use `-DPCA_ENGINE_BENCH=ON` for the live GP bridge and real-panel size experiments. It links the existing factor engine into these targets. Combined with CUDA, it also builds `pca_rankic_test`. See [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md) for the input contract, precision, and validation of GPU ranking reused by RankIC.

## Smoke checks without market data

CPU and single-GPU module checks generate small synthetic inputs. They cover all three layouts, fresh CPU ranking, Python admission, Lanczos, MPI archives, GPU precision, and CUDA-aware MPI. The GPU check also verifies that the independent cuBLAS reference and production targets have separate link dependencies.

```bash
mkdir -p pca/runs
qsub -v PCA_RUN="$PWD/pca/runs/smoke_cpu_01" \
  -o "$PWD/pca/runs/smoke_cpu_01.log" pca/smoke.pbs
qsub -v PCA_RUN="$PWD/pca/runs/smoke_gpu_01" \
  -o "$PWD/pca/runs/smoke_gpu_01.log" pca/smoke_gpu.pbs
```

Each output path must be a new child of `pca/runs/`. `SUCCESS` and `SUCCESS.json` are written only after every check passes. Python validation requires Python >= 3.11, NumPy, and pandas; plotting also needs matplotlib. Vanda scripts load the corresponding scientific-computing modules and require no market data.

## Use an existing export

`prepare.py` reads the supplied `R/meta.json`, `R/chunk_*.f32`, `scores.jsonl`, `signatures.f32`, and initial vector. Large matrices remain in their original location; only a private manifest is created.

```bash
python3 pca/prepare.py --export-dir /path/to/export \
  --init /path/to/pca_init.f8 --out pca/runs/real_input
mpirun -np 4 pca/build/pca_run \
  --manifest pca/runs/real_input/manifest.json \
  --out pca/runs/example_01 --count 40 --chunk 16 \
  --method power --steps 4 --layout date
python3 pca/validate.py --manifest pca/runs/real_input/manifest.json \
  --run pca/runs/example_01
```

| Option | Meaning |
| --- | --- |
| `--layout date` | Partition by date; reduce batched norms and right vectors with Allreduce |
| `--layout transpose` | Redistribute date blocks into complete factors with Alltoallv, then solve locally |
| `--layout factor` | Read complete factors directly as an independent input scenario |
| `--method power/lanczos` | Power iteration / bidiagonalization with two-pass full reorthogonalization and a handwritten Jacobi solver |
| `--steps k` | Iteration budget: 2k+1 matrix-vector products for power, usually 2k-1 for Lanczos; diagnostics are additional |
| `--rank-input 1` | Rank raw sampled values first; do not rank an existing R again |
| `--signature-input file` | Replay admission separately using supplied signatures |
| `--archive mpi` | Shard archive signatures; use MAXLOC to select the global nearest record |
| `--archive-input file` | Continue from a complete archive snapshot; otherwise start with an empty archive |
| `--device cuda --precision float/double` | Batched GPU power iteration; CUDA Lanczos is not implemented |
| `--device cuda --layout date --cuda-aware 1` | Reduce GPU memory directly through MPI |

Admission quality is absolute RankIC. The similarity threshold is 0.90; replacement requires quality strictly greater than the existing quality plus 0.002. Changes within a block are visible immediately. Exact ties select the lexicographically larger key. See WALKTHROUGH for details and interfaces.

Outputs include `signatures.f32`, `decisions.json`, `archive_keys.json`, `metrics.json`, and the `archive.json` / `archive_signatures.f32` snapshot for subsequent batches. `provenance.json` records input identity; validation writes `validation.json`. Total timing includes signature writes inside the loop, but excludes MPI startup, GPU initialization, and final snapshot/JSON serialization. Timed stages can overlap and must not simply be added together.

See [DELIVERY.md](DELIVERY.md) for snapshot format, separate validation fields, final CPU/GPU acceptance checks, and report packaging. Agreement with a reference at the same iteration budget and convergence to exact SVD are separate checks; an unconverged finite-budget approximation is not necessarily an implementation error.

## Experiments and reproduction

See [EXPERIMENTS.md](EXPERIMENTS.md) for P0-P6 scripts, validation scope, and report entry points, and [REPRODUCE.md](REPRODUCE.md) for stage dependencies and full-batch settings. Full report assembly requires complete, validated runs; the presence of a script or a submitted job does not establish completion.

All market data, factor expressions, manifests, outputs, figures, and reports remain in private, ignored run directories. The public repository contains source, synthetic test generators, job scripts, and documentation.

OpenAI Codex assisted with design, implementation, review, and orchestration. Numerical conclusions require actual execution evidence and independent Python reference checks.
