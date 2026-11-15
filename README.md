# Distributed GP Factor Miner

DSA5208 (NUS) Project 3 — Wu Jiarui, Hou Jiacheng, Pan Yuxiang.

The project parallelises a hierarchical genetic-programming (GP) factor miner for Chinese A-share data with **MPI + OpenMP**, plus an **MPI + CUDA** path for the PCA stage. The GP search loop is unchanged and stays in Python. The expensive part of every generation is rewritten in C++:

1. **Factor evaluation.** Each candidate expression is evaluated on the full panel (1461 days × 5876 stocks × 132 fields), and the engine computes its daily RankIC and fitness.
2. **PCA-QD admission.** The engine computes a PCA signature for each candidate, compares it with the diversity archive and admits candidates in order.

**Report:** [English PDF](report/main.pdf) · [中文 PDF](report/main_zh.pdf). The report covers the algorithm, design decisions, validation, scaling experiments on NUS Vanda, conclusions and AI usage.

---

## 1. How it works

```text
Python GP (selection, crossover, mutation)
        │  a batch of ~10,000 expressions per generation
        ▼
C++ factor engine ───────────────────────────────────────────────┐
  • merge the batch into one DAG (shared subexpressions computed  │
    once; wide levels evaluated in chunks)                        │
  • MPI: each rank owns a contiguous block of trading dates       │
      – rolling operators: halo exchange from the left neighbour  │
      – ema: affine-map scan passed from rank to rank             │
  • OpenMP: threads inside a rank work on different DAG nodes     │
  • daily RankIC locally, one MPI_Allreduce for fitness           │
        ▼                                                         │
PCA signature (same date partition)                               │
  • sample every 5th date, rank cross-sections                    │
  • 4-step power iteration, norms/vectors reduced with MPI        │
        ▼                                                         │
QD archive on rank 0: batched similarity, FP64 refinement,        │
sequential admission  ──► back to Python for the next generation ◄┘
```

### Final configuration and headline results

All numbers are for a real batch of 10,000 expressions; full details are in the report (§6–§7).

| Part | Final choice | Result |
|---|---|---|
| Factor engine | 1 MPI rank per machine, 72 OpenMP threads over DAG nodes, full halo exchange (`H0=0`) | 5060 s on 1 core → **27.8 s** on 4×72 cores (182×, efficiency 0.63) |
| Pure MPI (for comparison) | 288 ranks × 1 thread | 75.0 s (67.5×, efficiency 0.23) |
| PCA signature | `date` layout, 4-step power iteration (as in the original program), FP64 | 9388 matrices in 13.4 s on 72 ranks |
| Admission | central archive on rank 0, sequential | 1.5–8.1 % of PCA time (Amdahl limit) |
| Extension | Lanczos instead of power iteration | 20.3 % faster at equal accuracy; not used in production, so that the archive stays identical to the original program's |

---

## 2. Repository layout

```text
.
├── include/gpe/          C++ factor-engine headers (panel, DAG, operators, distributed API)
├── src/
│   ├── core/             serial engine: parser, DAG, operators, RankIC/fitness, ema scan, OpenMP
│   └── mpi/              distributed.cpp — date partition, halo exchange, ema chain, MPI batch eval
├── apps/gpe_bench.cpp    benchmark driver for a fixed expression batch
├── tests/                C++ engine tests (DAG, operators vs reference, metrics, MPI vs serial)
│
├── pca/                  PCA signatures, similarity and QD admission (own CMake project)
│   ├── core.*  gpu.*  main.cpp  mpi_archive.hpp     CPU power/Lanczos, CUDA kernels, driver, sharded archive
│   ├── engine_bridge.cpp  gp_bridge.py  run_gp.py   integration with the factor engine and Python GP
│   ├── *_bench.cpp  reference_cublas.cu             benchmarks and the cuBLAS reference
│   ├── test_*.cpp  validate*.py  *check.py          tests and validation against the Python implementation
│   ├── *.pbs                                        Vanda job scripts
│   └── README.md  WALKTHROUGH.md  GP_INTEGRATION.md  EXPERIMENTS.md  REPRODUCE.md
│
├── python/               original Python GP miner (package gp_factor_miner)
│   ├── src/gp_factor_miner/   data, expressions, hierarchical evolution, PCA-QD archive, screening
│   ├── scripts/               panel/reference export, Python benchmarks, plotting of Vanda results
│   └── tests/
│
├── dag/                  DAG analysis: sharing statistics, planner timing, memory/cache simulation
├── scripts/vanda/        Vanda environment, MPI binding and experiment job scripts for the engine
├── docs/                 design notes and module documents (engine/, dag/, pca/, design/)
└── report/               LaTeX sources (main.tex, main_zh.tex), PDFs and figures
```

Where each report section lives in the code:

| Report section | Code |
|---|---|
| §4.2 Subexpression DAG and memory | `src/core/dag.cpp`, `dag/` |
| §4.3 Date partition, halo, ema scan, threads | `src/mpi/distributed.cpp`, `src/core/halo_plan.cpp`, `src/core/ema_scan.cpp`, `src/core/evaluate.cpp` |
| §4.4–4.5 PCA, similarity, admission, Lanczos | `pca/core.cpp`, `pca/gpu.cu`, `pca/mpi_archive.hpp` |
| §5 Validation | `tests/`, `pca/validate*.py`, `pca/test_*.cpp`, `python/scripts/export_reference.py` |
| §6 Experiments | `scripts/vanda/*.pbs`, `pca/*.pbs`, `python/scripts/plot_vanda.py` |

---

## 3. Build

Requirements: C++17 compiler, CMake ≥ 3.18, OpenMP, MPI (e.g. OpenMPI 5); CUDA only for the GPU path; Python ≥ 3.11 for the GP loop.

**Factor engine**

```bash
cmake -S . -B build -DGPE_ENABLE_MPI=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

**PCA module and GP bridge**

```bash
python3 -m pip install -e ./python
cmake -S pca -B pca/build_live -DPCA_MPI=ON -DPCA_ENGINE_BENCH=ON
cmake --build pca/build_live -j 4 --target pca_engine_bridge pca_run
```

Add `-DPCA_CUDA=ON` in a separate build directory for the GPU version (one MPI rank per GPU, CUDA-aware MPI required).

On Vanda: `module load OpenMPI/5.0.8-GCC-14.3.0 CMake/4.0.3-GCCcore-14.3.0` first; see `scripts/vanda/env.sh`.

---

## 4. Run

**Benchmark one expression batch with the engine** (hybrid 4 machines × 72 threads, the best configuration in the report). A 72-thread rank spans both 36-core sockets, which OpenMPI will not bind, so ranks are left unbound and `omp_slice.sh` pins each rank's threads; `scripts/vanda/env.sh` chooses the placement for other rank/thread counts:

```bash
mpirun -np 4 --map-by ppr:1:node --bind-to none -x OMP_NUM_THREADS=72 \
  scripts/vanda/omp_slice.sh build/gpe_bench /path/to/panels /path/to/expressions.json \
  --threads node --chunk 2000
```

**Check the distributed engine against the serial one:**

```bash
mpirun -np 4 build/test_mpi_consistency /path/to/panels /path/to/expressions.json
```

**Run the GP with the C++ backend** (three depths, one generation each, as in report §6.3.4):

```bash
python3 pca/run_gp.py --panel-dir /path/to/evolution/panels \
  --engine-binary pca/build_live/pca_engine_bridge \
  --pca-binary pca/build_live/pca_run \
  --out pca/runs/example_gp_01 --ranks 16 --threads 1 \
  --population 64 --generations 1 --gene-pool 32 \
  --warm-start-multiplier 3 --progress-chunk 1000 \
  --windows 1 5 10 15 20 30 60
```

**Reproduce the scaling experiments on Vanda:**

```bash
qsub scripts/vanda/build_test.pbs        # build and validation tests
bash scripts/vanda/submit.sh             # factor-engine experiment jobs
python python/scripts/plot_vanda.py <csv_dir> <figure_dir>
```

PCA experiments and their job scripts are listed in [pca/REPRODUCE.md](pca/REPRODUCE.md).

---

## 5. Data

Market data, exported panels and run outputs are **not in the repository** (see `.gitignore`); they stay on the experiment machines. The panel format is described in [docs/engine/data_format.md](docs/engine/data_format.md).

## 6. Further documentation

- [docs/README.md](docs/README.md) — index of design and module documents
- [dag/README.md](dag/README.md) — DAG analysis tools
- [pca/README.md](pca/README.md), [pca/WALKTHROUGH.md](pca/WALKTHROUGH.md), [pca/GP_INTEGRATION.md](pca/GP_INTEGRATION.md) — PCA module and GP integration
- [python/README.md](python/README.md) — original Python GP miner
