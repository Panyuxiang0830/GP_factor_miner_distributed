# GP Factor Miner — Distributed Evaluation Engine

DSA5208 (NUS) Project 3. A C++ evaluation engine for a hierarchical PCA-QD genetic-programming (GP) factor miner, parallelised with **MPI + OpenMP** and **MPI + CUDA**.

The GP search loop (selection, crossover, mutation, archive bookkeeping) stays in Python (`python/`). The compute-heavy batch evaluation is rewritten in C++. It has two parts:

1. **Factor computation and evaluation (E1–E8):** parsing, DAG merging of shared subtrees, operator execution, daily RankIC, fitness.
2. **PCA signatures and factor similarity (E9–E14):** sampled cross-sectional ranks, batched power iteration, signature similarity, archive admission.

Distribution is by **trading date**:

- each MPI rank owns a contiguous block of dates for all stocks;
- rolling operators use redundant computation or halo exchange, controlled by the halo budget `H0`;
- `ema` uses an affine `MPI_Exscan`;
- statistics and power iterations use `MPI_Allreduce`.

## Repository layout

```
include/gpe/   public headers
src/core/      serial engine (reference implementation)
src/omp/       OpenMP backend
src/cuda/      CUDA kernels
src/mpi/       date decomposition, halo exchange, collectives
tests/         C++ tests
python/        original Python GP factor miner (gp_factor_miner)
scripts/       run scripts and PBS job scripts
docs/          design documents
```

## Build

```bash
cmake -S . -B build -DGPE_ENABLE_OPENMP=ON -DGPE_ENABLE_MPI=OFF -DGPE_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To build the other backends, set the matching option to `ON`: `GPE_ENABLE_MPI` (needs an MPI installation, `mpicxx`) or `GPE_ENABLE_CUDA` (needs `nvcc`; architecture `86` targets the RTX 3090).

## Data policy

Market data, exported panels and run outputs are **never committed** (see `.gitignore`). Data live outside the repository on the experiment machines.
