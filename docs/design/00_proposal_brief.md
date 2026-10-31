# DSA5208 Project 3 Proposal (Draft): Distributed Acceleration of Genetic-Programming Factor Mining

## 1. Background: GP factor mining

In quantitative equity research, a **factor** is a formula that turns raw stock data into a daily score for every stock. For example, `ema(turnover_rate * amount_skew, 5)`. A factor is good if, on each trading day, ranking stocks by their factor value predicts the ranking of their future returns. This is measured by the **RankIC**, the daily Spearman correlation between factor values and future returns, averaged over time.

All data are panels of shape `trading days × stocks`. In our case that is roughly 1,460 days (2016–2021) by about 5,000 Chinese A-share stocks, for 127 raw fields.

I already have a working **genetic programming (GP)** factor miner written in Python. It represents each factor as an expression tree:

- **Leaves:** raw fields.
- **Operators:**
  - element-wise: `+ − × ÷ log abs`;
  - rolling time-series: `mean, std, max, min, ema, lag, …`;
  - rolling pairwise: `correlation, beta`.
- **Window lengths:** 1–60 days.

It searches the expression space with the following steps:

1. **Fitness.** Each candidate is scored by |mean RankIC|, penalised by the instability of RankIC across four time segments and by its tree size.
2. **Layer-wise generation.** Evolution runs in three stages, with maximum tree depth 1, then 2, then 3. Each stage evolves a population of 10,000 for 10 generations and keeps a *gene pool* of its best 2,000 expressions. The next stage builds deeper trees mostly from these gene-pool subtrees. Good "building blocks" are found first and then combined.
3. **PCA-based diversity control.** Many candidates are near-duplicates of each other. To avoid keeping them all, each factor is compressed into a short **PCA signature**:
   1. take its cross-sectional ranks every 5th day, giving a matrix of about 290 × 5,000;
   2. extract the leading singular vector by power iteration;
   3. project onto it, which gives a vector of about 290 values.

   A candidate enters the archive only if its |correlation| with every archived signature is below 0.9. Otherwise it may replace its nearest neighbour, but only if its RankIC is better.
4. **Out-of-sample screening.** Finally, factors are screened on 2022 data with |RankIC| ≥ 0.07 and pairwise correlation < 0.75.

A previous run produced **1,481 out-of-sample factors**, with mean |RankIC| of 0.08 in 2022.

## 2. Bottleneck

One full run evaluates about 400,000 expressions. Each evaluation applies about 8 operators to full panels, sorts about 5,000 stocks for each of about 1,460 days to compute the RankIC, and then computes the PCA signature. The current Python implementation (pandas, 32 processes) parallelises only across expressions. It has three problems:

- It recomputes shared subtrees from scratch, even though deeper layers are built from earlier gene pools.
- It computes signatures in single-threaded NumPy.
- It uses the GPU only for the final small similarity matrix multiplication.

## 3. Proposed project

I propose to rewrite the **batch factor-evaluation engine** in C++. The GP selection and mutation logic is cheap, so it would stay in Python. I would then parallelise the engine with **MPI + OpenMP** and **MPI + CUDA**, focusing on two parts.

### (a) Distributed factor computation

- **Decomposition by trading date.** Each MPI rank owns a contiguous block of dates for all stocks. Cross-sectional operations (ranking, daily RankIC) are then local.
- **Rolling operators** need a **halo exchange** with the neighbouring rank. The halo width is derived recursively from the expression tree.
- **`ema`** is an inherently sequential recurrence. It is parallelised as an affine prefix scan using `MPI_Exscan` with a custom operator.
- **Reductions.** RankIC averages and segment statistics are combined with `MPI_Allreduce`.
- **Exploiting layer-wise generation.** A batch of expressions is merged into a DAG, so shared subtrees are computed once. The gene-pool results are cached between layers. For the deep layers, most of the tree is then already computed. One gene pool is about 58 GB, so this cache only fits in memory when it is distributed across ranks or GPUs.
- **Node-level parallelism.** OpenMP threads parallelise over stocks, with sliding-window O(T) operator implementations. On GPUs, a CUDA kernel computes the per-day ranking (one thread block per day, with the row held in shared memory).

### (b) Distributed and GPU PCA signatures

- **Distributed power iteration.** With rows (dates) partitioned across ranks:
  - computing $u = Rv$ is local;
  - computing $v = R^\top u$ requires an `MPI_Allreduce` of a length-N vector at every iteration.

  This is similar to the distributed iterative-solver pattern in Lecture 8. The vectors of a whole batch of factors are combined into one Allreduce, to reduce latency cost.
- **CUDA kernels:**
  - batched sampled ranking;
  - fused batched power iteration (memory-bound);
  - a hand-written tiled shared-memory GEMM for the candidate-versus-archive similarity matrix, compared against cuBLAS.
- **The serial part.** Admission to the archive is inherently sequential, so it stays on rank 0. It is a measurable serial fraction for the Amdahl / Karp–Flatt analysis.

### Validation and experiments

- **Validation.** Every operator, RankIC series, signature and archive decision is checked against the existing Python implementation, together with a reproduction of the 1,481 factors.
- **Experiments:**
  - strong and weak scaling;
  - MPI-process versus thread ratios;
  - time breakdown into compute, halo, Allreduce and serial admission;
  - CPU versus GPU;
  - cache on versus cache off.
- **Platforms:**
  - my own server (an RTX 3090 plus a multi-core CPU) for development and experiments;
  - Atlas or Vanda for multi-node runs where required.

## 4. Questions for discussion

1. Is it acceptable to build on my existing Python project by rewriting its evaluation engine in C++?
2. Would this be suitable as the topic for this project assignment?
3. May I run the experiments on my own server (an RTX 3090 plus a multi-core CPU, with more computing resources available later if needed)?

If convenient, I would like to talk with you briefly after next Tuesday's lecture (3 November) to discuss this proposal.
