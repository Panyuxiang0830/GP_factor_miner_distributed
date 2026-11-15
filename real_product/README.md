# real_product：单机 128 核 + 4×RTX 4090 的分层 GP 因子挖掘（进行中）

> 状态：**未完成**。CPU 路径已经实现并测试过；CUDA 源文件（`cuda/*.cu`、`tests/cpp/test_gpu.cpp`）还没写，`GPP_ENABLE_CUDA` 默认 OFF。完整的时间和内存估算章节也还没写完。

不改动 `report/ src/ include/ pca/ python/ dag/ scripts/`。课程的 C++ 核心（`../src/core`、`../pca/core.cpp`）以原文件编译进本目录的 CMake；Python 侧需要修改的模块复制到 `python/gpp/` 后再改（改动处标 `PROD:`）。

## 结构
- `src/engine.cpp`：共享内存 DAG 引擎。OpenMP 按节点并行，窄层退回算子内并行；chunk 大小由内存预算（`memory_budget_gb`）自动决定，超预算时把表达式组二分。根节点打分融合进节点任务：每天只排序一次，同时得到 RankIC、PCA 签名行和 OOS 秩向量。
- `src/archive.cpp`：PCA-QD 准入，语义与 `PCAQDArchive.consider_batch` 一致。FP32 粗算 GEMM 加误差界，再用 FP64 精修；每行只保留摘要，块内的顺序扫描是串行的（这是定义，见下）。
- `src/screening.cpp`：2022 样本外筛选。所有候选并入一个 DAG，一次求值；相关性由分块乘积给出近似值，阈值带内用两遍 FP64 公式精确计算，最后按原顺序贪心扫描。
- `src/c_api.cpp` + `python/gpp/native.py`：ctypes 进程内后端，取代原来的 mpirun/文件桥。
- `python/gpp/`：`config.py`（深度 1–5，各阶段节点上限 3/7/11/16/21，种群 100k）、`hier_engine.py`、`selection.py`（向量化 lexicase，随机抽取与原实现相同）、`backend.py`、`screening.py`、`cli.py`、`export.py`（可选 warm-up 行）、`synthetic.py`。
- `configs/host_*.json`：按主机内存给的预设，含 `memory_budget_gb` 和 chunk 设置。`scripts/`：build、smoke、export、full、bench。

## 验证（本机 macOS、单线程、无 CUDA）
- C++ `ctest`：引擎结果与课程串行引擎逐位一致（不限预算、紧预算、chunk=3 三种情况）；QD 决策与 `pca::Archive` 完全一致。
- pytest `tests/python`：14 个测试，13 个通过。
  - **因子值 / RankIC / 适应度 / 签名**：参照是课程 Python 算子，但每个节点的输出都舍入到 float32（C++ 面板就是 float32 存储），这样逐项一致（约 1e-17）。和原始 float64 pandas 相比：深度 ≤2 完全一致，深度 3–4 约有 2–11% 的表达式均值 RankIC 相差最多 3e-5。
  - **截面常数日**：某天的截面完全是常数时，C++ 返回 NaN（数学上的定义值）。pandas 这时会给出 NaN 或 1e-16 的舍入残差，取决于 N 和求和的内存布局，所以没有办法逐位复现。参照里对这种日也取 NaN。
  - **QD 准入**：与 Python 逐项一致（block 1/16/4096，且测了重新同步的情况）。
  - **OOS 筛选**：sample_step 5/20、带 limit 的情况下，两个 CSV 逐行一致。和未修改的 pandas 相比，3 组种子 × 2 种步长的入选因子全部相同，|ΔIC| ≤ 6.6e-4。
  - **lexicase**：随机抽取与原实现完全相同。
  - **gpp.hier_engine（纯 Python 路径）与课程引擎**：深度 1–3 结果完全一致。
  - **CLI smoke（深度 1–5 + 筛选）**：跑通。
  - **未通过：`test_native_backend_equals_python_path`**。加入 `-ffp-contract=off` 之前这个测试是通过的，之后失败，还没定位原因。
- 未验证：OpenMP 多线程（本机没有 libomp），以及全部 GPU 代码。

## 待办
1. 定位上面那个失败的测试。
2. 写 `cuda/gpu_scorer.cu`。设计：
   - 每天一个 block，用 cub BlockRadixSort 排序；
   - 平均秩通过 heads/tails 标记加 max-scan 得到；
   - Pearson 求和与幂迭代的点积都用和 CPU 相同的串行顺序，配合 `--fmad=false`，目标是和 CPU 逐位一致；
   - 每个 GPU 4 条 stream，pinned 暂存区。
3. 写 `cuda/gpu_products.cu`：
   - QD 的 SGEMM tile 回传主机，交给 `SummaryBuilder` 生成摘要；
   - 筛选的六个对和用 DGEMM。
   并在服务器上跑 `test_gpu.cpp`，和 CPU 逐位比对。
4. 补全内存/显存预算和时间估算（依据 `report/main_zh.tex` §6、`report/progress.md`）：
   - 串行 10k 表达式 5060 s，其中 RankIC 3522 s；
   - 4×72 核 27.8 s；
   - 一个面板 34.3 MB，字段 4.3 GB；
   - 按 RankIC 约占 70% 推算，每代 130k 个根：CPU 打分约 0.35 核·s/根，约 355 s/代；GPU 受 PCIe 限制，约 1.4 ms/面板，约 45 s/代（4 卡）。这些是推算，不是实测。
   - 24 GB 显存放不下因子面板：扣掉约 4.5 GB 字段后只够约 500 个面板。所以算子留在 CPU，GPU 只做打分。
   - QD 准入本身是串行的：记录 i 必须看到 0..i-1 的决定，只有块内的相似度乘积能并行。
