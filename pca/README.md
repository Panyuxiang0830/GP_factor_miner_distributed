# PCA 签名、相似度与顺序准入

本目录实现项目第③部分，任务见 [docs/pca/task.md](../docs/pca/task.md)。生产数值计算使用手写 C++17、OpenMP、MPI 和 CUDA；Python 用于输入准备、独立验证和实验分析。

```text
抽样因子值（可选排名）或已排名矩阵 R
  → 幂迭代 / Golub–Kahan Lanczos
  → PCA 签名
  → FP32 相似度粗算 + FP64 复核
  → 按候选顺序更新 QD 档案
```

这是可独立运行的模块，原 GP 引擎尚未自动调用该实现。与上游批次接口对接的约定见 [WALKTHROUGH.md](WALKTHROUGH.md)。本目录不依赖旧的 pca_workflow/、pca_step1/ 或已移除的 gpe_export 目标。

## 编译

以下命令均从仓库根目录执行；在集群计算节点构建和运行。

```bash
cmake -S pca -B pca/build -DPCA_MPI=ON
cmake --build pca/build -j 4
OMP_NUM_THREADS=2 ctest --test-dir pca/build --output-on-failure
```

需要支持 C++17 的编译器、CMake≥3.18、OpenMP 和 MPI；串行构建可设 `-DPCA_MPI=OFF`。GPU 使用独立目录，加 `-DPCA_CUDA=ON`，默认 CUDA 架构 86。Vanda 的模块配置见 PBS 脚本。

默认生产程序不链接 BLAS/LAPACK/cuBLAS/cuSOLVER。`-DPCA_CUBLAS_REFERENCE=ON` 仅为 CUDA 构建增加独立的 `pca_cublas_reference` 可执行文件；它不改变生产目标的链接依赖。

## 无行情数据的小规模检查

提供 CPU 和单 GPU 的完整模块检查，自动生成小合成输入，检查三种布局、CPU 新排名、Python 准入、Lanczos、MPI 档案，以及 GPU 精度和 CUDA-aware MPI。GPU 检查另验证独立 cuBLAS 参照与生产程序的链接分离。

```bash
mkdir -p pca/runs
qsub -v PCA_RUN="$PWD/pca/runs/smoke_cpu_01" \
  -o "$PWD/pca/runs/smoke_cpu_01.log" pca/smoke.pbs
qsub -v PCA_RUN="$PWD/pca/runs/smoke_gpu_01" \
  -o "$PWD/pca/runs/smoke_gpu_01.log" pca/smoke_gpu.pbs
```

输出路径必须是新的 `pca/runs/` 子目录。只有检查全部通过后才生成 SUCCESS 和 SUCCESS.json。Python 验证需要 Python≥3.11、NumPy、pandas；分析绘图另需 matplotlib。Vanda 脚本加载对应科学计算模块，不需要访问市场数据。

## 使用已有导出输入

`prepare.py` 读取已提供的 `R/meta.json`、`R/chunk_*.f32`、`scores.jsonl`、`signatures.f32` 和初始向量。大矩阵保留原位置，仅生成私有 manifest。

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

| 参数 | 含义 |
| --- | --- |
| `--layout date` | 按日期切分，整批范数和右向量用 Allreduce |
| `--layout transpose` | 从日期块 Alltoallv 重分布成完整因子，再本地求解 |
| `--layout factor` | 直接按完整因子读取的独立输入场景 |
| `--method power/lanczos` | 幂迭代 / 双对角化与两遍完全重正交化、自写 Jacobi |
| `--steps k` | 预算；power 为 2k+1 次矩阵乘，Lanczos 通常 2k−1 次，诊断另计 |
| `--rank-input 1` | 输入是未排名抽样值，先排名；现成 R 不应再次排名 |
| `--signature-input file` | 用提供的签名单独重放准入 |
| `--archive mpi` | 档案签名分片，MAXLOC 选择全局最近记录 |
| `--device cuda --precision float/double` | GPU 批量幂迭代；CUDA 尚未实现 Lanczos |
| `--device cuda --layout date --cuda-aware 1` | 直接对显存地址进行 MPI 归约 |

准入质量为 abs RankIC，相似度阈值 0.90，替换要求严格大于旧质量加 0.002。同块内修改立即可见，完全并列按 key 字典序选择较大者。详情与接口见 WALKTHROUGH。

运行输出为 signatures.f32、decisions.json、archive_keys.json 和 metrics.json；验证另写 validation.json。总计时包括循环内签名写入，不含 MPI 启动、GPU 初始化及最终 JSON 序列化，阶段存在嵌套，不能直接相加。

## 实验与复现

P0–P6 脚本、验证范围和报告入口见 [EXPERIMENTS.md](EXPERIMENTS.md)，完整阶段依赖与整代参数见 [REPRODUCE.md](REPRODUCE.md)。完整报告汇总只接受已齐备且通过验证的运行，不以脚本存在或成功提交作业代替完成。

所有行情、因子表达式、manifest、运行输出、图表和报告保留在私有且被忽略的 runs，不提交到公开仓库。仓库仅包含源码、人工合成测试生成器、作业脚本和说明。

开发使用 OpenAI Codex 辅助设计、实现、审查和编排；数值结论应由实际运行和独立 Python 参考验证支持。
