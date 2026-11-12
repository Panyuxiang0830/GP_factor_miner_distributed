# 原 GP 与 C++/MPI/CUDA 的实际衔接

本入口直接运行原项目的 run_hierarchical_evolution，不另写选择、交叉或变异算法。原程序新增可选 batch_backend，默认 None 时保持原数值路径；启用 CppGPBackend 后，进化期的批量因子求值、RankIC 与适应度交给已有 MPI 引擎，签名和准入交给 pca 模块，结果返回原 Python 程序继续下一代。

## 1. 数据经过哪些步骤

1. Python 按原有分层 GP 策略生成表达式，并使用原缓存、父代选择、交叉和变异。
2. 一个 GP 进度块作为一个 MPI 请求，C++ 按 engine_chunk 将其分为有界的表达式批次，构建 DAG 并调用 gpe::evaluate_batch_mpi。复用上游的 halo、ema、RankIC 和适应度实现，不重新实现因子算子。
3. 引擎暂时保留本批根面板。每个 rank 只从自己拥有的日期中按全局 t % sample_step == 0 抽样，排除负索引 halo。抽样行构成 R；CPU 在本地排名，CUDA 将同批因子行合并排名。幂迭代的范数与右向量由 MPI 归约；CUDA 直接使用显存指针。签名按全局日期顺序汇集、统一符号。
4. 返回与请求同序的 Candidate 和 FP32 签名。原 Python 程序按自己的适应度排序与家族/父代规则确定准入顺序；不能按 MPI 返回速度重排。
5. C++ QD 读取 Python 当前档案快照，按给定顺序准入，返回逐候选决定和完整新档案。Python 检查完整 key、质量和签名后更新原 archive 对象，继续生成下一代。

这里不需要先手工导出 R，也不保存整代全量因子值。引擎只保留有界表达式子批的根面板；其内存仍受 engine_chunk、dag_chunk、股票数和日期块大小影响。

## 2. 编译

```bash
cmake -S pca -B pca/build_live -DPCA_MPI=ON -DPCA_ENGINE_BENCH=ON
cmake --build pca/build_live -j 4 --target pca_engine_bridge pca_run
```

CUDA 使用独立目录并加 -DPCA_CUDA=ON。生产不链接 BLAS、cuBLAS 或 cuSOLVER；NumPy 只用于原 Python GP 的控制/准备、数据一致性检查和独立验证。GPU Lanczos 不在本次范围，CPU 后端可选 Lanczos。

## 3. 使用现有 EOD 命令行

原命令行新增以下可选参数；三项路径必须一起给出：

```bash
python3 -m gp_factor_miner.hierarchical_cli \
  --data-file /path/to/eod --label-file /path/to/label.pkl \
  --output-dir /path/to/new/private/output \
  --cpp-panel-dir /path/to/exported/evolution/panels \
  --cpp-engine-binary "$PWD/pca/build_live/pca_engine_bridge" \
  --cpp-pca-binary "$PWD/pca/build_live/pca_run" \
  --cpp-ranks 4 --cpp-device cpu
```

Python 环境需能导入 python/src/gp_factor_miner。提供 --cpp-device cuda 时须申请 GPU，并使用 CUDA 构建；建议每张卡一个 MPI rank。--cpp-mpiexec 可指定 MPI 启动器绝对路径，--cpp-precision 选择 float/double。原 CLI 的样本外 RankIC、相关性筛选和输出文件沿用原实现；本后端加速进化期，不宣称已经把后续研究流程全部移到 GPU。

--smoke 启用后，以指定 C++ 导出目录中的日期、股票顺序确定进化期的小样本，最多 256 天、512 只股票；原筛选期保留相同股票。启动时逐项核对 Python 与导出数据的坐标和数值，不允许同维度但不同日期/股票的数据混用。导出目录必须由相同清洗流程生成。

不提供这些 --cpp-* 参数时，原 CLI 保持原 Python 后端。源码包需要完整仓库目录结构，不能只复制 gp_bridge.py 单文件。

## 4. 直接从已有二进制字段目录启动

没有原 EOD pickle 时，也可使用现有字段导出，仍调用相同的三层 GP：

```bash
python3 pca/run_gp.py --panel-dir /path/to/evolution/panels \
  --engine-binary pca/build_live/pca_engine_bridge --pca-binary pca/build_live/pca_run \
  --out pca/runs/live_gp_01 --ranks 4 --device cpu \
  --population 24 --generations 1 --gene-pool 12 --windows 1 2 3
```

该入口默认是小规模配置，完成深度 1、2、3 各阶段及指定代数，输出原 GP 检查点、进度、完整 QD 档案、调用计时和 SUCCESS。它只使用指定进化期字段，不凭空生成样本外筛选数据。--resume 要求参数、数据与后端身份一致。

正式规模可显式使用 --population 10000 --generations 10 --gene-pool 2000 --warm-start-multiplier 3 --progress-chunk 1000 --windows 1 5 10 15 20 30 60；需另行申请适当计算资源和运行时间。小规模闭环通过不等于该完整预算已经实测。

集群入口为 live_gp.pbs，参数见脚本的 PCA_PANEL_DIR、PCA_ENGINE_BINARY、PCA_BINARY、PCA_RANKS、PCA_DEVICE 等变量。父 Python 进程从计算节点启动 MPI 子批，不能从登录节点运行大规模 GP。OpenMP 线程数通过 OMP_NUM_THREADS 设置，绑定时为每 rank 分配相应 PE。

## 5. 检查点与验收

原 GP 的 V4 检查点额外记录 batch_backend_identity，包括输入导出、矩阵坐标、文件大小与时间戳、可执行文件哈希、MPI rank 数、精度、方法及分块配置。未启用后端的旧检查点仍兼容；不能把它直接当成新后端的中间状态。不同数据或后端恢复时明确报错，不静默退回 Python 计算。

bridge_check.py 检查固定表达式的 Python/C++ 评分与签名、带历史档案的准入、三阶段完整进化、中断恢复和后端身份拒绝。闭环时将原 Python 评估/PCA/QD 入口设为调用即失败，确认不存在隐蔽回退。bridge_cpu.pbs 与 bridge_gpu.pbs 分别在计算节点运行。

cli_bridge_check.py 验证原 EOD CLI、三个深度的 GP、原样本外筛选和最终选因子文件，另核对已完成运行恢复。跨节点 GPU 另由 cross_node_gpu.pbs 提供实际不同主机上的显存通信证据。

## 6. 性能解释

一个 MPI 子进程处理一个 GP 进度块，不是每个因子启动一个进程。为先保证可靠衔接，评估与 QD 目前分别调用可执行文件；小批次下进程启动、输入准备和检查点 I/O 可能占主导。calls.json 的调用墙钟包含启动和最终输出，完整 GP 墙钟还包括原 Python 控制与检查点，不能由旧的独立 PCA 时间直接推算。

bridge metrics 的 total_seconds 为各 rank 最大值；engine_seconds、ranking_seconds 等阶段诊断目前来自 rank 0，不作为最慢 rank 阶段性能结论。配置、原始输出与计时均保存于私有运行目录。不同 MPI rank 数或精度可能引入细小评分差异，进化过程可能放大排序差异；固定输入、固定顺序的 QD 语义验证与跨配置整次进化结果相同是不同主张。
