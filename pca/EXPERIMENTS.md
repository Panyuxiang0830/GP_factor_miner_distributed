# PCA 实验入口

任务依据 [docs/pca/task.md](../docs/pca/task.md)。所有 PBS 命令从仓库根目录提交，`PCA_RUN` 指向新的私有输出目录；需要真实输入的脚本还需 `PCA_MANIFEST`。作业使用 CPU／GPU 计算节点，队列与资源按站点配置调整。

完整研究的阶段、输入和环境变量见 [REPRODUCE.md](REPRODUCE.md)。特别注意 study.pbs 默认只研究 32 个因子，整代研究必须显式设置 PCA_COUNT；仅生成数值检查点时设置 PCA_NUMERICAL_ONLY=1，随后用 sensitivity.pbs 并行重放准入。

| 范围 | 脚本 | 核对内容 |
| --- | --- | --- |
| 模块无数据检查 | smoke.pbs / smoke_gpu.pbs | 新目录、Python 参考、CPU/MPI/GPU 与独立 cuBLAS 链接分离 |
| P0 真实小样本 | validate_cpu.pbs / validate_gpu.pbs | 同输入签名、Python 准入、重复和精确 SVD 抽查 |
| P0 边界语义 | adversarial.pbs | 0.90 附近、并列、同 key、严格质量阈值、无效签名 |
| 提供签名重放 | gpu_api_test.pbs | 原始百分位 GPU 接口，以及 CPU/CUDA/MPI 准入重放 |
| P1 串行规模 | serial_scale.pbs | 合成输入分别改变 B、m、n、k；不是重新排名同一真实批次 |
| P2 数值研究 | study.pbs / sensitivity.pbs | 精确 SVD、两算法、多个起点、预算与顺序 QD 轨迹 |
| P2 同精度成本 | iteration_cost.pbs | 矩阵乘次数、通信载荷、归约次数和耗时 |
| P3 强扩展／混合 | scaling.pbs | date／transpose／factor，1/2/4 节点，各预热及三次计时 |
| 含排名扩展 | raw_scaling.pbs | 合成抽样数值先排名，单列排名、求解和重分布成本 |
| P4 弱扩展 | weak_input.pbs / weak.pbs | 固定每 rank 完整矩阵等价工作量，同一输入序列前缀 |
| P5 GPU kernel | gpu_roofline.pbs | 排名、批量 GEMV、分块相似度与名义 roofline |
| P5 单／双 GPU | validate_gpu_mpi.pbs | CUDA-aware MPI 显存归约，预热与重复计时 |
| P5 精度 | precision_full.pbs / precision_finish.pbs | FP32/FP64、同输入 Python 准入、CPU double 排名转置 |
| 独立库参照 | cublas_reference.pbs | 相同显存输入、严格 FP32，GEMM 与 GEMM+abs 分开计时 |
| P6 档案规模 | archive_scale.pbs | 集中与 MPI 分片档案，相似度／准入／通信分别记录 |
| 整批和扩展验收 | validate_full.pbs / validate_scaling.pbs / validate_synthetic_runs.pbs | 全量顺序 QD、重复产物、布局一致性，数值抽查范围另列 |

原始导出输入由 `prepare.py` 生成 manifest；大矩阵不复制。已有排名 R 不再次排名，原始抽样数值才使用 `--rank-input 1`。合成受控谱、合成弱扩展与真实提案是不同输入，不混称真实 GP 进化结果。

与交接文字的范围差异也应保留：P1 的 m/n 扩展采用独立合成矩阵维度，不是同一真实因子的重采样／股票子集重新排名；同精度 MPI 时间和通信比较为前 8 个真实因子的代表实验，不作为整代同精度加速结论。GPU 实现幂迭代，Lanczos 位于 CPU/MPI。RankIC 的 GPU 百分位复用接口已经提供，跨模块调用的确认与联调仍需另行完成。

`report.py` 可生成部分已完成研究的图；`assemble_report.py` 是整套实验的严格汇总门槛，使用该套脚本约定的目录和配置名，要求指定的验证标记、重复次数和完整规模齐备。它不是接受任意小规模 smoke 输出的通用报告工具。默认目录名可在其 CLI 帮助中查看。

```bash
python3 pca/assemble_report.py --runs pca/runs \
  --out pca/runs/report_01 \
  --scaling-validation scaling_validation_local_01 \
  --synthetic-validation synthetic_validation_01
```

只在证据完整时调用。`paper_sections.py` 据相同证据按 report/main_zh.tex 的 PCA 章节组织补稿；生成的报告与图仍只写入私有 runs。源仓库不包含实验结果或论文补稿。

性能每组预热一次、测量三次并取中位数；准确性与性能计时分开。GPU roofline 的字节量为名义下界，不能当作 profiler 实测 DRAM 流量。独立 kernel 加速不等于完整 GP 加速。完整 GP 的接口联调及同批次端到端测量仍需另行完成。
