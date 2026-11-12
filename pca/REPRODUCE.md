# 从私有输入到完整研究

从仓库根目录按阶段执行。先确认前一阶段的输出与成功标记，再提交依赖它的作业；PBS afterok 可用于自动连接这些阶段。本文不包含市场数据、表达式、运行输出或固定作业编号。所有输出目录须为新的 pca/runs 子目录，重跑时使用另一套目录名。

## 1. 输入与基础验证

使用 README 中的 prepare.py，将已交接的 R、scores、提供签名和初始向量映射成 pca/runs/real_input/manifest.json。大矩阵保持只读，不复制到仓库。

```bash
PCA_RUNS="$PWD/pca/runs"
PCA_REAL="$PCA_RUNS/real_input/manifest.json"
mkdir -p "$PCA_RUNS"
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/cpu_validation_01" pca/validate_cpu.pbs
```

无私有输入时可先用 smoke.pbs / smoke_gpu.pbs 做合成检查；它们不替代整代研究。准备参考初始向量、scores 和提供签名是 P0 的前提。

## 2. 整代数值研究与 QD 重放

study.pbs 默认 PCA_COUNT=32，属于小规模开发设置。对交接批次的 9388 张非恒定 R，必须显式指定如下参数；其他数据规模需同时调整严格汇总中的数据集检查。

```bash
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/study_full_01",PCA_COUNT=9388,PCA_WORKERS=8,PCA_NUMERICAL_ONLY=1 pca/study.pbs
```

该阶段只生成精确 SVD、两种 C++ 算法、三个初值与 k=1…10 的数值检查点；NUMERICAL_SUCCESS.json 表示数值阶段完成，不表示准入实验完成。等数值作业成功，再执行：

```bash
qsub -v PCA_MANIFEST="$PCA_REAL",PCA_STUDY="$PCA_RUNS/study_full_01",PCA_RUN="$PCA_RUNS/sensitivity_full_01" pca/sensitivity.pbs
```

完整的 60 条准入轨迹及 summary.json 在 sensitivity_full_01。基础 report.py 应同时接收 --study 与 --sensitivity。默认不加 PCA_NUMERICAL_ONLY 时，study.py 保留“数值研究后顺序重放”的原行为。

受控谱实验单独运行：PCA_SYNTHETIC=1、PCA_COUNT=24、PCA_RUN=study_synthetic_01，仍传 PCA_MANIFEST 以满足 PBS 入口，脚本会生成并改用自己的合成 manifest。它与弱扩展输入不同。

## 3. CPU 规模、强扩展、混合与弱扩展

serial_scale.pbs → serial_sizes_01；archive_scale.pbs → archive_sizes_01；iteration_cost.pbs 以真实 PCA_REAL 为输入 → iteration_cost_01。后者是前 8 个真实因子、8 ranks 的同精度代表实验。

scaling.pbs 分别提交 1/2/4 节点，输出 scaling_1node_01、scaling_2node_01、scaling_4node_01。每节点须显式申请 ncpus=72、mpiprocs=72、mem=180gb。四节点超过免费池限制，使用项目路由队列，例如：

```bash
qsub -q auto -l select=4:ncpus=72:mpiprocs=72:mem=180gb \
  -v PCA_MANIFEST="$PCA_REAL",PCA_RUN="$PCA_RUNS/scaling_4node_01" pca/scaling.pbs
```

每节点 4 ranks×18 threads 的混合实验由该脚本同时执行。1/2 节点按相同方式设置 select 与相应输出名。

weak_input.pbs → weak_input_01，生成 9216 张 293×5876 的合成 R。完成后，以 weak_input_01/manifest.json 为输入，分别运行 weak.pbs 的 1/2/4 节点版本 → weak_1/2/4node_01。每 rank 固定 32 张完整矩阵等价工作量，每个规模读取相同序列的前缀。

raw_scaling.pbs 也使用该完整合成 manifest，但将输入解释为未排名抽样数值，实际排名后计算 → raw_scaling_1/2/4node_01。排序会改变谱，不将其称为真实行情或保持受控谱的实验。所有 288 核作业均使用满足限制的项目路由队列。

## 4. GPU、精度与提供签名

CPU 强扩展成功后，可使用同输入串行签名作为参考：

```bash
PCA_BASELINE="$PCA_RUNS/scaling_1node_01/strong_date_p1_t1_r1/signatures.f32"
```

| 作业 | 额外输入 | 规范输出名 |
| --- | --- | --- |
| validate_gpu_mpi.pbs | PCA_MANIFEST=PCA_REAL | gpu_two_01 |
| gpu_roofline.pbs | 无市场输入 | gpu_roofline_01 |
| precision_full.pbs | PCA_MANIFEST=PCA_REAL，PCA_BASELINE | precision_full_03 |
| gpu_api_test.pbs | PCA_MANIFEST=PCA_REAL，PCA_BASELINE | gpu_api_01 |
| adversarial.pbs | 无市场输入 | adversarial_02 |
| cublas_reference.pbs | 无市场输入 | cublas_reference_01 |

每项都须设置对应 PCA_RUN。完整精度实验可以直接执行 precision_full.pbs；precision_finish.pbs 只用于复用已完成 GPU 数值结果的恢复场景，不是每次必需阶段。GPU 经 auto_free 或 auto 路由，不直接提交 from_route_only 的执行队列。单／双卡是 GPU 扩展，288 核是另一组 CPU 实验。

## 5. 独立验收与报告

| 验收 | 前置 | 输出名 |
| --- | --- | --- |
| validate_full.pbs | PCA_REAL、PCA_BASELINE、PCA_CPU_RUN=scaling_2node_01、PCA_GPU_RUN=gpu_two_01 | full_validation_01 |
| validate_scaling.pbs | scaling_1/4node_01、sensitivity_full_01 的默认轨迹 | scaling_validation_local_01 |
| validate_synthetic_runs.pbs | 三组 weak 和三组 raw，PCA_MANIFEST=weak_input_01/manifest.json，PCA_KIND=both | synthetic_validation_01 |

每项还须设置 PCA_RUN。合成验收核对全量顺序 QD、三次重复和布局一致性，数值独立重算为每组前 8 个因子；不把抽查写成全部数值逐项验证。

全部前置成功后，以新的 PCA_RUN 调用 final_report.pbs。它检查既定目录、计时重复、配置数及验证标记，写出完整报告、图表与 EVIDENCE.json；缺证据会报错。代码目录中没有实际结果，交付验收应在服务器上检查运行与报告，而不是只查看 GitHub 文件列表。
