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

新 study 同时生成 INPUT.json 绑定 manifest 和初始向量。复用没有该文件的历史研究时，prepare_completion.py 必须显式指定 --allow-legacy-study，并人工核对输入。历史结果不重写成新验收产生的结果。

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

## 5. 真实规模与整代同精度补充

本阶段依赖第 2 节的完整数值检查点。准备命令读取现有 manifest、SVD 研究及原始 scores，从中生成私有候选与步数计划；行情与导出 R 仍从原位置只读。将下面两个占位路径替换为已交接的 scores.jsonl 和原始行情字段目录（包含 meta.json 及字段文件）。

```bash
python3 pca/prepare_completion.py --manifest "$PCA_REAL" \
  --study "$PCA_RUNS/study_full_01" \
  --scores /path/to/export/scores.jsonl --panel-dir /path/to/panels \
  --out "$PCA_RUNS/completion_plan_01" --count 9388 --target 1e-6
qsub -v PCA_SOURCE="$PCA_RUNS/completion_plan_01/real_source.json",PCA_RUN="$PCA_RUNS/real_sizes_01" pca/real_size.pbs
qsub -v PCA_PLAN="$PCA_RUNS/completion_plan_01/accuracy_plan.json",PCA_RUN="$PCA_RUNS/matched_full_01" pca/matched_full.pbs
```

两项作业互不依赖。真实规模实验由引擎计算 128 个固定因子面板，19 个配置分别改变 B、采样日数 m、股票数 n、步数 k；每次都对所选股票重新排名。因子在完整股票池求值后固定，不在新股票池重新求值含截面算子的公式。准入使用原 scores，不重算质量。八线程生成面板的时间单列，PCA 计时使用一线程。每配置验证全量顺序 QD 与三次重复，前两个原始面板另做 pandas 排名和独立签名核对。

整代成本作业执行全部输入。两算法各自选择串行研究中首次达标的步数，十步未达标者仍以十步执行并单列；MPI 结果与相同步数串行签名比较，以误差上界认证精度。仅在两算法共同认证的同一集合比较成本。该作业是离线预算实验，两算法均逐因子归约、不计算残差诊断；不等同于生产驱动的批量 power 归约。

matched_full.pbs 预留 16 个 CPU 核，实际启动 8 ranks、每 rank 一线程。全集 FP32 缓存约需 64.7 GB，Vanda 可能按核数改写内存；作业先记录并检查实际分配资源，内存不足则退出。记录中的预留核数和活跃 ranks 要分别报告。真实规模和整代作业只有独立验证通过后才写 SUCCESS。

真实规模实验成功后，还可用其原始面板验证 RankIC 共用接口：

```bash
qsub -v PCA_REAL_SIZE="$PCA_RUNS/real_sizes_01",PCA_RUN="$PCA_RUNS/rankic_interface_01" pca/rankic_interface.pbs
```

不设置 PCA_REAL_SIZE 时执行合成边界检查；该作业验证 FP64 GPU 百分位与原引擎排名和实际 daily_rank_ic 消费者的一致性，不自动修改主 GP 后端。

## 6. 独立验收与报告

| 验收 | 前置 | 输出名 |
| --- | --- | --- |
| validate_full.pbs | PCA_REAL、PCA_BASELINE、PCA_CPU_RUN=scaling_2node_01、PCA_GPU_RUN=gpu_two_01 | full_validation_01 |
| validate_scaling.pbs | scaling_1/4node_01、sensitivity_full_01 的默认轨迹 | scaling_validation_local_01 |
| validate_synthetic_runs.pbs | 三组 weak 和三组 raw，PCA_MANIFEST=weak_input_01/manifest.json，PCA_KIND=both | synthetic_validation_01 |

每项还须设置 PCA_RUN。合成验收核对全量顺序 QD、三次重复和布局一致性，数值独立重算为每组前 8 个因子；不把抽查写成全部数值逐项验证。

全部前置成功后，以新的 PCA_RUN 调用 final_report.pbs。它检查既定目录、计时重复、配置数及验证标记，写出完整报告、图表与 EVIDENCE.json；缺证据会报错。代码目录中没有实际结果，交付验收应在服务器上检查运行与报告，而不是只查看 GitHub 文件列表。

final_report.pbs 保留基础实验的入口。包含本次两项补充的严格报告应在加载科学计算和 matplotlib 模块的计算节点上执行：

```bash
python3 pca/assemble_report.py --runs "$PCA_RUNS" \
  --out "$PCA_RUNS/report_completed_01" \
  --scaling-validation scaling_validation_local_01 \
  --synthetic-validation synthetic_validation_01 \
  --real-size "$PCA_RUNS/real_sizes_01" \
  --matched-full "$PCA_RUNS/matched_full_01" --require-completion
```

上述门槛要求交接的完整维度、9388 个因子、8 MPI ranks、19 个真实规模配置及双方成功标记；适用于本交接数据集。报告包含 completion_sections.md、两项新图及 CSV，只有补充验证完整才标记 completion_supplement_verified。迁移时可将 --runs 指向既有私有证据根；底层 report.py 的 --runs-root 同样允许显式指定根目录，输出仍须为其新的子目录。

## 7. 当前接口验收与交付

在基础实验之外，用 delivery_cpu.pbs / delivery_gpu.pbs 验证历史档案、连续批次及验收失败反例，然后按 [DELIVERY.md](DELIVERY.md) 打包当前源码和报告。跨节点 GPU 使用单独的 cross_node_gpu.pbs，须根据真实执行状态记录；现有同节点双卡结果不构成跨节点证据。实际 GP 自动衔接、原 EOD CLI 和闭环验收见 [GP_INTEGRATION.md](GP_INTEGRATION.md)；GPU Lanczos 不在本次范围。
