# PCA-QD 最终交付索引

本次交付为项目第③部分：PCA 签名、相似度、顺序 QD 准入，以及与原分层 GP 的可选后端衔接。开发分支为 pca-implementation；尚未合并 main。

## 阅读顺序

1. [README.md](README.md)：编译、模块运行和基本接口。
2. [GP_INTEGRATION.md](GP_INTEGRATION.md)：原 GP 与实际 MPI 引擎、PCA、QD、下一代和检查点的衔接。
3. [DELIVERY.md](DELIVERY.md)：档案格式、验收含义与私有报告打包。
4. [EXPERIMENTS.md](EXPERIMENTS.md) / [REPRODUCE.md](REPRODUCE.md)：实验目录和复现步骤。

算法与边界语义另见 [WALKTHROUGH.md](WALKTHROUGH.md)，GPU 百分位的 RankIC 消费契约见 [RANKIC_INTERFACE.md](RANKIC_INTERFACE.md)。

## 主报告

服务器上的当前报告目录为 `pca/runs/final_submission_04/`，相对于当前 Git 仓库根目录。正文沿用 report/main_zh.tex 的章序：背景与算法、数据、平台、并行设计、验证、实验、讨论和结论。实验部分先报告 PCA 签名与相似度，再比较幂迭代与 Lanczos，最后给出实际 GP 衔接结果。保留 7 张结果与并行汇总表、8 幅主要实验图，精简基础概念与重复图示；原始数值不变，前三版报告保留为历史资料。

- **主报告：`pca_paper.pdf`**。
- 可编辑正文：`pca_paper.md`；必要插图位于 `figures/`。
- 论文合并片段：`latex/pca_sections.tex`。
- 完整仓库源码：`source.tar.gz`，保留实际编译所需的上游源码布局。
- 验收摘要：`DELIVERY_STATUS.json` 与 `evidence/`；文件校验见 `MANIFEST_SHA256.json`。

报告合并 PCA 实验、接口验收、实际 GP 衔接和全部字段的完整数据维度预演。交接要求数据、表达式和实验输出不进入公开仓库，因此报告与结果包保留私有；GitHub 提交源码、人工合成检查、必要复现脚本和本文索引。

## 源码范围

| 内容 | 主要文件 |
|---|---|
| CPU/OpenMP、Lanczos | core.cpp / core.hpp |
| CUDA 排名、幂迭代与相似度 | gpu.cu / gpu.hpp |
| 顺序及 MPI 档案、快照 | mpi_archive.hpp / archive_io.hpp / archive_io.py |
| 独立模块入口 | main.cpp / prepare.py / validate.py |
| 实际因子引擎桥接与 GP 后端 | engine_bridge.cpp / gp_bridge.py / run_gp.py |
| 原项目可选接口 | ../python/src/gp_factor_miner/hierarchical_engine.py / hierarchical_cli.py |

其余 C++、Python、PBS 文件服务于独立参考、数值验收和报告中实验的复现。上游构建与 include/src 依赖保留，不把它们当成临时文件删除。生产目标不链接 BLAS/cuBLAS/cuSOLVER；库仅用于独立性能参照。

## 当前验收与边界

已通过 CPU/MPI、CUDA、历史档案接续、顺序准入、独立数值参考、完整快照与失败反例验证；实际 GP 三阶段闭环、中断恢复、原 CLI 与合成样本外流程均已验证。全部真实字段和完整数据维度的缩小进化预算预演已经完成。跨节点 GPU 验证与单节点双卡性能实验分别记录。

当前范围为使用已有进化期数据的 Python＋C++ 流程，2022 年样本外评估不纳入本次交付。正式人口与代数的运行已列入计划，目前正比较 CPU 配置，尚无正式规模结果。原 CLI 的合成样本外验证保留为接口检查。GPU Lanczos 不在本次范围，全组最终 LaTeX 合稿未在本次编译。

## 提交清单原则

保留源码、必要测试、可重复实验入口及说明。运行目录、历史版本报告、构建产物、缓存、临时文件、账户记录、行情和因子表达式不提交；旧输出仍留在私有运行目录用于追溯。一次调度故障期间使用的 resume_pending.py 已退出交付清单。
