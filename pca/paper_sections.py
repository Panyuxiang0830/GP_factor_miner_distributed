"""Write a PCA supplement following the supplied Chinese PDF's section order."""


def table(rows, keys):
    def cell(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    return "\n".join(["| " + " | ".join(keys) + " |",
                      "| " + " | ".join("---" for _ in keys) + " |"] +
                     ["| " + " | ".join(cell(row[key]) for key in keys) + " |" for row in rows])


def write_pdf_sections(out, context):
    """Called only by the full report assembler after its experiment gates pass."""
    c = context
    s = c["sensitivity"]
    times = c["gpu_times"]
    precision = c["precision"]
    notice = ("审阅草稿：仅整理当前已有结果；raw 排名扩展仍在排队，部分图片尚未生成。"
              "该文件不代表完整实验验收通过。" if c.get("draft") else
              "数值来自完整验收后的运行；完整实验及限制另见 report.md、EVIDENCE.json。")
    parts = [
        "# PCA 部分补稿：对应 report/main_zh.tex\n\n" + notice + "\n\n"
        "本补稿按 report/main_zh.tex 的 PCA 章节编排，供合入对应章节；不覆盖原有论文源文件。"
        "当前交付是独立 PCA/准入子流水线，尚不构成整代 GP 与新 GPU 实现已联调的证据。",

        "## 4.3 GPU 上的 PCA 签名与相似度\n\n"
        "每五个全局交易日抽样一天；本地只使用自有行，不能重复计入 halo。"
        "对抽样日的股票值做平均百分位排名，NaN 不参与排名，随后填 0.5 并逐行去均值，得到 R。"
        "GPU 用一行一个线程块的共享内存 bitonic 排序，补到二次幂长度，当前最多 8192 列。"
        "该排序网络为 O(n log²n)，与 CPU 比较排序的 O(n log n) 理论工作量不同。"
        "供 RankIC 复用的 gpu_percentile_rows 保留 NaN、不填充、不去均值；PCA 用 gpu_rank_rows。\n\n"
        "每个因子从相同初始右向量开始，交替计算并归一化 Rv 和 Rᵀu，最后再计算 Rv。"
        "Rv 按行归约，Rᵀu 按股票列并行；幂迭代支持 float/double 矩阵和向量。"
        "按日期切分时，Rv 是局部计算，但其范数需要全局归约；Rᵀu 归约后在各 rank 复制。"
        "一批 B 个因子把范数合为 B 个数、向量合为 B×n 个数，减少小消息。"
        "k 轮以及最终一次 Rv 共需 k+1 次范数归约和 k 次向量归约；最终签名按日期汇集后统一固定符号。"
        "CUDA-aware MPI 直接接收显存地址，并在运行时检查支持情况。\n\n"
        "比较签名前还要重新中心化、归一化，取绝对 Pearson 相关。GPU 使用手写 16×16 分块共享内存乘法粗算；"
        "CPU 通过保守误差界筛出可能为最大值、接近阈值或并列的记录，以 FP64 复核。"
        "候选依次处理，先前加入/替换立即可见，质量是 abs_rank_ic，替换要求严格高于旧值加 0.002。"
        "另有签名分片的 MPI 档案，以 MAXLOC 选择全局最近记录，复制小的 key/quality 元数据。"
        "默认 GPU 档案与 MPI 分片档案是两条实验路径，不能混成一次联合加速结论。\n\n"
        "cuBLAS 仅作为独立性能参照；生产求解与准入仍全为手写。"
        "独立参照程序不会被生产目标链接，动态依赖另作检查。",

        "## 4.4 幂迭代与 Lanczos\n\n"
        "幂迭代以固定起点反复施加 RᵀR，主、次方向分离速度取决于 σ₂/σ₁，"
        "比值接近 1 时四轮不足以保证主向量收敛。实现采用 Golub–Kahan 双对角化，"
        "保存左右 Krylov 基、进行两遍完全重正交化，再以自写 Jacobi 解小矩阵。"
        "不显式形成大的 RᵀR。幂迭代需 O(m+n) 向量空间，Lanczos 需 O(k(m+n)+k²)。"
        "本实现 k 轮 power 有 2k+1 次矩阵向量乘，k 步 Lanczos 通常有 2k−1 次；"
        "残差诊断另需两次。因此同名义步数不代表相同计算量。",

        "## 第 5 节：PCA 验证补充\n\n"
        f"真实行情上的深度 3 随机提案共有 {s['factors']} 个非恒定 R，"
        f"{c['provided']['candidates']} 个有效且可比较候选。它不是原文 6.2 的深度 2 引擎批次，"
        "也不是原文的 1481 个进化筛选因子。\n\n"
        "首先重放提供的签名，只隔离准入实现；CPU、CUDA、分片 MPI 与 Python 的逐候选动作、"
        "最近 key、替换 key、相似度及最终成员一致。其次从同一 FP32 R 重算，检查串行与并行签名，"
        "直接按因子读取和转置路径要求逐位一致。另用精确 NumPy SVD 研究近似算法误差。"
        "这三层验证不能相互替代。接近 0.90、最大值并列、同 key、质量差恰为 0.002、"
        "无效签名由独立边界用例覆盖。不能仅以误差比 0.90 小就断言决定不变。",

        "## 6.3.1 串行代价与 CPU 扩展\n\n"
        "本 PCA 驱动在 MPI_Barrier 后以 std::chrono::steady_clock 计时，各阶段取 rank 最大值；"
        "GPU kernel 使用 CUDA events。原稿因子引擎的 MPI_Wtime 计时另行说明。"
        "19 组 B/m/n/k 实验各预热一次、测量三次，以中位数对比 O(Bkmn) 的矩阵计算工作量。"
        "现成真实 R 已排名，因此该组 ranking_seconds=0。另对 9216 张合成数值面板实际排名，"
        "补测 72/144/288 核的完整阶段。合成面板排名后的谱不再套用受控谱参数。"
        "CPU 日期布局、先转置和直接因子读取分别报告；直接因子读取不含上游重分布。"
        "强扩展、固定每 rank 32 张完整矩阵等价工作量的弱扩展、MPI+OpenMP、1k–50k 档案另列，"
        "不可把 GPU 的双卡实验当作 288 核 CPU 实验。\n\n"
        "![串行代价](serial_problem_size.png)\n![完整排名流水线](raw_ranking_pipeline.png)\n"
        "![CPU 强扩展](strong_scaling.png)\n![弱扩展](weak_scaling.png)\n![档案规模](archive_scaling.png)",

        "## 6.3.2 GPU kernel 与 CPU 对比\n\n"
        "三个 kernel 独立计时，预热一次、测三次；float/double 指幂迭代精度，"
        "排名和粗算相似度在两组中都为 FP32。FP32 roofline 使用 A40 的非 Tensor 峰值，"
        "矩阵读取量和相似度必要输入输出量只是名义流量，不是 profiler 实测 DRAM 流量。\n\n" +
        table(c["kernels"],["iteration_precision","B","ranking_seconds","gemv_seconds","similarity_seconds"]) +
        f"\n\n同一批 {s['factors']} 个 R：72 核 CPU 日期布局 {times[0]:.3f} s，"
        f"单 A40 {times[1]:.3f} s，双 A40 {times[2]:.3f} s。"
        "子流水线计时包含读取、准备、传输、计算、相似度、准入及循环内签名写入；"
        "不含 MPI 启动、GPU 设备初始化与最终 JSON 序列化，签名写入也不代表 fsync 已完成。"
        "不能从 kernel 的吞吐直接推断流水线加速。\n\n"
        "![GPU kernel](gpu_kernel_times.png)\n![CPU/GPU](cpu_gpu_end_to_end.png)\n![Roofline](gpu_roofline.png)\n\n" +
        ("cuBLAS 独立参照使用相同显存输入和原手写 kernel，排除传输、分配、归一化；每块 50 次，"
         "预热块不计，三块取中位数。手写融合 abs，库的完整等价操作为 GEMM 加一个 abs kernel；"
         "纯 GEMM 另列。固定严格 FP32 模式、关闭 TF32，不把此结果称为库的所有模式最优值。"
         "依据 [NVIDIA cuBLAS 12.9 精度定义](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cublascomputetype-t)。\n\n" +
         table(c["cublas_rows"],["B","A","hand_similarity_seconds","cublas_gemm_seconds","cublas_similarity_seconds","cublas_full_semantics_speedup"]) +
         "\n\n![cuBLAS 参照](cublas_comparison.png)" if c.get("cublas_rows") else
         "cuBLAS 独立参照尚待实测，不在草稿中填入性能数字。"),

        "## 6.3.3 MPI + CUDA 扩展性\n\n"
        f"单卡到双卡的端到端加速比为 {times[1]/times[2]:.3f}。"
        "日期块分给不同 GPU，显存范数和 Rᵀu 直接用于 MPI_Allreduce；每组预热一次、测三次。"
        "实际硬件、布局和通信量见原始 metrics。不要用 FP64 kernel 的表现套用 FP32 的计算上限。",

        "## 6.3.4 签名与准入判定的一致性\n\n"
        f"整代 FP32 对 FP64：动作变化 {precision['fp32_vs_fp64_changed_actions']}，"
        f"决定三元组变化 {precision['fp32_vs_fp64_changed_decision_tuples']}，"
        f"最终成员对称差 {precision['archive_key_symmetric_difference']}。"
        f"提供签名对从 FP32 R 重算：动作变化 {c['provided']['provided_vs_exported_R_changed_actions']}，"
        f"成员对称差 {c['provided']['provided_vs_exported_R_archive_key_difference']}。"
        "这证明本批次的这些实现/精度选择保持决定一致，不证明其他初值或算法预算也一致。"
        "gpu_precision.csv 的单次正确性运行时间不用于性能加速比。",

        "## 6.4.1 精度与迭代次数\n\n"
        f"{s['fraction_ratio_above_09']:.1%} 的 R 满足 σ₂/σ₁>0.9。以 1−|cos|≤1e−6 为目标，"
        f"seed=0 的四步幂迭代有 {c['power4_not_converged']} 个因子未达到目标。"
        "因此四步不能被表述为普遍收敛。曲线同时以步数和矩阵向量乘次数作横轴，并报告多个初值。"
        "重根或很小谱间隙时应结合残差解释主方向误差。受控谱由四个解析正交 DCT 方向和小噪声构造，"
        "与原文 2.2 的按字段生成合成行情不同。\n\n![谱比](spectral_ratios.png)\n"
        "![误差曲线](convergence.png)\n![受控谱](controlled_spectrum_convergence.png)",

        "## 6.4.2 计算代价\n\n" +
        table(c["cost"],["method","converged","not_converged_by_k10","median_steps","median_matvecs"]) +
        "\n\n前 8 个真实 R 在 8 ranks 上达到同一目标精度的重复实测：\n\n" +
        table(c["matched"],["method","k","solve_seconds","allreduce_calls","communication_seconds","logical_payload_bytes"]) +
        "\n\n当前 power 已批量归约，Lanczos 逐因子执行且两遍重正交化。"
        "步数减少不等于 Allreduce 调用数减少；调用数、字节量、通信时间应一起报告。"
        "全代研究的每因子时间另含 C API 复制和残差诊断，不当作纯 kernel 时间。",

        "## 6.4.3 对准入判定的影响\n\n"
        "基准是 seed=0、power k=4；下表分别统计动作、最近 key 和最终成员对称差，"
        "后者同时计入新增和移出的成员，不等于替换次数。\n\n" +
        table(c["selected"],["method","k","changed_actions","changed_closest_keys","archive_key_symmetric_difference"]) +
        "\n\n实现与原基线完全一致，并不说明换成更收敛的算法后档案不变。60 条轨迹均保留，"
        "初值敏感性也按同一口径检查。\n\n![准入变化](decision_sensitivity.png)",

        "## 7.1 各部分如何并行：PCA 补充\n\n"
        "| 模块 | 技术 | 主要通信 | 限制 |\n| --- | --- | --- | --- |\n"
        "| 抽样排名 | CPU OpenMP / GPU 排序 | 日期本地，无截面全局排序 | 排序、传输 |\n"
        "| 签名 date | MPI + CPU/GPU | 每批归约 B 范数、B×n 向量 | 数据量不随 P 缩小 |\n"
        "| 签名 transpose | MPI Alltoallv + 本地求解 | 重分布与最终汇集 | 重分布、文件读取 |\n"
        "| 集中档案 | CPU/GPU 粗算、CPU FP64 复核 | 签名汇集 | 顺序准入和准备 |\n"
        "| 分片档案 | MPI MAXLOC | 每候选选择与元数据一致性 | 归约和重复元数据操作 |\n\n"
        "因子引擎 halo、ema、RankIC 的行沿用原文已验证部分；不同批次的通信时间不能直接相加。"
        "本部分计得的 C++ 准入时间不是原 Python GP 的串行比例。",

        "## 7.2 关于数据的结论：本批 PCA 档案\n\n"
        "下表比较随机提案中的有效候选与默认四步幂迭代得到的最终 QD 档案，质量为样本内 abs_rank_ic。"
        "这不是原稿 1481 个已选进化因子的分布，也不含新的样本外回测。\n\n" +
        table(c["rankic_rows"],["population","count","minimum","p25","median","p75","maximum"]) +
        "\n\nQD 同时保留行为多样性，不能预设其质量分布会整体向高值移动。"
        "FP32 导出影响、算法预算影响和投资有效性是不同问题。股票池扩大导致的引擎负载不均见原文，"
        "本部分的固定尺寸 R 不直接测量每日股票池扩大。\n\n![RankIC 分布](rankic_distributions.png)",

        "## 7.3 后续工作 / 8 结论\n\n"
        "已有结果支持独立子流水线的数值和顺序语义，以及所测配置的扩展结论。"
        "仍需与最新因子引擎在批次接口联调，核对全局日期、halo 排除、key 与返回顺序。"
        "CPU/GPU 异步重叠、GP 移民提前计算、显存预算自适应分块和基于残差停止属于后续工作，"
        "不能写成已经实现。四步幂迭代是预算选择；提高近似精度可能真实改变准入，"
        "而本批次从 FP32 改到 FP64 未改变最终成员。单个 kernel 的峰值表现不代表端到端优势。",

        "## AI 工具的使用 / 附录 A 补充\n\n"
        "使用 OpenAI Codex 协助设计讨论、代码起草、审查、实验编排和文字初稿。"
        "数值证据来自实际 PBS 作业，验证使用原 Python 准入和仅用于参照的精确 SVD；"
        "不以 AI 生成的数字替代实验。\n\n"
        "新增源码、CMake 和 PBS 脚本位于 pca，原有代码不改写；"
        "数据、输出、图表和本补稿仅位于忽略的 runs。README 说明编译参数、manifest 准备与作业提交，"
        "WALKTHROUGH 说明上游接口。CPU/GPU 需分别开启 PCA_MPI/PCA_CUDA；原项目的构建命令"
        "不会自动把本独立实现接入 GP。复现入口为 final_report.pbs 和 assemble_report.py。"
    ]
    (out / "pdf_sections.md").write_text("\n\n".join(parts) + "\n")
