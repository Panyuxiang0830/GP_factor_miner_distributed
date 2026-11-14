#!/usr/bin/env python3
"""Generate standalone plots and an evidence-based report from completed runs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path,
                        help="optional completed QD replay directory for a reused numerical checkpoint")
    parser.add_argument("--scaling", type=Path, nargs="*", default=[])
    parser.add_argument("--roofline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs-root",type=Path,default=Path(__file__).resolve().parent/"runs",
                        help="private results root; allows reporting an existing evidence tree after source migration")
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to(args.runs_root.resolve()):
        parser.error("output must be new and inside runs/")
    summary = json.loads(((args.sensitivity or args.study) / "summary.json").read_text())
    if not summary["completed"]:
        raise ValueError("study is incomplete")
    args.out.mkdir(parents=True)
    metrics = np.load(args.study / "metrics.npy", mmap_mode="r")
    ratios = np.load(args.study / "sigma2_over_sigma1.npy")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=30)
    ax.set(xlabel="sigma2 / sigma1", ylabel="Number of factors", title="Spectral gap distribution")
    save(fig, args.out, "spectral_ratios")
    k = np.arange(1, summary["maxsteps"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, name in enumerate(("Power", "Lanczos")):
        errors = np.asarray(metrics[:, 0, method, :, 0])
        median = np.median(errors, axis=0)
        lo, hi = np.quantile(errors, [.25, .75], axis=0)
        axes[0].plot(k, np.maximum(median, 1e-16), "o-", label=name)
        axes[0].fill_between(k, np.maximum(lo, 1e-16), np.maximum(hi, 1e-16), alpha=.2)
        matvecs = np.median(metrics[:, 0, method, :, 3], axis=0)
        axes[1].plot(matvecs, np.maximum(median, 1e-16), "o-", label=name)
    for ax in axes:
        ax.set(xscale="log", yscale="log", ylabel="Median 1 - abs(cosine) vs exact SVD")
        ax.legend(); ax.grid(alpha=.3)
    axes[0].set_xlabel("Requested iteration count")
    axes[1].set_xlabel("Actual matrix-vector products (excludes residual check)")
    save(fig, args.out, "convergence")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("power", "lanczos"):
        for seed in range(summary["seeds"]):
            points = [p for p in summary["decision_sensitivity"] if p["method"] == method and p["seed"] == seed]
            # Old preexperiment reports contain tuple changes only, explicitly labeled.
            field = "changed_actions" if "changed_actions" in points[0] else "changed_decisions"
            ax.plot([p["k"] for p in points], [p[field] for p in points], "o-", label=f"{method}, seed {seed}")
    ax.set(xscale="log", xlabel="Iterations", ylabel="Changes vs power k=4, seed 0", title="QD decision sensitivity")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, args.out, "decision_sensitivity")
    groups = {}
    for directory in args.scaling:
        for path in directory.glob("*/metrics.json"):
            match = re.fullmatch(r"strong_(date|transpose|factor)_p(\d+)_t(\d+)_r([123])", path.parent.name)
            if not match:
                continue
            row = json.loads(path.read_text())
            if match[3] != "1":
                continue
            group = (match[1], row["factors"], row["rows"], row["cols"])
            groups.setdefault(group, {}).setdefault(int(match[2]), []).append(row)
    scaling_table = []
    if groups:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for group, by_rank in groups.items():
            ranks = sorted(by_rank)
            if any(len(by_rank[p]) != 3 for p in ranks):
                raise ValueError("scaling run lacks three successful measured repetitions")
            times = [float(np.median([x["total_seconds"] for x in by_rank[p]])) for p in ranks]
            label = f"{group[0]}, B={group[1]}"
            axes[0].plot(ranks, times, "o-", label=label)
            if 1 in by_rank:
                baseline = times[ranks.index(1)]
                speedups = baseline / np.asarray(times)
                axes[1].plot(ranks, speedups, "o-", label=label)
                axes[2].plot(ranks, speedups / ranks, "o-", label=label)
            for p, t in zip(ranks, times):
                scaling_table.append(dict(layout=group[0], factors=group[1], rows=group[2], cols=group[3], ranks=p, median_seconds=t))
        for ax, ylabel in zip(axes, ("Total seconds", "Speedup", "Parallel efficiency")):
            ax.set(xscale="log", yscale="log", xlabel="MPI ranks", ylabel=ylabel)
            ax.grid(alpha=.3); ax.legend(fontsize=8)
        save(fig, args.out, "strong_scaling")
    (args.out / "scaling_summary.json").write_text(json.dumps(scaling_table, indent=2) + "\n")
    roofline_rows = []
    if args.roofline:
        if not (args.roofline / "SUCCESS").exists():
            raise ValueError("roofline run incomplete")
        fig, ax = plt.subplots(figsize=(7, 4))
        intensity = np.logspace(-2, 3, 300)
        # NVIDIA A40 datasheet, non-Tensor FP32 peak (not TF32/Tensor throughput).
        bandwidth, fp32_peak = 696e9, 37.4e12
        ax.loglog(intensity, np.minimum(bandwidth * intensity, fp32_peak), label="A40 nominal FP32 roofline", color="black")
        for B in (8, 32, 64):
            runs = [json.loads((args.roofline / f"float_B{B}_r{r}.json").read_text()) for r in (1, 2, 3)]
            for kernel, ai_field, perf_field in (
                ("GEMV", "gemv_arithmetic_intensity", "gemv_flops_per_second"),
                ("Similarity", "similarity_arithmetic_intensity", "similarity_flops_per_second")):
                ai = float(np.median([r[ai_field] for r in runs]))
                perf = float(np.median([r[perf_field] for r in runs]))
                ax.scatter(ai, perf, marker="o" if kernel == "GEMV" else "s", label=f"{kernel}, B={B}")
                roofline_rows.append(dict(kernel=kernel, B=B, nominal_intensity=ai, measured_flops_per_second=perf))
        ax.set(xlabel="Nominal arithmetic intensity (FLOP/byte)", ylabel="Measured FLOP/s", xscale="log", yscale="log")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        save(fig, args.out, "gpu_roofline")
        (args.out / "roofline_summary.json").write_text(json.dumps(roofline_rows, indent=2) + "\n")
    text = f"""# PCA 实验报告（由已完成结果生成）

本报告使用 {summary['factors']} 个 R 矩阵、{summary['seeds']} 个初始向量、两种 C++ 算法和 k=1…{summary['maxsteps']}。
PCA 签名是 R 的第一左奇异向量的近似；QD 是保持质量和行为多样性的顺序档案。
相似度为签名重新中心化后的绝对 Pearson 相关。谱比是第二与第一奇异值之比。

## 算法和验证

生产求解由手写 C++ 完成，不调用 BLAS/LAPACK；精确 SVD 仅用于 Python 验证。
两种算法使用同一份 FP32 导出 R，内部 double 运算，输出 float32。
图 convergence 同时按步数和实际矩阵向量乘次数对比；阴影为四分位区间。
时间数据包含两次额外残差检查的矩阵向量乘，不作为纯迭代性能。
每个 seed 的档案从空开始，保持同一候选顺序；区别最近 key、动作和最终成员变化。
初始向量 0 来自交接文件，其余来自 NumPy default_rng(seed)。

谱比分位数：`{json.dumps(summary['ratio_quantiles'])}`。

## 性能

已汇总 {len(scaling_table)} 个强扩展配置。每配置保留一次预热及三次计时，汇总取中位数。
date 为按日期归约，transpose 先重分布到因子，factor 是直接按因子读文件的独立输入场景。
没有单 rank 基线的组不计算加速比。各阶段最大时间不可直接相加，通信是计算/转置的子集。
运行时间、问题规模和处理器数相关图使用对数坐标。

## GPU roofline

若提供 roofline 运行，图中峰值采用 NVIDIA A40 的普通 FP32（非 Tensor）37.4 TFLOP/s 和 696 GB/s，
来源：[NVIDIA A40 datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)。
图只绘制 FP32 kernel；FP64 不套用 FP32 的计算上限。
GEMV 字节数只计 R 的读取，相似度字节数采用最低必要输入输出量，并非 profiler 实测 DRAM 流量；
因此图中的强度是名义估计，不能据此直接宣布某 kernel 已经达到计算或带宽极限。
排名是排序型工作负载，不用矩阵乘的 FLOP 模型解释；其独立耗时保存在原始 JSON 中。

## 范围与限制

这些是指定运行的证据，不能自动覆盖其他规模、GPU、弱扩展或档案实验；验证范围和实验入口见 ../../EXPERIMENTS.md。
真实交接因子的基因池来自随机提案，而非完整 GP 进化筛选；本实验评价其谱和准入行为，不能据此声称投资有效。
重根或很小谱间隙时，单一主向量方向可能不稳定，需结合残差解释。
AI 使用说明：代码实现、测试编排及报告初稿使用 OpenAI Codex 协助；实验结论须以运行输出和复核为准。
"""
    (args.out / "report.md").write_text(text)
    print(f"Report and figures written to {args.out}")


if __name__ == "__main__":
    main()
