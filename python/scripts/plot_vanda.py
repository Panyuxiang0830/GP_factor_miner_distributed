"""Figures for the Vanda experiments (X1-X5) from the gpe_bench CSV files.

    python plot_vanda.py <csv_dir> <figure_dir>

Time, problem size and core counts are on logarithmic axes.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

PHASES = ["load", "compute", "metrics", "halo", "scan", "wait"]


def read(csv_dir, name):
    path = Path(csv_dir) / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / f"{name}.pdf")
    fig.savefig(Path(out) / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_serial(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (labels, col, xlabel) in zip(
        axes,
        [
            (["T_real", "T_synthetic"], "T", "trading days T"),
            (["N_real", "N_synthetic"], "N", "assets N"),
            (["batch_real"], "exprs", "expressions in the batch"),
        ],
    ):
        for label in labels:
            d = df[df.label == label].sort_values(col)
            if d.empty:
                continue
            core = d.total - d.load
            ax.loglog(d[col], core, "o-", label=f"{label} (slope {slope(d[col], core):.2f})")
            ax.loglog(d[col], d.compute, "s--", alpha=0.6, label=f"{label}: operators")
            ax.loglog(d[col], d.metrics, "^--", alpha=0.6, label=f"{label}: RankIC")
        ax.set_xlabel(xlabel)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("time excluding load (s)")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "x1_serial")


def plot_strong(df, out):
    d = df[df.label == "strong"].sort_values("P")
    base = d[d.P == 1].total.iloc[0]
    p = d.P.to_numpy()
    speedup = base / d.total.to_numpy()
    eff = speedup / p
    kf = np.where(p > 1, (1 / speedup - 1 / p) / (1 - 1 / np.maximum(p, 2)), np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].loglog(p, d.total, "o-", label="measured")
    axes[0].loglog(p, base / p, "k--", label="ideal")
    axes[0].set_ylabel("time (s)")
    axes[1].loglog(p, speedup, "o-", label="speedup")
    axes[1].loglog(p, p, "k--", label="ideal")
    axes[1].set_ylabel("speedup")
    axes[2].semilogx(p, eff, "o-", label="efficiency")
    axes[2].semilogx(p, kf, "s-", label="Karp-Flatt e")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("MPI ranks (1 core each)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "x2_strong")
    breakdown(d, "P", out, "x2_strong_breakdown")


def plot_weak(df, out):
    d = df[df.label == "weak"].sort_values("P")
    base = d[d.P == 1].total.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogx(d.P, base / d.total, "o-", label="all phases")
    core0 = (d.total - d.load).iloc[0]
    ax.semilogx(d.P, core0 / (d.total - d.load), "s-", label="excluding load")
    ax.set_xlabel("MPI ranks (64 synthetic days each)")
    ax.set_ylabel("weak-scaling efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "x3_weak")
    breakdown(d, "P", out, "x3_weak_breakdown")


def breakdown(d, col, out, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    for ph in PHASES:
        ax.bar(x, d[ph], bottom=bottom, label=ph)
        bottom += d[ph].to_numpy()
    ax.set_xticks(x, d[col].astype(str))
    ax.set_xlabel(col)
    ax.set_ylabel("time (s, max over ranks)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    save(fig, out, name)


def plot_comm(df, strong, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    scan = df[df.label == "h0_scan"]
    for p, d in scan.groupby("P"):
        d = d.sort_values("H0")
        axes[0].semilogy(d.H0, d.total, "o-", label=f"{p} ranks")
        axes[1].semilogy(d.H0, d.halo + d.wait, "o--", label=f"{p} ranks: halo + wait")
    hyb = df[df.label == "h0_hybrid"]
    for (p, t), d in hyb.groupby(["P", "threads"]):
        axes[0].semilogy(d.sort_values("H0").H0, d.sort_values("H0").total, "s:", label=f"{p}x{t}")
    axes[0].set_xlabel("H0 (redundant-computation budget, days)")
    axes[0].set_ylabel("time (s)")
    axes[1].set_xlabel("H0")
    axes[1].set_ylabel("halo + wait (s)")
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    lo, hi = axes[0].get_ylim()
    ticks = [1, 1.25, 1.5, 2, 2.5, 3, 3.5, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    axes[0].set_yticks([y for y in ticks if lo <= y <= hi])
    # Same rank count packed on as few nodes as possible (the strong-scaling
    # runs) and spread over four nodes.
    spread = df[df.label == "spread4"]
    packed = strong[strong.label == "strong"] if strong is not None else spread.iloc[:0]
    rows = []
    for p in sorted(spread.P.unique()):
        for kind, d in (("packed", packed), ("4 nodes", spread)):
            r = d[d.P == p]
            if not r.empty:
                rows.append((f"{p} {kind}", r.halo.iloc[0], r.total.iloc[0]))
    if rows:
        x = np.arange(len(rows))
        axes[2].bar(x - 0.2, [r[1] for r in rows], 0.4, label="halo")
        axes[2].bar(x + 0.2, [r[2] for r in rows], 0.4, label="total")
        axes[2].set_xticks(x, [r[0] for r in rows], rotation=30, fontsize=7)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("time (s)")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "4-node runs pending", ha="center", va="center", transform=axes[2].transAxes)
    save(fig, out, "x4_comm")


def plot_hybrid(omp, hyb, strong, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if omp is not None:
        base = strong[(strong.label == "strong") & (strong.P == 1)].total.iloc[0]
        for m, d in omp.groupby("mode"):
            d = d.sort_values("threads")
            t = np.r_[1, d.threads]
            axes[0].loglog(t, base / np.r_[base, d.total], "o-", label=f"threads: {m}")
        axes[0].loglog([1, 72], [1, 72], "k--", label="ideal")
        axes[0].set_xlabel("OpenMP threads (1 rank)")
        axes[0].set_ylabel("speedup")
    if hyb is not None:
        pure = strong[(strong.label == "strong") & (strong.P == 288)]
        for m, d in hyb.groupby("mode"):
            d = pd.concat([pure.assign(mode=m), d]).sort_values("threads")
            axes[1].loglog(d.threads, d.total, "o-", label=f"threads: {m}")
        axes[1].set_xlabel("threads per rank (ranks x threads = 288)")
        axes[1].set_ylabel("time (s)")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "x5_hybrid")


def main(csv_dir, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    if (d := read(csv_dir, "x1_serial")) is not None:
        # A single full-size run ("full") is the end point of all three real
        # sweeps instead of being repeated in each.
        full = d[d.label == "full"]
        d = pd.concat([d] + [full.assign(label=l) for l in ("T_real", "N_real", "batch_real")])
        plot_serial(d, out)
    strong = read(csv_dir, "x2_strong")
    if strong is not None:
        plot_strong(strong, out)
    if (d := read(csv_dir, "x3_weak")) is not None:
        plot_weak(d, out)
    if (d := read(csv_dir, "x4_comm")) is not None:
        # The H0 = 0 point of the hybrid splits comes from the X5 runs.
        if (h := read(csv_dir, "x5_hybrid")) is not None:
            h0 = h[(h["mode"] == "op") & (h.H0 == 0)].assign(label="h0_hybrid")
            h0 = h0[h0.P.isin(d[d.label == "h0_hybrid"].P)]
            d = pd.concat([d, h0])
        plot_comm(d, strong, out)
    if strong is not None:
        plot_hybrid(read(csv_dir, "x5_omp"), read(csv_dir, "x5_hybrid"), strong, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
