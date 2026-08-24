#!/usr/bin/env python3
"""
Two standalone figures (same rcParams and fig size as plot_sift_space_power_qps.py):

1) Multi-node query-parallel: grouped QPS and QPS/W speedup vs A100×4 SIFT; y-axis log10;
   nodes 1–32 (64 omitted). Power: A100 3200 W, PIM N×450 W.
2) Grouped bars: UPMEM / PIM-HBM / AiM — speedup vs CPU vs vs GPU (A100×4 class baseline).

Figure 2 uses log-scaled Y so both series remain visible (0.45× vs 137×).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

LABEL_SIZE = 24
TICK_SIZE = 22
LEGEND_SIZE = 24

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": 1.2,
    # Longer y-axis tick marks (major default ~3.5 pt; minor default ~2 pt)
    "ytick.major.size": 5.2,
    "ytick.minor.size": 4.0,
})

# Match plot_sift_space_power_qps: SIFT row, A100 *4 column
A100_SIFT_QPS = 75740.4
A100_TOTAL_W = 3200.0
PIM_W_PER_NODE = 450.0

NODES = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
QP_QPS = np.array(
    [
        33489.12,
        28572.43,
        57236.30,
        114433.72,
        228904.33,
        458946.58,
        913998.21,
    ],
    dtype=float,
)

# Multinode grouped bars: QPS vs QPS/W (same palette as sift_space)
BAR_COLOR_QPS = "#B3CDE3"
BAR_COLOR_QPSW = "#FBB4AE"
BAR_WIDTH_MULTINODE = 0.32
# Short legend labels for PIM arch figure (CPU / GPU baselines)
LEGEND_VS_CPU = "vs CPU"
LEGEND_VS_GPU = "vs GPU"

FIGSIZE = (6.0, 3.65)
# Same margins so axes line up; avoid per-figure drift when not using tight bbox.
# left large enough for y-axis label "Speedup" + log tick labels (no tight crop).
SUBPLOT_ADJUST = dict(left=0.22, right=0.98, top=0.90, bottom=0.22)
# Y-axis title for both figures
Y_AXIS_LABEL = "Speedup"

# Shared ncol-2 legend spacing (multinode + pim_arch)
LEGEND_KW_SHARED = dict(
    ncol=2,
    frameon=False,
    fontsize=LEGEND_SIZE,
    columnspacing=0.55,
    handletextpad=0.35,
    borderaxespad=0,
)
OUT_DIR = Path(__file__).parent


def save_same_canvas(fig, base: Path, tag: str):
    """Save full figure at FIGSIZE (no tight crop) so both PDF/PNG heights match."""
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        path = base.with_suffix(ext)
        fig.savefig(
            str(path),
            facecolor="white",
            edgecolor="none",
            **kw,
        )
        print(f"[{tag}] Saved -> {path}")

# Figure 2: architectures and two speedup series (vs CPU, vs GPU)
ARCH_LABELS = ["UPMEM", "PIM-HBM", "AiM"]
SPEEDUP_VS_CPU = np.array([20.53064656, 99.56546, 137.3381], dtype=float)
SPEEDUP_VS_GPU = np.array([1.306795317, 6.337437, 8.741702], dtype=float)

# Same pair as plot_sift_space_power_qps: PIMCQG / A100 *4
COLOR_VS_CPU = "#B3CDE3"
COLOR_VS_GPU = "#FBB4AE"
BAR_WIDTH = 0.32


def y_tick_formatter_log_pow10(v, pos):
    """Label powers of ten as $10^n$ (e.g. 1 -> $10^0$, 10 -> $10^1$)."""
    if v <= 0 or not np.isfinite(v):
        return ""
    logv = np.log10(float(v))
    if abs(logv - round(logv)) > 1e-9:
        return ""
    e = int(round(logv))
    return rf"$10^{{{e}}}$"


def style_axes_grid(ax):
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def plot_multinode_qp_speedup():
    mask = NODES < 64
    nodes_plot = NODES[mask]
    qp_plot = QP_QPS[mask]

    speedup_qps = qp_plot / A100_SIFT_QPS
    pim_total_w = nodes_plot * PIM_W_PER_NODE
    qps_w_pim = qp_plot / pim_total_w
    qps_w_a100 = A100_SIFT_QPS / A100_TOTAL_W
    speedup_qpsw = qps_w_pim / qps_w_a100

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    x = np.arange(len(nodes_plot))
    w = BAR_WIDTH_MULTINODE
    off = w / 2

    ax.bar(
        x - off,
        speedup_qps,
        width=w,
        color=BAR_COLOR_QPS,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label="QPS",
    )
    ax.bar(
        x + off,
        speedup_qpsw,
        width=w,
        color=BAR_COLOR_QPSW,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label="QPS/W",
    )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}" for n in nodes_plot])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.65, zorder=2)
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_tick_formatter_log_pow10))
    style_axes_grid(ax)
    y_all = np.concatenate([speedup_qps, speedup_qpsw])
    y_min = max(y_all.min() * 0.85, 0.25)
    y_top_data = y_all.max() * 1.15
    # Include 10 so the 10^1 major tick label is visible
    ax.set_ylim(bottom=y_min, top=max(y_top_data, 10.5))

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        bbox_transform=ax.transAxes,
        **LEGEND_KW_SHARED,
    )
    fig.subplots_adjust(**SUBPLOT_ADJUST)

    base = OUT_DIR / "multinode_qp_speedup_vs_a100_sift"
    save_same_canvas(fig, base, "plot_multinode_qp")
    plt.close(fig)


def plot_pim_arch_speedup_bars():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    n = len(ARCH_LABELS)
    x = np.arange(n)

    ax.bar(
        x - BAR_WIDTH / 2,
        SPEEDUP_VS_CPU,
        width=BAR_WIDTH,
        color=COLOR_VS_CPU,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label=LEGEND_VS_CPU,
    )
    ax.bar(
        x + BAR_WIDTH / 2,
        SPEEDUP_VS_GPU,
        width=BAR_WIDTH,
        color=COLOR_VS_GPU,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        label=LEGEND_VS_GPU,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_LABELS)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_tick_formatter_log_pow10))
    # Wider log range: headroom inside axes for legend
    ax.set_ylim(0.35, 400)
    style_axes_grid(ax)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        bbox_transform=ax.transAxes,
        **LEGEND_KW_SHARED,
    )
    fig.subplots_adjust(**SUBPLOT_ADJUST)

    base = OUT_DIR / "pim_arch_speedup_cpu_gpu"
    save_same_canvas(fig, base, "plot_pim_arch")
    plt.close(fig)


def main():
    plot_multinode_qp_speedup()
    plot_pim_arch_speedup_bars()


if __name__ == "__main__":
    main()
