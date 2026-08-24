#!/usr/bin/env python3
"""
Multi-factor grouped bar chart: SIFT1B vs SPACEV1B, three hardware configs each.
Two panels side by side: QPS and QPS/W.
Visual style aligned with plot_overall_try.py (colors, bar edges, fonts, grid).
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
})

# Reuse pastel palette from plot_overall_try: map hardware to three hues
BAR_COLORS = {
    "PIMCQG": "#FBB4AE",
    "A100 *4": "#B3CDE3",
    "A100 *8": "#CCEBC5",
}
CONFIGS = ["PIMCQG", "A100 *4", "A100 *8"]
DATASETS = ["SIFT1B", "SPACEV1B"]

# Rows: dataset (SIFT1B, SPACEV1B); cols: config (PIMCQG, A100 *4, A100 *8)
QPS = np.array(
    [
        [34197.78, 75740.4, 122810.1],
        [33478.41, 35567.9, 51884.3],
    ],
    dtype=float,
)
QPS_PER_W = np.array(
    [
        [75.99506, 23.66888, 19.18908],
        [74.39646, 11.11497, 8.106922],
    ],
    dtype=float,
)

BAR_WIDTH = 0.22
# Small horizontal nudge for QPS/W ratio labels above A100 bars (data coords)
ANN_X_SHIFT = 0.03


def plot_metric_panel(ax, values, ylabel, show_ylabel=True):
    """Grouped bars: x = datasets, groups = configs."""
    n_ds = len(DATASETS)
    n_cf = len(CONFIGS)
    bar_width = BAR_WIDTH
    x = np.arange(n_ds)

    for i, cfg in enumerate(CONFIGS):
        offset = (i - (n_cf - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            values[:, i],
            width=bar_width,
            color=BAR_COLORS[cfg],
            label=cfg,
            edgecolor="black",
            linewidth=2,
            zorder=3,
            alpha=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def annotate_qps_per_w_vs_pimcqg(ax, values):
    """Above each A100 *4 / A100 *8 bar only: show PIMCQG / GPU (QPS/W ratio)."""
    n_ds, n_cf = values.shape[0], len(CONFIGS)
    x = np.arange(n_ds)
    idx_pim = 0
    y_max = float(values.max())
    ann_fs = max(TICK_SIZE - 10, 9)
    y_pad = y_max * 0.025

    for j in range(n_ds):
        pim = values[j, idx_pim]
        for idx_other in (1, 2):
            other = values[j, idx_other]
            if other <= 0:
                continue
            ratio = pim / other
            x_bar = x[j] + (idx_other - (n_cf - 1) / 2) * BAR_WIDTH + ANN_X_SHIFT
            ax.text(
                x_bar,
                other + y_pad,
                f"{ratio:.2f}×",
                ha="center",
                va="bottom",
                fontsize=ann_fs,
                color="black",
            )

    ax.set_ylim(0, y_max * 1.12)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.65), sharey=False)

    plot_metric_panel(
        axes[0],
        QPS,
        "QPS",
        show_ylabel=True,
    )
    axes[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v >= 1000 else f"{v:.0f}")
    )

    plot_metric_panel(
        axes[1],
        QPS_PER_W,
        "QPS/W",
        show_ylabel=True,
    )
    def _fmt_qps_per_w_y(v, _pos):
        v = float(v)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.1f}"

    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps_per_w_y))
    annotate_qps_per_w_vs_pimcqg(axes[1], QPS_PER_W)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.16, wspace=0.32)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BAR_COLORS[c], edgecolor="black", linewidth=2)
        for c in CONFIGS
    ]
    fig.legend(
        handles=handles,
        labels=CONFIGS,
        loc="upper center",
        ncol=len(CONFIGS),
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 1.07),
    )

    out_dir = Path(__file__).parent
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        path = out_dir / f"sift_space_power_qps{ext}"
        plt.savefig(str(path), bbox_inches="tight", **kw)
        print(f"[plot_sift_space_power_qps] Saved -> {path}")

    plt.show()


if __name__ == "__main__":
    main()
