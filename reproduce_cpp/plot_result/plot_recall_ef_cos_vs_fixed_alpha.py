#!/usr/bin/env python3
"""
Grouped bar chart: Recall vs EF for cos(theta) vs fixed alpha.
Style aligned with plot_overfetch_try.py (colors, fonts, grid).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
BAR_EDGE_WIDTH = 0.8

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "axes.linewidth": 1.2,
})

# ── Colors (same as plot_overfetch_try.py) ───────────────────────────────────
BAR_COLOR_COS = "#B3CDE3"
BAR_COLOR_FIX = "#FBB4AE"

# ── Data (from measurements table) ───────────────────────────────────────────
EF_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RECALL_OR = [56.331, 75.915, 84.57, 89.307, 92.208, 94.008, 95.326, 96.298, 96.994, 97.559]
RECALL_FI = [56.266, 75.874, 84.829, 89.611, 92.345, 94.165, 95.406, 96.31, 96.954, 97.479]
DIFFERENCE = [0.065, 0.041, -0.259, -0.304, -0.137, -0.157, -0.08, -0.012, 0.04, 0.08]


def main():
    x = np.arange(len(EF_VALUES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.bar(
        x - width / 2,
        RECALL_OR,
        width,
        label=r"$\cos(\theta)$",
        color=BAR_COLOR_COS,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        RECALL_FI,
        width,
        label=r"fixed $\alpha$",
        color=BAR_COLOR_FIX,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        zorder=2,
    )

    # Difference labels above the taller bar in each group
    y_pad = 0.35
    ann_size = TICK_SIZE - 2
    for i in range(len(EF_VALUES)):
        ymax = max(RECALL_OR[i], RECALL_FI[i])
        d = -DIFFERENCE[i]
        s = f"{d:.3f}"
        ax.text(
            x[i],
            ymax + y_pad,
            s,
            ha="center",
            va="bottom",
            fontsize=ann_size,
            color="black",
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in EF_VALUES])
    ax.set_xlabel("EF")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(bottom=50, top=105)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.06),
        frameon=False,
    )

    plt.tight_layout(pad=1.2)

    out_dir = Path(__file__).parent
    for ext in ("pdf", "png"):
        p = out_dir / f"recall_ef_cos_vs_fixed_alpha.{ext}"
        dpi = 200 if ext == "png" else None
        kwargs = {"bbox_inches": "tight"}
        if dpi:
            kwargs["dpi"] = dpi
        plt.savefig(str(p), **kwargs)
        print(f"[plot_recall_ef] Saved -> {p}")

    plt.show()


if __name__ == "__main__":
    main()
