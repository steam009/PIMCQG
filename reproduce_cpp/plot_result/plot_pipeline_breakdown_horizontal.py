#!/usr/bin/env python3
"""
Horizontal stacked breakdown: Total Time (1.0), Query Dispatch, DPU search,
Post-processor, Others — for SIFT1B, SPACEV1B, SSN1B.

Numbers in DATASETS are used as-is (already ratio / proportion values); no division by sum.
Total row is fixed at 1.0; other rows use the given segment widths directly.

Colors aligned with plot_sift_space_power_qps.py (pastel trio + extensions).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

LABEL_SIZE = 28
TICK_SIZE = 24
# Bottom dataset names (SIFT1B / SPACEV1B / SSN1B) only — larger than axis ticks & legend
DATASET_LABEL_FONTSIZE = 32

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": TICK_SIZE,
    "axes.linewidth": 1.2,
})

# From plot_sift_space_power_qps / plot_overall_try
PASTEL_PINK = "#FBB4AE"
PASTEL_BLUE = "#B3CDE3"
PASTEL_GREEN = "#CCEBC5"

# Extra harmonizing pastels (same family as ggplot2 Pastel1-adjacent)
PASTEL_LAVENDER = "#DECBE4"
PASTEL_PEACH = "#FED9A6"
PASTEL_SLATE = "#C8C8DC"
# Light gray for Total Time bar (not in legend)
TOTAL_LIGHT_GRAY = "#DCDCDC"

COLORS = {
    "total": TOTAL_LIGHT_GRAY,
    "prepare": PASTEL_PINK,
    "cpu_dpu": PASTEL_BLUE,
    "dpu_search": PASTEL_GREEN,
    "dpu_cpu": PASTEL_LAVENDER,
    "postprocess": PASTEL_PEACH,
    "other": PASTEL_SLATE,
}

ROW_LABELS = [
    "Total Time",
    "Query Dispatch",
    "DPU search",
    "Post-processor",
    "Others",
]

# Ratio values as in source table (plotted directly, not renormalized)
DATASETS: dict[str, dict[str, float]] = {
    "SIFT1B": {
        "prepare": 0.233266094,
        "cpu_dpu": 0.432651017,
        "dpu_search": 0.501174,
        "dpu_cpu": 0.434852443,
        "postprocess": 0.555175004,
        "other": 0.334078,
    },
    "SPACEV1B": {
        "prepare": 0.171387122,
        "cpu_dpu": 0.317297177,
        "dpu_search": 0.300636,
        "dpu_cpu": 0.543053231,
        "postprocess": 0.433143622,
        "other": 0.510356,
    },
    "SSN1B": {
        "prepare": 0.29958006,
        "cpu_dpu": 0.443634773,
        "dpu_search": 0.416953,
        "dpu_cpu": 0.444915143,
        "postprocess": 0.549652509,
        "other": 0.256785,
    },
}

BAR_HEIGHT = 0.62
EDGE_LW = 1.5


def _max_extent_for_dataset(d: dict[str, float]) -> float:
    qd = d["prepare"] + d["cpu_dpu"]
    pp = d["dpu_cpu"] + d["postprocess"]
    return max(1.0, qd, d["dpu_search"], pp, d["other"])


def plot_one_dataset(ax, name: str, d: dict[str, float], x_max: float) -> None:
    y = np.arange(len(ROW_LABELS), dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels(ROW_LABELS)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.55, len(ROW_LABELS) - 0.45)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(name, fontweight="normal", fontsize=DATASET_LABEL_FONTSIZE)

    # Row 0: Total = 1
    ax.barh(
        y[0], 1.0, height=BAR_HEIGHT, left=0.0,
        color=COLORS["total"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )

    # Row 1: Query Dispatch — Prepare + CPU-DPU (values as given)
    left = 0.0
    ax.barh(
        y[1], d["prepare"], height=BAR_HEIGHT, left=left,
        color=COLORS["prepare"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )
    left += d["prepare"]
    ax.barh(
        y[1], d["cpu_dpu"], height=BAR_HEIGHT, left=left,
        color=COLORS["cpu_dpu"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )

    # Row 2: DPU search
    ax.barh(
        y[2], d["dpu_search"], height=BAR_HEIGHT, left=0.0,
        color=COLORS["dpu_search"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )

    # Row 3: Post-processor — DPU-CPU + Postprocessor (values as given)
    left = 0.0
    ax.barh(
        y[3], d["dpu_cpu"], height=BAR_HEIGHT, left=left,
        color=COLORS["dpu_cpu"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )
    left += d["dpu_cpu"]
    ax.barh(
        y[3], d["postprocess"], height=BAR_HEIGHT, left=left,
        color=COLORS["postprocess"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )

    # Row 4: Others
    ax.barh(
        y[4], d["other"], height=BAR_HEIGHT, left=0.0,
        color=COLORS["other"], edgecolor="black", linewidth=EDGE_LW, zorder=3,
    )


def main() -> None:
    x_max = max(_max_extent_for_dataset(d) for d in DATASETS.values()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True, sharex=True)

    for ax, ds_name in zip(axes, ["SIFT1B", "SPACEV1B", "SSN1B"], strict=True):
        plot_one_dataset(ax, ds_name, DATASETS[ds_name], x_max)

    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.24, wspace=0.09)

    # Single-row legend: left-to-right order below
    legend_entries = [
        ("Prepare", COLORS["prepare"]),
        ("CPU–DPU", COLORS["cpu_dpu"]),
        ("DPU search", COLORS["dpu_search"]),
        ("DPU–CPU", COLORS["dpu_cpu"]),
        ("Postprocessor", COLORS["postprocess"]),
        ("Others", COLORS["other"]),
    ]
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", linewidth=EDGE_LW, label=lab)
        for lab, c in legend_entries
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.56, 1.02),
        fontsize=TICK_SIZE,
        columnspacing=0.5,
        handletextpad=0.35,
        handlelength=1.0,
        handleheight=0.9,
        borderpad=0.2,
    )

    out_dir = Path(__file__).parent
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        path = out_dir / f"pipeline_breakdown_horizontal{ext}"
        plt.savefig(str(path), bbox_inches="tight", **kw)
        print(f"[plot_pipeline_breakdown_horizontal] Saved -> {path}")

    plt.show()


if __name__ == "__main__":
    main()
