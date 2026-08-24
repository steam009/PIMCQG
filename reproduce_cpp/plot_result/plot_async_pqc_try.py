#!/usr/bin/env python3
"""
Bar chart comparing Per-Query, Batch Sync, PIMCQG_1, and PIMCQG
across three datasets: SIFT1B, SPACEV1B, SSN1B.
Broken y-axis to handle the large gap between Per-Query and the rest.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Global font settings (paper-ready) ──────────────────────────────────────
LABEL_SIZE  = 24
TICK_SIZE   = 18
# X-axis category labels (dataset names): larger than y tick numerals
XTICK_LABEL_SIZE = TICK_SIZE + 4
LEGEND_SIZE = 20
LINE_WIDTH  = 2.0

plt.rcParams.update({
    "font.size":        TICK_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.2,
})

# Same figure size as plot_Multifree.py (inches)
FIG_WIDTH, FIG_HEIGHT = 8.0, 5.5

# ── Data (replace with real measurements) ───────────────────────────────────
datasets = ["SIFT1B", "SPACEV1B", "SSN1B"]

# QPS for each dataset: [Per-Query, Batch Sync, PIMCQG_1, PIMCQG]
qps_per_query  = [220.89, 202.29389, 225.31]
qps_batch_sync = [22857, 24488.736, 10538.018]
qps_pimcqg_1   = [18483, 14443.909, 9482.8676]
qps_pimcqg     = [34197.77714, 33478.40643, 15807.0278]

# ── Broken-axis ranges (adjust to match your data) ──────────────────────────
BREAK_LOW_MAX  = 600      # top of the lower panel
BREAK_HIGH_MIN = 8000     # bottom of the upper panel


def draw_bars(ax, x, offsets, colors, labels, qps_list, bar_width):
    bars_list = []
    for offset, color, label, qps_data in zip(offsets, colors, labels, qps_list):
        bars = ax.bar(
            x + offset,
            qps_data,
            width=bar_width,
            color=color,
            label=label,
            linewidth=LINE_WIDTH,
            edgecolor="black",
            alpha=1,
        )
        bars_list.append(bars)
    return bars_list


def add_break_marks(ax_top, ax_bot, d=0.012):
    """Draw diagonal break marks on both axes to indicate the axis break."""
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def main():
    x = np.arange(len(datasets))
    bar_width = 0.2
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    colors = ["#DECBE4", "#CCEBC5", "#B3CDE3", "#FBB4AE"]
    labels = ["Per-Query", "Batch Sync", "PIMCQG_1", "PIMCQG"]
    qps    = [qps_per_query, qps_batch_sync, qps_pimcqg_1, qps_pimcqg]

    # upper panel ~80% height, lower panel ~20%
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    draw_bars(ax_top, x, offsets, colors, labels, qps, bar_width)
    draw_bars(ax_bot, x, offsets, colors, labels, qps, bar_width)

    # ── Set y ranges for each panel ─────────────────────────────────────────
    all_high = [v for row in [qps_batch_sync, qps_pimcqg_1, qps_pimcqg] for v in row]
    ax_top.set_ylim(BREAK_HIGH_MIN, max(all_high) * 1.2)
    ax_bot.set_ylim(0, BREAK_LOW_MAX)

    # ── Formatting ───────────────────────────────────────────────────────────
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}K" if v >= 1000 else f"{v:.0f}"
    ))
    ax_bot.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}"
    ))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(200))

    for ax in (ax_top, ax_bot):
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_xlim(x[0] - bar_width * 2.5, x[-1] + bar_width * 2.5)

    # hide inner spines to create visual break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(datasets, fontsize=XTICK_LABEL_SIZE)

    # shared y-label
    fig.text(0.01, 0.55, "QPS", va="center", rotation="vertical", fontsize=LABEL_SIZE)

    ax_top.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.07))

    add_break_marks(ax_top, ax_bot)

    plt.tight_layout(rect=[0.05, 0, 1, 1])

    from pathlib import Path
    out_dir = Path(__file__).parent
    pdf_path = out_dir / "async_pqc_comparison.pdf"
    png_path = out_dir / "async_pqc_comparison.png"

    plt.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[plot_async_pqc] Figure saved -> {pdf_path}")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_async_pqc] Figure saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
