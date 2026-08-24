#!/usr/bin/env python3
"""
Bar chart comparing w/ MultiFree (w/ MF) vs w/o MultiFree (w/o MF)
across three datasets: SIFT1B, SPACEV1B, SSN1B.
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

# Same figure size as plot_async_pqc_try.py (inches)
FIG_WIDTH, FIG_HEIGHT = 8.0, 5.5

# ── Data (replace with real measurements) ───────────────────────────────────
datasets = ["SIFT1B", "SPACEV1B", "SSN1B"]

# Time in seconds for each dataset: [w/ MF, w/o MF]
time_with_mf    = [0.06, 0.0502, 0.079133]
time_without_mf = [0.151371, 0.128, 0.156957]


def main():
    x = np.arange(len(datasets))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bars2 = ax.bar(
        x - bar_width / 2,
        time_without_mf,
        width=bar_width,
        color="#B3CDE3",
        label="w/o MF",
        linewidth=LINE_WIDTH,
        edgecolor="black",
        alpha=1,
    )
    bars1 = ax.bar(
        x + bar_width / 2,
        time_with_mf,
        width=bar_width,
        color="#FBB4AE",
        label="w/ MF",
        linewidth=LINE_WIDTH,
        edgecolor="black",
        alpha=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=XTICK_LABEL_SIZE)
    ax.set_ylabel("Time (s)")
    # Default AutoLocator uses dense ticks (e.g. 0.02 steps). With "%.1f", many
    # distinct positions format to the same string ("0.0", "0.1"), so labels repeat.
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.12, 1.05),
        frameon=False,
    )

    # ── Double-headed arrows + reduction percentage ──────────────────────────
    # Arrow is placed right against the right edge of the red bar (w/o MF)
    for i, (v_mf, v_wo) in enumerate(zip(time_with_mf, time_without_mf)):
        pct = (v_wo - v_mf) / v_wo * 100
        # right edge of the red bar = center - bar_width/2 + bar_width/2 = x[i]
        x_arrow = x[i] + 0.05
        # double-headed arrow from v_mf (bottom) to v_wo (top)
        ax.annotate(
            "",
            xy=(x_arrow, v_wo),
            xytext=(x_arrow, v_mf),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.4,
            ),
        )
        # percentage label beside the midpoint of the arrow
        ax.text(
            x_arrow + 0.05,
            (v_mf + v_wo) / 2,
            f"-{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=TICK_SIZE + 2,
            color="black",
        )

    ax.set_xlim(x[0] - bar_width * 1.2, x[-1] + bar_width * 1.2 + 0.45)

    plt.tight_layout()

    from pathlib import Path
    out_dir = Path(__file__).parent
    pdf_path = out_dir / "multi_free_comparison.pdf"
    png_path = out_dir / "multi_free_comparison.png"

    plt.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[plot_multi_free] Figure saved -> {pdf_path}")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_multi_free] Figure saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
