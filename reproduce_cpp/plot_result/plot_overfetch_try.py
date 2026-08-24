#!/usr/bin/env python3
"""
Dual-axis bar+line chart: Normalized QPS (bars, left axis) and
Normalized Recall (line, right axis) vs EF, for SIFT1B and SPACEV1B.
Left subplot: SIFT1B   |   Right subplot: SPACEV1B
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np

# ── Serif font for subplot titles (Times New Roman if available, else fallback) ─
TITLE_FONT = "Times New Roman"
if not any(f.name == "Times New Roman" for f in fm.fontManager.ttflist):
    for fallback in ("Liberation Serif", "DejaVu Serif", "serif"):
        if any(fallback in f.name for f in fm.fontManager.ttflist) or fallback == "serif":
            TITLE_FONT = fallback
            break

# ── Global font settings (paper-ready) ──────────────────────────────────────
LABEL_SIZE  = 20
TICK_SIZE   = 17
LEGEND_SIZE = 16
LINE_WIDTH  = 2.0

plt.rcParams.update({
    "font.size":        TICK_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.2,
})

# ── EF values (x-axis, shared) ───────────────────────────────────────────────
EF_VALUES = [30, 50, 70, 80, 100, 150, 200, 300]

# ── Raw data: fill in real measurements ──────────────────────────────────────
# Format: one value per EF entry above

# SIFT1B
SIFT1B_QPS    = [34197.77714, 28361.55308, 24273.80856, 22776.62208, 16242.10242, 14868.19346, 13374.10951, 7332.345911]   # replace with real values
SIFT1B_RECALL = [61.83, 68.18, 71.62, 72.72, 74.58, 77.11, 78.51, 80.14]  # replace with real values
SIFT1B_BASE_QPS    = 3422.515744   # normalization baseline for QPS
SIFT1B_BASE_RECALL = 76.364  # normalization baseline for Recall

# SPACEV1B
SPACEV1B_QPS    = [33478.40643, 27219.27851, 22317.57449, 21659.71149, 18335.83923, 14384.20555, 12577.5089
 , 6933.89914]   # replace with real values
SPACEV1B_RECALL = [67.48, 71.86, 73.95, 74.69, 75.87, 78.03, 79.12, 80.27]  # replace with real values
SPACEV1B_BASE_QPS    = 3123.630791   # normalization baseline for QPS
SPACEV1B_BASE_RECALL = 76.034  # normalization baseline for Recall

# ── Colors ───────────────────────────────────────────────────────────────────
BAR_COLOR    = "#B3CDE3"   # light blue bars for Normalized QPS
LINE_COLOR   = "#E41A1C"   # red line for Normalized Recall
BASELINE_COLOR = "#888888" # dashed gray for baseline = 1.0


def _fmt_norm_axis_y(v, _pos):
    """Hide trailing .00 for whole numbers; keep two decimals otherwise."""
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def plot_one(ax_left, ax_right, ef_vals, qps_raw, recall_raw, base_qps, base_recall, subplot_label, title):
    """Draw one subplot with dual y-axes. Title (subplot_label + title) is placed below in serif font."""
    x = np.arange(len(ef_vals))
    bar_width = 0.5

    norm_qps    = [v / base_qps    for v in qps_raw]
    norm_recall = [v / base_recall for v in recall_raw]

    # ── Left axis: Normalized QPS bars ──────────────────────────────────────
    bars = ax_left.bar(
        x,
        norm_qps,
        width=bar_width,
        color=BAR_COLOR,
        label="Norm. QPS",
        linewidth=LINE_WIDTH,
        edgecolor="black",
        alpha=1.0,
        zorder=2,
    )

    # baseline = 1.0 reference line for QPS
    ax_left.axhline(1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1.2, zorder=1)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(e) for e in ef_vals])
    ax_left.set_xlabel("EF")
    ax_left.set_ylabel("Normalized QPS", color="black")
    ax_left.tick_params(axis="y", labelcolor="black")
    ax_left.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_norm_axis_y))
    # Ensure baseline 1.0 is shown on the left y-axis
    yticks = list(ax_left.get_yticks())
    if not any(abs(t - 1.0) < 1e-9 for t in yticks):
        yticks.append(1.0)
        yticks.sort()
        ax_left.set_yticks(yticks)
    ax_left.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)

    # ── Right axis: Normalized Recall line ───────────────────────────────────
    ax_right.plot(
        x,
        norm_recall,
        color=LINE_COLOR,
        marker="o",
        markersize=6,
        linewidth=LINE_WIDTH,
        label="Norm. Recall",
        zorder=3,
    )

    # baseline = 1.0 reference line for Recall
    ax_right.axhline(1.0, color=BASELINE_COLOR, linestyle=":", linewidth=1.2, zorder=1)

    ax_right.set_ylabel("Normalized Recall", color="black")
    ax_right.tick_params(axis="y", labelcolor="black")
    ax_right.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_norm_axis_y))
    ax_right.set_ylim(bottom=0)

    # ── Combined legend (both axes), moved down a bit ─────────────────────────
    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    # ── Title below subplot: (a) SIFT1B / (b) SPACEV1B in serif font ───────────
    ax_left.text(
        0.5, -0.38,
        f"{subplot_label} {title}",
        transform=ax_left.transAxes,
        ha="center",
        fontsize=24,
        fontfamily=TITLE_FONT,
    )


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # SIFT1B – left subplot
    ax_l0 = axes[0]
    ax_r0 = ax_l0.twinx()
    plot_one(
        ax_l0, ax_r0,
        EF_VALUES,
        SIFT1B_QPS, SIFT1B_RECALL,
        SIFT1B_BASE_QPS, SIFT1B_BASE_RECALL,
        subplot_label="(a)",
        title="SIFT1B",
    )

    # SPACEV1B – right subplot
    ax_l1 = axes[1]
    ax_r1 = ax_l1.twinx()
    plot_one(
        ax_l1, ax_r1,
        EF_VALUES,
        SPACEV1B_QPS, SPACEV1B_RECALL,
        SPACEV1B_BASE_QPS, SPACEV1B_BASE_RECALL,
        subplot_label="(b)",
        title="SPACEV1B",
    )

    # w_pad: horizontal gap between columns (fraction of fontsize); lower = tighter
    plt.tight_layout(pad=1.5, w_pad=0.35, rect=(0, 0.12, 1, 1))

    from pathlib import Path
    out_dir = Path(__file__).parent
    pdf_path = out_dir / "overfetch_comparison.pdf"
    png_path = out_dir / "overfetch_comparison.png"

    plt.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[plot_overfetch] Figure saved -> {pdf_path}")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_overfetch] Figure saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
