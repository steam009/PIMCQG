#!/usr/bin/env python3
"""
Recall-QPS tradeoff curve plot, two subplots side by side:
  Left  subplot: SIFT1B
  Right subplot: SPACEV1B

- Competitor methods stop at ~60% recall; their curves end naturally.
- Our method reaches 95% recall at ~30x higher QPS.
- Log-scale Y-axis handles the large QPS gap.
- Shaded exclusive recall zone (60%->95%) highlights our recall advantage.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np

# ── Serif font for subplot titles ────────────────────────────────────────────
TITLE_FONT = "Times New Roman"
if not any(f.name == "Times New Roman" for f in fm.fontManager.ttflist):
    for fallback in ("Liberation Serif", "DejaVu Serif", "serif"):
        if any(fallback in f.name for f in fm.fontManager.ttflist) or fallback == "serif":
            TITLE_FONT = fallback
            break

# ── Global font settings (paper-ready) ──────────────────────────────────────
# Axis titles (x/y labels): LABEL_SIZE; tick numerals: one step smaller.
LABEL_SIZE  = 24
TICK_SIZE   = LABEL_SIZE - 1
LEGEND_SIZE = LABEL_SIZE  # match axis label size (shared fig legend)
LINE_WIDTH  = 2
PIN_LABEL_SIZE = 22

# Distance from each axes bottom to subplot caption, in figure coordinates.
# Adjust only this to move (a)/(b) up/down without changing subplot/axes height
# (unlike ax.text with transAxes y<0, which tight_layout treats as extra margin).
SUBPLOT_CAPTION_PAD = 0.17

# "PIMCQG reaches …" label: offset left of our_max as a fraction of visible x-span.
# A fixed recall delta (e.g. -20) is uneven: 20 points is a larger fraction of a
# narrow x-range on the right subplot than on the wide-range left subplot.
REACH_RECALL_LABEL_X_FRAC = 0.56
# SPACEV1B (right) subplot only: extra recall-axis shift for PIMCQG end label (positive = right).
OUR_METHOD_LABEL_X_SHIFT_SPACEV1B = 4.0
# Y-axis tick mark length in points (matplotlib default ~3.5).
Y_TICK_LENGTH_MAJOR = 8.0
Y_TICK_LENGTH_MINOR = 5.0
# Log y-axis: pad limits in multiplicative space so max QPS is not lost in decade-wide autoscale.
Y_LIM_PAD_LO = 0.88
Y_LIM_PAD_HI = 1.12

plt.rcParams.update({
    "font.size":        TICK_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.2,
})

# ── Visual style per method ──────────────────────────────────────────────────
STYLES = {
    "UpANNS": dict(color="#7FC97F", marker="s",  linestyle="-",  zorder=3),
    "PIMANN": dict(color="#BEAED4", marker="^",  linestyle="-",  zorder=3),
    "PIMCQG": dict(color="#E31A1C", marker="o",  linestyle="-",  zorder=5),
}

OUR_METHOD = "PIMCQG"

# ── Data: (recall%, qps) operating points per method ────────────────────────
# Competitors: QPS ~1000 level, max recall ~60%
# Ours:        QPS ~30000 level, max recall ~95%
# Replace with real measurements.
#9.288
SIFT1B_DATA = {
    "UpANNS": [(56.64, 2349.864), (57.48, 2015.5), (58.54, 1634.69), (59.29, 1383.92),  (60.03, 1068.12), (61.43, 705.88)],
    "PIMANN": [(56.64, 1322), (57.48, 1216), (58.54, 969), (59.29, 855),  (60.03, 643), (61.43, 424)],
    "PIMCQG": [(60.83, 34197.78), (68.18, 28361), (72.72, 22776), (74.58, 16242.1), (77.11, 14868.19), (81.75, 12804.43),
               (84.33, 8981.955), (89.06, 4262), (92.15, 2529.688),  (95.36, 1524)],
}
#7.614

SPACEV1B_DATA = {
    "UpANNS": [(60.4450, 2291.8), (61.5750, 1933.95), (63.3540, 1385.74), (64.3080, 1088.80),  (65.6100, 662.42), (66.7110, 258.876)],
    "PIMANN": [(60.4450, 3243.00), (61.5750, 2618.00), (63.3540, 1781.00), (64.3080, 1318.00),  (65.6100, 698.00), (66.7110, 207.00)],
    "PIMCQG": [(67, 33478.40643), (70, 30380.66979), (72, 27219.27851), (74, 19317.57449),
               (76, 18161.56528), (78, 14566.92531), (80, 11384.20555),  (82, 7141.207864), (84, 4911.615479), (86, 2366.264638), (88, 1367.26334), (90, 612.69)],
}


def place_subplot_captions(fig, axes, captions):
    """Draw (a)/(b) labels below each axes using figure coords (does not shrink axes)."""
    for ax, cap in zip(np.atleast_1d(axes), captions):
        pos = ax.get_position()
        fig.text(
            pos.x0 + 0.5 * pos.width,
            pos.y0 - SUBPLOT_CAPTION_PAD,
            cap,
            ha="center",
            va="top",
            fontsize=LABEL_SIZE,
            fontfamily=TITLE_FONT,
            transform=fig.transFigure,
        )


def plot_one(ax, data, show_ylabel=True, x_major_interval=None, our_method_label_x_shift=0.0):
    """Draw one Recall-QPS subplot with log-scale Y axis."""

    competitors = [m for m in data if m != OUR_METHOD]

    # Dynamically derive key recall thresholds from actual data
    rival_ceil  = max(max(x for x, _ in data[m]) for m in competitors)
    our_max     = max(x for x, _ in data[OUR_METHOD])
    # Shared comparison point: the competitor ceiling (exists in all curves)
    cmp_x       = rival_ceil

    for method, points in data.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        s  = STYLES[method]
        lw = LINE_WIDTH + 0.8 if method == OUR_METHOD else LINE_WIDTH
        ax.plot(xs, ys,
                color=s["color"], marker=s["marker"],
                linestyle=s["linestyle"], linewidth=lw,
                markersize=8 if method == OUR_METHOD else 7,
                label=method, zorder=s["zorder"])

    # Shaded exclusive recall zone (competitor ceiling → our max)
    ax.axvspan(rival_ceil, our_max, alpha=0.10, color="#E31A1C", zorder=1)

    # Competitor recall ceiling dashed line
    ax.axvline(x=rival_ceil, color="gray", linestyle="--", linewidth=1.6, zorder=2)

    # X limits (needed so annotation offsets use the same span as the final axis)
    all_x = [x for pts in data.values() for x, _ in pts]
    x_lo = min(all_x) - 2
    x_hi = max(all_x) + 2
    x_span = x_hi - x_lo

    # ── Annotations ──────────────────────────────────────────────────────────
    # QPS at (or nearest to) the shared comparison point
    _, ours_qps  = min(data[OUR_METHOD], key=lambda p: abs(p[0] - cmp_x))
    rival_qps    = max(
        min(data[m], key=lambda p: abs(p[0] - cmp_x))[1]
        for m in competitors
    )

    # Double-headed arrow: use PIMCQG's actual nearest x to cmp_x
    arrow_x, _ = min(data[OUR_METHOD], key=lambda p: abs(p[0] - cmp_x))
    ax.annotate("", xy=(arrow_x, ours_qps), xytext=(arrow_x, rival_qps),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ratio = int(round(ours_qps / rival_qps))
    ax.text(arrow_x + 0.8, (ours_qps * rival_qps) ** 0.48,
            f"~{ratio}×", fontsize=LABEL_SIZE, color="black", va="center")

    # Not "max recall": rivals plateau here (very slow gains beyond; ~+1% still possible).
    # "Ceiling" = best measured operating point for competitors in this sweep.
    ax.annotate(f"Reach saturation\n(~{rival_ceil:.1f}%)",
                xy=(rival_ceil, rival_qps),
                xytext=(rival_ceil + 4, rival_qps * 1.7),
                fontsize=PIN_LABEL_SIZE, color="gray", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

    # Label at our max recall (x-offset is fraction of span, not absolute recall %)
    # ax.annotate(f"{OUR_METHOD}\n(~{our_max:.1f}%)",
    #             xy=(our_max, data[OUR_METHOD][-1][1]),
    #             xytext=(
    #                 our_max - REACH_RECALL_LABEL_X_FRAC * x_span + our_method_label_x_shift,
    #                 data[OUR_METHOD][-1][1] * 2.3,
    #             ),
    #             fontsize=PIN_LABEL_SIZE, color="#E31A1C",
    #             arrowprops=dict(arrowstyle="->", color="#E31A1C", lw=1.0))

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_yscale("log")
    # Tighten y-limits to data (log autoscale often extends to a full decade above max,
    # which makes ~33K look "low" vs 30K: on log scale, 33.5K sits only ~38% of the way
    # from 30K to 40K, so it always looks close to the 30K grid line — that is correct.)
    all_y = [y for pts in data.values() for _, y in pts]
    y_data_lo, y_data_hi = min(all_y), max(all_y)
    ax.set_ylim(y_data_lo * Y_LIM_PAD_LO, y_data_hi * Y_LIM_PAD_HI)

    ax.set_xlim(x_lo, x_hi)
    if x_major_interval is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major_interval))
    ax.set_xlabel("Recall (%)", labelpad=5)
    if show_ylabel:
        ax.set_ylabel("QPS")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}K" if v >= 1000 else f"{v:.0f}")
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", axis="y",    linestyle=":",  alpha=0.25)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE, direction="in")
    ax.tick_params(axis="y", which="major", length=Y_TICK_LENGTH_MAJOR)
    ax.tick_params(axis="y", which="minor", length=Y_TICK_LENGTH_MINOR)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    plot_one(axes[0], SIFT1B_DATA, show_ylabel=True, x_major_interval=5)
    plot_one(
        axes[1],
        SPACEV1B_DATA,
        show_ylabel=False,
        x_major_interval=5,
        our_method_label_x_shift=OUR_METHOD_LABEL_X_SHIFT_SPACEV1B,
    )

    # bottom: room for xlabel + (a)/(b) caption; top: bring plots closer to legend
    plt.tight_layout(pad=1.5, rect=(0, 0.20, 1, 0.93))
    fig.subplots_adjust(wspace=0.14)

    # Shared legend: single row, centered above both subplots (like plot_overall_try.py)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles, labels=labels,
        loc="upper center", ncol=len(labels),
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.965),
    )

    # Captions after layout: position tweaks do not change axes bbox (see SUBPLOT_CAPTION_PAD).
    place_subplot_captions(fig, axes, ["(a) SIFT1B", "(b) SPACEV1B"])

    from pathlib import Path
    out_dir  = Path(__file__).parent
    pdf_path = out_dir / "recall_qps_comparison.pdf"
    png_path = out_dir / "recall_qps_comparison.png"

    plt.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[plot_recall_qps] Saved -> {pdf_path}")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_recall_qps] Saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
