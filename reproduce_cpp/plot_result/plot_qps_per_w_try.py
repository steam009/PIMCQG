#!/usr/bin/env python3
"""
Recall–(QPS/W) bar chart with broken Y-axis, three subplots side by side:
  (a) SIFT1B   (b) SPACEV1B   (c) SSN1B

Uses the same recall–QPS data as plot_overall_try.py; QPS is divided by
per-system power (W) to obtain efficiency (QPS/W).

Power assumptions (W):
  SymphonyQG: 410,  PIMCQG: 450,  GGNN: 810
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np

# ── Serif font for subplot titles ────────────────────────────────────────────
TITLE_FONT = "Times New Roman"
if not any(f.name == "Times New Roman" for f in fm.fontManager.ttflist):
    for fallback in ("Liberation Serif", "DejaVu Serif", "serif"):
        if any(fallback in f.name for f in fm.fontManager.ttflist) or fallback == "serif":
            TITLE_FONT = fallback
            break

LABEL_SIZE  = 18
TICK_SIZE   = 13
LEGEND_SIZE = 17

plt.rcParams.update({
    "font.size":        TICK_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.2,
})

BAR_COLORS = {
    "SymphonyQG": "#B3CDE3",
    "GGNN":       "#CCEBC5",
    "PIMCQG":     "#FBB4AE",
}

METHODS = ["SymphonyQG", "GGNN", "PIMCQG"]

LEGEND_LABELS = {
    "SymphonyQG": "SymphonyQG (CPU)",
    "GGNN": "GGNN (GPU)",
    "PIMCQG": "PIMCQG",
}

BAR_EDGE_WIDTH = 0.4

# Power (W) per method — user-specified for QPS/W
POWER_W = {
    "SymphonyQG": 410.0,
    "GGNN":       810.0,
    "PIMCQG":     450.0,
}

# ── Same recall–QPS data as plot_overall_try.py ─────────────────────────────
SIFT1B_DATA = {
    "SymphonyQG": [(62, 4820.95), (70, 3953.71), (76, 3422.52),  (78, 3093.36), (80, 2648.94), (82, 2351.45), (84, 2011.43),  (86, 1721.58), (88, 1501.26), (90, 1231.37), (92, 720.32), (94, 615.75)],
    "GGNN":       [(62, 2387), (70, 2399), (76, 2422),  (78, 2439), (80, 2451), (82, 2463), (84, 2375),  (86, 2387), (88, 2399), (90, 2311), (92, 2423), (94, 2335)],
    "PIMCQG":     [(62, 34197.78), (70, 26273.81), (76, 18737.12),  (78, 14868.19), (80, 14032.40), (82, 12804.43), (84, 8823.53),  (86, 4373.83), (88, 4262.84), (90, 3157.89), (92, 2529.69), (94, 1824.00)],
}

SPACEV1B_DATA = {
    "SymphonyQG": [(67, 4812.46), (70, 4377.90706), (72, 3818.923592),  (74, 3405.440341), (76, 3123.630791), (78, 2318.347182), (80, 1534.35165),  (82, 1262.17458), (84, 804.576157), (86, 356.2382018), (88, 206.7183268), (90, 102.9227786)],
    "GGNN":       [(67, 2100), (70, 2101), (72, 2122),  (74, 2153), (76, 2142), (78, 2103), (80, 2106),  (82, 2097), (84, 2108), (86, 2080), (88, 2159), (90, 2111)],
    "PIMCQG":     [(67, 33478.40643), (70, 30380.66979), (72, 27219.27851),  (74, 19317.57449), (76, 18161.56528), (78, 14566.92531), (80, 11384.20555),  (82, 7141.207864), (84, 4911.615479), (86, 2366.264638), (88, 1367.26334), (90, 1012.69)],
}

SSN1B_DATA = {
    "SymphonyQG": [(64, 1740.85684), (66, 1380.113392), (68, 1227.609411),  (70, 1077.387474), (72, 779.1086158), (76, 617.699956), (78, 398.4610121), (80, 313.2693759), (82, 260.9410737), (84, 155.6072338), (86, 130.3961329), (88, 66.19692772)],
    "GGNN":       [(64, 1018), (66, 1008), (68, 1008),  (70, 1006), (72, 1013), (76, 1002), (78, 999), (80, 986), (82, 1026), (84, 992), (86, 1007), (88, 1012)],
    "PIMCQG":     [(64, 15807.0278), (66, 15584.17271), (68, 14238.86165),  (70, 13068.64962), (72, 10587.53776), (76, 9042.926773), (78, 6037.11619), (80, 4370.505209), (82, 2553.495736), (84, 2268.705477), (86, 1529.019256), (88, 1325.642837)],
}


def build_bar_groups_qps_per_w(data, recall_ticks):
    """For each recall tick, nearest QPS per method, then divide by POWER_W."""
    out = {}
    for method in METHODS:
        points = data.get(method, [])
        pw = POWER_W[method]
        vals = []
        for tick in recall_ticks:
            if points:
                _, qps = min(points, key=lambda p: abs(p[0] - tick))
                vals.append(qps / pw)
            else:
                vals.append(0.0)
        out[method] = vals
    return out


def plot_one_broken(ax_low, ax_high, data, recall_ticks, subplot_label, title,
                    show_ylabel=True, y_low=(0, 16), y_high=(20, 82)):
    """Grouped-bar Recall–(QPS/W) subplot with broken Y-axis."""

    qpw_by_method = build_bar_groups_qps_per_w(data, recall_ticks)

    n_ticks  = len(recall_ticks)
    n_groups = len(METHODS)
    bar_width = 0.22
    x = np.arange(n_ticks)

    for ax in [ax_low, ax_high]:
        for i, method in enumerate(METHODS):
            offset = (i - (n_groups - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                qpw_by_method[method],
                width=bar_width,
                color=BAR_COLORS[method],
                label=method,
                edgecolor="black",
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
                alpha=1,
            )

    ax_low.set_ylim(*y_low)
    ax_high.set_ylim(*y_high)

    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)
    ax_high.tick_params(bottom=False)
    ax_high.set_xticks([])

    d = 0.02
    break_kw = dict(color="k", clip_on=False, linewidth=1.5)
    kw = {**break_kw, "transform": ax_high.transAxes}
    ax_high.plot((-d, +d), (-d, +d), **kw)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw = {**break_kw, "transform": ax_low.transAxes}
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_low.set_xticks(x)
    ax_low.set_xticklabels([f"{r}%" for r in recall_ticks])
    ax_low.set_xlabel("Recall (%)")

    if show_ylabel:
        ax_low.set_ylabel("QPS/W")
        # Nudge label slightly upward (axes coords; default vertical center is ~0.5)
        ax_low.yaxis.set_label_coords(-0.09, 0.68)

    # Lower panel: avoid MaxNLocator + int() rounding mismatch (e.g. y in (0,6) gives
    # ticks at 0, 1.5, 3, 4.5 but labels "0,2,3,4" — equal pixel steps look like unequal value steps).
    # Omit the top y_low limit (6 / 16) next to the axis break — label would crowd the gap.
    y_lo_max = y_low[1]
    if y_lo_max <= 6:
        ax_low.yaxis.set_major_locator(
            ticker.FixedLocator(np.arange(0, y_lo_max, 2))
        )
    elif y_lo_max <= 16:
        ax_low.yaxis.set_major_locator(
            ticker.FixedLocator(np.arange(0, y_lo_max, 4))
        )
    else:
        ax_low.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="upper"))

    def _fmt_qpw_y(v, _pos):
        v = float(v)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"

    fmt = ticker.FuncFormatter(_fmt_qpw_y)
    ax_low.yaxis.set_major_formatter(fmt)
    ax_high.yaxis.set_major_formatter(fmt)

    ax_low.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax_high.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax_low.set_axisbelow(True)
    ax_high.set_axisbelow(True)

    ax_low.text(0.5, -0.7, f"{subplot_label} {title}",
                transform=ax_low.transAxes, ha="center",
                fontsize=LABEL_SIZE, fontfamily=TITLE_FONT)


def main():
    SIFT1B_RECALL_TICKS   = [62, 70, 76, 80, 84, 86, 88, 90, 92, 94]
    SPACEV1B_RECALL_TICKS = [67, 72, 76, 78, 80, 82, 84, 86, 88, 90]
    SSN1B_RECALL_TICKS    = [64, 68, 72, 76, 78, 80, 82, 84, 86, 88]

    # y_low / y_high chosen so low panel fits SymphonyQG+GGNN; high panel fits PIMCQG peaks
    datasets = [
        (SIFT1B_DATA,   SIFT1B_RECALL_TICKS,   "(a)", "SIFT1B",   True,
         (0, 16), (20, 82)),
        (SPACEV1B_DATA, SPACEV1B_RECALL_TICKS, "(b)", "SPACEV1B", False,
         (0, 16), (20, 82)),
        (SSN1B_DATA,    SSN1B_RECALL_TICKS,    "(c)", "SSN1B",    False,
         (0, 6), (8, 38)),
    ]

    fig = plt.figure(figsize=(18, 3.5))
    outer_gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)

    for col, (data, ticks, label, title, show_ylabel, y_lo, y_hi) in enumerate(datasets):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[col],
            height_ratios=[1, 2], hspace=0.045
        )
        ax_high = fig.add_subplot(inner_gs[0])
        ax_low  = fig.add_subplot(inner_gs[1])

        plot_one_broken(ax_low, ax_high, data, ticks, label, title,
                        show_ylabel=show_ylabel, y_low=y_lo, y_high=y_hi)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.28, wspace=0.1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS[m], label=LEGEND_LABELS[m])
        for m in METHODS
    ]
    fig.legend(handles=handles, labels=[LEGEND_LABELS[m] for m in METHODS],
               loc="upper center", ncol=len(METHODS),
               frameon=False,
               fontsize=LEGEND_SIZE,
               bbox_to_anchor=(0.5, 0.95))

    from pathlib import Path
    out_dir  = Path(__file__).parent
    pdf_path = out_dir / "overall_qps_per_w.pdf"
    png_path = out_dir / "overall_qps_per_w.png"

    plt.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[plot_qps_per_w_try] Saved -> {pdf_path}")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_qps_per_w_try] Saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
