#!/usr/bin/env python3
"""
Parse DATA_SIZE, avg_push_time (Host->DPU), avg_pull_time (DPU->Host)
from push_xfer_size_bench.log and plot transfer time vs. data size.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Global font settings (paper-ready) ──────────────────────────────────────
LABEL_SIZE  = 20   # axis label font size
TICK_SIZE   = 14   # tick label font size
LEGEND_SIZE = 17   # legend font size
LINE_WIDTH  = 2.0
MARKER_SIZE = 7

plt.rcParams.update({
    "font.size":        TICK_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "axes.linewidth":   1.2,
})


def parse_log(path: Path):
    sizes    = []
    avg_push = []
    avg_pull = []

    current_size = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # Match "DATA_SIZE = 8192 bytes"
            m_size = re.search(r"DATA_SIZE\s*=\s*(\d+)\s*bytes", line)
            if m_size:
                current_size = int(m_size.group(1))
                continue

            # Match "avg_push_time (Host->DPU) : 0.019588779 s"
            m_push = re.search(r"avg_push_time\s*\(Host->DPU\)\s*:\s*([\d.]+)\s*s", line)
            if m_push and current_size is not None:
                sizes.append(current_size)
                avg_push.append(float(m_push.group(1)))
                continue

            # Match "avg_pull_time (DPU->Host) : 0.012087266 s"
            m_pull = re.search(r"avg_pull_time\s*\(DPU->Host\)\s*:\s*([\d.]+)\s*s", line)
            if m_pull and len(avg_pull) < len(sizes):
                avg_pull.append(float(m_pull.group(1)))

    n = min(len(sizes), len(avg_push), len(avg_pull))
    return sizes[:n], avg_push[:n], avg_pull[:n]


def main():
    log_path = Path(__file__).with_name("push_xfer_size_bench.log")
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    sizes, avg_push, avg_pull = parse_log(log_path)

    x = list(range(len(sizes)))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(x, avg_push, marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color="#1f77b4", label="Host → PIM")
    ax.plot(x, avg_pull, marker="s", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color="#d62728", label="PIM → Host")

    ymax = max(max(avg_push), max(avg_pull))
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
    ax.set_xlabel("Data Size (bytes)")
    ax.set_ylabel("Transfer Time (s)")
    # Y-axis only: shorter tick strings + slightly smaller font so labels fit without changing axes box.
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    # Nudge y-axis title slightly downward (axes coords; default vertical center is 0.5).
    ax.yaxis.set_label_coords(-0.23, 0.39, transform=ax.transAxes)

    out_path = Path(__file__).with_name("push_xfer_size_bench.pdf")
    plt.savefig(str(out_path), bbox_inches="tight")
    print(f"[plot_push_xfer_bench] Figure saved -> {out_path}")
    png_path = out_path.with_suffix(".png")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_push_xfer_bench] Figure saved -> {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
