#!/usr/bin/env python3
"""
Plot results from TransferBwFixture.bw_comparison_table (TransferBwTest.cpp).

Usage
-----
  # Step 1: run the simulator and save output
  ./sim --gtest_filter="TransferBwFixture.bw_comparison_table" | tee transfer_bw_result.log

  # Step 2: plot
  python3 plot_hbm_pim_host_pim.py                    # reads transfer_bw_result.log
  python3 plot_hbm_pim_host_pim.py path/to/other.log  # custom log file

Output
------
  transfer_bw_result.png  – saved in the same directory as this script
"""

import re
import sys
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


# ─────────────────────────────────────────────
#  Parse the comparison table from the log
# ─────────────────────────────────────────────
def parse_log(path: Path):
    """
    Parse lines of the form:
        <data_size_int>   <write_time_float>   <read_time_float>
    which appear inside the '====..====' block of bw_comparison_table.

    Returns (sizes, write_times, read_times) as lists.
    """
    sizes       = []
    write_times = []
    read_times  = []

    # Pattern: optional leading whitespace, integer, two scientific floats
    pattern = re.compile(
        r"^\s+(\d+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s*$"
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                sizes.append(int(m.group(1)))
                write_times.append(float(m.group(2)))
                read_times.append(float(m.group(3)))

    return sizes, write_times, read_times


# ─────────────────────────────────────────────
#  Draw the figure
# ─────────────────────────────────────────────
def plot(sizes, write_times, read_times, out_path: Path):
    x        = list(range(len(sizes)))
    x_labels = [str(s) for s in sizes]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(x, write_times, marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color="#1f77b4", label="Host → PIM")
    ax.plot(x, read_times,  marker="s", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color="#d62728", label="PIM → Host")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Data Size (bytes)")
    ax.set_ylabel("Transfer Time (s)")
    # Y-axis only: shorter tick strings + slightly smaller font so labels fit without changing axes box.
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    # Nudge y-axis title slightly downward (axes coords; default vertical center is 0.5).
    ax.yaxis.set_label_coords(-0.23, 0.39, transform=ax.transAxes)

    plt.savefig(str(out_path), bbox_inches="tight")
    print(f"[plot_hbm_pim_host_pim] Figure saved -> {out_path}")
    png_path = out_path.with_suffix(".png")
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    print(f"[plot_hbm_pim_host_pim] Figure saved -> {png_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = Path(__file__).with_name("transfer_bw_result.log")

    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        print("  Run first:  ./sim --gtest_filter='TransferBwFixture.bw_comparison_table'"
              " | tee transfer_bw_result.log")
        sys.exit(1)

    sizes, write_times, read_times = parse_log(log_path)

    if not sizes:
        print(f"[ERROR] No data rows found in {log_path}.")
        print("  Make sure the file contains output from bw_comparison_table.")
        sys.exit(1)

    print(f"[plot_hbm_pim_host_pim] Parsed {len(sizes)} data points from {log_path}")
    print(f"  Data sizes : {sizes}")
    print(f"  Write (s)  : {[f'{v:.3e}' for v in write_times]}")
    print(f"  Read  (s)  : {[f'{v:.3e}' for v in read_times]}")

    out_path = Path(__file__).with_name("transfer_bw_result.pdf")
    plot(sizes, write_times, read_times, out_path)


if __name__ == "__main__":
    main()
