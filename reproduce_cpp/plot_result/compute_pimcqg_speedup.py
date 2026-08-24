#!/usr/bin/env python3
"""
Speedup of PIMCQG QPS relative to Per-Query, Batch Sync, and PIMCQG_1
for each dataset. Data aligned with plot_async_pqc_try.py.
"""

# Same as plot_async_pqc_try.py (update both when measurements change)
datasets = ["SIFT1B", "SPACEV1B", "SSN1B"]

qps_per_query = [220.89, 202.29389, 225.31]
qps_batch_sync = [22857, 24488.736, 10538.018]
qps_pimcqg_1 = [18483, 14443.909, 9482.8676]
qps_pimcqg = [34197.77714, 33478.40643, 15807.0278]

BASELINES = {
    "Per-Query": qps_per_query,
    "Batch Sync": qps_batch_sync,
    "PIMCQG_1": qps_pimcqg_1,
}


def main() -> None:
    n = len(datasets)
    print("PIMCQG speedup = QPS(PIMCQG) / QPS(baseline)\n")

    for baseline_name, baseline_qps in BASELINES.items():
        print(f"--- vs {baseline_name} ---")
        for i in range(n):
            s = qps_pimcqg[i] / baseline_qps[i]
            print(
                f"  {datasets[i]:10s}  "
                f"PIMCQG={qps_pimcqg[i]:.4f}  "
                f"baseline={baseline_qps[i]:.4f}  "
                f"speedup={s:.4f}x"
            )
        print()

    # Compact table: rows = datasets, columns = baselines
    header = "dataset".ljust(12) + "".join(f"{k:>14}" for k in BASELINES)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = datasets[i].ljust(12)
        for baseline_name, baseline_qps in BASELINES.items():
            s = qps_pimcqg[i] / baseline_qps[i]
            row += f"{s:>14.4f}"
        print(row)


if __name__ == "__main__":
    main()
