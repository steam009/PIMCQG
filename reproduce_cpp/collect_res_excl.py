import os
import re
import glob
import pandas as pd

# Base directory (script lives inside reproduce_cpp/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_PATTERN = os.path.join(BASE_DIR, "nohup_dpu*_vdpu*_nprobe*_ef*_postef*_subset_ssn1b_top10.out")
OUTPUT_EXCEL = os.path.join(BASE_DIR, "summary_results.xlsx")

# When no "BATCH SIZE:" / "BATCH_SIZE:" line appears in the log
DEFAULT_BATCH_SIZE = 3000

# Fields that must be present for a row to be exported (batch size may default)
REQUIRED_FIELDS = ("EF", "POST EF", "NPROBE", "DPU Search Time (s)")

# Match result line: "EF: 30, POST EF: 32, NPROBE: 16, QPS: 90.99, Recall: 0.06%"
re_result_line = re.compile(
    r"EF:\s*(\d+),\s*POST EF:\s*(\d+),\s*NPROBE:\s*(\d+),\s*QPS:\s*[\d\.]+,\s*Recall:\s*([\d\.]+)%"
)

# Match DPU search time line: "DPU search time (FIFO mode): 0.126691 seconds"
re_dpu_time = re.compile(r"DPU search time \(FIFO mode\):\s*([\d\.]+)\s*seconds")

# Match batch size: "BATCH SIZE:  %d" or "BATCH_SIZE: %d" (C printf)
re_batch_size = re.compile(r"BATCH[\s_]+SIZE:\s*(\d+)")


def parse_file(path: str):
    """Parse metrics from a single .out file; returns None if incomplete."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    result = {
        "EF": None,
        "POST EF": None,
        "NPROBE": None,
        "BATCH SIZE": None,
        "DPU Search Time (s)": None,
    }

    # Scan from the bottom to avoid matching intermediate debug output
    for line in reversed(lines):
        stripped = line.strip()

        # Match summary result line
        if result["EF"] is None:
            m = re_result_line.search(stripped)
            if m:
                result["EF"] = int(m.group(1))
                result["POST EF"] = int(m.group(2))
                result["NPROBE"] = int(m.group(3))

        # Match batch size line
        if result["BATCH SIZE"] is None:
            b = re_batch_size.search(stripped)
            if b:
                result["BATCH SIZE"] = int(b.group(1))

        # Match DPU search time line
        if result["DPU Search Time (s)"] is None:
            t = re_dpu_time.search(stripped)
            if t:
                result["DPU Search Time (s)"] = float(t.group(1))

        # Stop only when required fields and batch are both resolved, or batch is absent
        # (scan entire file). Otherwise we break right after "DPU search time" and miss
        # "BATCH_SIZE:" which appears above it in the log (earlier in reverse order).
        if all(result[k] is not None for k in REQUIRED_FIELDS) and result["BATCH SIZE"] is not None:
            break

    if result["BATCH SIZE"] is None:
        result["BATCH SIZE"] = DEFAULT_BATCH_SIZE

    # If any required field is missing, skip this file
    if any(result[k] is None for k in REQUIRED_FIELDS):
        missing = [k for k in REQUIRED_FIELDS if result[k] is None]
        print(f"[WARN] {os.path.basename(path)} incomplete (missing: {missing}), skipped.")
        return None

    return result


def main():
    files = sorted(glob.glob(SEARCH_PATTERN))
    if not files:
        print(f"No matching result files found: {SEARCH_PATTERN}")
        return

    rows = []
    for fp in files:
        print(f"Parsing: {os.path.basename(fp)}")
        data = parse_file(fp)
        if data is not None:
            rows.append(data)

    if not rows:
        print("No results parsed successfully; Excel will not be generated.")
        return

    df = pd.DataFrame(rows, columns=[
        "EF",
        "POST EF",
        "NPROBE",
        "BATCH SIZE",
        "DPU Search Time (s)",
    ])

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()