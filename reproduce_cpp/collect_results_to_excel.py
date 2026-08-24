import os
import re
import glob
import pandas as pd

# Root directory (modify as needed; use '.' if script is placed in reproduce_cpp)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_PATTERN = os.path.join(BASE_DIR, "nohup_dpu*_vdpu*_nprobe*_ef*_postef*_sift100M_top10.out")
OUTPUT_EXCEL = os.path.join(BASE_DIR, "summary_results.xlsx")

# Regular expression
re_query_prepare = re.compile(r"Query prepare time:\s*([\d\.]+)\s*seconds")
re_cpu_dpu = re.compile(r"CPU-DPU data transfer time:\s*([\d\.]+)\s*seconds")
re_dpu_search = re.compile(r"DPU search time:\s*([\d\.]+)\s*seconds")
re_dpu_cpu = re.compile(r"DPU-CPU data transfer time:\s*([\d\.]+)\s*seconds")
re_cpu_post = re.compile(r"CPU Post process time:\s*([\d\.]+)\s*seconds")
re_params = re.compile(r"EF:\s*(\d+),\s*POST EF:\s*(\d+),\s*NPROBE:\s*(\d+)")

def parse_file(path: str):
    """Parse required metrics from a single .out file; return None on failure"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Scan from end to avoid matching intermediate debug output
    result = {
        "EF": None,
        "POST EF": None,
        "NPROBE": None,
        "Query prepare": None,
        "CPU-DPU": None,
        "DPU search": None,
        "DPU-CPU": None,
        "CPU Post process": None,
    }

    for line in reversed(lines):
        line = line.strip()

        m = re_params.search(line)
        if m:
            result["EF"] = int(m.group(1))
            result["POST EF"] = int(m.group(2))
            result["NPROBE"] = int(m.group(3))
            continue

        m = re_query_prepare.search(line)
        if m:
            result["Query prepare"] = float(m.group(1))
            continue

        m = re_cpu_dpu.search(line)
        if m:
            result["CPU-DPU"] = float(m.group(1))
            continue

        m = re_dpu_search.search(line)
        if m:
            result["DPU search"] = float(m.group(1))
            continue

        m = re_dpu_cpu.search(line)
        if m:
            result["DPU-CPU"] = float(m.group(1))
            continue

        m = re_cpu_post.search(line)
        if m:
            result["CPU Post process"] = float(m.group(1))
            continue

    # If EF / POST EF / NPROBE is empty, consider the file result incomplete and skip
    if result["EF"] is None or result["POST EF"] is None or result["NPROBE"] is None:
        print(f"[WARN] {os.path.basename(path)} result section incomplete, skip.")
        return None

    # Some timing fields may be missing; allow None here
    return result

def main():
    files = sorted(glob.glob(SEARCH_PATTERN))
    if not files:
        print(f"No matching result files found: {SEARCH_PATTERN}")
        return

    rows = []
    for fp in files:
        print(f"Parsing file: {os.path.basename(fp)}")
        data = parse_file(fp)
        if data is not None:
            rows.append(data)

    if not rows:
        print("No results successfully parsed; Excel will not be generated.")
        return

    df = pd.DataFrame(rows, columns=[
        "EF",
        "POST EF",
        "NPROBE",
        "Query prepare",
        "CPU-DPU",
        "DPU search",
        "DPU-CPU",
        "CPU Post process",
    ])

    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()