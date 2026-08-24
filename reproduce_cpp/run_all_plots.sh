#!/bin/bash
#
# run_all_plots.sh - Run all plot scripts in plot_result/ and collect output to figure/
#
# Usage:
#   ./run_all_plots.sh            # run all plotting scripts
#   ./run_all_plots.sh --list     # list available scripts without running
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_DIR="$SCRIPT_DIR/plot_result"
FIGURE_DIR="$SCRIPT_DIR/figure"

# ── Plot scripts that generate figures (in dependency order) ─────────────────
# compute_pimcqg_speedup.py is a text-only script (no figure); run first for console output.
PLOT_SCRIPTS=(
    "compute_pimcqg_speedup.py"
    "plot_overall_try.py"
    "plot_Multifree.py"
    "plot_async_pqc_try.py"
    "plot_hbm_pim_host_pim.py"
    "plot_pipeline_breakdown_horizontal.py"
    "plot_multinode_qp_and_pim_speedup.py"
    "plot_recall_qps.py"
    "plot_qps_per_w_try.py"
    "plot_overfetch_try.py"
    "plot_sift_space_power_qps.py"
    "plot_recall_ef_cos_vs_fixed_alpha.py"
    "plot_push_xfer_bench.py"
)

if [[ "$1" == "--list" ]]; then
    echo "Available plot scripts (in run order):"
    for i in "${!PLOT_SCRIPTS[@]}"; do
        printf "  %2d. %s\n" $((i + 1)) "${PLOT_SCRIPTS[$i]}"
    done
    exit 0
fi

# ── Prepare figure directory ─────────────────────────────────────────────────
mkdir -p "$FIGURE_DIR"

# Track prior state so we only move newly-generated files
BEFORE_FILES=$(mktemp)
find "$PLOT_DIR" -maxdepth 1 \( -name "*.pdf" -o -name "*.png" \) -print0 2>/dev/null \
    | xargs -0 -r ls -1 2>/dev/null > "$BEFORE_FILES" || true

echo "======================================================================"
echo "=== Running All Plot Scripts ==="
echo "======================================================================"
echo "Plot directory : $PLOT_DIR"
echo "Figure output  : $FIGURE_DIR"
echo ""

FAILED=()
PASSED=0

for script in "${PLOT_SCRIPTS[@]}"; do
    script_path="$PLOT_DIR/$script"
    if [[ ! -f "$script_path" ]]; then
        echo "[SKIP] $script — file not found"
        continue
    fi

    echo "----------------------------------------------------------------------"
    echo "Running: $script"
    echo "----------------------------------------------------------------------"

    # Determine interpreter: .py → python3
    case "$script" in
        *.py)  INTERP="python3" ;;
        *)     INTERP="bash" ;;
    esac

    cd "$PLOT_DIR"
    if $INTERP "$script_path"; then
        echo "[PASS] $script"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $script (exit code $?)"
        FAILED+=("$script")
    fi
    cd "$SCRIPT_DIR"
    echo ""
done

# ── Move newly-generated figures to figure/ ──────────────────────────────────
echo "----------------------------------------------------------------------"
echo "Collecting generated figures..."

MOVED=0
while IFS= read -r -d '' f; do
    fname=$(basename "$f")
    # Only move if not present before this run
    if ! grep -qxF "$f" "$BEFORE_FILES" 2>/dev/null; then
        mv "$f" "$FIGURE_DIR/"
        echo "  → figure/$fname"
        MOVED=$((MOVED + 1))
    fi
done < <(find "$PLOT_DIR" -maxdepth 1 \( -name "*.pdf" -o -name "*.png" \) -print0 2>/dev/null)

# Also collect any .pdf/.png already in figure/ that were moved in prior runs
for ext in pdf png; do
    for f in "$FIGURE_DIR"/*."$ext"; do
        [[ -f "$f" ]] || continue
        if [[ $MOVED -eq 0 ]]; then
            fname=$(basename "$f")
            echo "  (existing) figure/$fname"
            MOVED=$((MOVED + 1))
        fi
    done
done

rm -f "$BEFORE_FILES"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "=== Plot Generation Summary ==="
echo "======================================================================"
echo "Passed : $PASSED / ${#PLOT_SCRIPTS[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed : ${#FAILED[@]}"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "Figures in figure/ : $(find "$FIGURE_DIR" -maxdepth 1 \( -name '*.pdf' -o -name '*.png' \) | wc -l)"
echo ""
echo "Done."
