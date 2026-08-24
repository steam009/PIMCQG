#!/bin/bash

set -e  # Exit on error

# ======================================================================
# Source UPMEM SDK environment (must use the SDK that has dpu_get_rank_allocator_id)
# ======================================================================
# First check if SDK is already sourced; if not, try to auto-detect
if [ -z "$UPMEM_HOME" ] || ! command -v dpu-pkg-config >/dev/null 2>&1; then
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    UPMEM_SDK_DIR="${SCRIPT_DIR}/../upmem-2024.2.0-Linux-x86_64"
    if [ ! -f "$UPMEM_SDK_DIR/upmem_env.sh" ]; then
        UPMEM_SDK_DIR="${SCRIPT_DIR}/../../upmem-2024.2.0-Linux-x86_64"
    fi
    if [ -f "$UPMEM_SDK_DIR/upmem_env.sh" ]; then
        source "$UPMEM_SDK_DIR/upmem_env.sh" > /dev/null 2>&1
    fi
fi
if ! command -v dpu-pkg-config >/dev/null 2>&1; then
    echo "ERROR: UPMEM SDK not found. Please source upmem_env.sh first."
    exit 1
fi

# ======================================================================
# Mode selection: "perf", "recall", or "both"
#   perf   - DPU FIFO performance testing (bin/host_code)
#   recall - Pure CPU recall testing (bin/host_recall)
#   both   - Run performance first, then recall
# ======================================================================
MODE="${1:-both}"

if [[ "$MODE" != "perf" && "$MODE" != "recall" && "$MODE" != "both" ]]; then
    echo "Usage: $0 [perf|recall|both]"
    echo "  perf   - DPU FIFO performance testing only"
    echo "  recall - Pure CPU recall testing only"
    echo "  both   - Run both performance and recall (default)"
    exit 1
fi

echo "======================================================================"
echo "=== Batch Run Script - Mode: $MODE ==="
echo "======================================================================"
echo ""

# ======================================================================
# Common configuration
# ======================================================================

# Define physical DPU configurations to test (perf mode only)
DPU_CONFIGS=(2560)

# Define virtual DPU count (perf mode: for load balancing simulation)
VIRTUAL_DPU_CONFIGS=(2560)

# Define nprobe values to test (used by both modes)
NPROBE_VALUES=(8)

# Define EF and POST_EF combinations to test (format: "EF:POST_EF")
EF_POST_EF_COMBINATIONS=(
    "40:48"
)

# Other fixed parameters
export NR_TASKLETS=11
export BATCH_SIZE=3000
export DIMM=128
export PADDED_DIMM=128
export MAX_SIZE_PER_DPU=400000
export DEGREE=32
export MAX_QUERY_CLUSTER_PER_DPU=2000
export TOPK=10

# Conda library path (for runtime)
CONDA_LIB_PATH=${CONDA_LIB_PATH:-$CONDA_PREFIX/lib}

# ======================================================================
# Helper: setup runtime library path
# ======================================================================
setup_runtime_lib() {
    if [[ "$LD_LIBRARY_PATH" != *"$CONDA_LIB_PATH"* ]]; then
        export LD_LIBRARY_PATH=$CONDA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi
}

# Helper: strip conda from LD_LIBRARY_PATH for compilation
strip_conda_ld() {
    OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    if [[ -n "$LD_LIBRARY_PATH" ]]; then
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^$CONDA_LIB_PATH$" | tr '\n' ':' | sed 's/:$//')
    fi
}

restore_ld() {
    export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
}

# ======================================================================
# Performance mode
# ======================================================================
run_perf() {
    echo "######################################################################"
    echo "### PERFORMANCE MODE (DPU FIFO) ###"
    echo "######################################################################"
    echo ""

    for nr_dpus in "${DPU_CONFIGS[@]}"; do
        for virtual_dpus in "${VIRTUAL_DPU_CONFIGS[@]}"; do
            echo "======================================================================"
            echo "=== Testing with NR_DPUS = $nr_dpus, VIRTUAL_DPUS = $virtual_dpus ==="
            echo "======================================================================"
            echo ""

            export NR_DPUS=$nr_dpus
            export VIRTUAL_DPUS=$virtual_dpus

            for combination in "${EF_POST_EF_COMBINATIONS[@]}"; do
                ef=$(echo $combination | cut -d':' -f1)
                post_ef=$(echo $combination | cut -d':' -f2)

                echo "----------------------------------------------------------------------"
                echo "Testing: NR_DPUS=$nr_dpus, VIRTUAL_DPUS=$virtual_dpus, EF=$ef, POST_EF=$post_ef"
                echo "----------------------------------------------------------------------"

                export EF=$ef
                export POST_EF=$post_ef

                echo "Configuration:"
                echo "  NR_DPUS: $NR_DPUS"
                echo "  VIRTUAL_DPUS: $VIRTUAL_DPUS"
                echo "  BATCH_SIZE: $BATCH_SIZE"
                echo "  DEGREE: $DEGREE"
                echo "  EF: $EF"
                echo "  POST_EF: $POST_EF"
                echo ""

                # Step 1: Clean
                echo "--- Step 1: Cleaning previous builds ---"
                make -C dpu clean >/dev/null 2>&1 || true
                make -f Makefile clean >/dev/null 2>&1 || true
                make -f Makefile.mixed clean >/dev/null 2>&1 || true
                echo "Clean completed"
                echo ""

                # Step 2: Build (DPU + host_code)
                echo "--- Step 2: Building (DPU + host_code) ---"
                ./build.sh

                if [ ! -f "bin/host_code" ]; then
                    echo "ERROR: Build failed - bin/host_code not found!"
                    exit 1
                fi
                echo "Build completed"
                echo ""

                # Step 3: Run
                echo "--- Step 3: Running performance tests ---"
                setup_runtime_lib

                for nprobe in "${NPROBE_VALUES[@]}"; do
                    OUTPUT_FILE="nohup_perf_dpu${NR_DPUS}_vdpu${VIRTUAL_DPUS}_nprobe${nprobe}_ef${EF}_postef${POST_EF}.out"
                    echo "  Running: nprobe=$nprobe -> $OUTPUT_FILE"
                    ./bin/host_code $nprobe > "$OUTPUT_FILE" 2>&1
                done

                echo "Tests completed for EF=$EF, POST_EF=$POST_EF"
                echo ""
            done

            echo "All perf tests completed for NR_DPUS=$NR_DPUS, VIRTUAL_DPUS=$VIRTUAL_DPUS"
            echo ""
        done
    done
}

# ======================================================================
# Recall mode
# ======================================================================
run_recall() {
    echo "######################################################################"
    echo "### RECALL MODE (Pure CPU) ###"
    echo "######################################################################"
    echo ""

    for combination in "${EF_POST_EF_COMBINATIONS[@]}"; do
        ef=$(echo $combination | cut -d':' -f1)
        post_ef=$(echo $combination | cut -d':' -f2)

        echo "----------------------------------------------------------------------"
        echo "Testing recall: EF=$ef, POST_EF=$post_ef"
        echo "----------------------------------------------------------------------"

        export EF=$ef
        export POST_EF=$post_ef
        # Recall mode uses VIRTUAL_DPUS for load balancing in compute_query_to_dpu_mapping
        export VIRTUAL_DPUS=2560

        echo "Configuration:"
        echo "  BATCH_SIZE: $BATCH_SIZE"
        echo "  DEGREE: $DEGREE"
        echo "  EF: $EF"
        echo "  POST_EF: $POST_EF"
        echo "  TOPK: $TOPK"
        echo ""

        # Step 1: Clean
        echo "--- Step 1: Cleaning previous builds ---"
        make -f Makefile.mixed clean >/dev/null 2>&1 || true
        echo "Clean completed"
        echo ""

        # Step 2: Build recall target
        echo "--- Step 2: Building recall target (host_recall) ---"
        make -f Makefile.mixed recall NR_DPUS=64 VIRTUAL_DPUS=2560 \
            BATCH_SIZE=$BATCH_SIZE DIMM=$DIMM PADDED_DIMM=$PADDED_DIMM \
            MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU DEGREE=$DEGREE \
            MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
            EF=$EF POST_EF=$POST_EF TOPK=$TOPK

        if [ ! -f "bin/host_recall" ]; then
            echo "ERROR: Build failed - bin/host_recall not found!"
            exit 1
        fi
        echo "Build completed"
        echo ""

        # Step 3: Run
        echo "--- Step 3: Running recall tests ---"
        setup_runtime_lib

        for nprobe in "${NPROBE_VALUES[@]}"; do
            OUTPUT_FILE="nohup_recall_nprobe${nprobe}_ef${EF}_postef${POST_EF}.out"
            echo "  Running: nprobe=$nprobe -> $OUTPUT_FILE"
            ./bin/host_recall $nprobe > "$OUTPUT_FILE" 2>&1
        done

        echo "Recall tests completed for EF=$EF, POST_EF=$POST_EF"
        echo ""
    done
}

# ======================================================================
# Main execution
# ======================================================================
setup_runtime_lib

case "$MODE" in
    perf)
        run_perf
        ;;
    recall)
        run_recall
        ;;
    both)
        run_perf
        run_recall
        ;;
esac

echo "======================================================================"
echo "=== All configurations tested successfully! (Mode: $MODE) ==="
echo "======================================================================"
