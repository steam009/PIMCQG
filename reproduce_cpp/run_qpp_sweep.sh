#!/bin/bash

set -e

echo "======================================================================"
echo "=== QUERIES_PER_PUSH Sweep - Testing Multiple QPP Configurations ==="
echo "======================================================================"
echo ""

# QUERIES_PER_PUSH values to test
QPP_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# Per-QPP FIFO pointer width N (capacity = 1 << N slots); index i is for QUERIES_PER_PUSH = i+1.
# Must stay consistent with the #if QUERIES_PER_PUSH chain in include/fifo_types.h.
INPUT_FIFO_SIZE=(4 4 3 3 3 2 2 2 2 1 1 1 1 1 1)
OUTPUT_FIFO_SIZE=(4 4 3 3 3 2 2 2 2 1 1 1 1 1 1)

# Fixed parameters
export NR_DPUS=2560
export VIRTUAL_DPUS=2560
export NR_TASKLETS=11
export BATCH_SIZE=3000
export DIMM=128
export PADDED_DIMM=128
export MAX_SIZE_PER_DPU=440000
export DEGREE=32
export MAX_QUERY_CLUSTER_PER_DPU=2000
export TOPK=10
export EF=100
export POST_EF=128

NPROBE_VALUES=(8)

echo "Fixed configuration:"
echo "  NR_DPUS:       $NR_DPUS"
echo "  VIRTUAL_DPUS:  $VIRTUAL_DPUS"
echo "  BATCH_SIZE:    $BATCH_SIZE"
echo "  EF:            $EF"
echo "  POST_EF:       $POST_EF"
echo "  NPROBE:        ${NPROBE_VALUES[*]}"
echo "  FIFO: per-QPP INPUT_FIFO_SIZE / OUTPUT_FIFO_SIZE (see script arrays + fifo_types.h)"
echo ""

nq=${#QPP_VALUES[@]}
if [ ${#INPUT_FIFO_SIZE[@]} -ne "$nq" ] || [ ${#OUTPUT_FIFO_SIZE[@]} -ne "$nq" ]; then
    echo "Error: INPUT_FIFO_SIZE and OUTPUT_FIFO_SIZE must each have ${nq} entries (one per QPP_VALUES)."
    exit 1
fi

fifo_n=${#INPUT_FIFO_SIZE[@]}
for qpp in "${QPP_VALUES[@]}"; do
    if [ "$qpp" -lt 1 ] || [ "$qpp" -gt "$fifo_n" ]; then
        echo "Error: QUERIES_PER_PUSH=$qpp out of range for INPUT_FIFO_SIZE / OUTPUT_FIFO_SIZE (length $fifo_n, index qpp-1)."
        exit 1
    fi
    export QUERIES_PER_PUSH=$qpp
    idx=$((qpp - 1))
    export INPUT_FIFO_PTR_SIZE="${INPUT_FIFO_SIZE[$idx]}"
    export OUTPUT_FIFO_PTR_SIZE="${OUTPUT_FIFO_SIZE[$idx]}"

    echo "======================================================================"
    echo "=== Testing QUERIES_PER_PUSH = $qpp ==="
    echo "  INPUT_FIFO_SIZE=${INPUT_FIFO_SIZE[$idx]} OUTPUT_FIFO_SIZE=${OUTPUT_FIFO_SIZE[$idx]} (capacity in/out: $((1 << INPUT_FIFO_PTR_SIZE)) $((1 << OUTPUT_FIFO_PTR_SIZE)) slots)"
    echo "======================================================================"

    # Step 1: Clean
    echo "--- Step 1: Cleaning previous builds ---"
    make -C dpu clean >/dev/null 2>&1 || true
    make -f Makefile clean >/dev/null 2>&1 || true
    make -f Makefile.mixed clean >/dev/null 2>&1 || true
    echo "Clean completed"

    # Step 2: Build
    echo "--- Step 2: Building with QUERIES_PER_PUSH=$qpp ---"
    ./build.sh

    if [ ! -f "bin/host_code" ]; then
        echo "Error: Build failed for QUERIES_PER_PUSH=$qpp!"
        exit 1
    fi
    echo "Build completed"

    # Step 3: Run
    echo "--- Step 3: Running tests ---"
    for nprobe in "${NPROBE_VALUES[@]}"; do
        OUTPUT_FILE="nohup_dpu${NR_DPUS}_vdpu${VIRTUAL_DPUS}_nprobe${nprobe}_ef${EF}_postef${POST_EF}_qpp${qpp}_sift.out"
        echo "  Running: nprobe=$nprobe, QUERIES_PER_PUSH=$qpp -> $OUTPUT_FILE"
        ./bin/host_code $nprobe > "$OUTPUT_FILE" 2>&1
        echo "  Done: $OUTPUT_FILE"
    done

    echo "Completed QUERIES_PER_PUSH=$qpp"
    echo ""
done

echo "======================================================================"
echo "=== All QUERIES_PER_PUSH configurations tested! ==="
echo "======================================================================"
echo ""
echo "Output files generated:"
for qpp in "${QPP_VALUES[@]}"; do
    for nprobe in "${NPROBE_VALUES[@]}"; do
        f="nohup_dpu${NR_DPUS}_vdpu${VIRTUAL_DPUS}_nprobe${nprobe}_ef${EF}_postef${POST_EF}_qpp${qpp}.out"
        if [ -f "$f" ]; then
            echo "  [OK] $f"
        else
            echo "  [MISSING] $f"
        fi
    done
done
