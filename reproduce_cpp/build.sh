#!/bin/bash

# Build script for reproduce_cpp (DPU + C/C++ hybrid version)

set -e  # Exit on error

# Check if debug mode
DEBUG_MODE=${DEBUG_MODE:-""}
if [[ "$1" == "debug" ]] || [[ "$DEBUG_MODE" == "1" ]]; then
    echo "=== Building reproduce_cpp (DEBUG MODE) ==="
    DEBUG_FLAGS="-g -O0"
    BUILD_TYPE="debug"
else
    echo "=== Building reproduce_cpp (RELEASE MODE) ==="
    DEBUG_FLAGS=""
    BUILD_TYPE="release"
fi
echo "Build type: $BUILD_TYPE"
echo ""

# DPU parameter configuration (can be overridden via environment variables)
NR_DPUS=${NR_DPUS:-600}
VIRTUAL_DPUS=${VIRTUAL_DPUS:-$NR_DPUS}
NR_TASKLETS=${NR_TASKLETS:-1}
BATCH_SIZE=${BATCH_SIZE:-1000}
DIMM=${DIMM:-128}
PADDED_DIMM=${PADDED_DIMM:-128}
MAX_SIZE_PER_DPU=${MAX_SIZE_PER_DPU:-30000}
DEGREE=${DEGREE:-32}
MAX_QUERY_CLUSTER_PER_DPU=${MAX_QUERY_CLUSTER_PER_DPU:-2000}
EF=${EF:-300}
POST_EF=${POST_EF:-300}
TOPK=${TOPK:-100}
QUERIES_PER_PUSH=${QUERIES_PER_PUSH:-}
INPUT_FIFO_PTR_SIZE=${INPUT_FIFO_PTR_SIZE:-}
OUTPUT_FIFO_PTR_SIZE=${OUTPUT_FIFO_PTR_SIZE:-}

echo "DPU Configuration:"
echo "  NR_DPUS: $NR_DPUS"
echo "  VIRTUAL_DPUS: $VIRTUAL_DPUS"
echo "  NR_TASKLETS: $NR_TASKLETS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  DIMM: $DIMM"
echo "  PADDED_DIMM: $PADDED_DIMM"
echo "  DEGREE: $DEGREE"
echo "  MAX_SIZE_PER_DPU: $MAX_SIZE_PER_DPU"
echo "  MAX_QUERY_CLUSTER_PER_DPU: $MAX_QUERY_CLUSTER_PER_DPU"
echo "  EF: $EF"
echo "  POST_EF: $POST_EF"
echo "  TOPK: $TOPK"
if [ -n "$QUERIES_PER_PUSH" ]; then
    echo "  QUERIES_PER_PUSH: $QUERIES_PER_PUSH"
fi
if [ -n "$INPUT_FIFO_PTR_SIZE" ]; then
    echo "  INPUT_FIFO_PTR_SIZE: $INPUT_FIFO_PTR_SIZE"
fi
if [ -n "$OUTPUT_FIFO_PTR_SIZE" ]; then
    echo "  OUTPUT_FIFO_PTR_SIZE: $OUTPUT_FIFO_PTR_SIZE"
fi
echo ""

# Create necessary directories
mkdir -p bin build

# ======== Step 1: Compile DPU code using Makefile ========
echo "=== Step 1: Building DPU code ==="
make -C dpu \
    NR_DPUS=$NR_DPUS \
    NR_TASKLETS=$NR_TASKLETS \
    BATCH_SIZE=$BATCH_SIZE \
    DIMM=$DIMM \
    PADDED_DIMM=$PADDED_DIMM \
    MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU \
    DEGREE=$DEGREE \
    MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
    EF=$EF \
    POST_EF=$POST_EF \
    TOPK=$TOPK \
    ${QUERIES_PER_PUSH:+QUERIES_PER_PUSH=$QUERIES_PER_PUSH} \
    ${INPUT_FIFO_PTR_SIZE:+INPUT_FIFO_PTR_SIZE=$INPUT_FIFO_PTR_SIZE} \
    ${OUTPUT_FIFO_PTR_SIZE:+OUTPUT_FIFO_PTR_SIZE=$OUTPUT_FIFO_PTR_SIZE}

if [ -f "bin/dpu_code" ]; then
    echo "✓ DPU code compiled successfully: bin/dpu_code"
else
    echo "✗ Error: DPU code compilation failed!"
    exit 1
fi
echo ""

# ======== Step 2: Compile Host code using Makefile.mixed (with C++ library) ========
echo "=== Step 2: Building Host code (with symqg C++ library) ==="
if [[ "$BUILD_TYPE" == "debug" ]]; then
    # Get DPU compile options and add debug options
    DPU_CFLAGS_VALUE=$(dpu-pkg-config --cflags dpu 2>/dev/null || echo "")
    # Build DPU parameter macro definitions
    QPP_DEFINE=${QUERIES_PER_PUSH:+" -DQUERIES_PER_PUSH=$QUERIES_PER_PUSH"}
    IFP_DEFINE=${INPUT_FIFO_PTR_SIZE:+" -DINPUT_FIFO_PTR_SIZE=$INPUT_FIFO_PTR_SIZE"}
    OFP_DEFINE=${OUTPUT_FIFO_PTR_SIZE:+" -DOUTPUT_FIFO_PTR_SIZE=$OUTPUT_FIFO_PTR_SIZE"}
    DPU_DEFINES_VALUE="-DNR_DPUS=$NR_DPUS -DVIRTUAL_DPUS=$VIRTUAL_DPUS -DBATCH_SIZE=$BATCH_SIZE -DDIMM=$DIMM -DPADDED_DIMM=$PADDED_DIMM -DMAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU -DDEGREE=$DEGREE -DMAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU -DPOST_EF=$POST_EF -DEF=$EF$QPP_DEFINE$IFP_DEFINE$OFP_DEFINE"
    make -f Makefile.mixed host \
        NR_DPUS=$NR_DPUS \
        VIRTUAL_DPUS=$VIRTUAL_DPUS \
        BATCH_SIZE=$BATCH_SIZE \
        DIMM=$DIMM \
        PADDED_DIMM=$PADDED_DIMM \
        MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU \
        DEGREE=$DEGREE \
        MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
        POST_EF=$POST_EF \
        EF=$EF \
        ${QUERIES_PER_PUSH:+QUERIES_PER_PUSH=$QUERIES_PER_PUSH} \
        ${INPUT_FIFO_PTR_SIZE:+INPUT_FIFO_PTR_SIZE=$INPUT_FIFO_PTR_SIZE} \
        ${OUTPUT_FIFO_PTR_SIZE:+OUTPUT_FIFO_PTR_SIZE=$OUTPUT_FIFO_PTR_SIZE} \
        CFLAGS="-std=c11 -O0 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -I./include -I../symqglib $DPU_CFLAGS_VALUE $DPU_DEFINES_VALUE"
else
    make -f Makefile.mixed host \
        NR_DPUS=$NR_DPUS \
        VIRTUAL_DPUS=$VIRTUAL_DPUS \
        BATCH_SIZE=$BATCH_SIZE \
        DIMM=$DIMM \
        PADDED_DIMM=$PADDED_DIMM \
        MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU \
        DEGREE=$DEGREE \
        MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
        POST_EF=$POST_EF \
        EF=$EF \
        ${QUERIES_PER_PUSH:+QUERIES_PER_PUSH=$QUERIES_PER_PUSH} \
        ${INPUT_FIFO_PTR_SIZE:+INPUT_FIFO_PTR_SIZE=$INPUT_FIFO_PTR_SIZE} \
        ${OUTPUT_FIFO_PTR_SIZE:+OUTPUT_FIFO_PTR_SIZE=$OUTPUT_FIFO_PTR_SIZE}
fi

if [ -f "bin/host_code" ]; then
    echo "✓ Host code compiled successfully: bin/host_code"
else
    echo "✗ Error: Host code compilation failed!"
    exit 1
fi

if [ -f "bin/host_code.S" ]; then
    echo "✓ Host assembly generated successfully: bin/host_code.S"
else
    echo "✗ Warning: Host assembly file not generated"
fi
echo ""

# ======== Step 3: Optional - Compile standard version using Makefile.mixed ========
echo "=== Step 3: Building standard version (optional) ==="
make -f Makefile.mixed all \
    NR_DPUS=$NR_DPUS \
    VIRTUAL_DPUS=$VIRTUAL_DPUS \
    BATCH_SIZE=$BATCH_SIZE \
    DIMM=$DIMM \
    PADDED_DIMM=$PADDED_DIMM \
    MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU \
    DEGREE=$DEGREE \
    MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
    POST_EF=$POST_EF \
    EF=$EF \
    ${QUERIES_PER_PUSH:+QUERIES_PER_PUSH=$QUERIES_PER_PUSH} \
    ${INPUT_FIFO_PTR_SIZE:+INPUT_FIFO_PTR_SIZE=$INPUT_FIFO_PTR_SIZE} \
    ${OUTPUT_FIFO_PTR_SIZE:+OUTPUT_FIFO_PTR_SIZE=$OUTPUT_FIFO_PTR_SIZE}

if [ -f "bin/reproduce_c" ]; then
    echo "✓ Standard version compiled successfully: bin/reproduce_c"
else
    echo "✗ Warning: Standard version compilation failed (non-critical)"
fi
echo ""

# ======== Build Complete Summary ========
echo "==================================================================="
echo "Build completed successfully!"
echo "==================================================================="
echo ""
echo "Generated files:"
if [ -f "bin/dpu_code" ]; then
    echo "  ✓ bin/dpu_code       - DPU executable"
fi
if [ -f "bin/host_code" ]; then
    echo "  ✓ bin/host_code      - Host executable (with symqg C++ library)"
fi
if [ -f "bin/host_code.S" ]; then
    echo "  ✓ bin/host_code.S    - Host assembly code"
fi
if [ -f "bin/reproduce_c" ]; then
    echo "  ✓ bin/reproduce_c    - Standard version"
fi
echo ""
echo "To run the DPU version:"
echo "  export DPU_BINARY=./reproduce_cpp/bin/dpu_code"
echo "  cd .."
echo "  ./reproduce_cpp/bin/host_code"
echo ""
echo "To clean build artifacts:"
echo "  make -f Makefile clean"
echo "  make -f Makefile.mixed clean"
echo ""