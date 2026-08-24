#!/bin/bash

# DPU code compilation script
# Compile only dpu/task.c into DPU executable

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Display help information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Display this help information"
    echo "  -c, --clean             Clean build artifacts"
    echo "  -d, --debug             Enable debug mode"
    echo "  -v, --verbose           Show verbose build information"
    echo ""
    echo "DPU parameters (can be set via environment variables):"
    echo "  NR_DPUS                 Number of DPUs (default: 1)"
    echo "  NR_TASKLETS             Number of tasklets (default: 16)"
    echo "  BATCH_SIZE              Batch size (default: 64)"
    echo "  DIMM                    Dimension (default: 128)"
    echo "  MAX_SIZE_PER_DPU        Max size per DPU (default: 1024)"
    echo "  DEGREE                  Graph degree (default: 32)"
    echo "  MAX_QUERY_CLUSTER_PER_DPU  Max query clusters per DPU (default: 32)"
    echo "  EF                      Search candidate set size (default: 400)"
    echo "  POST_EF                 Post-processing candidate set size (default: 100)"
    echo "  TOPK                    Number of results to return (default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Compile with default parameters"
    echo "  $0 --clean                            # Clean build artifacts"
    echo "  NR_DPUS=4 NR_TASKLETS=16 $0           # Compile with custom parameters"
    echo "  DEBUG=1 $0                            # Enable DEBUG mode compilation"
    echo ""
}

# Parse command line arguments
CLEAN_MODE=0
DEBUG_MODE=0
VERBOSE_MODE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_MODE=1
            shift
            ;;
        -d|--debug)
            DEBUG_MODE=1
            shift
            ;;
        -v|--verbose)
            VERBOSE_MODE=1
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help to view help information"
            exit 1
            ;;
    esac
done

# DPU parameter configuration (can be overridden via environment variables)
NR_DPUS=${NR_DPUS:-1}
NR_TASKLETS=${NR_TASKLETS:-16}
BATCH_SIZE=${BATCH_SIZE:-64}
DIMM=${DIMM:-128}
MAX_SIZE_PER_DPU=${MAX_SIZE_PER_DPU:-1024}
DEGREE=${DEGREE:-32}
MAX_QUERY_CLUSTER_PER_DPU=${MAX_QUERY_CLUSTER_PER_DPU:-32}
EF=${EF:-400}
POST_EF=${POST_EF:-100}
TOPK=${TOPK:-10}

# If debug mode is enabled
if [ $DEBUG_MODE -eq 1 ]; then
    export DEBUG=1
fi

# Display banner
echo "=========================================="
echo "  DPU Code Compilation Script"
echo "=========================================="
echo ""

# Clean mode
if [ $CLEAN_MODE -eq 1 ]; then
    print_info "Cleaning DPU build artifacts..."
    make -C dpu clean
    if [ -f "bin/dpu_code" ]; then
        rm -f bin/dpu_code
        print_success "Deleted bin/dpu_code"
    fi
    print_success "Cleanup complete!"
    exit 0
fi

# Display build configuration
print_info "Build configuration:"
echo "  NR_DPUS:                  $NR_DPUS"
echo "  NR_TASKLETS:              $NR_TASKLETS"
echo "  BATCH_SIZE:               $BATCH_SIZE"
echo "  DIMM:                     $DIMM"
echo "  MAX_SIZE_PER_DPU:         $MAX_SIZE_PER_DPU"
echo "  DEGREE:                   $DEGREE"
echo "  MAX_QUERY_CLUSTER_PER_DPU: $MAX_QUERY_CLUSTER_PER_DPU"
echo "  EF:                       $EF"
echo "  POST_EF:                  $POST_EF"
echo "  TOPK:                     $TOPK"

if [ $DEBUG_MODE -eq 1 ]; then
    echo "  DEBUG:                    Enabled"
fi

echo ""

# Create necessary directories
mkdir -p bin

# Start compilation
print_info "Starting DPU code compilation..."
echo ""

# Set make parameters
MAKE_ARGS=""
if [ $VERBOSE_MODE -eq 0 ]; then
    MAKE_ARGS="-s"
fi

# Compile DPU code
if make $MAKE_ARGS -C dpu \
    NR_DPUS=$NR_DPUS \
    NR_TASKLETS=$NR_TASKLETS \
    BATCH_SIZE=$BATCH_SIZE \
    DIMM=$DIMM \
    MAX_SIZE_PER_DPU=$MAX_SIZE_PER_DPU \
    DEGREE=$DEGREE \
    MAX_QUERY_CLUSTER_PER_DPU=$MAX_QUERY_CLUSTER_PER_DPU \
    EF=$EF \
    POST_EF=$POST_EF \
    TOPK=$TOPK; then
    
    echo ""
    
    # Check generated files
    if [ -f "bin/dpu_code" ]; then
        FILE_SIZE=$(ls -lh bin/dpu_code | awk '{print $5}')
        print_success "DPU code compiled successfully!"
        echo ""
        echo "Generated files:"
        echo "  ✓ bin/dpu_code        ($FILE_SIZE)"
        echo ""
        echo "Usage:"
        echo "  export DPU_BINARY=./reproduce_cpp/bin/dpu_code"
        echo "  ./reproduce_cpp/bin/host_code"
        echo ""
    else
        print_error "Compilation failed: bin/dpu_code not found"
        exit 1
    fi
else
    echo ""
    print_error "DPU code compilation failed!"
    exit 1
fi

print_success "Compilation complete!"

