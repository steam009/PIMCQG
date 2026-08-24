# Makefile for reproduce_cpp (C/C++ version, supports DPU)

# ========== Compiler Settings ==========
CC = gcc
CXX = g++

# DPU compiler (for CPU-side DPU API)
DPU_CC = gcc
DPU_CFLAGS = $(shell dpu-pkg-config --cflags dpu)
DPU_LDFLAGS = $(shell dpu-pkg-config --libs dpu)

# Standard compile flags
CFLAGS = -std=c11 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -I./include
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -I./include
LDFLAGS = -lm -fopenmp

# ========== DPU Parameter Configuration ==========
NR_DPUS ?= 1
NR_TASKLETS ?= 16
BATCH_SIZE ?= 64
DIMM ?= 128
MAX_SIZE_PER_DPU ?= 1024
DEGREE ?= 32
MAX_QUERY_CLUSTER_PER_DPU ?= 32
EF ?= 400
POST_EF ?= 100
TOPK ?= 10

# ========== Build Directory ==========
BUILD_DIR = build

# ========== CPU source files (non-DPU version) ==========
CPU_SRCS = src/main.c src/utils.c src/ivf_search.c src/hashtable.c src/error.c
CPU_OBJS = $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(CPU_SRCS))

# ========== CPU source files (with DPU support) ==========
CPU_DPU_SRCS = src/main.c src/utils.c src/ivf_search.c src/hashtable.c src/error.c
CPU_DPU_OBJS = $(patsubst src/%.c,$(BUILD_DIR)/dpu_cpu/%.o,$(CPU_DPU_SRCS))

# Example program source files
EXAMPLE_SRCS = example_usage.c src/utils.c src/ivf_search.c src/hashtable.c src/error.c
EXAMPLE_OBJS = $(patsubst %.c,$(BUILD_DIR)/example/%.o,$(EXAMPLE_SRCS))

# ========== Target Files ==========
# Original CPU target
TARGET = bin/reproduce_c
EXAMPLE_TARGET = bin/example_usage

# New DPU-related targets
DPU_CODE = bin/dpu_code
HOST_CODE = bin/host_code
HOST_CODE_ASM = bin/host_code.S

# ========== Default Target ==========
all: $(TARGET)

# Build version with DPU support (HOST_CODE_ASM depends on HOST_CODE, DPU_CODE compiled independently)
dpu: $(HOST_CODE_ASM) $(DPU_CODE)

# ========== Main program (without DPU) ==========
$(TARGET): $(CPU_OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build completed: $(TARGET)"

# ========== HOST code (with DPU support) ==========
$(HOST_CODE): $(CPU_DPU_OBJS)
	@mkdir -p bin
	$(DPU_CC) $(CFLAGS) $(DPU_CFLAGS) -o $@ $^ $(LDFLAGS) $(DPU_LDFLAGS)
	@echo "Build completed: $(HOST_CODE)"

# Generate HOST code assembly (disassemble from final binary)
$(HOST_CODE_ASM): $(HOST_CODE)
	@mkdir -p bin
	objdump -d -S $< > $@
	@echo "Generated assembly: $(HOST_CODE_ASM)"

# Compile CPU object files with DPU support
$(BUILD_DIR)/dpu_cpu/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(DPU_CC) $(CFLAGS) $(DPU_CFLAGS) \
		-DNR_DPUS=$(NR_DPUS) \
		-DNR_TASKLETS=$(NR_TASKLETS) \
		-DBATCH_SIZE=$(BATCH_SIZE) \
		-DDIMM=$(DIMM) \
		-DPADDED_DIMM=$(PADDED_DIMM) \
		-DMAX_SIZE_PER_DPU=$(MAX_SIZE_PER_DPU) \
		-DDEGREE=$(DEGREE) \
		-DMAX_QUERY_CLUSTER_PER_DPU=$(MAX_QUERY_CLUSTER_PER_DPU) \
		-DEF=$(EF) \
		-DPOST_EF=$(POST_EF) \
		-DTOPK=$(TOPK) \
		-c $< -o $@

# ========== DPU Code ==========
$(DPU_CODE):
	@mkdir -p bin
	@echo "Compiling DPU code..."
	$(MAKE) -C dpu \
		NR_DPUS=$(NR_DPUS) \
		NR_TASKLETS=$(NR_TASKLETS) \
		BATCH_SIZE=$(BATCH_SIZE) \
		DIMM=$(DIMM) \
		PADDED_DIMM=$(PADDED_DIMM) \
		MAX_SIZE_PER_DPU=$(MAX_SIZE_PER_DPU) \
		DEGREE=$(DEGREE) \
		MAX_QUERY_CLUSTER_PER_DPU=$(MAX_QUERY_CLUSTER_PER_DPU) \
		EF=$(EF) \
		POST_EF=$(POST_EF) \
		TOPK=$(TOPK)

# ========== Example Program ==========
example: $(EXAMPLE_TARGET)

$(EXAMPLE_TARGET): $(EXAMPLE_OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build completed: $(EXAMPLE_TARGET)"

$(BUILD_DIR)/example/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/example/src/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# ========== Standard CPU Object File Compilation Rules ==========
$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# ========== Clean ==========
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(CPU_OBJS) $(EXAMPLE_OBJS) $(TARGET) $(EXAMPLE_TARGET)
	$(MAKE) -C dpu clean
	@echo "Clean completed"

# Clean all results
clean-all: clean
	rm -rf bin results
	@echo "Clean all completed"

# ========== Run Programs ==========
run: $(TARGET)
	cd .. && ./reproduce_cpp/$(TARGET)

run-dpu: $(HOST_CODE) $(DPU_CODE)
	cd .. && DPU_BINARY=./reproduce_cpp/$(DPU_CODE) ./reproduce_cpp/$(HOST_CODE)

# Run example
run-example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

# ========== Memory Check ==========
valgrind: $(TARGET)
	cd .. && valgrind --leak-check=full --show-leak-kinds=all ./reproduce_cpp/$(TARGET)

valgrind-example: $(EXAMPLE_TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(EXAMPLE_TARGET)

# ========== Help Info ==========
help:
	@echo "Available targets:"
	@echo "  all            - Build main program (default, without DPU)"
	@echo "  dpu            - Build version with DPU support (host_code + host_code.S + dpu_code)"
	@echo "  example        - Build example program"
	@echo "  clean          - Remove object files and executables"
	@echo "  clean-all      - Remove all generated files (including results)"
	@echo "  run            - Build and run main program"
	@echo "  run-dpu        - Build and run DPU version"
	@echo "  run-example    - Build and run example program"
	@echo "  valgrind       - Check main program memory with valgrind"
	@echo "  valgrind-example - Check example memory with valgrind"
	@echo "  help           - Show this help info"
	@echo ""
	@echo "DPU output files:"
	@echo "  bin/host_code   - Host-side executable"
	@echo "  bin/host_code.S - Host-side assembly code"
	@echo "  bin/dpu_code    - DPU executable"
	@echo ""
	@echo "DPU parameters (can be overridden via command line):"
	@echo "  NR_DPUS=$(NR_DPUS)"
	@echo "  NR_TASKLETS=$(NR_TASKLETS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  DIMM=$(DIMM)"
	@echo "  PADDED_DIMM=$(PADDED_DIMM)"
	@echo "  DEGREE=$(DEGREE)"
	@echo "  EF=$(EF)"
	@echo "  POST_EF=$(POST_EF)"
	@echo ""
	@echo "Example: make dpu NR_DPUS=4 NR_TASKLETS=16"

.PHONY: all dpu example clean clean-all run run-dpu run-example valgrind valgrind-example help