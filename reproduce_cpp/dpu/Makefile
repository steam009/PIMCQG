# DPU Makefile - compile task.c into DPU executable
# Called by the main Makefile

# Compiler and flags
CC           = dpu-upmem-dpurte-clang
CFLAGS       = --std=c11 -O2 -flto -Wall -Wextra -Wno-unused-parameter

# If debug info is needed
ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG
endif

# If test mode is needed
ifeq ($(TEST), 1)
	CFLAGS += -DTEST
endif

# If DPU logs need to be displayed
ifeq ($(SHOW_DPU_LOGS), 1)
	CFLAGS += -DSHOW_DPU_LOGS
endif

# Source files and target
APP_MAIN     = task.c

# Target binary filename
DPU_TARGET   = ../bin/dpu_code

# Include paths
INC          = -I../include -I../support

# Add required compile-time definitions
CFLAGS      += -DNR_DPUS=$(NR_DPUS) \
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
               $(if $(QUERIES_PER_PUSH),-DQUERIES_PER_PUSH=$(QUERIES_PER_PUSH),) \
               $(if $(INPUT_FIFO_PTR_SIZE),-DINPUT_FIFO_PTR_SIZE=$(INPUT_FIFO_PTR_SIZE),) \
               $(if $(OUTPUT_FIFO_PTR_SIZE),-DOUTPUT_FIFO_PTR_SIZE=$(OUTPUT_FIFO_PTR_SIZE),)

# Default target
all: $(DPU_TARGET)

# Create DPU executable
$(DPU_TARGET): $(APP_MAIN)
	@mkdir -p ../bin
	@echo "compile DPU code: $@"
	$(CC) $(CFLAGS) $(INC) -o $@ $<

# Clean
clean:
	rm -f $(DPU_TARGET)
	@echo "DPU clear finished"

.PHONY: all clean

