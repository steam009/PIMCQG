#include <perfcounter.h>
#include <stdio.h>
#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <sem.h>
#include <barrier.h>
#include <mram_unaligned.h>
#include <wramfifo.h>
#include "../support/common.h"
#include "buffer.h"
#include "../include/fifo_types.h"

#if BATCH_SIZE < 8
#define BATCH_SIZE 32
#endif

#define LOWBIT(x) ((x) & (-(x)))

/* ============================================
 * Custom FIFO init macros (optimized for process_inputs_all_tasklets)
 *
 * Original INPUT_FIFO_INIT / OUTPUT_FIFO_INIT allocate tmp_data per
 *   NR_TASKLETS × DATA_SIZE
 * tasklet, designed for process_inputs_each_tasklet (each tasklet processes one input independently).
 *
 * However, process_inputs_all_tasklets implementation (wramfifo.c):
 *   - input:  uses ring buffer pointer from input_fifo_peek() directly, never accesses tmp_data
 *             → input tmp_data is completely wasted (originally NR_TASKLETS×INPUT_DATA_SIZE bytes)
 *   - output: only tasklet 0 calls reduce, writes to tmp_data copy 0
 *             → output tmp_data only needs 1 copy (originally NR_TASKLETS×OUTPUT_DATA_SIZE, N-1 copies wasted)
 *
 * Optimization effect (NR_TASKLETS=11, INPUT_DATA_SIZE=1360, OUTPUT_DATA_SIZE=2600):
 *   save input  tmp_data: 11×1360 - 8      = 14,952 B
 *   save output tmp_data: 11×2600 - 2600   = 26,000 B
 *   total saved ≈ 40,952 B
 * ============================================ */

/* Input FIFO: tmp_data occupies 8-byte placeholder (not used by process_inputs_all_tasklets) */
#define INPUT_FIFO_INIT_ALL(name, PTR_SIZE, DATA_SIZE)                              \
    _Static_assert((PTR_SIZE) <= 10,                                                \
        "wram fifo error: invalid input pointer size, should be <= 10");            \
    _Static_assert((DATA_SIZE) && ((DATA_SIZE) & 7) == 0,                          \
        "wram fifo error: invalid input data size, must be > 0 and multiple of 8");\
    __host uint8_t __get_fifo_data_name(name)[(1 << (PTR_SIZE)) * (DATA_SIZE)];    \
    __dma_aligned uint8_t __get_fifo_tmp_data_name(name)[8];                        \
    __host struct dpu_input_fifo_t name                                             \
        = { 0, 0, __get_fifo_data_name(name), __get_fifo_tmp_data_name(name),      \
            (PTR_SIZE), (DATA_SIZE) };

/* Output FIFO: tmp_data allocated 1 copy (only for tasklet 0 reduce write) */
#define OUTPUT_FIFO_INIT_ALL(name, PTR_SIZE, DATA_SIZE)                             \
    _Static_assert((PTR_SIZE) <= 10,                                                \
        "wram fifo error: invalid output pointer size, should be <= 10");           \
    _Static_assert((DATA_SIZE) && ((DATA_SIZE) & 7) == 0,                          \
        "wram fifo error: invalid output data size, must be > 0 and multiple of 8");\
    __host uint8_t __get_fifo_data_name(name)[(1 << (PTR_SIZE)) * (DATA_SIZE)];    \
    __dma_aligned uint8_t __get_fifo_tmp_data_name(name)[(DATA_SIZE)];              \
    __host struct dpu_output_fifo_t name                                            \
        = { 0, 0, __get_fifo_data_name(name), __get_fifo_tmp_data_name(name),      \
            (PTR_SIZE), (DATA_SIZE), 0 };

/* ============================================
 * FIFO initialization - transfer complete data directly to WRAM
 * Each FIFO entry is now fifo_batch_input_t (containing QUERIES_PER_PUSH queries)
 * ============================================ */
INPUT_FIFO_INIT_ALL(input_fifo, INPUT_FIFO_PTR_SIZE, INPUT_FIFO_DATA_SIZE);
OUTPUT_FIFO_INIT_ALL(output_fifo, OUTPUT_FIFO_PTR_SIZE, OUTPUT_FIFO_DATA_SIZE);

/* Loop control flag - DPU exits loop when CPU sets this to 0 */
__host volatile uint64_t active = 1;

/* ============================================
 * Data storage (transferred from CPU)
 * ============================================ */
__host PID entry_point[128];
__host int32_t centroid_ids[128];
__host int32_t cluster_offset[128];
__mram_noinit PID neighbor_id[MAX_SIZE_PER_DPU * DEGREE];
__mram_noinit uint64_t rabit_code[(PADDED_DIMM >> 6) * MAX_SIZE_PER_DPU];
__mram_noinit int32_t factor[MAX_SIZE_PER_DPU];

/* ============================================
 * WRAM buffer storage
 * ============================================ */
__dma_aligned ResultID result_ids[POST_EF];
__dma_aligned int32_t result_dists[POST_EF];

/* HashBasedBooleanSet storage
 * Capacity must cover all distinct nodes visited during beam search.
 * Worst case: EF * DEGREE ≈ 100 * 32 = 3200 distinct nodes examined.
 * 4096 main slots (power-of-2, ~16 KB) with load factor ≈ 0.78 gives very
 * few collisions; 512 overflow slots ensure hashset_set never silently fails.
 * Silent failure → node not marked visited → re-selected indefinitely → ∞ loop.
 */
__dma_aligned PID hash_table[1024];
__dma_aligned PID overflow_table[512];

__dma_aligned int32_t cur_appro_dist[DEGREE];
__dma_aligned PID cur_neighbor[DEGREE];
__dma_aligned result_buffer_t result_buf;
__dma_aligned hashset_t visited_set;

/* tasklet-local buffers for lock-free accumulation */
#define LOCAL_BUF_CAPACITY ((EF + NR_TASKLETS - 1) / NR_TASKLETS)
__dma_aligned candidate_t local_buf_data[NR_TASKLETS][LOCAL_BUF_CAPACITY];
__dma_aligned search_buffer_t local_bufs[NR_TASKLETS];

/* Current batch input data pointer (points to FIFO buffer, avoids redundant copy) */
__dma_aligned volatile fifo_batch_input_t* cur_batch_ptr;

/* Current batch output data */
__dma_aligned fifo_batch_output_t batch_output;

/* int16_t LUT computed per query (reused, only one copy needed) */
__dma_aligned int16_t int16_lut[PADDED_DIMM * 4];

/* Synchronization barrier - unified single barrier */
BARRIER_INIT(barrier, NR_TASKLETS);
BARRIER_INIT(barrier_sync, NR_TASKLETS);
BARRIER_INIT(barrier_work, NR_TASKLETS);
BARRIER_INIT(barrier_end, NR_TASKLETS);

/* Shared variables */
__dma_aligned PID cur_id;
__dma_aligned int32_t shared_current_dist;
__dma_aligned volatile int32_t skip_iteration;
__dma_aligned volatile int32_t continue_loop;
__dma_aligned volatile int32_t selected_buf_idx;  /* index of selected buffer */
__dma_aligned volatile int32_t should_abort;  /* flag whether processing should abort */

/* Current query being processed (selected from cur_batch_ptr) */
__dma_aligned volatile fifo_input_data_t* cur_query_ptr;  /* points to cur_batch_ptr->queries[q] */
__dma_aligned int16_t cur_sumq;   /* sumq of current query */
__dma_aligned volatile int32_t cur_centroid_id_shared;   /* centroid index of current query (shared) */
__dma_aligned volatile int32_t cur_cluster_shared;       /* cluster ID of current query (shared) */

/* Performance counters */
__dma_aligned uint32_t ins_search;
__dma_aligned uint32_t ins_neighbor;          /* neighbor compute wall-clock time (measured by tasklet 0) */
__dma_aligned uint32_t ins_result;            /* query_reduce output copy time (tasklet 0) */
__dma_aligned uint32_t ins_check_loop;        /* beam-search loop control time (tasklet 0) */
__dma_aligned uint32_t ins_insert_result_buf; /* dpu_result_buffer_insert time (tasklet 0) */
__dma_aligned uint32_t ins_insert_local_buf;  /* search_buffer_insert aggregate time (summed across tasklets, then aggregated) */
__dma_aligned uint32_t ins_lut_compute;       /* LUT computation time */

/* Per-tasklet parallel-section local counters (avoid contention on shared variables) */
__dma_aligned uint32_t _local_ins_insert_local_buf[NR_TASKLETS];

/* ============================================
 * Context structure for passing data between compute and reduce
 * ============================================ */
typedef struct {
    uint8_t* bytes;
} query_context_t;

__dma_aligned query_context_t query_ctx[NR_TASKLETS];

/* kPos[j] returns the rq component index (i.e. 3-k) for the lowest set bit k of j.
 * Identical to kPos used in pack_float_lut_impl in fastscan_impl.hpp.
 * Before fix: kPos[10]=1, kPos[12]=2, kPos[14]=1 (reversed vs CPU side, causing lut[10↔12] swap). */
static const uint8_t kPos[16] = {3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3};

/* ============================================
 * pack_int16_lut_impl: compute int16_t LUT from rotated_query (int16_t)
 * Logic identical to CPU-side pack_float_lut_impl / pack_lut_impl:
 *   For each group of 4 dimensions (dim/4 groups total), build 16 subset sums.
 * ============================================ */
static void pack_int16_lut_impl(int dim, const int16_t* rq, int16_t* lut) {
    int nb = dim >> 2;  /* dim / 4 groups */
    for (int i = 0; i < nb; i++) {
        lut[0] = 0;
        for (int j = 1; j < 16; j++) {
            lut[j] = lut[j - LOWBIT(j)] + rq[kPos[j]];
        }
        lut += 16;
        rq  += 4;
    }
}


/* ============================================
 * process_single_query: process a single query on the tasklet group.
 * Before call: tasklet 0 must set cur_query_ptr, cur_sumq and complete LUT computation (int16_lut).
 *         visited_set, result_buf, local_bufs already cleared by tasklet 0.
 * After call: results are in result_buf.
 * ============================================ */
static void process_single_query(int t_id, int cur_centroid_id, int cur_cluster,
                                 uint8_t* bytes) {
    uint32_t _ts;   /* tasklet-private timestamp (stack var, independent per tasklet) */

    if (t_id == 0) {
        search_buffer_insert(&local_bufs[0], entry_point[cur_centroid_id],
                             cur_query_ptr->ep_dist);
    }
    barrier_wait(&barrier_sync);

    /* Beam Search main loop
     * MAX_BEAM_ITERS: safety upper bound to prevent infinite loops caused by
     * hashset overflow (silent failure when overflow_table is full causes nodes
     * to be re-selected indefinitely). With EF=100 and DEGREE=32 the search
     * visits at most ~EF*DEGREE nodes in normal operation; multiply by 4 for
     * a generous budget that still terminates in bounded time.
     */
    const int MAX_BEAM_ITERS = EF * 8;
    int beam_iter = 0;
    while (1) {
        if (beam_iter++ >= MAX_BEAM_ITERS) {
            /* Hashset overflow caused re-visitation loop; force exit. */
            if (t_id == 0) continue_loop = 0;
            barrier_wait(&barrier_sync);
            break;
        }
        /* ---- ins_check_loop timing (tasklet 0 only, serial section) ---- */
        if (t_id == 0) {
            _ts = perfcounter_get();
            /* Find element with smallest distance across all local_bufs */
            int32_t min_dist = 0x7FFFFFFF;
            int min_buf_idx = -1;

            for (int b = 0; b < NR_TASKLETS; b++) {
                if (search_buffer_has_next(&local_bufs[b])) {
                    DIST next_dist = local_bufs[b].data[local_bufs[b].cur].distance;
                    if (next_dist < min_dist) {
                        min_dist = next_dist;
                        min_buf_idx = b;
                    }
                }
            }

            if (min_buf_idx >= 0) {
                continue_loop = 1;
                selected_buf_idx = min_buf_idx;
                shared_current_dist = local_bufs[min_buf_idx].data[local_bufs[min_buf_idx].cur].distance;
                cur_id = local_bufs[min_buf_idx].data[local_bufs[min_buf_idx].cur].id;

                if (hashset_get(&visited_set, cur_id)) {
                    skip_iteration = 1;
                } else {
                    skip_iteration = 0;
                    hashset_set(&visited_set, cur_id);
                }
            } else {
                continue_loop = 0;
            }
            ins_check_loop += perfcounter_get() - _ts;
        }
        barrier_wait(&barrier_sync);

        if (!continue_loop) {
            break;
        }

        /* Let the selected tasklet clear its own buffer */
        PID next_id = local_bufs[t_id].data[local_bufs[t_id].cur].id;
        if (next_id == cur_id) {
            search_buffer_pop_with_dist(&local_bufs[t_id], &shared_current_dist);
        }

        if (skip_iteration) {
            barrier_wait(&barrier_work);
            barrier_wait(&barrier_end);
            continue;
        }

        int cluster_begin = cluster_offset[cur_centroid_id];
        if (t_id == 0) {
            /* ---- ins_insert_result_buf timing (tasklet 0 only) ---- */
            _ts = perfcounter_get();
            dpu_result_buffer_insert(&result_buf, cur_id, shared_current_dist, cur_cluster);
            ins_insert_result_buf += perfcounter_get() - _ts;

            mram_read(&neighbor_id[cur_id * DEGREE + cluster_begin * DEGREE],
                      &cur_neighbor, ALIGN(DEGREE * sizeof(PID), 8));
        }
        barrier_wait(&barrier_work);

        /* ---- ins_neighbor wall-clock timing (tasklet 0 measures entire parallel section) ---- */
        if (t_id == 0) _ts = perfcounter_get();

        for (int k = t_id; k < DEGREE; k += NR_TASKLETS) {
            if (hashset_get(&visited_set, cur_neighbor[k])) {
                continue;
            }
            uint64_t cur_rabit_code[PADDED_DIMM >> 6];
            mram_read(&rabit_code[cur_neighbor[k] * (PADDED_DIMM >> 6) +
                                  cluster_begin * (PADDED_DIMM >> 6)],
                      &cur_rabit_code, ALIGN((PADDED_DIMM >> 6) * sizeof(uint64_t), 8));
            int32_t cur_factor[2];
            mram_read(&factor[cur_neighbor[k] + cluster_begin], &cur_factor,
                      ALIGN(sizeof(int32_t), 8));

            int lut_base = 0;
            int accumulator = 0;
            for (int l = 0; l < (PADDED_DIMM >> 6); l++) {
                uint64_t code = cur_rabit_code[l];
                for (int b = 0; b < 8; b++) {
                    bytes[b] = (code >> (b * 8)) & 0xFF;
                }
                for (int b = 0; b < 8; ++b) {
                    if (lut_base >= PADDED_DIMM * 4) break;

                    uint8_t val = bytes[b];
                    uint8_t lo4 = val & 0x0F;
                    uint8_t hi4 = (val >> 4) & 0x0F;

                    accumulator += int16_lut[lut_base + lo4];
                    lut_base += 16;
                    if (lut_base < PADDED_DIMM * 4) {
                        accumulator += int16_lut[lut_base + hi4];
                        lut_base += 16;
                    }
                }
            }
            cur_appro_dist[k] = (int32_t)(((accumulator << 1) - cur_sumq) >> 2) +
                                 cur_factor[((cur_neighbor[k] + cluster_begin) % 2 == 1)];
            // printf("cur_cluster: %d, cur_neighbor: %d, cur_appro_dist[%d]: %d\n", cur_cluster, cur_neighbor[k], k, cur_appro_dist[k]);

            if (!search_buffer_is_full(&local_bufs[t_id], cur_appro_dist[k])) {
                /* ---- ins_insert_local_buf timing (per-tasklet, avoids contention) ---- */
                uint32_t _ins = perfcounter_get();
                search_buffer_insert(&local_bufs[t_id], cur_neighbor[k], cur_appro_dist[k]);
                _local_ins_insert_local_buf[t_id] += perfcounter_get() - _ins;
            }
        }
        barrier_wait(&barrier_end);

        if (t_id == 0) ins_neighbor += perfcounter_get() - _ts;
    }

    if (t_id == 0) {
        dpu_result_buffer_set_finished(&result_buf);
    }
}

/* ============================================
 * query_compute: compute function executed by all tasklets
 * Processes QUERIES_PER_PUSH queries in fifo_batch_input_t
 * ============================================ */
void query_compute(uint8_t* input, void* ctx) {
    int t_id = me();
    query_context_t *contexts = (query_context_t*)ctx;
    uint8_t* bytes = contexts[t_id].bytes;
    uint32_t start_search_time = perfcounter_get();

    /* tasklet 0 saves FIFO input pointer and pre-initializes all output slots as invalid (no bulk copy) */
    if (t_id == 0) {
        cur_batch_ptr = (fifo_batch_input_t*)input;
        should_abort = 0;
        /* Pre-initialize all output results as invalid to prevent query_reduce from reading stale data */
        for (int q = 0; q < QUERIES_PER_PUSH; q++) {
            batch_output.results[q].query_id    = cur_batch_ptr->queries[q].query_id;
            batch_output.results[q].centroid_id = (int16_t)cur_batch_ptr->queries[q].centroid_id;
            batch_output.results[q].flag        = 0;
            batch_output.results[q].results[0]  = -1;
        }
    }
    barrier_wait(&barrier_sync);

    /* Process each query in the batch sequentially */
    for (int q = 0; q < QUERIES_PER_PUSH; q++) {
        /* tasklet 0 prepares current query's LUT and state */
        if (t_id == 0) {
            cur_query_ptr = &cur_batch_ptr->queries[q];
            should_abort = 0;

            /* Empty query marker: skip */
            if (cur_query_ptr->query_id == -1) {
                should_abort = 1;
            }
        }
        barrier_wait(&barrier_sync);

        if (should_abort) {
            /* Current slot empty, skip (output already marked invalid in query_reduce) */
            break;
        }

        /* Find cluster index for centroid_id (tasklet 0 handles, result written to shared variables) */
        if (t_id == 0) {
            int32_t cur_cluster = cur_query_ptr->centroid_id;
            cur_cluster_shared  = cur_cluster;
            cur_centroid_id_shared = -1;
            for (int c = 0; c < 128; c++) {
                if (centroid_ids[c] == cur_cluster) {
                    cur_centroid_id_shared = c;
                    break;
                }
            }
            if (cur_centroid_id_shared == -1) {
                printf("cur_centroid_id not found, cur_centroid: %d\n", cur_cluster);
                should_abort = 1;
            }
        }
        barrier_wait(&barrier_sync);

        if (should_abort) {
            continue;
        }

        /* tasklet 0: compute int16_t LUT and clear buffers */
        if (t_id == 0) {
            uint32_t _ts = perfcounter_get();
            /* Build LUT from rotated_query */
            pack_int16_lut_impl(PADDED_DIMM,
                                (const int16_t*)cur_query_ptr->rotated_query,
                                int16_lut);
            /* Save sumq (use incoming value directly) */
            cur_sumq = cur_query_ptr->sumq;

            /* Clear search state */
            dpu_result_buffer_clear(&result_buf);
            hashset_clear(&visited_set);
            ins_lut_compute += perfcounter_get() - _ts;
        }
        /* Each tasklet clears its own local_buf */
        search_buffer_clear(&local_bufs[t_id]);
        barrier_wait(&barrier_sync);

        /* Execute single-query beam search (all tasklets use shared centroid info) */
        process_single_query(t_id, cur_centroid_id_shared, cur_cluster_shared, bytes);

        /* tasklet 0 stores result in batch_output immediately after beam search completes */
        if (t_id == 0) {
            fifo_query_output_t *qout = &batch_output.results[q];
            qout->query_id    = cur_query_ptr->query_id;
            qout->centroid_id = (int16_t)cur_query_ptr->centroid_id;
            qout->flag        = 1;
            int copy_count = (int)result_buf.size < POST_EF ? (int)result_buf.size : POST_EF;
            for (int i = 0; i < copy_count; i++) {
                qout->results[i] = result_ids[i].id;
            }
            if (copy_count < POST_EF) {
                qout->results[copy_count] = -1;
            }
        }
        barrier_wait(&barrier_sync);
    }
    if (t_id == 0) {
        ins_search += perfcounter_get() - start_search_time;
    }
}

/* ============================================
 * query_reduce: reduction function executed by tasklet 0
 * Copies batch_output to output FIFO buffer
 * ============================================ */
void query_reduce(uint8_t* input, uint8_t* output, void* ctx) {
    /* ---- ins_result timing (only called by tasklet 0, no contention) ---- */
    uint32_t _ts = perfcounter_get();
    fifo_batch_output_t *out = (fifo_batch_output_t*)output;

    /* Copy batch_output from query_compute phase to output FIFO */
    for (int q = 0; q < QUERIES_PER_PUSH; q++) {
        out->results[q] = batch_output.results[q];
    }
    ins_result += perfcounter_get() - _ts;
}

/* ============================================
 * Main function - use process_inputs_all_tasklets to process FIFO
 * ============================================ */
int main() {
    int t_id = me();
    perfcounter_config(COUNT_CYCLES, true);

    /* Initialize */
    if (t_id == 0) {
        mem_reset();
        dpu_result_buffer_init(&result_buf, result_ids, result_dists, POST_EF);
        hashset_init(&visited_set, hash_table, 1024, overflow_table, 512);
    }
    search_buffer_init(&local_bufs[t_id], local_buf_data[t_id], LOCAL_BUF_CAPACITY);
    query_ctx[t_id].bytes = (uint8_t*)mem_alloc(8);
    barrier_wait(&barrier_sync);

    /* Use process_inputs_all_tasklets to handle FIFO loop
     * This function repeatedly fetches data from input FIFO, calls compute and reduce functions,
     * until active flag is set to 0 and input FIFO is empty
     */
    process_inputs_all_tasklets(&input_fifo, &output_fifo,
        query_compute, query_reduce, query_ctx, &barrier, &active);
    

    /* Aggregate per-tasklet parallel-section local counters */
    barrier_wait(&barrier);
    if (t_id == 0) {
        for (int i = 0; i < NR_TASKLETS; i++) {
            ins_insert_local_buf += _local_ins_insert_local_buf[i];
        }

        if(visited_set.overflow_full) {
            printf("hashset overflow\n");
        }
        /* ---- Performance breakdown print ---- */
        printf("=== DPU Performance Breakdown ===\n");
        printf("[total  ] ins_search           : %10u cycles, %.6f s\n",
               ins_search, (double)ins_search / 350000000.0);
        printf("[beam   ] ins_check_loop        : %10u cycles, %.6f s  (beam-search loop select)\n",
               ins_check_loop, (double)ins_check_loop / 350000000.0);
        printf("[beam   ] ins_insert_result_buf : %10u cycles, %.6f s  (result buffer insert)\n",
               ins_insert_result_buf, (double)ins_insert_result_buf / 350000000.0);
        printf("[neigh  ] ins_neighbor          : %10u cycles, %.6f s  (parallel neighbor compute wall)\n",
               ins_neighbor, (double)ins_neighbor / 350000000.0);
        printf("[neigh  ] ins_insert_local_buf  : %10u cycles, %.6f s  (search buffer insert, %d tasklet total)\n",
               ins_insert_local_buf, (double)ins_insert_local_buf / 350000000.0, NR_TASKLETS);
        printf("[reduce ] ins_result            : %10u cycles, %.6f s  (output FIFO copy)\n",
               ins_result, (double)ins_result / 350000000.0);
        printf("[lut    ] ins_lut_compute       : %10u cycles, %.6f s  (LUT computation)\n",
               ins_lut_compute, (double)ins_lut_compute / 350000000.0);
        printf("=================================\n");
    }

    return 0;
}
