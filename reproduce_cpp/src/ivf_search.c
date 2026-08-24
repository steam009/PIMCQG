#include "../include/ivf_search.h"
#include "../include/settings.h"
#include "../include/symqg_c_binding.h"
#include "../include/error.h"
#include "../support/timer.h"
#include "../include/buffer.h"
#include "../include/fifo_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
/* Explicitly declare usleep to avoid implicit declaration warnings (some standards modes don't provide the prototype in <unistd.h>) */
extern int usleep(unsigned int usec);

/* =============== IndexWrapper Implementation =============== */

IndexWrapper* index_wrapper_create(const char* index_type,
                                  const char* metric,
                                  size_t num_elements,
                                  size_t dimension,
                                  size_t degree_bound,
                                  int allocate_vectors) {
    if (strcmp(metric, "L2") != 0) {
        SET_ERROR(ERR_UNSUPPORTED, "Only L2 distance supported currently");
        return NULL;
    }

    if (degree_bound < 32 || degree_bound % 32 != 0) {
        SET_ERROR(ERR_INVALID_PARAM, "The degree bound must be a multiple of 32");
        return NULL;
    }

    IndexWrapper* wrapper = (IndexWrapper*)malloc(sizeof(IndexWrapper));
    CHECK_NULL_RETURN(wrapper, NULL, "Failed to allocate IndexWrapper");

    wrapper->cluster_id = 0;

    if (strcmp(index_type, "QG") == 0) {
        /* Use C binding to create QuantizedGraph */
        wrapper->index = qg_create(num_elements, degree_bound, dimension, allocate_vectors);
        if (!wrapper->index) {
            SET_ERROR(ERR_OPERATION_FAILED, "Failed to create QuantizedGraph");
            free(wrapper);
            return NULL;
        }
    } else {
        SET_ERROR(ERR_UNSUPPORTED, "Index type [%s] not supported", index_type);
        free(wrapper);
        return NULL;
    }
    
    return wrapper;
}

void index_wrapper_free(IndexWrapper* wrapper) {
    if (wrapper) {
        if (wrapper->index) {
            qg_free(wrapper->index);
        }
        free(wrapper);
    }
}

int index_wrapper_load(IndexWrapper* wrapper, const char* filename) {
    if (!wrapper || !wrapper->index) {
        SET_ERROR(ERR_INVALID_PARAM, "Invalid wrapper or index");
        return -1;
    }
    
    if (qg_load_index(wrapper->index, filename) != 0) {
        SET_ERROR(ERR_FILE_READ, "Failed to load index from %s", filename);
        return -1;
    }
    return 0;
}

int index_wrapper_save(IndexWrapper* wrapper, const char* filename) {
    if (!wrapper || !wrapper->index) {
        SET_ERROR(ERR_INVALID_PARAM, "Invalid wrapper or index");
        return -1;
    }
    
    if (qg_save_index(wrapper->index, filename) != 0) {
        SET_ERROR(ERR_FILE_WRITE, "Failed to save index to %s", filename);
        return -1;
    }
    return 0;
}

void index_wrapper_set_ef(IndexWrapper* wrapper, size_t ef_search) {
    if (wrapper && wrapper->index) {
        qg_set_ef(wrapper->index, ef_search);
    }
}

void index_wrapper_set_cluster(IndexWrapper* wrapper, size_t c_id) {
    if (wrapper) {
        wrapper->cluster_id = c_id;
        if (wrapper->index) {
            qg_set_cluster(wrapper->index, c_id);
        }
    }
}

void index_wrapper_set_post_ef(IndexWrapper* wrapper, size_t ef_search) {
    if (wrapper && wrapper->index) {
        qg_set_post_ef(wrapper->index, ef_search);
    }
}

void index_wrapper_enable_profiling(IndexWrapper* wrapper, int enable) {
    if (wrapper && wrapper->index) {
        qg_enable_profiling(wrapper->index, enable);
    }
}

void index_wrapper_get_lut_sumq(IndexWrapper* wrapper, const float* query, float* lut, float* sumq) {
    if (wrapper && wrapper->index) {
        qg_get_lut_sumq(wrapper->index, query, lut, sumq);
    }
}

void index_wrapper_get_rotated_query(IndexWrapper* wrapper, const float* query,
                                     float* rotated_query_out, float* sumq_out) {
    if (wrapper && wrapper->index) {
        qg_get_rotated_query(wrapper->index, query, rotated_query_out, sumq_out);
    }
}

void index_wrapper_report_timings(IndexWrapper* wrapper) {
    if (wrapper && wrapper->index) {
        qg_report_timings(wrapper->index);
    }
}

IntArray* index_wrapper_search(IndexWrapper* wrapper, const FloatArray* query, int k) {
    if (!wrapper || !wrapper->index) {
        SET_ERROR(ERR_INVALID_PARAM, "Invalid wrapper or index");
        return int_array_create(0);
    }
    
    if (!query || !query->data) {
        SET_ERROR(ERR_INVALID_PARAM, "Invalid query vector");
        return int_array_create(0);
    }
    
    /* Allocate result array */
    uint32_t* results = (uint32_t*)malloc(k * sizeof(uint32_t));
    if (!results) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate results array");
        return int_array_create(0);
    }
    
    /* Call underlying search function; the last parameter should pass ep_dist */
    int num_results = qg_search(wrapper->index, query->data, (uint32_t)k, results, 0);
    if (num_results < 0) {
        SET_ERROR(ERR_OPERATION_FAILED, "Search operation failed");
        free(results);
        return int_array_create(0);
    }
    
    /* Convert result format */
    IntArray* result_vec = int_array_create(num_results);
    for (int i = 0; i < num_results; i++) {
        int_array_push(result_vec, (int)results[i]);
    }
    
    free(results);
    return result_vec;
}

size_t index_wrapper_get_num_elements(const IndexWrapper* wrapper) {
    if (!wrapper || !wrapper->index) {
        return 0;
    }
    return qg_get_num_elements(wrapper->index);
}

size_t index_wrapper_get_dimension(const IndexWrapper* wrapper) {
    if (!wrapper || !wrapper->index) {
        return 0;
    }
    return qg_get_dimension(wrapper->index);
}

const float* index_wrapper_get_vector(const IndexWrapper* wrapper, uint32_t data_id) {
    if (!wrapper || !wrapper->index) {
        return NULL;
    }
    return qg_get_vector(wrapper->index, data_id);
}

const uint64_t* index_wrapper_get_packed_code(const IndexWrapper* wrapper, uint32_t data_id) {
    if (!wrapper || !wrapper->index) {
        return NULL;
    }
    return qg_get_packed_code(wrapper->index, data_id);
}

const float* index_wrapper_get_factor(const IndexWrapper* wrapper, uint32_t data_id) {
    if (!wrapper || !wrapper->index) {
        return NULL;
    }
    return qg_get_factor(wrapper->index, data_id);
}

const uint32_t* index_wrapper_get_neighbors(const IndexWrapper* wrapper, uint32_t data_id) {
    if (!wrapper || !wrapper->index) {
        return NULL;
    }
    return qg_get_neighbors(wrapper->index, data_id);
}

uint32_t index_wrapper_get_entry_point(const IndexWrapper* wrapper) {
    if (!wrapper || !wrapper->index) {
        return 0;
    }
    return qg_get_entry_point(wrapper->index);
}

/* =============== IVFSearcher Implementation =============== */

IVFSearcher* ivf_searcher_create(void) {
    IVFSearcher* searcher = (IVFSearcher*)malloc(sizeof(IVFSearcher));
    if (!searcher) return NULL;
    
    searcher->indices = NULL;
    searcher->num_indices = 0;
    searcher->centroids = NULL;
    searcher->cluster_to_original = NULL;
    searcher->base_data = NULL;
    searcher->total_num_elements = 0;
    /* Initialize lazy-loading related fields */
    searcher->index_paths = NULL;
    searcher->index_num_elements = NULL;
    searcher->index_degree = 0;
    return searcher;
}

void ivf_searcher_free(IVFSearcher* searcher) {
    if (searcher) {
        if (searcher->indices) {
            for (int i = 0; i < searcher->num_indices; i++) {
                if (searcher->indices[i]) {
                    index_wrapper_free(searcher->indices[i]);
                }
            }
            free(searcher->indices);
        }
        /* Free lazy-loading related fields */
        if (searcher->index_paths) {
            for (int i = 0; i < searcher->num_indices; i++) {
                if (searcher->index_paths[i]) {
                    free(searcher->index_paths[i]);
                }
            }
            free(searcher->index_paths);
        }
        if (searcher->index_num_elements) {
            free(searcher->index_num_elements);
        }
        if (searcher->centroids) {
            float_matrix_free(searcher->centroids);
        }
        if (searcher->cluster_to_original) {
            cluster_mapping_table_free(searcher->cluster_to_original);
        }
        if (searcher->base_data) {
            float_matrix_free(searcher->base_data);
        }
        free(searcher);
    }
}

int ivf_searcher_load_centroids(IVFSearcher* searcher, const char* centroids_path) {
    if (!searcher) return -1;
    
    /* Free existing centroids */
    if (searcher->centroids) {
        float_matrix_free(searcher->centroids);
        searcher->centroids = NULL;
    }
    
    searcher->centroids = read_fvecs(centroids_path);
    if (!searcher->centroids) {
        return -1;
    }
    return 0;
}

int ivf_searcher_load_cluster_mapping(IVFSearcher* searcher, const char* mapping_path) {
    if (!searcher) return -1;
    
    /* Free existing cluster mapping */
    if (searcher->cluster_to_original) {
        cluster_mapping_table_free(searcher->cluster_to_original);
        searcher->cluster_to_original = NULL;
    }
    
    searcher->cluster_to_original = load_mapping(mapping_path);
    if (!searcher->cluster_to_original) {
        return -1;
    }
    return 0;
}

int ivf_searcher_load_base_data(IVFSearcher* searcher, const char* base_path) {
    if (!searcher) return -1;

    /* Free existing base data */
    if (searcher->base_data) {
        float_matrix_free(searcher->base_data);
        searcher->base_data = NULL;
    }

    /* Detect format by filename: SPACE1B directory, u8bin, or bvecs (SIFT1B) */
    const char* basename = strrchr(base_path, '/');
    basename = basename ? basename + 1 : base_path;
    const char* ext = strrchr(base_path, '.');
    if (strcmp(basename, "vectors.bin") == 0) {
        /* SPACE1B: directory path ending with vectors.bin */
        searcher->base_data = read_space1b_base_as_float(base_path, 1000000000LL);
    } else if (ext && strcmp(ext, ".u8bin") == 0) {
        /* SSN: single-file uint8 database */
        searcher->base_data = read_u8bin_as_float(base_path, 0);
    } else {
        /* SIFT1B: bvecs format (uint8) */
        searcher->base_data = read_bvecs_as_float(base_path);
    }

    if (!searcher->base_data) {
        return -1;
    }
    return 0;
}

/* Helper function: check if a file exists */
static int file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int ivf_searcher_load_indices(IVFSearcher* searcher, const char* dataset, int degree, int C) {
    if (!searcher) return -1;
    
    /* Clean up old indices (already loaded) */
    if (searcher->indices) {
        for (int i = 0; i < searcher->num_indices; i++) {
            if (searcher->indices[i]) {
                index_wrapper_free(searcher->indices[i]);
            }
        }
        free(searcher->indices);
    }
    /* Clean up old lazy-loading metadata */
    if (searcher->index_paths) {
        for (int i = 0; i < searcher->num_indices; i++) {
            if (searcher->index_paths[i]) free(searcher->index_paths[i]);
        }
        free(searcher->index_paths);
    }
    if (searcher->index_num_elements) {
        free(searcher->index_num_elements);
    }
    
    /* Save degree for creating IndexWrapper / paths during lazy loading */
    searcher->index_degree = degree;
    
    /* Allocate index array (all initialized to NULL, lazy-loaded) and metadata arrays
     * Note: index_paths is only used during build phase; paths are rebuilt on demand during lazy loading */
    searcher->num_indices = C;
    searcher->indices = (IndexWrapper**)calloc(C, sizeof(IndexWrapper*));
    searcher->index_paths = (char**)calloc(C, sizeof(char*));
    searcher->index_num_elements = (size_t*)calloc(C, sizeof(size_t));
    if (!searcher->indices || !searcher->index_paths || !searcher->index_num_elements) {
        return -1;
    }
    
    /* SPACE1B uses "IVF-reduceMem" subdirectory; other datasets use "IVF-reduceMem-8192" */
    const char* ivf_subdir = (strcmp(dataset, "SPACE1B") == 0)
                             ? "IVF-reduceMem"
                             : "IVF-reduceMem-8192";
    if (strcmp(dataset, "SSN") == 0) {
        ivf_subdir = "IVF-reduceMem";
    }

    int available_count = 0;
    for (int i = 0; i < C; i++) {
        /* Check if the cluster has enough data points */
        IntArray* indices = cluster_mapping_table_get(searcher->cluster_to_original, i);
        if (indices && indices->size > 1) {
            char index_path[1024];
            snprintf(index_path, sizeof(index_path),
                    "../data/%s/%s/symphonyqg_%d_cluster_%d.index",
                    dataset, ivf_subdir, degree, i);
            
            if (file_exists(index_path)) {
                /* Only record path and element count, do not load index
                 * Note: avoid using strdup directly (may lack prototype under some compile options, causing return value to be truncated to int) */
                size_t len = strlen(index_path);
                char *path_copy = (char*)malloc(len + 1);
                if (!path_copy) {
                    fprintf(stderr, "Warning: failed to allocate memory for index path of cluster %d\n", i);
                    continue;
                }
                memcpy(path_copy, index_path, len + 1);  /* include trailing '\0' */
                searcher->index_paths[i] = path_copy;
                searcher->index_num_elements[i] = indices->size;
                available_count++;
            }
        }
    }
    
    printf("Lazy load mode: %d/%d cluster indices available (not yet loaded, will load on demand)\n",
           available_count, C);
    return 0;
}

/* SearchResult has been replaced by ResultBuffer */

/* Helper structure: for DPU load balancing */
typedef struct {
    int centroid_id;       // original centroid ID
    int replica_id;        // replica number (0, 1, 2, ...)
    int64_t workload;
    int num_elements;
} CentroidInfo;

/* DPU input buffer circular queue structure */
typedef struct {
    fifo_input_data_t* buffer;  /* buffer array */
    int head;                    /* queue head pointer (next position to dequeue) */
    int tail;                    /* queue tail pointer (next position to enqueue) */
    int count;                   /* current number of elements in buffer */
    int capacity;                /* buffer capacity */
} dpu_input_buffer_t;

/* Function declarations */
static float compute_sqr_dist_cpu(const float* query, const float* vector, int dim);

static double batch_push_to_dpus_time = 0.0;

static int total_queries_sent = 0;

/* ============================================================
 * CPU-server baseline for flush_output_fifo_callback (seconds/round).
 *
 * How to obtain this value:
 *   1.  On a DPU-free server, compile and run:
 *         gcc -O2 -o test_flush test_flush_callback.c -lm
 *         ./test_flush [total_rounds] [bench_iters]
 *   2.  Copy the printed "Per-round (parallel, /N ranks) avg" value here
 *       (in seconds).
 *   3.  Define it at compile time:
 *         -DCPU_FLUSH_CALLBACK_TIME_PER_ROUND=0.016059
 *       or edit the default value in this file directly.
 *
 * When this macro is defined (non-zero), ivf_search_batch_fifo() will
 * print a comparison between the measured DPU xfer+callback time and the
 * pure CPU callback computation time.
 * ============================================================ */
/* Uncomment and set after running test_flush_callback on the CPU server: */
// #define CPU_FLUSH_CALLBACK_TIME_PER_ROUND  0.016

/* Timing breakdown statistics variables */
static double lut_sumq_time = 0.0;
static double cluster_check_time = 0.0;
static double replica_selection_time = 0.0;
static double buffer_add_time = 0.0;
static double batch_push_check_time = 0.0;
static double output_flush_time = 0.0;
static double batch_send_time = 0.0;


/* ============================================
 * New architecture: 1 xfer thread + N per-rank processing threads
 *
 *  xfer thread:
 *    - calls dpu_fifo_push_xfer(full dpu_set, ...) to pull all rank outputs
 *    - sem_post wakes all per-rank processing threads (with last-round flag)
 *    - sem_wait waits for all processing threads to finish, then next round
 *
 *  per-rank processing threads (nr_ranks total, pre-created and reused):
 *    - sem_wait waits for xfer thread signal
 *    - calls flush_output_fifo_callback to process this rank's results
 *    - sem_post notifies xfer thread of completion
 *
 * Effect: correctly uses full dpu_set for xfer while preserving per-rank parallel processing,
 *       eliminating per-round thread create/destroy overhead of dpu_callback.
 * ============================================ */

/* ============================================================
 * NUM_OUTPUT_BUFS: pipeline depth for double-buffer output drain.
 *
 * Root cause of DPU_ERR_WRAM_FIFO_FULL:
 *   The xfer thread previously waited T_callback (~16 ms) between
 *   consecutive output flushes.  If DPU produces results faster than
 *   OUTPUT_FIFO_CAPACITY * T_dpu < T_callback, the output FIFO
 *   fills up, DPU stalls, input FIFO backs up, and push fails.
 *
 * Fix: rotate through NUM_OUTPUT_BUFS snapshot buffers.  The xfer
 * thread flushes into buf[round % N] and only waits for the callback
 * that last used THAT buffer (N rounds ago) before reusing it.
 * The output FIFO is therefore drained every T_flush (~1 ms) instead
 * of every T_flush + T_callback (~17 ms).
 *
 * Rule of thumb: NUM_OUTPUT_BUFS >= ceil(T_callback / T_flush).
 * Start with 4; increase if DPU_ERR_WRAM_FIFO_FULL still occurs.
 * ============================================================ */
#define NUM_OUTPUT_BUFS 128

/* per-rank processing thread args (callback only, no xfer) */
typedef struct {
    struct dpu_set_t        rank;               /* rank handled by this thread */
    uint32_t                rank_id;            /* global rank number */

    /* ---- Per-buffer independent semaphores + data (critical design) ----
     *
     * Root cause: using a single sem_start + single cur_buf_idx/stop_after_this,
     * the xfer thread can post multiple times before rank thread processes, overwriting cur_buf_idx/
     * stop_after_this. The rank thread sees only the last write on wakeup, causing:
     *   - get_fifo_size called with wrong buf → sz=0 → queries_sent not decremented
     *   - stop=1 read early → rank thread breaks early → sem_buf_done never posted
     *   - xfer thread shutdown loop waits for sem_buf_done → deadlock
     *
     * Solution: each buf has independent sem_start_buf[b], stop_flag_buf[b], local_args_buf[b].
     * xfer thread posts sem_start_buf[buf]; rank threads wait in round-robin for each buf
     * dedicated semaphore; each sem_wait maps to a unique buf, no overwrite.
     * Also, xfer does sem_wait(sem_buf_done[buf]) before posting when round r >= NUM_OUTPUT_BUFS,
     * ensuring each buf's sem_start_buf count is at most 1,
     * completely eliminating signal accumulation. */
    flush_callback_args_t   local_args_buf[NUM_OUTPUT_BUFS]; /* per-buf independent args (including output_link) */
    sem_t                   sem_start_buf[NUM_OUTPUT_BUFS];  /* per-buf start signal */
    sem_t                   sem_buf_done[NUM_OUTPUT_BUFS];   /* per-buf done signal */
    volatile int            stop_flag_buf[NUM_OUTPUT_BUFS];  /* per-buf stop flag */

} rank_process_args_t;

/* Single xfer thread args */
typedef struct {
    struct dpu_set_t       *dpu_set;            /* full dpu_set (for dpu_fifo_push_xfer) */
    /* ---- New: N output links + buffers ---- */
    struct dpu_fifo_link_t  output_links[NUM_OUTPUT_BUFS]; /* each link pre-prepared to corresponding buf */
    uint8_t                *output_bufs[NUM_OUTPUT_BUFS];  /* N independent snapshot buffers */
    size_t                  buf_bytes;                     /* bytes per buffer */
    rank_process_args_t    *proc_workers;       /* per-rank processing thread array */
    uint32_t                nr_ranks;
    volatile int            stop_flag;          /* set to 1 by main thread → xfer thread exits after final round */
    volatile int            already_stopped;
    double                  elapsed_time;       /* xfer thread total runtime */
} xfer_worker_args_t;

static rank_process_args_t *g_proc_workers      = NULL;  /* per-rank processing thread arg array */
static pthread_t           *g_proc_threads      = NULL;  /* per-rank processing thread handle array */
static xfer_worker_args_t   g_xfer_worker_args;          /* Single xfer thread args */
static pthread_t            g_xfer_thread;               /* xfer thread handle */
static uint32_t             g_nr_of_ranks        = 0;
static bool                 flush_worker_initialized = false;
static FloatArray* zero_vector = NULL;

/* Comparison function: sort by workload in descending order */
static int compare_centroid_info(const void* a, const void* b) {
    const CentroidInfo* ca = (const CentroidInfo*)a;
    const CentroidInfo* cb = (const CentroidInfo*)b;
    // Descending: cb->workload - ca->workload
    int64_t diff = cb->workload - ca->workload;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

/* ============================================
 * FIFO output callback function - read results directly from output FIFO
 * ============================================ */
static dpu_error_t flush_output_fifo_callback(struct dpu_set_t rank,
                                              uint32_t rank_id,
                                              void *args) {
#ifdef CPU_FLUSH_CALLBACK_TIME_PER_ROUND
    /* Replace real computation with a sleep equal to the CPU-server measured
     * time.  Obtained by running test_flush_callback.c on a DPU-free server:
     *   gcc -O2 -o test_flush test_flush_callback.c -lm && ./test_flush
     * Then recompile with -DCPU_FLUSH_CALLBACK_TIME_PER_ROUND=<seconds>. */
    (void)rank; (void)rank_id; (void)args;
    // usleep((unsigned int)((CPU_FLUSH_CALLBACK_TIME_PER_ROUND) * 1e6));
    return DPU_OK;
#else
    flush_callback_args_t *flush_args = (flush_callback_args_t*)args;
    struct dpu_set_t dpu;
    uint8_t *output_data = flush_args->output_fifo_data;
    struct dpu_fifo_link_t *output_link = (struct dpu_fifo_link_t*)flush_args->output_link;
    IVFSearcher *searcher = (IVFSearcher*)flush_args->searcher;
    const FloatMatrix *queries = (const FloatMatrix*)flush_args->queries;
    ResultBuffer_C ***all_results = (ResultBuffer_C***)flush_args->all_results;
    size_t output_fifo_size = (1 << OUTPUT_FIFO_PTR_SIZE) * OUTPUT_FIFO_DATA_SIZE;
    const uint16_t max_fifo_capacity = (1 << OUTPUT_FIFO_PTR_SIZE);

    /* Single-threaded sequential traversal of all DPU output FIFOs in this rank */
    int dpu_count = 0;
    DPU_FOREACH(rank, dpu, dpu_count) {
        (void)dpu_count;
        int dpu_id = GET_DPU_ID_BY_DPU(dpu, rank_id);
        uint16_t sz = get_fifo_size(output_link, dpu);
        if (sz == 0 || sz > max_fifo_capacity) continue; /* no data or pointer anomaly, skip */

        for (int fi = 0; fi < (int)sz; fi++) {
            fifo_batch_output_t *batch_out = (fifo_batch_output_t*)get_fifo_elem(
                output_link, dpu, &output_data[dpu_id * output_fifo_size], fi);

            for (int q = 0; q < QUERIES_PER_PUSH; q++) {
                fifo_query_output_t *output = &batch_out->results[q];

                int query_id    = output->query_id;
                int centroid_id = output->centroid_id;
                int flag        = output->flag;

                /* flag=0 means DPU broke early: all remaining slots are also empty */
                if (flag != 1) break;

                /* Defensive bounds check for data corruption (isolated to this entry) */
                if (query_id < 0 || query_id >= BATCH_SIZE ||
                    centroid_id < 0 || centroid_id >= searcher->num_indices) {
                    continue;
                }

                /* dpu_id is globally unique; each rank callback writes to independent rank_id slot, no race */
                flush_args->queries_sent[dpu_id]--;
                if (flush_args->queries_sent[dpu_id] < 0)
                    flush_args->queries_sent[dpu_id] = 0;

                IntArray *cluster_indices = cluster_mapping_table_get(
                    searcher->cluster_to_original, centroid_id);
                if (!cluster_indices) continue;

                for (int r = 0; r < POST_EF; r++) {
                    int32_t local_idx = output->results[r];
                    if (local_idx == -1) break;
                    if (local_idx >= 0 && local_idx < (int)cluster_indices->size) {
                        uint64_t global_idx = cluster_indices->data[local_idx];
                        if (global_idx < (uint64_t)searcher->total_num_elements) {
                            float dist = sqr_dist(&queries->rows[query_id], zero_vector);
                            result_buffer_insert(all_results[rank_id][query_id],
                                                 global_idx, dist);
                        }
                    }
                }
            }
        }
    }
    return DPU_OK;
#endif
}

/* ============================================
 * per-rank processing thread function
 * - waits for xfer thread signal (sem_start)
 * - calls flush_output_fifo_callback to process this rank's output FIFO
 * ============================================ */
static void* rank_process_thread_fn(void* arg) {
    rank_process_args_t* rp = (rank_process_args_t*)arg;
    uint64_t rank_round = 0;

    while (true) {
        /* Round-robin: buf index is determined by the rank's own round counter,
         * matching the xfer thread's round-robin order exactly.
         * This avoids the race where a single cur_buf_idx field gets overwritten
         * before the rank thread reads it. */
        int buf = (int)(rank_round % NUM_OUTPUT_BUFS);

        /* Wait on the BUFFER-SPECIFIC semaphore.  The xfer thread guarantees
         * at most 1 pending post per buf (it waits on sem_buf_done[buf] before
         * re-posting sem_start_buf[buf]), so no accumulation occurs. */
        sem_wait(&rp->sem_start_buf[buf]);

        /* stop flag and local_args for this buf were written by xfer thread
         * before sem_post(sem_start_buf[buf]) → happens-before via sem. */
        int should_stop = rp->stop_flag_buf[buf];

        flush_output_fifo_callback(rp->rank, rp->rank_id, &rp->local_args_buf[buf]);

        sem_post(&rp->sem_buf_done[buf]);

        rank_round++;
        if (should_stop) break;
    }
    return NULL;
}

/* ============================================
 * xfer thread function
 * - calls dpu_fifo_push_xfer with full dpu_set (correct usage)
 * - sem_post wakes all per-rank processing threads
 * - sem_wait waits for all processing threads to finish, then next round
 * - is_last mode: do one more round after stop_flag to drain FIFO
 * ============================================ */
static void* xfer_worker_thread_fn(void* arg) {
    xfer_worker_args_t* xw = (xfer_worker_args_t*)arg;
    bool is_last = false;
    uint64_t round = 0;   /* monotonically increasing round counter */

    while (true) {
        int buf = (int)(round % NUM_OUTPUT_BUFS);

        /* --------------------------------------------------------
         * KEY: Wait for the callback that LAST used buf[buf] to
         * finish before we overwrite it with fresh DPU data.
         * That callback was started (NUM_OUTPUT_BUFS) rounds ago.
         * For the first NUM_OUTPUT_BUFS rounds there is no prior use
         * of this slot, so no wait needed.
         * -------------------------------------------------------- */
        if (round >= (uint64_t)NUM_OUTPUT_BUFS) {
            for (uint32_t i = 0; i < xw->nr_ranks; i++) {
                sem_wait(&xw->proc_workers[i].sem_buf_done[buf]);
            }
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Flush ALL ranks' output FIFOs into snapshot buffer buf[buf].
         * output_links[buf] was pre-configured via dpu_fifo_prepare_xfer
         * to write into output_bufs[buf] during init. */
        DPU_ASSERT(dpu_fifo_push_xfer(*xw->dpu_set,
                                      &xw->output_links[buf],
                                      DPU_XFER_NO_RESET));

        clock_gettime(CLOCK_MONOTONIC, &t1);
        xw->elapsed_time += (t1.tv_sec - t0.tv_sec) +
                            (t1.tv_nsec - t0.tv_nsec) / 1e9;

        /* Signal all per-rank threads to process buf[buf].
         * Write to the PER-BUFFER slots before posting to the PER-BUFFER semaphore.
         * This guarantees:
         *  1. output_link is the one whose FIFO pointers were updated by the flush
         *     → get_fifo_size() returns the correct non-zero size.
         *  2. stop_flag_buf[buf] is written to the buffer-specific slot, not a
         *     shared field → the rank thread reads the correct stop flag for
         *     THIS buffer, never an accidentally overwritten value from a later
         *     round that could cause it to exit prematurely.
         * The xfer thread already waited for sem_buf_done[buf] at the top of this
         * iteration (when round >= NUM_OUTPUT_BUFS), so sem_start_buf[buf] is
         * guaranteed to have count 0 before this post → at most 1 pending per buf. */
        for (uint32_t i = 0; i < xw->nr_ranks; i++) {
            xw->proc_workers[i].local_args_buf[buf].output_fifo_data = xw->output_bufs[buf];
            xw->proc_workers[i].local_args_buf[buf].output_link      = &xw->output_links[buf];
            xw->proc_workers[i].stop_flag_buf[buf]                   = is_last ? 1 : 0;
            sem_post(&xw->proc_workers[i].sem_start_buf[buf]);
        }

        /* NOTE: we do NOT wait for callbacks here.  The xfer thread
         * immediately proceeds to the next round, using a different
         * buffer slot.  The wait at the TOP of the loop ensures we
         * never overwrite a buffer that is still being read. */

        round++;
        if (is_last) break;
        if (xw->stop_flag) is_last = true;
    }

    /* --------------------------------------------------------
     * Shutdown: wait for all in-flight callbacks to finish.
     * At most (NUM_OUTPUT_BUFS - 1) rounds of callbacks can still
     * be running when we exit the loop.
     * -------------------------------------------------------- */
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
        /* Only wait for buffers that were actually dispatched */
        if (round > (uint64_t)b) {
            for (uint32_t i = 0; i < xw->nr_ranks; i++) {
                sem_wait(&xw->proc_workers[i].sem_buf_done[b]);
            }
        }
    }

    xw->already_stopped = 1;
    return NULL;
}

/* ============================================
 * Initialize: create per-rank processing threads + single xfer thread
 * ============================================ */
static void init_flush_worker_thread(struct dpu_set_t*       dpu_set,
                                     struct dpu_fifo_link_t* output_link_template,
                                     flush_callback_args_t*  callback_args) {
    if (flush_worker_initialized) {
        return;
    }

    dpu_get_nr_ranks(*dpu_set, &g_nr_of_ranks);

    /* ---- Compute per-buffer size from callback_args ---- */
    size_t buf_bytes = callback_args->output_fifo_bytes;

    /* ---- Allocate NUM_OUTPUT_BUFS output snapshot buffers ---- */
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
        g_xfer_worker_args.output_bufs[b] = (uint8_t*)calloc(buf_bytes, 1);
        if (!g_xfer_worker_args.output_bufs[b]) {
            fprintf(stderr, "Failed to allocate output_buf[%d]\n", b);
            /* cleanup already allocated bufs */
            for (int j = 0; j < b; j++) free(g_xfer_worker_args.output_bufs[j]);
            return;
        }
    }
    g_xfer_worker_args.buf_bytes = buf_bytes;

    /* ---- Create NUM_OUTPUT_BUFS output FIFO links, each pre-configured
     *      with dpu_fifo_prepare_xfer pointing to its own buffer.
     *
     *  Why separate links?  dpu_fifo_prepare_xfer stores a per-DPU
     *  host-side buffer pointer inside the link's transfer_matrix.
     *  By creating N independent links we avoid re-calling prepare_xfer
     *  on every flush round, making the xfer thread as fast as possible.
     * -------------------------------------------------------------------- */
    size_t output_fifo_size = buf_bytes / (g_nr_of_ranks * 64);  /* bytes per DPU slot */
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
        /* Deep-copy the template link (symbol address, FIFO parameters, etc.) */
        DPU_ASSERT(dpu_link_output_fifo(*dpu_set,
                                        &g_xfer_worker_args.output_links[b],
                                        "output_fifo"));
        /* Point each DPU's slot to the corresponding position in output_bufs[b] */
        struct dpu_set_t rank_it2, dpu_it2;
        uint32_t rank_idx2;
        DPU_RANK_FOREACH(*dpu_set, rank_it2, rank_idx2) {
            DPU_FOREACH(rank_it2, dpu_it2) {
                int local_id  = (int)dpu_get_slice_id(dpu_it2.dpu) * 8
                              + (int)dpu_get_member_id(dpu_it2.dpu);
                int global_id = (int)rank_idx2 * 64 + local_id;
                DPU_ASSERT(dpu_fifo_prepare_xfer(
                    dpu_it2,
                    &g_xfer_worker_args.output_links[b],
                    (uint8_t *)&g_xfer_worker_args.output_bufs[b][global_id * output_fifo_size]));
            }
        }
    }

    /* ---- Allocate per-rank worker structs ---- */
    g_proc_workers = (rank_process_args_t*)calloc(g_nr_of_ranks, sizeof(rank_process_args_t));
    g_proc_threads = (pthread_t*)calloc(g_nr_of_ranks, sizeof(pthread_t));
    if (!g_proc_workers || !g_proc_threads) {
        fprintf(stderr, "Failed to allocate proc worker arrays\n");
        free(g_proc_workers); free(g_proc_threads);
        g_proc_workers = NULL; g_proc_threads = NULL;
        return;
    }

    /* ---- Init and start per-rank process threads ---- */
    struct dpu_set_t rank_it;
    uint32_t rank_idx;
    DPU_RANK_FOREACH(*dpu_set, rank_it, rank_idx) {
        rank_process_args_t* rp = &g_proc_workers[rank_idx];
        rp->rank    = rank_it;
        rp->rank_id = rank_idx;
        /* Copy template args into EACH per-buffer slot.
         * output_link and output_fifo_data will be overwritten per-round by
         * xfer_worker_thread_fn before sem_post(sem_start_buf[buf]).
         * Per-buffer copy prevents any two rounds racing on shared pointers. */
        for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
            rp->local_args_buf[b]  = *callback_args;
            rp->stop_flag_buf[b]   = 0;
            sem_init(&rp->sem_start_buf[b], 0, 0);
            sem_init(&rp->sem_buf_done[b],  0, 0);
        }

        int ret = pthread_create(&g_proc_threads[rank_idx], NULL,
                                 rank_process_thread_fn, rp);
        if (ret != 0) {
            fprintf(stderr, "Failed to create proc thread for rank %u: %d\n", rank_idx, ret);
        }
    }

    /* ---- Init and start the single xfer thread ---- */
    g_xfer_worker_args.dpu_set           = dpu_set;
    g_xfer_worker_args.proc_workers      = g_proc_workers;
    g_xfer_worker_args.nr_ranks          = g_nr_of_ranks;
    g_xfer_worker_args.stop_flag         = 0;
    g_xfer_worker_args.already_stopped   = 0;
    g_xfer_worker_args.elapsed_time      = 0.0;

    int ret = pthread_create(&g_xfer_thread, NULL, xfer_worker_thread_fn, &g_xfer_worker_args);
    if (ret != 0) {
        fprintf(stderr, "Failed to create xfer thread: %d\n", ret);
    }

    printf("Started xfer thread + %u per-rank process thread(s) [NUM_OUTPUT_BUFS=%d]\n",
           g_nr_of_ranks, NUM_OUTPUT_BUFS);
    flush_worker_initialized = true;
}

/* ============================================
 * Stop xfer thread and all per-rank processing threads, aggregate statistics
 * ============================================ */
/* Step 1: signal threads to stop and join them (must happen before dpu_sync).
 * sem_destroy / free / dpu_fifo_link_free are intentionally deferred so they
 * are NOT counted inside the query-latency timing window. */
static void stop_flush_worker_threads(void) {
    if (!flush_worker_initialized) {
        return;
    }

    /* Signal xfer thread to exit after finishing the current round */
    g_xfer_worker_args.stop_flag = 1;

    /* Wait for xfer thread to exit */
    pthread_join(g_xfer_thread, NULL);

    /* Wait for all processing threads to exit */
    for (uint32_t i = 0; i < g_nr_of_ranks; i++) {
        pthread_join(g_proc_threads[i], NULL);
    }

    /* Capture elapsed time while still inside the timing window */
    output_flush_time = g_xfer_worker_args.elapsed_time;
}

/* Step 2: release resources – sem_destroy, free, dpu_fifo_link_free.
 * Call this AFTER the end_query timestamp has been recorded. */
static void free_flush_worker_resources(void) {
    if (!flush_worker_initialized) {
        return;
    }

    /* Destroy semaphores */
    for (uint32_t i = 0; i < g_nr_of_ranks; i++) {
        for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
            sem_destroy(&g_proc_workers[i].sem_start_buf[b]);
            sem_destroy(&g_proc_workers[i].sem_buf_done[b]);
        }
    }

    /* Free double-buffered output buffers and FIFO links */
    for (int b = 0; b < NUM_OUTPUT_BUFS; b++) {
        if (g_xfer_worker_args.output_bufs[b]) {
            free(g_xfer_worker_args.output_bufs[b]);
            g_xfer_worker_args.output_bufs[b] = NULL;
        }
        dpu_fifo_link_free(&g_xfer_worker_args.output_links[b]);
    }

    free(g_proc_workers);
    free(g_proc_threads);
    g_proc_workers = NULL;
    g_proc_threads = NULL;
    g_nr_of_ranks  = 0;
    flush_worker_initialized = false;
}

/* Convenience wrapper: stop threads then free resources (used by destructor
 * and early-exit paths where timing is not a concern). */
static void cleanup_flush_worker_thread(void) {
    stop_flush_worker_threads();
    free_flush_worker_resources();
}

/* Use constructor and destructor attributes for auto init and cleanup */
__attribute__((constructor))
static void auto_init_flush_worker(void) {
    /* Note: not auto-initialized here; initialized on first use */
    /* Because dpu_set may not be ready at this point */
}

__attribute__((destructor))
static void auto_cleanup_flush_worker(void) {
    cleanup_flush_worker_thread();
}

/* ============================================
 * FIFO output post-processing function - direct call version (no dpu_callback)
 * Directly traverse all ranks and DPUs, read results from output FIFO
 * ============================================ */
__attribute__((unused))
static void flush_output_fifo_direct(struct dpu_set_t *dpu_set,
                                     flush_callback_args_t *flush_args) {
    uint8_t *output_data = flush_args->output_fifo_data;
    struct dpu_fifo_link_t *output_link = (struct dpu_fifo_link_t*)flush_args->output_link;
    IVFSearcher *searcher = (IVFSearcher*)flush_args->searcher;
    const FloatMatrix *queries = (const FloatMatrix*)flush_args->queries;
    ResultBuffer_C ***all_results = (ResultBuffer_C***)flush_args->all_results;
    size_t output_fifo_size = (1 << OUTPUT_FIFO_PTR_SIZE) * OUTPUT_FIFO_DATA_SIZE;
    const uint16_t max_fifo_capacity = (1 << OUTPUT_FIFO_PTR_SIZE);  /* FIFO max capacity */
    
    /* Traverse all ranks */
    struct dpu_set_t rank_it, dpu;
    uint32_t rank_idx;
    DPU_RANK_FOREACH(*dpu_set, rank_it, rank_idx) {
        int dpu_count = 0;
        DPU_FOREACH(rank_it, dpu, dpu_count) {
            uint16_t sz = get_fifo_size(output_link, dpu);
            int dpu_id = GET_DPU_ID_BY_DPU(dpu, rank_idx);
            
            /* Boundary check: sz must not exceed FIFO max capacity, otherwise pointer data is invalid */
            if (sz > max_fifo_capacity) {
                /* FIFO pointer data anomaly, possibly caused by race condition, skip this read */
                continue;
            }
            printf("[DEBUG] DPU#%d: FIFO size=%d\n", dpu_id, sz);
            /* Accumulate queries processed by this DPU (sz = batch count in output FIFO) */
            for (int fifo_idx = 0; fifo_idx < sz; ++fifo_idx) {
                /* Read complete batch result directly from output FIFO (fifo_batch_output_t) */
                fifo_batch_output_t *batch_out = (fifo_batch_output_t*)get_fifo_elem(
                    output_link, dpu, &output_data[dpu_id * output_fifo_size], fifo_idx);

                /* Traverse each query result in the batch */
                for (int q = 0; q < QUERIES_PER_PUSH; q++) {
                    fifo_query_output_t *output = &batch_out->results[q];
                    int query_id = output->query_id;
                    int centroid_id = output->centroid_id;
                    int flag = output->flag;

                    /* Validate data */
                    if (flag != 1 || query_id < 0 || query_id >= BATCH_SIZE ||
                        centroid_id < 0 || centroid_id >= searcher->num_indices) {
                        if (dpu_count == 0 && fifo_idx == 0 && q == 0) {
                            printf("[DEBUG] Direct: Invalid data at fifo_idx=0,q=0: query_id=%d, centroid_id=%d, flag=%d\n",
                                   query_id, centroid_id, flag);
                        }
                        continue;
                    }
                    flush_args->queries_sent[dpu_id]--;
                    if (flush_args->queries_sent[dpu_id] < 0)
                        flush_args->queries_sent[dpu_id] = 0;

                    /* Get cluster index mapping */
                    IntArray* cluster_indices = cluster_mapping_table_get(
                        searcher->cluster_to_original, centroid_id);
                    if (!cluster_indices) continue;

                    /* Process each result */
                    for (int r = 0; r < POST_EF; r++) {
                        int32_t local_idx = output->results[r];
                        if (local_idx == -1) break;
                        if (local_idx >= 0 && local_idx < (int)cluster_indices->size) {
                            uint64_t global_idx = cluster_indices->data[local_idx];
                            if (global_idx < (uint64_t)searcher->total_num_elements) {
                                float dist = sqr_dist(&queries->rows[query_id],
                                                     zero_vector);
                                result_buffer_insert(all_results[0][query_id], global_idx, dist);
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Batch push function: push buffer data to all DPUs (using circular queue to avoid data movement)
 * Each push carries QUERIES_PER_PUSH queries (packed as fifo_batch_input_t).
 */
static void batch_push_to_dpus(struct dpu_set_t* dpu_set,
                                struct dpu_set_t* dpu_vec,
                                dpu_input_buffer_t* dpu_buffers,
                                struct dpu_fifo_link_t* input_link,
                                fifo_batch_input_t* input_data, flush_callback_args_t* callback_args) {
    struct timespec start_time, end_time;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    struct dpu_fifo_link_t *output_link = (struct dpu_fifo_link_t*)callback_args->output_link;
    /* Prepare a batch of up to QUERIES_PER_PUSH queries for each DPU */
    for (int dpu_id = 0; dpu_id < NR_DPUS; dpu_id++) {
        int queries_filled = 0;
        for (int q = 0; q < QUERIES_PER_PUSH; q++) {
            if (dpu_buffers[dpu_id].count > 0) {
                /* Has data: take element from head position (circular via modulo) */
                int head_idx = dpu_buffers[dpu_id].head;
                memcpy(&input_data[dpu_id].queries[q],
                       &dpu_buffers[dpu_id].buffer[head_idx],
                       sizeof(fifo_input_data_t));
                /* Update head pointer (circular via modulo) */
                dpu_buffers[dpu_id].head =
                    (dpu_buffers[dpu_id].head + 1) % dpu_buffers[dpu_id].capacity;
                dpu_buffers[dpu_id].count--;
                queries_filled++;
            } else {
                /* No more data: fill remaining slots with query_id = -1 */
                input_data[dpu_id].queries[q].query_id = -1;
            }
        }
        if (queries_filled > 0) {
            callback_args->queries_sent[GET_DPU_ID_BY_DPU(dpu_vec[dpu_id], callback_args->dpu_rank_id[dpu_id])] += queries_filled;
        }
    }
    total_queries_sent += 1;  /* increment on each batch push */

    /* Batch push - use retry mechanism to handle full FIFO */
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    dpu_error_t err = dpu_fifo_push_xfer(*dpu_set, input_link, DPU_XFER_NO_RESET);
    if (err != DPU_OK) {
        /* If push fails (FIFO full), sync flush output FIFO then retry */
        printf("Warning: dpu_fifo_push_xfer failed with error %d, retrying...\n", err);
        int retry_count = 0;
        while (err != DPU_OK && retry_count < 200) {
            /* Wait and retry input push; sem already handled in push_xfer phase above,
             * worker will asynchronously release output FIFO slot to unblock DPU */
            usleep(500000);  /* wait 500ms, giving worker time to complete callback */
            err = dpu_fifo_push_xfer(*dpu_set, input_link, DPU_XFER_NO_RESET);
            retry_count++;
        }
        if (err != DPU_OK) {
            fprintf(stderr, "Error: dpu_fifo_push_xfer failed after %d retries: %d\n", retry_count, err);
            DPU_ASSERT(err);  /* trigger assertion on final failure */
        } else {
            printf("Recovered after %d retries\n", retry_count);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    batch_send_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    batch_push_to_dpus_time += (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
}

IntArray** ivf_searcher_search_batch(struct dpu_set_t* dpu_set,
                                        IVFSearcher* searcher,
                                        const FloatMatrix* queries,
                                        int batch_size,
                                        int ef,
                                        int topk,
                                        int nprobe,
                                        int post_ef,
                                        int nr_dpus) {
    if (!searcher || !queries || batch_size <= 0) {
        return NULL;
    }
    uint32_t nr_of_ranks;
    dpu_get_nr_ranks(*dpu_set, &nr_of_ranks);
    
    /* Ensure nr_dpus is at least NR_DPUS */
    if (nr_dpus < NR_DPUS) {
        nr_dpus = NR_DPUS;
    }
    printf("FIFO Mode: Virtual DPUs: %d, Physical DPUs: %d\n", nr_dpus, NR_DPUS);

    if (!zero_vector) {
        zero_vector = float_array_create(DIMM);
        for(size_t j=0; j<DIMM; j++){
            zero_vector->data[j] = 0.0f;
        }
        zero_vector->size = DIMM;
    }
    
    // /* Allocate result array */
    // IntArray** batch_results = (IntArray**)malloc(batch_size * sizeof(IntArray*));
    // if (!batch_results) {
    //     SET_ERROR(ERR_NOMEM, "Failed to allocate batch results array");
    //     return NULL;
    // }
    
    // /* Search for each query */
    // for (int query_idx = 0; query_idx < batch_size; query_idx++) {
    //     if (query_idx < (int)queries->num_rows) {
    //         batch_results[query_idx] = ivf_searcher_search(dpu_set, searcher, 
    //                                                       &queries->rows[query_idx],
    //                                                       ef, k, nprobe, post_ef);
    //     } else {
    //         /* If not enough queries, create empty result */
    //         batch_results[query_idx] = int_array_create(0);
    //     }
    // }
    
    // return batch_results;

    // Timer declaration (unused for now)
    // Timer timer;
    
    /* Find the nearest nprobe centroids */
    DistIndexArray* nearest_centroids = find_nearest_centroids_batch(queries, 
                                                                batch_size,
                                                              searcher->centroids, 
                                                              nprobe);
    
    /* Search in each selected cluster */
    ResultBuffer_C*** all_results = (ResultBuffer_C***)malloc(nr_of_ranks * sizeof(ResultBuffer_C**));
    for (uint32_t i = 0; i < nr_of_ranks; i++) {
        all_results[i] = (ResultBuffer_C**)malloc(batch_size * sizeof(ResultBuffer_C*));
        for (int j = 0; j < batch_size; j++) {
            all_results[i][j] = result_buffer_create(topk);
        }
    }

    // Count how many times each centroid is accessed (selected as nprobe neighbor)
    int num_centroids = searcher->centroids ? (int)searcher->centroids->num_rows : 0;
    int* centroid_visit_counts = NULL;
    if (num_centroids > 0) {
        centroid_visit_counts = (int*)calloc(num_centroids, sizeof(int));
        if (!centroid_visit_counts) {
            SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_visit_counts array");
            // Handle memory allocation failure
        } else {
            for (size_t i = 0; i < (size_t)batch_size; i++) {
                for (int j = 0; j < nprobe; j++) {
                    int centroid_id = nearest_centroids->data[i * nprobe + j].index;
                    if (centroid_id >= 0 && centroid_id < num_centroids) {
                        centroid_visit_counts[centroid_id]++;
                    }
                }
            }
        }
        // Statistics can be printed here or used for debugging
        // for (int c = 0; c < num_centroids; c++) {
        //     printf("Centroid %d visited %d times\n", c, centroid_visit_counts[c]);
        // }
    }
    int64_t* centroid_workload = (int64_t*)malloc(num_centroids * sizeof(int64_t));
    if (!centroid_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_workload array");
    }
    int64_t max_workload = 0LL;
    int64_t min_workload = INT64_MAX;
    int64_t total_workload = 0LL;
    int num_non_empty_centroids = 0;
    for (int i = 0; i < num_centroids; i++) {
        /* Use pre-recorded element count (independent of index loading status) */
        size_t n_elem = (searcher->index_num_elements ? searcher->index_num_elements[i] : 0);
        centroid_workload[i] = centroid_visit_counts[i] * (int64_t)n_elem;
        // printf("Centroid %d cluster size: %d\n", i, index_wrapper_get_num_elements(searcher->indices[i]));
        // printf("Centroid %d workload: %d\n", i, centroid_workload[i]);
        if(centroid_workload[i] > 0){
            if (centroid_workload[i] > max_workload) {
                max_workload = centroid_workload[i];
            }
            if (centroid_workload[i] < min_workload) {
                min_workload = centroid_workload[i];
            }
            total_workload += centroid_workload[i];
            num_non_empty_centroids++;
        }
    }
    int64_t average_workload = total_workload / nr_dpus;
    // printf("workload imbalance ratio (virtual): %f\n", (float)(max_workload) / (float)(average_workload));
    // printf("max_workload: %ld\n", max_workload);
    // printf("average_workload (per virtual DPU): %ld\n", average_workload);
    
    // Calculate how many replicas each centroid needs
    int* N_replicate_by_workload = (int*)malloc(num_centroids * sizeof(int));
    if (!N_replicate_by_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate N_replicate_by_workload array");
    }
    int total_replicated_centroids = 0;
    for (int i = 0; i < num_centroids; i++) {
        if (centroid_workload[i] > 0 && average_workload > 0) {
            // N_replicate_by_workload = ceil(centroid_workload[i] / average_workload)
            N_replicate_by_workload[i] = (centroid_workload[i] + average_workload - 1) / average_workload;
            if (N_replicate_by_workload[i] < 1) {
                N_replicate_by_workload[i] = 1;
            }
        } else {
            N_replicate_by_workload[i] = 0;
        }
        if (N_replicate_by_workload[i] > 1) {
            // printf("Centroid %d needs %d replicas (workload=%ld, avg=%ld)\n", 
            //        i, N_replicate_by_workload[i], centroid_workload[i], average_workload);
        }
        total_replicated_centroids += N_replicate_by_workload[i];
    }
    printf("Total centroids after replication: %d\n", total_replicated_centroids);
    // Use nr_dpus as the virtual DPU count for load balancing
    int64_t* dpu_workload = (int64_t*)malloc(nr_dpus * sizeof(int64_t));
    if (!dpu_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate dpu_workload array");
    }
    for (int i = 0; i < nr_dpus; i++) {
        dpu_workload[i] = 0LL;
    }
    PID** dpu_entry_point = (PID**)malloc(nr_dpus * sizeof(PID*));
    for (int i = 0; i < nr_dpus; i++) {
        dpu_entry_point[i] = (PID*)malloc(MAX_CLUSTER_PER_DPU * sizeof(PID)); 
    }
    // Record the centroid_id for each cluster position in each virtual DPU
    int** dpu_centroid_ids = (int**)malloc(nr_dpus * sizeof(int*));
    for (int i = 0; i < nr_dpus; i++) {
        dpu_centroid_ids[i] = (int*)malloc(MAX_CLUSTER_PER_DPU * sizeof(int));
        // Initialize to -1 to indicate unassigned
        for (int j = 0; j < MAX_CLUSTER_PER_DPU; j++) {
            dpu_centroid_ids[i][j] = -1;
        }
    }
    int32_t** dpu_cluster_offset = (int32_t**)malloc(nr_dpus * sizeof(int32_t*));
    for (int i = 0; i < nr_dpus; i++) {
        dpu_cluster_offset[i] = (int32_t*)malloc(MAX_CLUSTER_PER_DPU * sizeof(int32_t));
    }
    /* dpu_neighbor_id / dpu_rabit_code / dpu_factor are allocated later
     * (after virtual-to-physical mapping), only for active vDPUs */
    PID** dpu_neighbor_id = NULL;
    uint64_t** dpu_rabit_code = NULL;
    int32_t** dpu_factor = NULL;

    // ========== Load Balancing Assignment Algorithm Start ==========
    // Goal: assign each centroid to DPUs based on workload
    // Constraints: 1) DPU load as balanced as possible  2) each DPU's num_elements <= MAX_SIZE_PER_DPU
    
    // Create expanded centroid array including all replicas
    CentroidInfo* centroid_info = (CentroidInfo*)malloc(total_replicated_centroids * sizeof(CentroidInfo));
    if (!centroid_info) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_info array");
    }
    
    // Initialize centroid info including all replicas
    int info_idx = 0;
    for (int i = 0; i < num_centroids; i++) {
        int num_replicas = N_replicate_by_workload[i];
        if (num_replicas == 0) continue;
        
        // Get basic info for this centroid
        int num_elements = (int)searcher->index_num_elements[i];

        
        // Calculate workload per replica (evenly divided)
        int64_t workload_per_replica = centroid_workload[i] / num_replicas;
        
        // Create an entry for each replica
        for (int replica = 0; replica < num_replicas; replica++) {
            centroid_info[info_idx].centroid_id = i;
            centroid_info[info_idx].replica_id = replica;
            centroid_info[info_idx].workload = workload_per_replica;
            centroid_info[info_idx].num_elements = num_elements;
            info_idx++;
        }
    }
    
    // Sort by workload in descending order (using qsort, O(n log n))
    qsort(centroid_info, total_replicated_centroids, sizeof(CentroidInfo), compare_centroid_info);
    
    // Record the number of assigned centroids and elements for each virtual DPU
    int* dpu_cluster_count = (int*)calloc(nr_dpus, sizeof(int));
    int* dpu_element_count = (int*)calloc(nr_dpus, sizeof(int));
    if (!dpu_cluster_count || !dpu_element_count) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate DPU tracking arrays");
    }
    for (int i = 0; i < nr_dpus; i++) {
        dpu_cluster_count[i] = 0;
        dpu_element_count[i] = 0;
    }
    
    // Record which DPU each replica of each centroid is assigned to
    // centroid_to_dpu[centroid_id * MAX_REPLICAS + replica_id] = dpu_id
    // Assume at most 16 replicas
    #define MAX_REPLICAS 16
    int* centroid_to_dpu = (int*)malloc(num_centroids * MAX_REPLICAS * sizeof(int));
    if (!centroid_to_dpu) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_to_dpu array");
    }
    for (int i = 0; i < num_centroids * MAX_REPLICAS; i++) {
        centroid_to_dpu[i] = -1;  // Initialize to unassigned
    }
    
    // Use greedy algorithm to assign centroids to DPUs (including all replicas)
    for (int i = 0; i < total_replicated_centroids; i++) {
        int cid = centroid_info[i].centroid_id;
        int replica_id = centroid_info[i].replica_id;
        int64_t work = centroid_info[i].workload;
        int num_elem = centroid_info[i].num_elements;
        
        // Skip empty centroids or centroids not visited
        if (num_elem == 0 || work == 0) {
            continue;
        }
        
        // Find the virtual DPU with the minimum load that satisfies constraints
        int best_dpu = -1;
        int64_t min_workload = INT64_MAX;
        
        for (int dpu_id = 0; dpu_id < nr_dpus; dpu_id++) {
            // Check constraints
            if (dpu_cluster_count[dpu_id] >= MAX_CLUSTER_PER_DPU) {
                continue;  // This DPU is full, cannot assign more clusters
            }
            if (dpu_element_count[dpu_id] + num_elem > MAX_SIZE_PER_DPU) {
                continue;  // Adding this centroid would exceed element count limit
            }
            
            // Select the DPU with minimum load
            if (dpu_workload[dpu_id] < min_workload) {
                min_workload = dpu_workload[dpu_id];
                best_dpu = dpu_id;
            }
        }
        
        // If a suitable DPU is found, perform assignment
        if (best_dpu >= 0) {
            int cluster_idx = dpu_cluster_count[best_dpu];
            int elem_offset = dpu_element_count[best_dpu];
            
            // Record assignment relationship (including replica number)
            centroid_to_dpu[cid * MAX_REPLICAS + replica_id] = best_dpu;
            
            // Record centroid_id and offset for this position
            // entry_point / neighbor_id / rabit_code / factor will be filled
            // after virtual to physical DPU mapping is determined, during on-demand index loading
            dpu_centroid_ids[best_dpu][cluster_idx] = cid;
            
            // Update counters
            dpu_workload[best_dpu] += work;
            dpu_element_count[best_dpu] += num_elem;
            dpu_cluster_count[best_dpu]++;
            dpu_cluster_offset[best_dpu][cluster_idx] = elem_offset;
        } else {
            // Cannot assign this centroid (all DPUs full or cannot accommodate)
            // printf("Warning: Cannot assign centroid %d replica %d (num_elements=%d, workload=%ld) to any DPU\n", 
            //        cid, replica_id, num_elem, work);
        }
    }
    
    // Print assignment result statistics
    // printf("\n=== Virtual DPU Load Balancing Results ===\n");
    int64_t max_workload_after_replication = 0LL;
    for (int dpu_id = 0; dpu_id < nr_dpus; dpu_id++) {
        if (dpu_workload[dpu_id] > max_workload_after_replication) {
            max_workload_after_replication = dpu_workload[dpu_id];
        }
    }
    
    // ========== Load Balancing Assignment Algorithm End ==========
    // ========== Virtual DPU to Physical DPU Mapping ==========
    
    // Sort virtual DPUs by workload in descending order, select top NR_DPUS
    typedef struct {
        int virtual_dpu_id;
        int64_t workload;
    } VirtualDpuWorkload;
    
    VirtualDpuWorkload* vdpu_workloads = (VirtualDpuWorkload*)malloc(nr_dpus * sizeof(VirtualDpuWorkload));
    for (int i = 0; i < nr_dpus; i++) {
        vdpu_workloads[i].virtual_dpu_id = i;
        vdpu_workloads[i].workload = dpu_workload[i];
    }
    
    // Sort by workload in descending order
    for (int i = 0; i < nr_dpus - 1; i++) {
        for (int j = i + 1; j < nr_dpus; j++) {
            if (vdpu_workloads[j].workload > vdpu_workloads[i].workload) {
                VirtualDpuWorkload tmp = vdpu_workloads[i];
                vdpu_workloads[i] = vdpu_workloads[j];
                vdpu_workloads[j] = tmp;
            }
        }
    }
    
    // Create virtual DPU to physical DPU mapping
    // virtual_to_physical[virtual_dpu_id] = physical_dpu_id (-1 means not mapped, use random results)
    int* virtual_to_physical = (int*)malloc(nr_dpus * sizeof(int));
    for (int i = 0; i < nr_dpus; i++) {
        virtual_to_physical[i] = -1;  // Default: not mapped
    }
    
    // Select top NR_DPUS virtual DPUs with highest load to map to physical DPUs
    int num_active_dpus = (nr_dpus < NR_DPUS) ? nr_dpus : NR_DPUS;
    for (int i = 0; i < num_active_dpus; i++) {
        int vdpu_id = vdpu_workloads[i].virtual_dpu_id;
        virtual_to_physical[vdpu_id] = i;  // Map to the i-th physical DPU
    }
    
    printf("\n=== Virtual to Physical DPU Mapping ===\n");
    printf("Active physical DPUs: %d (out of %d virtual DPUs)\n", num_active_dpus, nr_dpus);
    int unmapped_count = 0;
    for (int i = 0; i < nr_dpus; i++) {
        if (virtual_to_physical[i] == -1) {
            unmapped_count++;
        }
    }
    printf("Unmapped virtual DPUs (will use random results): %d\n", unmapped_count);

    /* ============================================
     * Allocate large data arrays only for virtual DPUs that map to physical DPUs.
     * This avoids wasting ~113 GB on the 2058 unmapped vDPUs (with MAX_SIZE_PER_DPU=400000).
     * ============================================ */
    dpu_neighbor_id = (PID**)calloc(nr_dpus, sizeof(PID*));
    dpu_rabit_code  = (uint64_t**)calloc(nr_dpus, sizeof(uint64_t*));
    dpu_factor      = (int32_t**)calloc(nr_dpus, sizeof(int32_t*));
    if (dpu_neighbor_id && dpu_rabit_code && dpu_factor) {
        for (int i = 0; i < nr_dpus; i++) {
            if (virtual_to_physical[i] >= 0) {
                dpu_neighbor_id[i] = (PID*)malloc(MAX_SIZE_PER_DPU * DEGREE * sizeof(PID));
                dpu_rabit_code[i]  = (uint64_t*)malloc((PADDED_DIMM >> 6) * MAX_SIZE_PER_DPU * sizeof(uint64_t));
                dpu_factor[i]      = (int32_t*)malloc(MAX_SIZE_PER_DPU * sizeof(int32_t));
            }
            /* else: stays NULL — free(NULL) is safe, and lazy-loading skips these */
        }
    }

    /* ============================================
     * Lazy loading: after mapping is determined, only load cluster indices assigned to actual DPUs
     * and fill entry_point / neighbor_id / rabit_code / factor for corresponding virtual DPUs
     * ============================================ */
    {
        /* Collect all centroid IDs in virtual DPUs that map to physical DPUs
         * Use searcher->num_indices as upper bound to avoid out-of-bounds access to index_paths/index_num_elements */
        int safe_num = (num_centroids < searcher->num_indices) ? num_centroids : searcher->num_indices;
        int* centroid_to_load = (int*)calloc(safe_num, sizeof(int));
        if (!centroid_to_load) {
            fprintf(stderr, "Error: Failed to allocate centroid_to_load array (size=%d)\n", safe_num);
        } else {
            for (int vdpu = 0; vdpu < nr_dpus; vdpu++) {
                if (virtual_to_physical[vdpu] >= 0) {
                    for (int slot = 0; slot < dpu_cluster_count[vdpu]; slot++) {
                        int cid = dpu_centroid_ids[vdpu][slot];
                        if (cid >= 0 && cid < safe_num) {
                            centroid_to_load[cid] = 1;
                        }
                    }
                }
            }
        }

        /* Load these indices */
        printf("Lazy loading cluster indices for active physical DPUs...\n");
        int loaded_count = 0;
        if (centroid_to_load && searcher->index_paths && searcher->index_num_elements) {
            for (int i = 0; i < safe_num; i++) {
                if (!centroid_to_load[i]) continue;
                if (searcher->indices[i]) continue;       /* already loaded */
                if (!searcher->index_paths[i]) continue;  /* no file for this cluster */
                if (searcher->index_num_elements[i] <= 1) continue; /* too few elements */

                IndexWrapper* wrapper = index_wrapper_create(
                    "QG", "L2",
                    searcher->index_num_elements[i],
                    DIMM,
                    (size_t)searcher->index_degree,
                    0);  /* allocate_vectors=0: skip original_data_ (~89% RAM savings) */
                if (!wrapper) {
                    fprintf(stderr, "Warning: index_wrapper_create failed for cluster %d\n", i);
                    continue;
                }
                /* index_wrapper_load returns 0 on success; original code ignored return value */
                int load_ret = index_wrapper_load(wrapper, searcher->index_paths[i]);
                if (load_ret != 0) {
                    /* Try treating non-zero as "still OK" (some bindings return 0 on success) */
                    /* If the index data is corrupt/missing, free and skip */
                    fprintf(stderr, "Warning: index_wrapper_load returned %d for cluster %d, skipping\n",
                            load_ret, i);
                    index_wrapper_free(wrapper);
                    continue;
                }
                index_wrapper_set_cluster(wrapper, i);
                searcher->indices[i] = wrapper;
                loaded_count++;
            }
        }
        printf("Lazy loaded %d cluster indices (only for active physical DPUs)\n", loaded_count);
        if (centroid_to_load) free(centroid_to_load);

        /* Fill entry_point and data buffers for each active virtual DPU */
        for (int vdpu = 0; vdpu < nr_dpus; vdpu++) {
            if (virtual_to_physical[vdpu] < 0) continue;  /* Skip unmapped virtual DPUs */
            for (int slot = 0; slot < dpu_cluster_count[vdpu]; slot++) {
                int cid = dpu_centroid_ids[vdpu][slot];
                if (cid < 0 || !searcher->indices[cid]) continue;
                IndexWrapper* wrapper = searcher->indices[cid];
                int elem_offset = dpu_cluster_offset[vdpu][slot];
                int num_elem = (int)index_wrapper_get_num_elements(wrapper);

                /* Set entry_point */
                dpu_entry_point[vdpu][slot] = index_wrapper_get_entry_point(wrapper);

                /* Fill neighbor_id / rabit_code / factor */
                for (int elem_idx = 0; elem_idx < num_elem; elem_idx++) {
                    int dpu_idx = elem_offset + elem_idx;
                    if (dpu_idx >= MAX_SIZE_PER_DPU) break;

                    const uint32_t* neighbors = index_wrapper_get_neighbors(wrapper, elem_idx);
                    if (neighbors) {
                        for (int k = 0; k < DEGREE; k++) {
                            dpu_neighbor_id[vdpu][dpu_idx * DEGREE + k] = neighbors[k];
                        }
                    }

                    const uint64_t* packed_code = index_wrapper_get_packed_code(wrapper, elem_idx);
                    if (packed_code) {
                        int code_size = (PADDED_DIMM >> 6);
                        for (int k = 0; k < code_size; k++) {
                            dpu_rabit_code[vdpu][dpu_idx * code_size + k] = packed_code[k];
                        }
                    }

                    const float* factor_ptr = index_wrapper_get_factor(wrapper, elem_idx);
                    if (factor_ptr) {
                        dpu_factor[vdpu][dpu_idx] = (int32_t)(*factor_ptr);
                    }
                }
            }
        }
    }

    // Allocate data structures for physical DPUs
    PID** physical_dpu_entry_point = (PID**)malloc(NR_DPUS * sizeof(PID*));
    int** physical_dpu_centroid_ids = (int**)malloc(NR_DPUS * sizeof(int*));
    int32_t** physical_dpu_cluster_offset = (int32_t**)malloc(NR_DPUS * sizeof(int32_t*));
    PID** physical_dpu_neighbor_id = (PID**)malloc(NR_DPUS * sizeof(PID*));
    uint64_t** physical_dpu_rabit_code = (uint64_t**)malloc(NR_DPUS * sizeof(uint64_t*));
    int32_t** physical_dpu_factor = (int32_t**)malloc(NR_DPUS * sizeof(int32_t*));
    int* physical_dpu_cluster_count = (int*)calloc(NR_DPUS, sizeof(int));
    int* physical_dpu_element_count = (int*)calloc(NR_DPUS, sizeof(int));
    
    for (int i = 0; i < NR_DPUS; i++) {
        physical_dpu_entry_point[i] = (PID*)malloc(MAX_CLUSTER_PER_DPU * sizeof(PID));
        physical_dpu_centroid_ids[i] = (int*)malloc(MAX_CLUSTER_PER_DPU * sizeof(int));
        for (int j = 0; j < MAX_CLUSTER_PER_DPU; j++) {
            physical_dpu_centroid_ids[i][j] = -1;
        }
        physical_dpu_cluster_offset[i] = (int32_t*)malloc(MAX_CLUSTER_PER_DPU * sizeof(int32_t));
        physical_dpu_neighbor_id[i] = (PID*)malloc(MAX_SIZE_PER_DPU * DEGREE * sizeof(PID));
        physical_dpu_rabit_code[i] = (uint64_t*)malloc((PADDED_DIMM >> 6) * MAX_SIZE_PER_DPU * sizeof(uint64_t));
        physical_dpu_factor[i] = (int32_t*)malloc(MAX_SIZE_PER_DPU * sizeof(int32_t));
    }
    
    // Copy data from virtual DPUs to physical DPUs
    for (int vdpu = 0; vdpu < nr_dpus; vdpu++) {
        int pdpu = virtual_to_physical[vdpu];
        if (pdpu >= 0) {
            // Copy data
            memcpy(physical_dpu_entry_point[pdpu], dpu_entry_point[vdpu], 
                   MAX_CLUSTER_PER_DPU * sizeof(PID));
            memcpy(physical_dpu_centroid_ids[pdpu], dpu_centroid_ids[vdpu], 
                   MAX_CLUSTER_PER_DPU * sizeof(int));
            memcpy(physical_dpu_cluster_offset[pdpu], dpu_cluster_offset[vdpu], 
                   MAX_CLUSTER_PER_DPU * sizeof(int32_t));
            memcpy(physical_dpu_neighbor_id[pdpu], dpu_neighbor_id[vdpu], 
                   MAX_SIZE_PER_DPU * DEGREE * sizeof(PID));
            memcpy(physical_dpu_rabit_code[pdpu], dpu_rabit_code[vdpu], 
                   (PADDED_DIMM >> 6) * MAX_SIZE_PER_DPU * sizeof(uint64_t));
            memcpy(physical_dpu_factor[pdpu], dpu_factor[vdpu], 
                   MAX_SIZE_PER_DPU * sizeof(int32_t));
            physical_dpu_cluster_count[pdpu] = dpu_cluster_count[vdpu];
            physical_dpu_element_count[pdpu] = dpu_element_count[vdpu];
        }
    }

    /* Free virtual DPU arrays immediately after copy — data is now in physical arrays.
     * This saves ~28 GB peak memory by not keeping both copies alive. */
    for (int i = 0; i < nr_dpus; i++) {
        free(dpu_entry_point[i]);
        free(dpu_centroid_ids[i]);
        free(dpu_cluster_offset[i]);
        free(dpu_neighbor_id[i]);
        free(dpu_rabit_code[i]);
        free(dpu_factor[i]);
    }
    free(dpu_entry_point);      dpu_entry_point = NULL;
    free(dpu_centroid_ids);     dpu_centroid_ids = NULL;
    free(dpu_cluster_offset);   dpu_cluster_offset = NULL;
    free(dpu_neighbor_id);      dpu_neighbor_id = NULL;
    free(dpu_rabit_code);       dpu_rabit_code = NULL;
    free(dpu_factor);           dpu_factor = NULL;

    int max_cluster_per_dpu = 0;
    for (int i = 0; i < NR_DPUS; i++) {
        if (physical_dpu_cluster_count[i] > max_cluster_per_dpu) {
            max_cluster_per_dpu = physical_dpu_cluster_count[i];
        }
    }
    int max_size_per_dpu = 0;
    for (int i = 0; i < NR_DPUS; i++) {
        if (physical_dpu_element_count[i] > max_size_per_dpu) {
            max_size_per_dpu = physical_dpu_element_count[i];
        }
    }
    printf("max_size_per_dpu (physical): %d\n", max_size_per_dpu);
    printf("max_cluster_per_dpu (physical): %d\n", max_cluster_per_dpu);
    printf("max_workload_after_replication (virtual): %ld\n", max_workload_after_replication);
    printf("workload imbalance ratio (virtual): %f\n", (float)(max_workload_after_replication) / (float)(average_workload));
    struct dpu_set_t dpu;
    int dpu_id;
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_entry_point[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "entry_point", 0, ALIGN(max_cluster_per_dpu*sizeof(PID),8), DPU_XFER_DEFAULT));

    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_centroid_ids[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "centroid_ids", 0, ALIGN(max_cluster_per_dpu*sizeof(int),8), DPU_XFER_DEFAULT));
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_cluster_offset[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "cluster_offset", 0, ALIGN(max_cluster_per_dpu*sizeof(int32_t),8), DPU_XFER_DEFAULT));
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_neighbor_id[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "neighbor_id", 0, ALIGN(max_size_per_dpu*DEGREE*sizeof(PID),8), DPU_XFER_DEFAULT));
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_rabit_code[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "rabit_code", 0, ALIGN((PADDED_DIMM >> 6)*max_size_per_dpu*sizeof(uint64_t),8), DPU_XFER_DEFAULT));
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        dpu_prepare_xfer(dpu, physical_dpu_factor[dpu_id]);
    }
    DPU_ASSERT(dpu_push_xfer(*dpu_set,DPU_XFER_TO_DPU, "factor", 0, ALIGN(max_size_per_dpu*sizeof(int32_t),8), DPU_XFER_DEFAULT));

    struct dpu_set_t* dpu_vec = (struct dpu_set_t*)malloc(NR_DPUS * sizeof(struct dpu_set_t));
    uint8_t *dpu_rank_id = (uint8_t*)malloc(NR_DPUS * sizeof(uint8_t));
    dpu_id = 0;
    struct dpu_set_t rank_it, dpu_it;
    uint32_t rank_idx;
    DPU_RANK_FOREACH(*dpu_set, rank_it, rank_idx){
        DPU_FOREACH(rank_it, dpu_it){
            dpu_rank_id[dpu_id] = (uint8_t)rank_idx;
            dpu_vec[dpu_id] = dpu_it;
            dpu_id++;
        }
    }

    /* ============================================
     * FIFO mode: initialization and asynchronous launch
     * Transfer complete query data directly to WRAM via FIFO
     * ============================================ */
    printf("Initializing FIFO mode (direct WRAM)...\n");
    printf("[DEBUG] Host POST_EF=%d, OUTPUT_FIFO_DATA_SIZE=%zu, OUTPUT_FIFO_PTR_SIZE=%d, QUERIES_PER_PUSH=%d\n",
           POST_EF, (size_t)OUTPUT_FIFO_DATA_SIZE, OUTPUT_FIFO_PTR_SIZE, QUERIES_PER_PUSH);

    /* Create input FIFO link (output links created internally by init_flush_worker_thread, NUM_OUTPUT_BUFS count) */
    struct dpu_fifo_link_t input_link, output_link_template;
    DPU_ASSERT(dpu_link_input_fifo(*dpu_set, &input_link, "input_fifo"));
    /* output_link_template only used to pass symbol name during init; actual flush uses g_xfer_worker_args.output_links[] */
    DPU_ASSERT(dpu_link_output_fifo(*dpu_set, &output_link_template, "output_fifo"));

    // /* Enable input FIFO push retry to prevent immediate error when DPU briefly full */
    // dpu_fifo_set_push_max_retries(&input_link, *dpu_set, 500000); /* max 500000 retries */
    // dpu_fifo_set_time_for_push_retries(&input_link, *dpu_set, 100);  /* wait 100 us per retry */

    /* output_fifo_size / output_fifo_bytes still needed for host-side array allocation like queries_sent */
    size_t output_fifo_size = (1 << OUTPUT_FIFO_PTR_SIZE) * OUTPUT_FIFO_DATA_SIZE;
    size_t output_fifo_bytes = nr_of_ranks * 64 * output_fifo_size;
    int *queries_sent = (int*)calloc(nr_of_ranks * 64, sizeof(int));

    /* Allocate input FIFO data buffers:
     * To avoid reusing the same host buffer causing "data overwritten by next write / sent to wrong DPU",
     * allocate independent input_data for each physical DPU (each is fifo_batch_input_t, containing QUERIES_PER_PUSH queries).
     */
    fifo_batch_input_t *input_data = (fifo_batch_input_t*)calloc(NR_DPUS, sizeof(fifo_batch_input_t));
    DPU_FOREACH(*dpu_set, dpu, dpu_id){
        DPU_ASSERT(dpu_fifo_prepare_xfer(dpu, &input_link, (uint8_t *)&input_data[dpu_id]));
    }

    /* Create circular queue buffer for each DPU */
    dpu_input_buffer_t* dpu_buffers = (dpu_input_buffer_t*)calloc(NR_DPUS, sizeof(dpu_input_buffer_t));
    for (int i = 0; i < NR_DPUS; i++) {
        dpu_buffers[i].buffer = (fifo_input_data_t*)malloc(DPU_INPUT_BUFFER_CAPACITY * sizeof(fifo_input_data_t));
        dpu_buffers[i].capacity = DPU_INPUT_BUFFER_CAPACITY;
        dpu_buffers[i].head = 0;
        dpu_buffers[i].tail = 0;
        dpu_buffers[i].count = 0;
    }

    /* Prepare callback argument template (output_fifo_data updated by xfer thread to corresponding buf pointer before each flush round) */
    flush_callback_args_t callback_args;
    callback_args.output_fifo_data  = NULL;          /* placeholder; overwritten per-round by xfer thread */
    callback_args.output_fifo_bytes = output_fifo_bytes;
    callback_args.output_link       = &output_link_template;
    callback_args.searcher          = searcher;
    callback_args.queries           = (void*)queries;
    callback_args.all_results       = all_results;
    callback_args.cluster_mapping   = searcher->cluster_to_original;
    callback_args.base_data         = searcher->base_data;
    callback_args.topk              = topk;
    callback_args.nr_dpus           = NR_DPUS;
    callback_args.batch_size        = batch_size;
    callback_args.queries_sent      = queries_sent;
    callback_args.dpu_rank_id       = dpu_rank_id;

    /* Start xfer + per-rank worker threads (internally allocates NUM_OUTPUT_BUFS output buffers + links) */
    init_flush_worker_thread(dpu_set, &output_link_template, &callback_args);

    /* Launch DPU asynchronously */
    printf("Launching DPU asynchronously...\n");
    DPU_ASSERT(dpu_launch(*dpu_set, DPU_ASYNCHRONOUS));
    
    /* Track the number of queries assigned to each replica of each centroid */
    int* replica_query_count = (int*)calloc(num_centroids * MAX_REPLICAS, sizeof(int));
    if (!replica_query_count) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate replica_query_count array");
    }
    
    /* rotated_query buffer (float, staged on CPU then converted to int16_t for FIFO struct) */
    float *rotated_query_float_buf = (float*)malloc(PADDED_DIMM * sizeof(float));
    
    struct timespec start_query, end_query;
    
    
    /* ============================================
     * FIFO mode: per-query preprocessing and push
     * Transfer complete data directly to DPU WRAM via FIFO
     * ============================================ */
    printf("Processing queries via FIFO (direct WRAM)...\n");
    
    /* Find any loaded index to use for get_rotated_query (all same-dataset indices share rotation matrix) */
    IndexWrapper* rotation_index = NULL;
    for (int i = 0; i < num_centroids && !rotation_index; i++) {
        if (searcher->indices[i]) rotation_index = searcher->indices[i];
    }
    if (!rotation_index) {
        fprintf(stderr, "Error: No loaded cluster index found; cannot compute rotated query!\n");
    }
    
    /* Initialize timing breakdown statistics variables */
    lut_sumq_time = 0.0;
    cluster_check_time = 0.0;
    replica_selection_time = 0.0;
    buffer_add_time = 0.0;
    batch_push_check_time = 0.0;
    output_flush_time = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start_query);
    
    
    for (int query_idx = 0; query_idx < batch_size; query_idx++) {
        /* Compute rotated_query and sumq (replaces original LUT computation) */
        struct timespec t_start, t_end;
        float sumq;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        index_wrapper_get_rotated_query(rotation_index, queries->rows[query_idx].data,
                                        rotated_query_float_buf, &sumq);
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        lut_sumq_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        
        /* Process each nprobe centroid for this query */
        for (int probe_idx = 0; probe_idx < nprobe; probe_idx++) {
            int centroid_id = nearest_centroids->data[query_idx * nprobe + probe_idx].index;
            
            /* Check if the cluster exists and has sufficient data */
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            IntArray* cluster_indices = cluster_mapping_table_get(
                searcher->cluster_to_original, centroid_id);
            if (!cluster_indices || cluster_indices->size < 2) {
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                cluster_check_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
                continue;
            }
            
            if (centroid_id >= searcher->num_indices || !searcher->indices[centroid_id]) {
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                cluster_check_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
                continue;
            }
            
            // IndexWrapper* wrapper = searcher->indices[centroid_id];
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            cluster_check_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
            
            /* Select the replica of this centroid with the fewest usages */
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            int best_replica = 0;
            int assigned_dpu = -1;
            int min_usage = INT_MAX;
            int num_replicas = N_replicate_by_workload[centroid_id];
            
            for (int r = 0; r < num_replicas; r++) {
                int dpu_id_check = centroid_to_dpu[centroid_id * MAX_REPLICAS + r];
                if (dpu_id_check >= 0) {
                    int usage = replica_query_count[centroid_id * MAX_REPLICAS + r];
                    if (usage < min_usage) {
                        min_usage = usage;
                        best_replica = r;
                    }
                }
            }
            
            assigned_dpu = centroid_to_dpu[centroid_id * MAX_REPLICAS + best_replica];
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            replica_selection_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
            
            if (assigned_dpu < 0) {
                printf("[WARNING] No DPU assigned for centroid %d\n", centroid_id);
                continue;
            }
            
            /* Update the usage count of this replica */
            replica_query_count[centroid_id * MAX_REPLICAS + best_replica]++;
            
            /* Get physical DPU ID */
            int physical_dpu_id = virtual_to_physical[assigned_dpu];
            if (physical_dpu_id < 0) {
                /* This virtual DPU is not mapped to a physical DPU, skip */
                continue;
            }
            
            /* Get entry point distance */
            int32_t ep_dist = 999999999;  /* Use a large default value */
            
            /* Add data to the corresponding DPU's circular queue buffer */
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            if (dpu_buffers[physical_dpu_id].count < dpu_buffers[physical_dpu_id].capacity) {
                /* Insert into circular queue using tail pointer and modulo */
                int tail_idx = dpu_buffers[physical_dpu_id].tail;
                fifo_input_data_t* buf_slot = &dpu_buffers[physical_dpu_id].buffer[tail_idx];
                buf_slot->query_id = query_idx;
                buf_slot->centroid_id = centroid_id;
                buf_slot->ep_dist = ep_dist;
                buf_slot->sumq = (int16_t)sumq;
                buf_slot->pad = 0;
                for (int l = 0; l < PADDED_DIMM; l++) {
                    buf_slot->rotated_query[l] = (int16_t)rotated_query_float_buf[l];
                }
                /* Update tail pointer (circular via modulo) */
                dpu_buffers[physical_dpu_id].tail = (dpu_buffers[physical_dpu_id].tail + 1) % dpu_buffers[physical_dpu_id].capacity;
                dpu_buffers[physical_dpu_id].count++;
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                buffer_add_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
                
                /* Check if batch push is needed */
                clock_gettime(CLOCK_MONOTONIC, &t_start);
                int dpus_with_input = 0;
                for (int i = 0; i < NR_DPUS; i++) {
                    if (dpu_buffers[i].count >= QUERIES_PER_PUSH) {
                        dpus_with_input++;
                    }
                }
                
                int threshold = (int)(NR_DPUS * DPU_BATCH_PUSH_THRESHOLD_RATIO);
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                batch_push_check_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
                
                /* If threshold reached, execute batch push (timing tracked in batch_send_time) */
                if (dpus_with_input >= threshold) {
                    batch_push_to_dpus(dpu_set, dpu_vec, dpu_buffers, &input_link, input_data, &callback_args);
                }
            } else {
                /* Buffer full, force push */
                printf("[WARNING] DPU#%d buffer full, forcing push\n", physical_dpu_id);
                batch_push_to_dpus(dpu_set, dpu_vec, dpu_buffers, &input_link, input_data, &callback_args);
                /* Re-add to buffer (count reduced, can insert now) */
                int tail_idx = dpu_buffers[physical_dpu_id].tail;
                fifo_input_data_t* buf_slot = &dpu_buffers[physical_dpu_id].buffer[tail_idx];
                buf_slot->query_id = query_idx;
                buf_slot->centroid_id = centroid_id;
                buf_slot->ep_dist = ep_dist;
                buf_slot->sumq = (int16_t)sumq;
                buf_slot->pad = 0;
                for (int l = 0; l < PADDED_DIMM; l++) {
                    buf_slot->rotated_query[l] = (int16_t)rotated_query_float_buf[l];
                }
                dpu_buffers[physical_dpu_id].tail = (dpu_buffers[physical_dpu_id].tail + 1) % dpu_buffers[physical_dpu_id].capacity;
                dpu_buffers[physical_dpu_id].count++;
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                buffer_add_time += (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
            }
        }
    }
    
    /* ============================================
     * Flush remaining data in all DPU buffers
     * ============================================ */
    printf("Flushing remaining input buffers...\n");
    while (1) {
        int dpus_with_input = 0;
        for (int i = 0; i < NR_DPUS; i++) {
            if (dpu_buffers[i].count > 0) {
                dpus_with_input++;
            }
        }
        if (dpus_with_input == 0) break;
        
        batch_push_to_dpus(dpu_set, dpu_vec, dpu_buffers, &input_link, input_data, &callback_args);
        // DPU_ASSERT(dpu_fifo_push_xfer(*dpu_set, &output_link, DPU_XFER_NO_RESET));
        // // flush_output_fifo_direct(dpu_set, &callback_args);
        // DPU_ASSERT(dpu_callback(*dpu_set, flush_output_fifo_callback, 
        //                    &callback_args, DPU_CALLBACK_PARALLEL));
    }
    
    int in_sz = get_fifo_size(&input_link, dpu_vec[0]);
    printf("input_fifo size: %d\n", in_sz);
    printf("Total queries sent to FIFO: %d\n", total_queries_sent);
    
    
    /* ============================================
     * FIFO mode: send termination signal and wait for DPU to complete
     *
     * Deadlock analysis:
     *   dpu_sync() holds rank-level lock and blocks, waiting for DPU to exit.
     *   Meanwhile DPU may be spinning in output_fifo_push() (output FIFO full with only 2 slots).
     *   The xfer thread needs to call dpu_fifo_push_xfer() to drain output FIFO, but this call
     *   also needs the rank lock → forming classic deadlock:
     *     dpu_sync holds lock waiting for DPU → DPU waits for FIFO drain → xfer thread waits for rank lock
     *
     * Fix:
     *   Use non-blocking dpu_status() polling instead of blocking dpu_sync(),
     *   allowing xfer thread to continuously drain output FIFO; call dpu_sync after DPU exits naturally.
     * ============================================ */
    printf("Sending termination signal to DPU...\n");
    
    /* Set active = 0 to notify DPU to stop processing */
    uint64_t active_signal = 0;
    DPU_ASSERT(dpu_prepare_xfer(*dpu_set, &active_signal));
    DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, "active", 0,
                             sizeof(uint64_t), DPU_XFER_PARALLEL));

    /* Use non-blocking dpu_status polling to wait for DPU completion; xfer thread continues draining output FIFO
     * (using dpu_sync would hold rank lock, preventing xfer thread from draining FIFO → deadlock) */
    printf("Stopping busy-poll worker thread (final drain)...\n");
    /* Only join threads here – sem_destroy/free/dpu_fifo_link_free are deferred
     * so that they are excluded from the query-latency timing window. */
    stop_flush_worker_threads();

    dpu_sync(*dpu_set);
    clock_gettime(CLOCK_MONOTONIC, &end_query);

    /* Resource teardown happens AFTER the timestamp is captured. */
    free_flush_worker_resources();

    // printf("Synchronizing DPU...\n");
    // /* DPU has exited, dpu_sync returns immediately */
    // DPU_ASSERT(dpu_sync(*dpu_set));

    /* Check for unprocessed queries (diagnostic only; worker has exited, no race) */
    printf("All queries processed by worker thread successfully.\n");

    double dpu_search_time = (end_query.tv_sec - start_query.tv_sec) + 
                        (end_query.tv_nsec - start_query.tv_nsec) / 1e9;
    printf("BATCH_SIZE: %d\n", batch_size);
    printf("DPU search time (FIFO mode): %f seconds\n", dpu_search_time);
    printf("QPS: %f \n", (double)batch_size / dpu_search_time);
    printf("batch_push_to_dpus time: %f seconds\n", batch_push_to_dpus_time);
    printf("\n=== Timing Breakdown ===\n");
    printf("rotated_query and sumq computation time: %f seconds (%.2f%%)\n", lut_sumq_time, (lut_sumq_time / dpu_search_time) * 100.0);
    printf("Cluster check and validation time: %f seconds (%.2f%%)\n", cluster_check_time, (cluster_check_time / dpu_search_time) * 100.0);
    printf("Replica selection time: %f seconds (%.2f%%)\n", replica_selection_time, (replica_selection_time / dpu_search_time) * 100.0);
    printf("Buffer add time: %f seconds (%.2f%%)\n", buffer_add_time, (buffer_add_time / dpu_search_time) * 100.0);
    printf("Batch push check time (check logic only): %f seconds (%.2f%%)\n", batch_push_check_time, (batch_push_check_time / dpu_search_time) * 100.0);
    printf("Batch push execution time (dpu_fifo_push_xfer): %f seconds (%.2f%%)\n", batch_send_time, (batch_send_time / dpu_search_time) * 100.0);
    printf("Output FIFO flush time (xfer thread total elapsed): %f seconds (%.2f%%)\n", output_flush_time, (output_flush_time / dpu_search_time) * 100.0);
    double accounted_time = lut_sumq_time + cluster_check_time + replica_selection_time + 
                           buffer_add_time + batch_push_check_time + batch_send_time;
    printf("Other time (including init and sync wait): %f seconds (%.2f%%)\n", 
           dpu_search_time - accounted_time,
           ((dpu_search_time - accounted_time) / dpu_search_time) * 100.0);
    printf("===================\n\n");
    /* Display DPU logs */
    printf("Display DPU Logs (FIFO mode)\n");
    DPU_FOREACH (*dpu_set, dpu, dpu_id) {
        printf("DPU#%d:\n", dpu_id);
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
        break;
    }
    
    /* ============================================
     * Create batch result array
     * ============================================ */
    IntArray** batch_results = (IntArray**)malloc(batch_size * sizeof(IntArray*));
    
    for (int i = 0; i < batch_size; i++) {
        batch_results[i] = int_array_create(topk);
        /* Merge starting from rank 1; rank 0 is the target buffer, avoid self-merge causing duplicate entries filling capacity */
        for (int j = 1; j < nr_of_ranks; j++) {
            result_buffer_merge(all_results[0][i], all_results[j][i]);
        }
        ResultBuffer_C* result_buffer = all_results[0][i];
        
        /* Copy results from ResultBuffer */
        uint32_t* result_ids_out = (uint32_t*)malloc(topk * sizeof(uint32_t));
        size_t copied = result_buffer_copy_results(result_buffer, result_ids_out, NULL, topk);
        
        for (size_t j = 0; j < copied && j < (size_t)topk; j++) {
            int_array_push(batch_results[i], (int)result_ids_out[j]);
        }
        
        free(result_ids_out);
        result_buffer_free(all_results[0][i]);
    }
    
    /* Free result buffers of other ranks */
    for (uint32_t rank = 1; rank < nr_of_ranks; rank++) {
        for (int i = 0; i < batch_size; i++) {
            if (all_results[rank][i]) {
                result_buffer_free(all_results[rank][i]);
            }
        }
    }
    
    /* ============================================
     * Lazy loading: unload cluster indices loaded during this search to free memory
     * ============================================ */
    {
        int unloaded_count = 0;
        for (int i = 0; i < num_centroids; i++) {
            if (searcher->indices[i]) {
                index_wrapper_free(searcher->indices[i]);
                searcher->indices[i] = NULL;
                unloaded_count++;
            }
        }
        printf("Unloaded %d cluster indices to free memory.\n", unloaded_count);
    }
    
    /* ============================================
     * Clean up FIFO related resources
     * output_bufs[] and output_links[] already freed in cleanup_flush_worker_thread()
     * ============================================ */
    free(queries_sent);

    /* Free DPU input buffers */
    for (int i = 0; i < NR_DPUS; i++) {
        if (dpu_buffers[i].buffer) {
            free(dpu_buffers[i].buffer);
        }
    }
    free(dpu_buffers);
    
    free(input_data);
    dpu_fifo_link_free(&input_link);
    dpu_fifo_link_free(&output_link_template);
    free(rotated_query_float_buf);
    
    
    /* Clean up load balancing related memory */
    free(dpu_vec);
    free(dpu_rank_id);
    free(all_results);
    dist_index_array_free(nearest_centroids);
    free(centroid_visit_counts);
    free(centroid_workload);
    free(N_replicate_by_workload);
    free(centroid_info);
    free(dpu_cluster_count);
    free(dpu_element_count);
    free(centroid_to_dpu);
    free(replica_query_count);
    
    /* Free virtual DPU to physical DPU mapping related memory */
    free(vdpu_workloads);
    free(virtual_to_physical);
    
    /* Free physical DPU related memory */
    for (int i = 0; i < NR_DPUS; i++) {
        free(physical_dpu_entry_point[i]);
        free(physical_dpu_centroid_ids[i]);
        free(physical_dpu_cluster_offset[i]);
        free(physical_dpu_neighbor_id[i]);
        free(physical_dpu_rabit_code[i]);
        free(physical_dpu_factor[i]);
    }
    free(physical_dpu_entry_point);
    free(physical_dpu_centroid_ids);
    free(physical_dpu_cluster_offset);
    free(physical_dpu_neighbor_id);
    free(physical_dpu_rabit_code);
    free(physical_dpu_factor);
    free(physical_dpu_cluster_count);
    free(physical_dpu_element_count);
    
    /* Free virtual DPU related memory
     * (may have been freed early after copy to physical arrays; guard with NULL checks) */
    if (dpu_entry_point) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_entry_point[i]);
        free(dpu_entry_point);
    }
    if (dpu_centroid_ids) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_centroid_ids[i]);
        free(dpu_centroid_ids);
    }
    if (dpu_cluster_offset) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_cluster_offset[i]);
        free(dpu_cluster_offset);
    }
    if (dpu_neighbor_id) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_neighbor_id[i]);
        free(dpu_neighbor_id);
    }
    if (dpu_rabit_code) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_rabit_code[i]);
        free(dpu_rabit_code);
    }
    if (dpu_factor) {
        for (int i = 0; i < nr_dpus; i++) free(dpu_factor[i]);
        free(dpu_factor);
    }
    free(dpu_workload);
    
    printf("FIFO mode search completed.\n");
    return batch_results;
}


/* Helper function: check if an integer is in an array */
static int int_in_array(int value, const IntArray* arr) {
    for (size_t i = 0; i < arr->size; i++) {
        if (arr->data[i] == value) return 1;
    }
    return 0;
}

IntArray* ivf_searcher_find_EFS(struct dpu_set_t* dpu_set,
                               IVFSearcher* searcher,
                               const FloatMatrix* query,
                               int NQ,
                               int nprobe,
                               const IntMatrix* gt) {
    printf("Finding suitable EF values for IVF search...\n");
    
    IntArray* EFS = int_array_create(10);
    BeamSizeGenerator* W = beam_size_generator_create(TOPK);
    float prev_recall = 0.0f;
    
    while (1) {
        int ef_value = beam_size_generator_next(W);
        int_array_push(EFS, ef_value);
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        IntArray** results = (IntArray**)malloc(NQ * sizeof(IntArray*));
        for (int i = 0; i < NQ; i++) {
            // Create FloatMatrix containing a single query
            FloatMatrix* single_query = float_matrix_create(1);
            
            // Copy query row
            FloatArray* query_row = float_array_create(query->rows[i].size);
            for (size_t j = 0; j < query->rows[i].size; j++) {
                query_row->data[j] = query->rows[i].data[j];
            }
            query_row->size = query->rows[i].size;
            float_matrix_add_row(single_query, query_row);
            
            IntArray** batch_result = ivf_searcher_search_batch(dpu_set, searcher, single_query, 1, ef_value, TOPK, nprobe, ef_value, NR_DPUS);
            results[i] = batch_result[0];
            free(batch_result);
            float_matrix_free(single_query);
            printf("Query %d/%d completed\n", i + 1, NQ);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        /* Calculate recall */
        int total_num = NQ * TOPK;
        int total_correct = 0;
        
        for (int i = 0; i < NQ; i++) {
            for (int j = 0; j < TOPK && j < (int)gt->rows[i].size; j++) {
                if (int_in_array(gt->rows[i].data[j], results[i])) {
                    total_correct++;
                }
            }
        }
        
        double qps = NQ / total_time;
        double recall = (total_correct * 100.0) / total_num;
        
        printf("Recall: %.2f%%, prev_recall: %.2f%%, QPS: %.2f, EF: %d, nprobe: %d\n", 
               recall, prev_recall, qps, ef_value, nprobe);
        
        /* Free results */
        for (int i = 0; i < NQ; i++) {
            int_array_free(results[i]);
        }
        free(results);
        
        if (recall > 99.8 || (recall - prev_recall) < 0.05 || qps < 10) {
            break;
        }
        prev_recall = recall;
    }
    
    beam_size_generator_free(W);
    return EFS;
}

size_t ivf_searcher_get_memory_usage(const IVFSearcher* searcher) {
    (void)searcher;  /* unused parameter */
    return get_memory_usage();
}

void ivf_searcher_free_batch_results(IntArray** batch_results, int batch_size) {
    if (batch_results) {
        for (int i = 0; i < batch_size; i++) {
            if (batch_results[i]) {
                int_array_free(batch_results[i]);
            }
        }
        free(batch_results);
    }
}

void ivf_searcher_clear_indices(IVFSearcher* searcher) {
    if (searcher && searcher->indices) {
        for (int i = 0; i < searcher->num_indices; i++) {
            if (searcher->indices[i]) {
                index_wrapper_free(searcher->indices[i]);
                searcher->indices[i] = NULL;
            }
        }
    }
}

/* =============== Helper Function Implementation =============== */

/* Compute squared distance between two vectors */
static float compute_sqr_dist_cpu(const float* query, const float* vector, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = query[i] - vector[i];
        dist += diff * diff;
    }
    return dist;
}

/* =============== Load Balancing and Query Assignment Functions =============== */

/**
 * Compute query-to-DPU mapping
 * @param searcher IVF searcher
 * @param queries query vector matrix
 * @param batch_size batch size
 * @param nprobe number of centroids to probe per query
 * @param nr_dpus number of DPUs
 * @param query_centroid_to_dpu_out output: mapping from centroid_id to assigned_dpu for each query (HashTable array)
 * @return query-to-DPU-set mapping (HashTable array)
 */
HashTable** compute_query_to_dpu_mapping(
    IVFSearcher* searcher,
    const FloatMatrix* queries,
    int batch_size,
    int nprobe,
    int nr_dpus,
    HashTable*** query_centroid_to_dpu_out) {
    
    if (!searcher || !queries || batch_size <= 0 || nr_dpus <= 0) {
        SET_ERROR(ERR_INVALID_PARAM, "Invalid parameters");
        return NULL;
    }
    
    /* Find the nearest nprobe centroids */
    DistIndexArray* nearest_centroids = find_nearest_centroids_batch(queries, 
                                                                      batch_size,
                                                                      searcher->centroids, 
                                                                      nprobe);
    if (!nearest_centroids) {
        SET_ERROR(ERR_OPERATION_FAILED, "Failed to find nearest centroids");
        return NULL;
    }
    
    // Count how many times each centroid is accessed
    int num_centroids = searcher->centroids ? (int)searcher->centroids->num_rows : 0;
    int* centroid_visit_counts = (int*)calloc(num_centroids, sizeof(int));
    if (!centroid_visit_counts) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_visit_counts array");
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < nprobe; j++) {
            int centroid_id = nearest_centroids->data[i * nprobe + j].index;
            if (centroid_id >= 0 && centroid_id < num_centroids) {
                centroid_visit_counts[centroid_id]++;
            }
        }
    }
    
    // Compute workload for each centroid
    int64_t* centroid_workload = (int64_t*)malloc(num_centroids * sizeof(int64_t));
    if (!centroid_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_workload array");
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    int64_t total_workload = 0;
    int num_non_empty_centroids = 0;
    for (int i = 0; i < num_centroids; i++) {
        centroid_workload[i] = centroid_visit_counts[i] * 
                               index_wrapper_get_num_elements(searcher->indices[i]);
        if (centroid_workload[i] > 0) {
            total_workload += centroid_workload[i];
            num_non_empty_centroids++;
        }
    }
    
    int64_t average_workload = (num_non_empty_centroids > 0) ? (total_workload / nr_dpus) : 0;
    printf("Average workload per DPU: %lld\n", average_workload);
    
    // Calculate how many replicas each centroid needs
    int* N_replicate_by_workload = (int*)malloc(num_centroids * sizeof(int));
    if (!N_replicate_by_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate N_replicate_by_workload array");
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    int total_replicated_centroids = 0;
    for (int i = 0; i < num_centroids; i++) {
        if (centroid_workload[i] > 0 && average_workload > 0) {
            N_replicate_by_workload[i] = 1;
        } else {
            N_replicate_by_workload[i] = 0;
        }
        total_replicated_centroids += N_replicate_by_workload[i];
    }
    
    // printf("Total centroids after replication: %d\n", total_replicated_centroids);
    
    // Create expanded centroid array including all replicas
    CentroidInfo* centroid_info = (CentroidInfo*)malloc(total_replicated_centroids * sizeof(CentroidInfo));
    if (!centroid_info) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_info array");
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    // Initialize centroid info including all replicas
    int info_idx = 0;
    for (int i = 0; i < num_centroids; i++) {
        int num_replicas = N_replicate_by_workload[i];
        if (num_replicas == 0) continue;
        
        int num_elements = 0;
        if (searcher->indices[i]) {
            num_elements = (int)index_wrapper_get_num_elements(searcher->indices[i]);
        }
        
        int64_t workload_per_replica = centroid_workload[i] / num_replicas;
        
        for (int replica = 0; replica < num_replicas; replica++) {
            centroid_info[info_idx].centroid_id = i;
            centroid_info[info_idx].replica_id = replica;
            centroid_info[info_idx].workload = workload_per_replica;
            centroid_info[info_idx].num_elements = num_elements;
            info_idx++;
        }
    }
    
    // Sort by workload in descending order
    qsort(centroid_info, total_replicated_centroids, sizeof(CentroidInfo), compare_centroid_info);
    
    // Record the number of assigned centroids and elements for each DPU
    int* dpu_cluster_count = (int*)calloc(nr_dpus, sizeof(int));
    int* dpu_element_count = (int*)calloc(nr_dpus, sizeof(int));
    int64_t* dpu_workload = (int64_t*)calloc(nr_dpus, sizeof(int64_t));
    if (!dpu_cluster_count || !dpu_element_count || !dpu_workload) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate DPU tracking arrays");
        free(centroid_info);
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    // Record which DPU each replica of each centroid is assigned to
    #define MAX_REPLICAS 16
    int* centroid_to_dpu = (int*)malloc(num_centroids * MAX_REPLICAS * sizeof(int));
    if (!centroid_to_dpu) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate centroid_to_dpu array");
        free(dpu_workload);
        free(dpu_element_count);
        free(dpu_cluster_count);
        free(centroid_info);
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    for (int i = 0; i < num_centroids * MAX_REPLICAS; i++) {
        centroid_to_dpu[i] = -1;  // Initialize to unassigned
    }
    
    // Use greedy algorithm to assign centroids to DPUs (including all replicas)
    for (int i = 0; i < total_replicated_centroids; i++) {
        int cid = centroid_info[i].centroid_id;
        int replica_id = centroid_info[i].replica_id;
        int64_t work = centroid_info[i].workload;
        int num_elem = centroid_info[i].num_elements;
        
        if (num_elem == 0 || work == 0) {
            continue;
        }
        
        // Find the DPU with minimum load that satisfies constraints
        int best_dpu = -1;
        int64_t min_workload = INT64_MAX;
        
        for (int dpu_id = 0; dpu_id < nr_dpus; dpu_id++) {
            if (dpu_cluster_count[dpu_id] >= MAX_CLUSTER_PER_DPU) {
                continue;
            }
            if (dpu_element_count[dpu_id] + num_elem > MAX_SIZE_PER_DPU) {
                continue;
            }
            
            if (dpu_workload[dpu_id] < min_workload) {
                min_workload = dpu_workload[dpu_id];
                best_dpu = dpu_id;
            }
        }
        
        if (best_dpu >= 0) {
            centroid_to_dpu[cid * MAX_REPLICAS + replica_id] = best_dpu;
            dpu_workload[best_dpu] += work;
            dpu_element_count[best_dpu] += num_elem;
            dpu_cluster_count[best_dpu]++;
        } else {
            printf("Warning: Cannot assign centroid %d replica %d to any DPU, workload: %lld, num_elem: %d\n", cid, replica_id, work, num_elem);
        }
    }
    
    // Print assignment result statistics
    printf("\n=== DPU Load Balancing Results ===\n");
    for (int dpu_id = 0; dpu_id < nr_dpus; dpu_id++) {
        // printf("DPU %d: clusters=%d, elements=%d, workload=%d\n",
            //    dpu_id, dpu_cluster_count[dpu_id], dpu_element_count[dpu_id], dpu_workload[dpu_id]);
    }
    // max workload and average workload 
    int64_t max_workload = 0;
    for (int dpu_id = 0; dpu_id < nr_dpus; dpu_id++) {
        if (dpu_workload[dpu_id] > max_workload) {
            max_workload = dpu_workload[dpu_id];
        }
    }
    printf("max_workload: %lld\n", max_workload);
    // Total centroid count
    int total_centroid_count = 0;
    for (int i = 0; i < num_centroids; i++) {
        total_centroid_count += N_replicate_by_workload[i];
    }
    printf("total_centroid_count: %d\n", total_centroid_count);

    // printf("===================================\n\n");
    
    // Track the number of queries assigned to each replica of each centroid
    int* replica_query_count = (int*)calloc(num_centroids * MAX_REPLICAS, sizeof(int));
    if (!replica_query_count) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate replica_query_count array");
        free(centroid_to_dpu);
        free(dpu_workload);
        free(dpu_element_count);
        free(dpu_cluster_count);
        free(centroid_info);
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    // Create query_to_dpu_set
    HashTable** query_to_dpu_set = (HashTable**)malloc(batch_size * sizeof(HashTable*));
    if (!query_to_dpu_set) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate query_to_dpu_set array");
        free(replica_query_count);
        free(centroid_to_dpu);
        free(dpu_workload);
        free(dpu_element_count);
        free(dpu_cluster_count);
        free(centroid_info);
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    // Create query_centroid_to_dpu mapping (centroid_id -> assigned_dpu per query)
    HashTable** query_centroid_to_dpu = (HashTable**)malloc(batch_size * sizeof(HashTable*));
    if (!query_centroid_to_dpu) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate query_centroid_to_dpu array");
        free(query_to_dpu_set);
        free(replica_query_count);
        free(centroid_to_dpu);
        free(dpu_workload);
        free(dpu_element_count);
        free(dpu_cluster_count);
        free(centroid_info);
        free(N_replicate_by_workload);
        free(centroid_workload);
        free(centroid_visit_counts);
        dist_index_array_free(nearest_centroids);
        return NULL;
    }
    
    for (int i = 0; i < batch_size; i++) {
        query_to_dpu_set[i] = hashtable_create(nprobe);
        query_centroid_to_dpu[i] = hashtable_create(nprobe);
    }
    
    // Assign each query to corresponding DPU
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < nprobe; j++) {
            int centroid_id = nearest_centroids->data[i * nprobe + j].index;
            IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, 
                                                                   centroid_id);
            
            if (!cluster_indices || cluster_indices->size < 2) {
                continue;
            }
            
            if (centroid_id < searcher->num_indices && searcher->indices[centroid_id]) {
                // Select the replica of this centroid with the fewest usages
                int best_replica = 0;
                int min_usage = INT_MAX;
                int num_replicas = N_replicate_by_workload[centroid_id];
                
                for (int r = 0; r < num_replicas; r++) {
                    int dpu_id = centroid_to_dpu[centroid_id * MAX_REPLICAS + r];
                    if (dpu_id >= 0) {
                        int usage = replica_query_count[centroid_id * MAX_REPLICAS + r];
                        if (usage < min_usage) {
                            min_usage = usage;
                            best_replica = r;
                        }
                    }
                }
                
                // Get the DPU corresponding to the selected replica
                int assigned_dpu = centroid_to_dpu[centroid_id * MAX_REPLICAS + best_replica];
                if (assigned_dpu < 0) {
                    continue;
                }
                
                // Update the usage count of this replica
                replica_query_count[centroid_id * MAX_REPLICAS + best_replica]++;
                
                if (!hashtable_contains(query_to_dpu_set[i], assigned_dpu)) {
                    hashtable_insert(query_to_dpu_set[i], assigned_dpu, cluster_indices);
                }
                
                // Record centroid_id -> assigned_dpu mapping
                // Create an IntArray with a single assigned_dpu value
                IntArray* dpu_array = int_array_create(1);
                int_array_push(dpu_array, assigned_dpu);
                hashtable_insert(query_centroid_to_dpu[i], centroid_id, dpu_array);
                free(dpu_array);
            }
        }
    }
    
    // Print query distribution statistics per replica
    // printf("\n=== Query Distribution Across Replicas ===\n");
    // for (int i = 0; i < num_centroids; i++) {
    //     if (N_replicate_by_workload[i] > 1) {
    //         // printf("Centroid %d (%d replicas):\n", i, N_replicate_by_workload[i]);
    //         for (int r = 0; r < N_replicate_by_workload[i]; r++) {
    //             // int dpu_id = centroid_to_dpu[i * MAX_REPLICAS + r];
    //             // int query_count = replica_query_count[i * MAX_REPLICAS + r];
    //             // printf("  Replica %d -> DPU %d, queries: %d\n", r, dpu_id, query_count);
    //             (void)centroid_to_dpu;  /* suppress unused variable warning */
    //             (void)replica_query_count;  /* suppress unused variable warning */
    //         }
    //     }
    // }
    // printf("==========================================\n\n");
    
    // Clean up temporary arrays
    free(centroid_to_dpu);
    free(N_replicate_by_workload);
    free(replica_query_count);
    free(dpu_workload);
    free(dpu_element_count);
    free(dpu_cluster_count);
    free(centroid_info);
    free(centroid_workload);
    free(centroid_visit_counts);
    dist_index_array_free(nearest_centroids);
    
    // Set output parameters
    if (query_centroid_to_dpu_out) {
        *query_centroid_to_dpu_out = query_centroid_to_dpu;
    }
    
    return query_to_dpu_set;
}

/* =============== CPU Version Search Function Implementation =============== */

/* Compute approximate distance using packed codes (simulating DPU logic) */
static float compute_appro_dist_cpu(const float* lut, float sumq, 
                                     const uint64_t* rabit_code, 
                                     const float* factor) {
    int lut_base = 0;
    float accumulator = 0.0f;
    
    // Process packed codes, consistent with DPU code
    for (int l = 0; l < (PADDED_DIMM >> 6); l++) {
        uint64_t code = rabit_code[l];
        // printf("code: %llu\n", code);
        uint8_t bytes[8];
        
        // Extract bytes
        for (int b = 0; b < 8; b++) {
            bytes[b] = (code >> (b * 8)) & 0xFF;
        }
        
        // Directly extract 4-bit codes and accumulate in correct order
        for (int b = 0; b < 8; ++b) {
            if (lut_base >= PADDED_DIMM * 4) break;
            
            uint8_t val = bytes[b];
            uint8_t lo4 = val & 0x0F;
            uint8_t hi4 = (val >> 4) & 0x0F;
            
            // Use LUT to accumulate
            accumulator += (float)(lut[lut_base + lo4]);
            lut_base += 16;
            if (lut_base < PADDED_DIMM * 4) {
                accumulator += (float)(lut[lut_base + hi4]);
                lut_base += 16;
            }
        }
    }
    
    // Compute final distance: ((accumulator << 1) - sumq) >> 2) + factor
    // printf("accumulator: %f, sumq: %f, factor: %f, result: %f\n", accumulator, sumq, *factor, (((accumulator * 2) - (float)(sumq)) / 4) + (float)(*factor));
    return (((accumulator * 2) - (float)(sumq)) / 4) + (float)(*factor);
}

IntArray** calculate_recall_cpu(
                                    struct dpu_set_t* dpu_set,
                                    IVFSearcher* searcher,
                                    const FloatMatrix* queries,
                                    int batch_size,
                                    int ef,
                                    int topk,
                                    int nprobe,
                                    int post_ef,
                                    HashTable** query_to_dpu_set,
                                    HashTable** query_centroid_to_dpu) {
    if (!searcher || !queries || batch_size <= 0) {
        return NULL;
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    /* Allocate result array */
    IntArray** batch_results = (IntArray**)malloc(batch_size * sizeof(IntArray*));
    if (!batch_results) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate batch results array");
        return NULL;
    }
    
    /* Find the nearest nprobe centroids */
    DistIndexArray* nearest_centroids = find_nearest_centroids_batch(queries, 
                                                                    batch_size,
                                                                    searcher->centroids, 
                                                                    nprobe);
    
    /* Search for each query */
    for (int query_idx = 0; query_idx < batch_size; query_idx++) {
        // printf("Processing query %d/%d\n", query_idx + 1, batch_size);

        /* Get all assigned_dpus visited by this query */
        size_t num_dpu_for_query = query_to_dpu_set[query_idx]->size;
        int* assigned_dpus = hashtable_get_all_keys(query_to_dpu_set[query_idx]);
        
        /* Create independent result_buffer for each assigned_dpu - use array storage to avoid pointer conversion issues */
        result_buffer_t** dpu_result_buffers = (result_buffer_t**)malloc(num_dpu_for_query * sizeof(result_buffer_t*));
        HashTable* dpu_to_buffer_index = hashtable_create(num_dpu_for_query);
        
        for (size_t dpu_idx = 0; dpu_idx < num_dpu_for_query; dpu_idx++) {
            int dpu_id = assigned_dpus[dpu_idx];
            result_buffer_t* result_buf = (result_buffer_t*)malloc(sizeof(result_buffer_t));
            ResultID* result_ids = (ResultID*)malloc(post_ef * sizeof(ResultID));
            DIST* result_dists = (DIST*)malloc(post_ef * sizeof(DIST));
            
            if (!result_buf || !result_ids || !result_dists) {
                fprintf(stderr, "Failed to allocate result buffer for DPU %d\n", dpu_id);
                continue;
            }
            
            dpu_result_buffer_init(result_buf, result_ids, result_dists, post_ef);
            dpu_result_buffers[dpu_idx] = result_buf;
            
            // Store index instead of pointer - use int_array_push
            IntArray* idx_wrapper = int_array_create(1);
            int_array_push(idx_wrapper, (int)dpu_idx);
            hashtable_insert(dpu_to_buffer_index, dpu_id, idx_wrapper);
            // hashtable_insert copies IntArray struct contents, so only free the shell, not the data
            free(idx_wrapper);
        }

        /* Search in each selected cluster */
        ResultBuffer_C* all_results = result_buffer_create(topk);
        
        /* Initialize CPU version search buffers */
        candidate_t* search_data = (candidate_t*)malloc(ef * sizeof(candidate_t));
        PID* hash_table = (PID*)malloc(1024 * sizeof(PID));
        PID* overflow_table = (PID*)malloc(512 * sizeof(PID));
        
        if (!search_data || !hash_table || !overflow_table) {
            SET_ERROR(ERR_NOMEM, "Failed to allocate CPU buffers");
            // Clean up allocated memory - must check for NULL
            if (search_data) free(search_data);
            if (hash_table) free(hash_table);
            if (overflow_table) free(overflow_table);
            if (all_results) result_buffer_free(all_results);
            // Clean up DPU buffers
            for (size_t i = 0; i < num_dpu_for_query; i++) {
                if (dpu_result_buffers[i]) {
                    if (dpu_result_buffers[i]->ids) free(dpu_result_buffers[i]->ids);
                    if (dpu_result_buffers[i]->distances) free(dpu_result_buffers[i]->distances);
                    free(dpu_result_buffers[i]);
                }
            }
            free(dpu_result_buffers);
            free(assigned_dpus);
            // Create empty result
            batch_results[query_idx] = int_array_create(0);
            continue;
        }
        
        search_buffer_t search_buf;
        hashset_t visited_set;
        
        /* Initialize buffers - use DPU buffer functions, capacity matching allocation size */
        search_buffer_init(&search_buf, search_data, ef);
        hashset_init(&visited_set, hash_table, 1024, overflow_table, 512);
        
        /* Get query vector */
        const FloatArray* query_vec = &queries->rows[query_idx];
        
        /* Search each selected centroid for this query */
        for (int probe_idx = 0; probe_idx < nprobe; probe_idx++) {
            search_buffer_clear(&search_buf);
            hashset_clear(&visited_set);
            int centroid_id = nearest_centroids->data[query_idx * nprobe + probe_idx].index;
            // printf("centroid_id: %d\n", centroid_id);
            
            /* Check if the cluster exists and has sufficient data */
            IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, centroid_id);
            if (!cluster_indices || cluster_indices->size < 2) {
                continue;
            }
            
            if (centroid_id >= searcher->num_indices || !searcher->indices[centroid_id]) {
                continue;
            }
            
            /* Get the assigned_dpu for this centroid */
            IntArray* dpu_wrapper = hashtable_get(query_centroid_to_dpu[query_idx], centroid_id);
            if (!dpu_wrapper || dpu_wrapper->size == 0) {
                continue;
            }
            int assigned_dpu_for_centroid = dpu_wrapper->data[0];
            
            /* Get the result_buffer for this DPU */
            IntArray* buf_wrapper = hashtable_get(dpu_to_buffer_index, assigned_dpu_for_centroid);
            if (!buf_wrapper) {
                continue;
            }
            result_buffer_t* result_buf_for_dpu = dpu_result_buffers[buf_wrapper->data[0]];
            
            IndexWrapper* wrapper = searcher->indices[centroid_id];
            
            /* Set search parameters */
            index_wrapper_set_ef(wrapper, ef);
            index_wrapper_set_post_ef(wrapper, post_ef);
            
            /* Get LUT and sumq for this query */
            float* lut = (float*)malloc((PADDED_DIMM >> 2) * 16 * sizeof(float));
            float sumq;
            if (!lut) continue;
            
            index_wrapper_get_lut_sumq(wrapper, query_vec->data, lut, &sumq);

            // printf("sumq: %f\n", sumq);
            //print lut
            // for (int i = 0; i < (PADDED_DIMM >> 2) * 16; i++) {
            //     printf("lut[%d]: %f\n", i, lut[i]);
            // }
            
            /* Get entry point */
            uint32_t entry_point = index_wrapper_get_entry_point(wrapper);
            const float* ep_vector = searcher->base_data->rows[cluster_indices->data[entry_point]].data;
            float ep_dist = compute_sqr_dist_cpu(query_vec->data, ep_vector, query_vec->size);
            
            /* Insert entry point into search buffer */
            search_buffer_insert(&search_buf, (PID)entry_point, (float)ep_dist);
            
            /* Beam search main loop */
            while (search_buffer_has_next(&search_buf)) {
                float current_dist;
                PID cur_id = search_buffer_pop_with_dist(&search_buf, &current_dist);
                
                if (cur_id == -1) break;
                
                /* Check if already visited */
                if (hashset_get(&visited_set, cur_id)) {
                    continue;
                }
                
                /* Mark as visited */
                hashset_set(&visited_set, cur_id);
                
                /* Insert into result buffer - use the result_buffer for this DPU */
                dpu_result_buffer_insert(result_buf_for_dpu, cur_id, current_dist, centroid_id);
                // printf("Inserted result %d with distance %f into result buffer\n", cur_id, current_dist);
                
                /* Get neighbors */
                const uint32_t* neighbors = index_wrapper_get_neighbors(wrapper, cur_id);
                if (!neighbors) continue;
                
                /* Expand neighbors */
                for (int k = 0; k < DEGREE; k++) {
                    PID neighbor_id = (PID)neighbors[k];
                    if ((unsigned int)neighbor_id == KPID_MAX || hashset_get(&visited_set, neighbor_id)) {
                        continue;
                    }
                    /* Get packed data of neighbor */
                    const uint64_t* rabit_code = index_wrapper_get_packed_code(wrapper, neighbor_id);
                    const float* factor = index_wrapper_get_factor(wrapper, neighbor_id);
                    // printf("factor: %f\n", *factor);
                    //print rabit_code
                    // for (int i = 0; i < (PADDED_DIMM >> 6); i++) {
                    //     printf("rabit_code[%d]: %llu\n", i, rabit_code[i]);
                    // }
                    
                    if (!rabit_code || !factor) continue;
                    // printf("neighbor_id: %d, ", neighbor_id);
                    /* Compute approximate distance */
                    float appro_dist = compute_appro_dist_cpu(lut, sumq, rabit_code, factor);
                    // printf("appro_dist: %f\n", appro_dist);
                    
                    /* Check if should be inserted into search buffer */
                    if (!search_buffer_is_full(&search_buf, appro_dist)) {
                        search_buffer_insert(&search_buf, neighbor_id, appro_dist);
                    }
                }
            }
            
            // Must free lut; it is allocated in each loop iteration
            free(lut);
        }
        
        /* Create result for this query */
        batch_results[query_idx] = int_array_create(topk);
        
        /* Collect results from all assigned_dpu result_buffers, convert to global indices and compute exact distances */
        for (size_t dpu_idx = 0; dpu_idx < num_dpu_for_query; dpu_idx++) {
            result_buffer_t* result_buf_for_dpu = dpu_result_buffers[dpu_idx];
            // printf("query_idx: %d, dpu_idx: %d, result_buf_for_dpu->size: %d\n", query_idx, dpu_idx, result_buf_for_dpu->size);
            
            /* Convert results in this DPU's ResultBuffer to global indices and compute exact distances */
            ResultID* cpu_results = (ResultID*)malloc(result_buf_for_dpu->size * sizeof(ResultID));
            if (cpu_results) {
                dpu_result_buffer_copy_results(result_buf_for_dpu, cpu_results, result_buf_for_dpu->size);
                
                for (size_t i = 0; i < result_buf_for_dpu->size; i++) {
                    PID local_idx = cpu_results[i].id;
                    int32_t centroid_id = cpu_results[i].centroid_id;
                    
                    IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, centroid_id);
                    if (cluster_indices && local_idx < (int)cluster_indices->size) {
                        uint32_t global_idx = cluster_indices->data[local_idx];
                        if (global_idx < searcher->base_data->num_rows) {
                            float exact_dist = sqr_dist(query_vec, &searcher->base_data->rows[global_idx]);
                            result_buffer_insert(all_results, global_idx, exact_dist);
                            // printf("global_idx: %d, exact_dist: %f\n", global_idx, exact_dist);
                        }
                    }
                }
                free(cpu_results);
            }
        }
        
        /* Extract topk from merged results */
        uint32_t* result_ids_final = (uint32_t*)malloc(topk * sizeof(uint32_t));
        if (result_ids_final) {
            size_t copied = result_buffer_copy_results(all_results, result_ids_final, NULL, topk);
            for (size_t i = 0; i < copied; i++) {
                int_array_push(batch_results[query_idx], (int)result_ids_final[i]);
            }
            free(result_ids_final);
        }
        
        /* Clean up buffers - this is critical; all allocated memory must be freed */
        free(search_data);
        free(hash_table);
        free(overflow_table);
        result_buffer_free(all_results);
        
        /* Clean up result_buffer for each DPU */
        for (size_t dpu_idx = 0; dpu_idx < num_dpu_for_query; dpu_idx++) {
            result_buffer_t* result_buf = dpu_result_buffers[dpu_idx];
            if (result_buf) {
                if (result_buf->ids) free(result_buf->ids);
                if (result_buf->distances) free(result_buf->distances);
                free(result_buf);
            }
        }
        free(dpu_result_buffers);
        
        // Use hashtable_free to free hash table and its data pointers
        // (cannot use int_array_free because hashtable_get returns internal pointers)
        hashtable_free(dpu_to_buffer_index);
        free(assigned_dpus);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("Total time: %f seconds\n", total_time);
    
    /* Clean up nearest_centroids */
    dist_index_array_free(nearest_centroids);
    
    printf("CPU version search completed\n");
    return batch_results;
}

IntArray** ivf_searcher_search_batch_only_CPU(
                                                struct dpu_set_t* dpu_set,
                                                IVFSearcher* searcher,
                                              const FloatMatrix* queries,
                                              int batch_size,
                                              int ef,
                                              int topk,
                                              int nprobe,
                                              int post_ef) {
    if (!searcher || !queries || batch_size <= 0) {
        return NULL;
    }
    
    printf("Starting CPU version search for DPU code accuracy verification...\n");

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    /* Allocate result array */
    IntArray** batch_results = (IntArray**)malloc(batch_size * sizeof(IntArray*));
    if (!batch_results) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate batch results array");
        return NULL;
    }
    
    /* Find the nearest nprobe centroids */
    DistIndexArray* nearest_centroids = find_nearest_centroids_batch(queries, 
                                                                    batch_size,
                                                                    searcher->centroids, 
                                                                    nprobe);
    
    /* Search for each query */
    for (int query_idx = 0; query_idx < batch_size; query_idx++) {
        // printf("Processing query %d/%d\n", query_idx + 1, batch_size);

        /* Search in each selected cluster */
        ResultBuffer_C* all_results = result_buffer_create(topk);

        
        /* Initialize CPU version buffers */
        candidate_t* search_data = (candidate_t*)malloc(ef * sizeof(candidate_t));
        ResultID* result_ids = (ResultID*)malloc(post_ef * sizeof(ResultID));
        float* result_dists = (float*)malloc(post_ef * sizeof(float));
        // ResultID* tmp_result_ids = (ResultID*)malloc(topk * sizeof(ResultID));
        // float* tmp_result_dists = (float*)malloc(topk * sizeof(float));
        PID* hash_table = (PID*)malloc(1024 * sizeof(PID));
        PID* overflow_table = (PID*)malloc(512 * sizeof(PID));
        
        if (!search_data || !result_ids || !result_dists || !hash_table || !overflow_table) {
            SET_ERROR(ERR_NOMEM, "Failed to allocate CPU buffers");
            // Clean up allocated memory
            free(search_data);
            free(result_ids);
            free(result_dists);
            // free(tmp_result_ids);
            // free(tmp_result_dists);
            free(hash_table);
            free(overflow_table);
            continue;
        }
        
        search_buffer_t search_buf;
        result_buffer_t result_buf;
        // result_buffer_t tmp_result_buf;
        hashset_t visited_set;
        
        /* Initialize buffers - use DPU buffer functions */
        search_buffer_init(&search_buf, search_data, ef);
        dpu_result_buffer_init(&result_buf, result_ids, result_dists, post_ef);  // This is the DPU version
        // dpu_result_buffer_init(&tmp_result_buf, tmp_result_ids, tmp_result_dists, topk);
        hashset_init(&visited_set, hash_table, 1024, overflow_table, 512);
        
        /* Get query vector */
        const FloatArray* query_vec = &queries->rows[query_idx];
        
        /* Search each selected centroid for this query */
        for (int probe_idx = 0; probe_idx < nprobe; probe_idx++) {
            // dpu_result_buffer_clear(&result_buf);
            search_buffer_clear(&search_buf);
            hashset_clear(&visited_set);
            int centroid_id = nearest_centroids->data[query_idx * nprobe + probe_idx].index;
            // printf("centroid_id: %d\n", centroid_id);
            
            /* Check if the cluster exists and has sufficient data */
            IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, centroid_id);
            if (!cluster_indices || cluster_indices->size < 2) {
                continue;
            }
            
            if (centroid_id >= searcher->num_indices || !searcher->indices[centroid_id]) {
                continue;
            }
            
            IndexWrapper* wrapper = searcher->indices[centroid_id];
            
            /* Set search parameters */
            index_wrapper_set_ef(wrapper, ef);
            index_wrapper_set_post_ef(wrapper, post_ef);
            
            /* Get LUT and sumq for this query */
            float* lut = (float*)malloc((PADDED_DIMM >> 2) * 16 * sizeof(float));
            float sumq;
            if (!lut) continue;
            
            index_wrapper_get_lut_sumq(wrapper, query_vec->data, lut, &sumq);

            // printf("sumq: %f\n", sumq);
            //print lut
            // for (int i = 0; i < (PADDED_DIMM >> 2) * 16; i++) {
            //     printf("lut[%d]: %f\n", i, lut[i]);
            // }

            /* Get entry point */
            uint32_t entry_point = index_wrapper_get_entry_point(wrapper);
            const float* ep_vector = searcher->base_data->rows[cluster_indices->data[entry_point]].data;
            int32_t ep_dist = (int32_t)compute_sqr_dist_cpu(query_vec->data, ep_vector, query_vec->size);
            
            /* Insert entry point into search buffer */
            search_buffer_insert(&search_buf, (PID)entry_point, ep_dist);
            
            /* Beam search main loop */
            while (search_buffer_has_next(&search_buf)) {
                DIST current_dist;
                PID cur_id = search_buffer_pop_with_dist(&search_buf, &current_dist);
                
                if (cur_id == -1) break;
                
                /* Check if already visited */
                if (hashset_get(&visited_set, cur_id)) {
                    continue;
                }
                
                /* Mark as visited */
                hashset_set(&visited_set, cur_id);
                
                /* Insert into result buffer - use DPU buffer functions */
                dpu_result_buffer_insert(&result_buf, cur_id, current_dist, centroid_id);
                // uint32_t global_idx = cluster_indices->data[cur_id];
                // printf("Inserted result %d with distance %d into result buffer\n", cur_id, current_dist);
                
                /* Get neighbors */
                const uint32_t* neighbors = index_wrapper_get_neighbors(wrapper, cur_id);
                if (!neighbors) continue;
                
                /* Expand neighbors */
                for (int k = 0; k < DEGREE; k++) {
                    PID neighbor_id = (PID)neighbors[k];
                    if ((unsigned int)neighbor_id == KPID_MAX || hashset_get(&visited_set, neighbor_id)) {
                        continue;
                    }
                    /* Get packed data of neighbor */
                    const uint64_t* rabit_code = index_wrapper_get_packed_code(wrapper, neighbor_id);
                    const float* factor = index_wrapper_get_factor(wrapper, neighbor_id);
                    // printf("factor: %f\n", *factor);
                    //print rabit_code
                    // for (int i = 0; i < (PADDED_DIMM >> 6); i++) {
                    //     printf("rabit_code[%d]: %llu\n", i, rabit_code[i]);
                    // }
                    
                    if (!rabit_code || !factor) continue;
                    
                    /* Compute approximate distance */
                    float appro_dist = compute_appro_dist_cpu(lut, sumq, rabit_code, factor);
                    // printf("neighbor_id: %d, appro_dist: %d\n", neighbor_id, appro_dist);
                    
                    /* Check if should be inserted into search buffer */
                    if (!search_buffer_is_full(&search_buf, appro_dist)) {
                        search_buffer_insert(&search_buf, neighbor_id, appro_dist);
                    }
                }
            }
            
            free(lut);
            // ResultID* tmp_cpu_result_ids = (ResultID*)malloc(result_buf.size * sizeof(ResultID));
            // dpu_result_buffer_copy_results(&result_buf, tmp_cpu_result_ids, result_buf.size);  // This is the DPU version
            // for (size_t i = 0; i < result_buf.size; i++) {
            //     PID local_idx = tmp_cpu_result_ids[i].id;
            //     int32_t centroid_id = tmp_cpu_result_ids[i].centroid_id;

            //     IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, centroid_id);
            //     if (cluster_indices && local_idx < (int)cluster_indices->size) {
            //         uint32_t global_idx = cluster_indices->data[local_idx];
            //         if (global_idx < searcher->base_data->num_rows) {
            //             float exact_dist = sqr_dist(query_vec, &searcher->base_data->rows[global_idx]);
            //             dpu_result_buffer_insert(&tmp_result_buf, local_idx, exact_dist, centroid_id);
            //         }
            //     }
            // }
            // free(tmp_cpu_result_ids);
        }
        
        /* Create result for this query */
        batch_results[query_idx] = int_array_create(topk);
        
        /* Convert results in ResultBuffer to global indices and compute exact distances */
        ResultID* cpu_results = (ResultID*)malloc(result_buf.size * sizeof(ResultID));
        if (cpu_results) {
            dpu_result_buffer_copy_results(&result_buf, cpu_results, result_buf.size);  // This is the DPU version
            // printf("query_idx: %d, result_buf.size: %d\n", query_idx, result_buf.size);
            if (cpu_results) {
                for (size_t i = 0; i < result_buf.size; i++) {
                    PID local_idx = cpu_results[i].id;
                    int32_t centroid_id = cpu_results[i].centroid_id;
                    
                    IntArray* cluster_indices = cluster_mapping_table_get(searcher->cluster_to_original, centroid_id);
                    if (cluster_indices && local_idx < (int)cluster_indices->size) {
                        uint32_t global_idx = cluster_indices->data[local_idx];
                        if (global_idx < (uint32_t)searcher->total_num_elements) {
                            float exact_dist = sqr_dist(query_vec, &searcher->base_data->rows[global_idx]);
                            result_buffer_insert(all_results, global_idx, exact_dist);
                            // printf("global_idx: %d, exact_dist: %f\n", global_idx, exact_dist);
                        }
                    }
                }
                
                uint32_t* result_ids_final = (uint32_t*)malloc(topk * sizeof(uint32_t));
                result_buffer_copy_results(all_results, result_ids_final, NULL, topk); 
                for (int i = 0; i < topk; i++) {
                    int_array_push(batch_results[query_idx], (int)result_ids_final[i]);
                    // printf("Inserted result %d into batch results\n", result_ids_final[i]);
                }
                free(result_ids_final);
                free(cpu_results);
            }
        }
        
        /* Clean up buffers */
        free(search_data);
        free(result_ids);
        free(result_dists);
        // free(tmp_result_ids);
        // free(tmp_result_dists);
        free(hash_table);
        free(overflow_table);
        result_buffer_free(all_results);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("Total time: %f seconds\n", total_time);
    /* Clean up */
    dist_index_array_free(nearest_centroids);
    
    printf("CPU version search completed\n");
    return batch_results;
}


