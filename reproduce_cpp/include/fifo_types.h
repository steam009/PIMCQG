#ifndef FIFO_TYPES_H
#define FIFO_TYPES_H

#include <stdint.h>

/* Include configuration constants */
#ifndef PADDED_DIMM
#define PADDED_DIMM 512
#endif

#ifndef POST_EF
#define POST_EF 30
#endif

#ifndef MAX_QUERY_CLUSTER_PER_DPU
#define MAX_QUERY_CLUSTER_PER_DPU 2000
#endif

/* ============================================
 * FIFO configuration constants
 * ============================================ */

/* Number of queries carried per FIFO push; override with -D at compile time */
#ifndef QUERIES_PER_PUSH
#define QUERIES_PER_PUSH 15
#endif

/* rotated_query size: PADDED_DIMM int16_t values */
#define FIFO_ROTATED_QUERY_SIZE (PADDED_DIMM)

/* FIFO depth (capacity = 1 << PTR_SIZE). Override with -D at compile time if needed. */
#ifndef INPUT_FIFO_PTR_SIZE
#define INPUT_FIFO_PTR_SIZE 1
#endif

/* ============================================
 * Shared data type definitions (common to CPU/DPU)
 * ============================================ */

/* Point ID type */
#ifndef PID_DEFINED
#define PID_DEFINED
typedef int32_t PID;
#endif

/* ============================================
 * Input FIFO data structure - transferred directly to WRAM via FIFO
 * Processing request for a single query-centroid pair
 * ============================================ */
typedef struct {
    int32_t query_id;                           /* query ID (for result matching) */
    int32_t centroid_id;                        /* target centroid ID */
    int32_t ep_dist;                            /* entry point distance */
    int16_t sumq;                               /* sumq value (int16_t, can be self-computed from rotated_query on DPU) */
    int16_t pad;                                /* alignment padding */
    int16_t rotated_query[FIFO_ROTATED_QUERY_SIZE]; /* rotated query vector (replaces original lut[]) */
} fifo_input_data_t;

/* ============================================
 * Batch input structure: QUERIES_PER_PUSH queries packed into one FIFO push
 * ============================================ */
typedef struct {
    fifo_input_data_t queries[QUERIES_PER_PUSH];
} fifo_batch_input_t;

/* Input FIFO data size (in batch units) */
#define INPUT_FIFO_DATA_SIZE (sizeof(fifo_batch_input_t))

/* ============================================
 * Output FIFO data structure - transferred directly via FIFO
 * Search result for a single query-centroid pair
 * ============================================ */
/* FIFO depth (capacity = 1 << PTR_SIZE). Override with -D at compile time if needed. */
#ifndef OUTPUT_FIFO_PTR_SIZE
#define OUTPUT_FIFO_PTR_SIZE 1
#endif

typedef struct {
    int32_t query_id;               /* query ID */
    int16_t centroid_id;            /* centroid ID */
    int16_t flag;                   /* flag */
    int32_t results[POST_EF];       /* search result array */
} fifo_query_output_t;

/* ============================================
 * Batch output structure: QUERIES_PER_PUSH results packed into one FIFO output
 * ============================================ */
typedef struct {
    fifo_query_output_t results[QUERIES_PER_PUSH];
} fifo_batch_output_t;

/* Output FIFO data size (in batch units) */
#define OUTPUT_FIFO_DATA_SIZE (sizeof(fifo_batch_output_t))

/* Input FIFO capacity */
#define INPUT_FIFO_CAPACITY (1 << INPUT_FIFO_PTR_SIZE)

/* ============================================
 * FIFO flush interval configuration
 * ============================================ */
#ifndef FIFO_FLUSH_INTERVAL
#define FIFO_FLUSH_INTERVAL 1  /* sync flush on every batch push, full FIFO state control */
#endif

/* ============================================
 * DPU input buffer configuration
 * ============================================ */
#ifndef DPU_INPUT_BUFFER_CAPACITY
#define DPU_INPUT_BUFFER_CAPACITY 512  /* input buffer capacity per DPU */
#endif

#ifndef DPU_BATCH_PUSH_THRESHOLD_RATIO
#define DPU_BATCH_PUSH_THRESHOLD_RATIO 1  /* DPU ratio threshold to trigger batch push (0.0-1.0) */
#endif

/* ============================================
 * Special flag values
 * ============================================ */
#define FIFO_FLAG_NORMAL    0       /* normal processing */
#define FIFO_FLAG_TERMINATE (-1)    /* termination signal */
#define FIFO_FLAG_ERROR     (-2)    /* error signal */

#endif /* FIFO_TYPES_H */
