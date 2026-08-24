/* ===========================================================================
 * test_flush_callback.c
 *
 * Standalone CPU-only benchmark that replicates the computation core of
 * flush_output_fifo_callback() in src/ivf_search.c, without any DPU SDK
 * dependency. Use this on a server that does NOT have DPU hardware to
 * measure the pure CPU time of the callback processing step.
 *
 * Compile:
 *   gcc -O2 -o test_flush test_flush_callback.c -lm
 *
 * Run:
 *   ./test_flush [total_rounds] [bench_iters]
 *     total_rounds : simulated xfer rounds per iteration  (default = 200)
 *     bench_iters  : number of repeated timing iterations (default = 10)
 *
 * After running, copy the printed "per-round time (ms)" into ivf_search.c:
 *   #define CPU_FLUSH_CALLBACK_TIME_PER_ROUND  <value_in_seconds>
 *
 * How 'total_rounds' maps to the real workload:
 *   rounds ≈ (BATCH_SIZE × nprobe) / (NR_DPUS × QUERIES_PER_PUSH)
 *   Example: (200 × 128) / (100 × 16) = 16 rounds per batch call
 *   Use a larger value (e.g. 200) for stable measurement then scale down.
 * ===========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ===========================================================================
 * Section 1: Configuration – match these with your actual system settings
 *            (same as settings.h / fifo_types.h in the main project)
 * ===========================================================================*/

#define NR_DPUS              2560   /* total number of physical DPUs */
#define DPUS_PER_RANK        64    /* DPUs per rank – rank 0 gets 64, rank 1 gets the rest */
#define NR_RANKS             40     /* number of DPU ranks */
#define BATCH_SIZE           10000   /* queries per search batch */
#define QUERIES_PER_PUSH     16    /* queries packed into one FIFO push */
#define POST_EF              128    /* max candidate IDs returned per DPU per query */
#define OUTPUT_FIFO_PTR_SIZE 1     /* FIFO slot count = 1 << OUTPUT_FIFO_PTR_SIZE */
#define DIMM                 128   /* query vector dimension */
#define NUM_CLUSTERS         128  /* number of IVF clusters (searcher->num_indices) */
#define CLUSTER_SIZE         10000 /* elements per cluster (approx) */
#define TOPK                 10    /* top-K result buffer capacity */

#define DEFAULT_TOTAL_ROUNDS 1   /* default simulation rounds */
#define DEFAULT_BENCH_ITERS  10    /* default repeated timing runs */

/* Derived constants */
#define OUTPUT_FIFO_CAPACITY (1 << OUTPUT_FIFO_PTR_SIZE)   /* 2 slots */
#define TOTAL_ELEMENTS       ((uint64_t)NUM_CLUSTERS * CLUSTER_SIZE)

/* ===========================================================================
 * Section 2: Data structures – copied/simplified from project headers
 * ===========================================================================*/

/* ----- FIFO output types (must match fifo_types.h exactly) ----- */
typedef struct {
    int32_t query_id;            /* query ID */
    int16_t centroid_id;         /* centroid ID */
    int16_t flag;                /* 1 = valid result */
    int32_t results[POST_EF];    /* local candidate indices within the cluster */
} fifo_query_output_t;

typedef struct {
    fifo_query_output_t results[QUERIES_PER_PUSH];
} fifo_batch_output_t;

#define OUTPUT_FIFO_DATA_SIZE  ((size_t)sizeof(fifo_batch_output_t))
#define OUTPUT_FIFO_SIZE       (OUTPUT_FIFO_CAPACITY * OUTPUT_FIFO_DATA_SIZE)

/* ----- Minimal integer array ----- */
typedef struct {
    int   *data;
    int    size;
} IntArray;

/* ----- Minimal float vector (for query) ----- */
typedef struct {
    float *data;
    int    size;
} FloatVec;

/* ----- Simple bounded sorted result buffer (replaces C++ ResultBuffer_C) ----- */
typedef struct {
    uint64_t *ids;
    float    *dists;
    int       size;
    int       capacity;
} ResultBuf;

/* ===========================================================================
 * Section 3: Global simulation state
 * ===========================================================================*/

static IntArray   *g_clusters[NUM_CLUSTERS];          /* cluster -> local index array */
static FloatVec   *g_queries[BATCH_SIZE];              /* query vectors               */
static ResultBuf  **g_results[NR_RANKS];              /* [rank_id][query_id]          */
static uint8_t    *g_output_fifo_data = NULL;          /* DPU output FIFO buffer       */
static uint16_t    g_fifo_sz[NR_DPUS];                /* mock FIFO read size per DPU  */
static int         g_queries_sent[NR_DPUS];            /* diagnostic counter           */

/* Precomputed squared norms of queries (||q||^2, used as distance proxy) */
static float       g_query_sq_norm[BATCH_SIZE];

/* ===========================================================================
 * Section 4: Helper – result_buf_insert (sorted insert, bounded capacity)
 *            Replicates the C++ ResultBuffer::insert behaviour
 * ===========================================================================*/
static void result_buf_insert(ResultBuf *buf, uint64_t id, float dist)
{
    if (buf->size == buf->capacity && dist >= buf->dists[buf->size - 1])
        return; /* worse than worst – discard */

    /* Binary search for insertion position */
    int lo = 0, hi = buf->size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (buf->dists[mid] < dist) lo = mid + 1;
        else                        hi = mid;
    }

    /* Shift elements right to make room */
    int move = buf->size - lo;
    if (buf->size == buf->capacity && move > 0)
        move--; /* overwrite the last slot */
    if (move > 0) {
        memmove(&buf->ids[lo + 1],   &buf->ids[lo],   (size_t)move * sizeof(uint64_t));
        memmove(&buf->dists[lo + 1], &buf->dists[lo], (size_t)move * sizeof(float));
    }

    buf->ids[lo]   = id;
    buf->dists[lo] = dist;
    if (buf->size < buf->capacity)
        buf->size++;
}

/* ===========================================================================
 * Section 5: Mock DPU FIFO accessors
 *
 * In the real DPU SDK:
 *   get_fifo_size(link, dpu)            – reads head/tail pointers from DPU WRAM
 *   get_fifo_elem(link, dpu, base, fi)  – returns pointer to the fi-th element
 *                                         using the circular read pointer
 *
 * Here we use a simple linear layout (read pointer starts at 0).
 * ===========================================================================*/

static inline uint16_t mock_get_fifo_size(int dpu_id)
{
    return g_fifo_sz[dpu_id];
}

static inline fifo_batch_output_t *mock_get_fifo_elem(int dpu_id, int fi)
{
    uint8_t *base = &g_output_fifo_data[(size_t)dpu_id * OUTPUT_FIFO_SIZE];
    return (fifo_batch_output_t *)(base + (size_t)fi * OUTPUT_FIFO_DATA_SIZE);
}

/* Compute DPU global ID: rank_id * 64 + local_dpu_index_within_rank
 * Mirrors GET_DPU_ID_BY_DPU in ivf_search.h */
static inline int mock_get_dpu_id(uint32_t rank_id, int local_idx)
{
    return (int)(rank_id * DPUS_PER_RANK + local_idx);
}

/* ===========================================================================
 * Section 6: Core computation – mirrors flush_output_fifo_callback()
 *
 * Parameters:
 *   rank_id    – rank being processed
 *   dpu_start  – first DPU global ID in this rank
 *   dpu_end    – one-past-last DPU global ID
 *   num_clusters – searcher->num_indices
 *   total_elems  – searcher->total_num_elements
 * ===========================================================================*/
static void mock_flush_output_fifo_callback(
        uint32_t rank_id,
        int      dpu_start,
        int      dpu_end,
        int      num_clusters,
        uint64_t total_elems)
{
    const uint16_t max_fifo_capacity = (uint16_t)OUTPUT_FIFO_CAPACITY;

    /* Iterate over every DPU in this rank – mirrors DPU_FOREACH */
    for (int dpu_id = dpu_start; dpu_id < dpu_end; dpu_id++) {

        uint16_t sz = mock_get_fifo_size(dpu_id);
        if (sz == 0 || sz > max_fifo_capacity) continue;

        for (int fi = 0; fi < (int)sz; fi++) {
            fifo_batch_output_t *batch_out = mock_get_fifo_elem(dpu_id, fi);

            for (int q = 0; q < QUERIES_PER_PUSH; q++) {
                fifo_query_output_t *out = &batch_out->results[q];

                int query_id    = out->query_id;
                int centroid_id = (int)(int16_t)out->centroid_id;
                int flag        = (int)(int16_t)out->flag;

                if (flag != 1 ||
                    query_id    < 0 || query_id    >= BATCH_SIZE   ||
                    centroid_id < 0 || centroid_id >= num_clusters)
                    continue;

                /* Mirrors: flush_args->queries_sent[dpu_id]-- */
                g_queries_sent[dpu_id]--;
                if (g_queries_sent[dpu_id] < 0)
                    g_queries_sent[dpu_id] = 0;

                IntArray *cluster_indices = g_clusters[centroid_id];
                if (!cluster_indices) continue;

                for (int r = 0; r < POST_EF; r++) {
                    int32_t local_idx = out->results[r];
                    if (local_idx == -1) break;
                    if (local_idx >= 0 && local_idx < cluster_indices->size) {
                        uint64_t global_idx = (uint64_t)cluster_indices->data[local_idx];
                        if (global_idx < total_elems) {
                            /*
                             * Original: sqr_dist(&queries->rows[query_id], zero_vector)
                             * Since zero_vector is all-zeros, this equals ||query||^2.
                             * Note: the original code recomputes this for every r, which
                             * is redundant – we replicate the same pattern for timing
                             * accuracy.  g_query_sq_norm[] is precomputed but the loop
                             * is kept to match the original memory access pattern.
                             */
                            float dist = 0.0f;
                            const float *qdata = g_queries[query_id]->data;
                            for (int d = 0; d < DIMM; d++)
                                dist += qdata[d] * qdata[d];

                            result_buf_insert(g_results[rank_id][query_id],
                                              global_idx, dist);
                        }
                    }
                }
            }
        }
    }
}

/* ===========================================================================
 * Section 7: Data initialisation
 * ===========================================================================*/

/* Simple LCG pseudo-random generator (avoids stdlib rand() non-determinism) */
static uint32_t lcg_state = 12345;
static inline uint32_t lcg_rand(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

static void init_data(void)
{
    /* ---- cluster mappings ---- */
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        g_clusters[c] = (IntArray *)malloc(sizeof(IntArray));
        g_clusters[c]->data = (int *)malloc((size_t)CLUSTER_SIZE * sizeof(int));
        g_clusters[c]->size = CLUSTER_SIZE;
        for (int i = 0; i < CLUSTER_SIZE; i++)
            g_clusters[c]->data[i] = c * CLUSTER_SIZE + i; /* unique global IDs */
    }

    /* ---- query vectors ---- */
    for (int qi = 0; qi < BATCH_SIZE; qi++) {
        g_queries[qi] = (FloatVec *)malloc(sizeof(FloatVec));
        g_queries[qi]->data = (float *)malloc((size_t)DIMM * sizeof(float));
        g_queries[qi]->size = DIMM;
        float sq_norm = 0.0f;
        for (int d = 0; d < DIMM; d++) {
            float v = (float)(lcg_rand() % 1000) / 1000.0f - 0.5f;
            g_queries[qi]->data[d] = v;
            sq_norm += v * v;
        }
        g_query_sq_norm[qi] = sq_norm;
    }

    /* ---- result buffers ---- */
    for (int ri = 0; ri < NR_RANKS; ri++) {
        g_results[ri] = (ResultBuf **)malloc((size_t)BATCH_SIZE * sizeof(ResultBuf *));
        for (int qi = 0; qi < BATCH_SIZE; qi++) {
            ResultBuf *rb = (ResultBuf *)malloc(sizeof(ResultBuf));
            rb->ids    = (uint64_t *)malloc((size_t)TOPK * sizeof(uint64_t));
            rb->dists  = (float *)   malloc((size_t)TOPK * sizeof(float));
            rb->size     = 0;
            rb->capacity = TOPK;
            g_results[ri][qi] = rb;
        }
    }

    /* ---- output FIFO buffer ----
     * Allocated as: NR_DPUS × OUTPUT_FIFO_SIZE bytes.
     * Each DPU's FIFO region starts at dpu_id * OUTPUT_FIFO_SIZE.
     * Filled with synthetic valid results. */
    size_t fifo_total = (size_t)NR_DPUS * OUTPUT_FIFO_SIZE;
    g_output_fifo_data = (uint8_t *)calloc(fifo_total, 1);

    for (int dpu_id = 0; dpu_id < NR_DPUS; dpu_id++) {
        for (int fi = 0; fi < OUTPUT_FIFO_CAPACITY; fi++) {
            fifo_batch_output_t *batch =
                (fifo_batch_output_t *)&g_output_fifo_data[
                    (size_t)dpu_id * OUTPUT_FIFO_SIZE + fi * OUTPUT_FIFO_DATA_SIZE];

            for (int q = 0; q < QUERIES_PER_PUSH; q++) {
                fifo_query_output_t *fq = &batch->results[q];
                fq->flag        = 1;                             /* valid */
                fq->query_id    = (int32_t)(lcg_rand() % BATCH_SIZE);
                fq->centroid_id = (int16_t)(lcg_rand() % NUM_CLUSTERS);
                for (int r = 0; r < POST_EF; r++)
                    fq->results[r] = (int32_t)(lcg_rand() % CLUSTER_SIZE);
            }
        }
        /* Set each DPU to have 1 readable FIFO entry per round */
        g_fifo_sz[dpu_id]      = 1;
        g_queries_sent[dpu_id] = QUERIES_PER_PUSH;
    }
}

/* Reset result buffers between rounds so insertion logic is exercised */
static void reset_results(void)
{
    for (int ri = 0; ri < NR_RANKS; ri++)
        for (int qi = 0; qi < BATCH_SIZE; qi++)
            g_results[ri][qi]->size = 0;
    for (int dpu_id = 0; dpu_id < NR_DPUS; dpu_id++)
        g_queries_sent[dpu_id] = QUERIES_PER_PUSH;
}

/* ===========================================================================
 * Section 8: Timing utilities
 * ===========================================================================*/

static inline double timespec_to_sec(const struct timespec *ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec * 1e-9;
}

static inline double elapsed_sec(const struct timespec *t0, const struct timespec *t1)
{
    return timespec_to_sec(t1) - timespec_to_sec(t0);
}

/* ===========================================================================
 * Section 9: Main
 * ===========================================================================*/

int main(int argc, char *argv[])
{
    int total_rounds = DEFAULT_TOTAL_ROUNDS;
    int bench_iters  = DEFAULT_BENCH_ITERS;

    if (argc >= 2) total_rounds = atoi(argv[1]);
    if (argc >= 3) bench_iters  = atoi(argv[2]);
    if (total_rounds <= 0) total_rounds = DEFAULT_TOTAL_ROUNDS;
    if (bench_iters  <= 0) bench_iters  = DEFAULT_BENCH_ITERS;

    printf("=== flush_output_fifo_callback CPU Benchmark ===\n");
    printf("Configuration:\n");
    printf("  NR_DPUS          = %d\n", NR_DPUS);
    printf("  NR_RANKS         = %d\n", NR_RANKS);
    printf("  DPUS_PER_RANK    = %d\n", DPUS_PER_RANK);
    printf("  BATCH_SIZE       = %d\n", BATCH_SIZE);
    printf("  QUERIES_PER_PUSH = %d\n", QUERIES_PER_PUSH);
    printf("  POST_EF          = %d\n", POST_EF);
    printf("  NUM_CLUSTERS     = %d\n", NUM_CLUSTERS);
    printf("  CLUSTER_SIZE     = %d\n", CLUSTER_SIZE);
    printf("  DIMM             = %d\n", DIMM);
    printf("  TOPK             = %d\n", TOPK);
    printf("  FIFO_CAPACITY    = %d\n", OUTPUT_FIFO_CAPACITY);
    printf("  total_rounds     = %d\n", total_rounds);
    printf("  bench_iters      = %d\n", bench_iters);
    printf("\n");

    printf("Initialising synthetic data ... ");
    fflush(stdout);
    init_data();
    printf("done.\n\n");

    /* Compute rank boundaries (same logic as ivf_search.c):
     *   rank 0 : DPU IDs [0 .. DPUS_PER_RANK)
     *   rank 1 : DPU IDs [DPUS_PER_RANK .. NR_DPUS)
     * For a single rank configuration set NR_RANKS = 1. */
    int rank_start[NR_RANKS], rank_end[NR_RANKS];
    for (int ri = 0; ri < NR_RANKS; ri++) {
        rank_start[ri] = ri * DPUS_PER_RANK;
        rank_end[ri]   = rank_start[ri] + DPUS_PER_RANK;
        if (rank_end[ri] > NR_DPUS) rank_end[ri] = NR_DPUS;
    }

    /* ---- Benchmark loop ---- */
    double iter_times[DEFAULT_BENCH_ITERS + 1]; /* max bench_iters */
    double *times = (double *)malloc((size_t)(bench_iters + 1) * sizeof(double));
    (void)iter_times;

    struct timespec t_start, t_end;

    for (int iter = 0; iter < bench_iters; iter++) {
        reset_results();

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        /* Simulate 'total_rounds' xfer rounds.
         * Each round: call the callback once per rank (serially here).
         * In the real system the per-rank callbacks run in parallel threads;
         * the CPU time measured here represents the SERIAL sum across ranks.
         * To get the parallel wall-clock equivalent, divide by NR_RANKS. */
        for (int round = 0; round < total_rounds; round++) {
            for (int ri = 0; ri < NR_RANKS; ri++) {
                if (rank_start[ri] >= NR_DPUS) continue;
                mock_flush_output_fifo_callback(
                    (uint32_t)ri,
                    rank_start[ri], rank_end[ri],
                    NUM_CLUSTERS,
                    TOTAL_ELEMENTS);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        times[iter] = elapsed_sec(&t_start, &t_end);
    }

    /* ---- Compute statistics ---- */
    double sum = 0.0, min_t = times[0], max_t = times[0];
    for (int i = 0; i < bench_iters; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }
    double avg = sum / bench_iters;

    printf("=== Timing Results (%d rounds × %d iters) ===\n",
           total_rounds, bench_iters);
    printf("  Total time avg   : %.6f s\n",   avg);
    printf("  Total time min   : %.6f s\n",   min_t);
    printf("  Total time max   : %.6f s\n",   max_t);
    printf("\n");
    printf("  Per-round (serial, all ranks) avg : %.6f ms\n",
           avg / total_rounds * 1e3);
    printf("  Per-round (parallel, /%d ranks) avg : %.6f ms   <-- wall-clock estimate\n",
           NR_RANKS, avg / total_rounds / NR_RANKS * 1e3);
    printf("\n");

    /* ---- Usage guidance ---- */
    double per_round_s = avg / total_rounds / NR_RANKS;
    printf("=== To use in ivf_search.c ===\n");
    printf("In ivf_search.c, add near the top of the file:\n\n");
    printf("  /* CPU baseline (measured by test_flush_callback on a DPU-free server) */\n");
    printf("  #ifndef CPU_FLUSH_CALLBACK_TIME_PER_ROUND\n");
    printf("  #define CPU_FLUSH_CALLBACK_TIME_PER_ROUND  %.9f  /* seconds */\n",
           per_round_s);
    printf("  #endif\n\n");
    printf("Then in ivf_search.c, after the 'output_flush_time' print,\n");
    printf("the comparison line will show this value × total_rounds.\n");

    free(times);
    /* (Data structures are intentionally not freed for brevity of benchmark) */
    return 0;
}
