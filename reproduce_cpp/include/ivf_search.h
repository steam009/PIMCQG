#ifndef IVF_SEARCH_H
#define IVF_SEARCH_H

#include "utils.h"
#include <stddef.h>
#include <stdint.h>
#include <dpu.h>
#include <dpu_log.h>
#include <dpu_management.h>
#include <pthread.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

/* DPU constant definitions */
#ifndef MAX_CLUSTER_PER_DPU
#define MAX_CLUSTER_PER_DPU 256
#endif

#ifndef POST_EF
#define POST_EF 100
#endif

#ifndef MAX_SIZE_PER_DPU
#define MAX_SIZE_PER_DPU 1024
#endif

/* ALIGN macro - for DPU memory alignment */
#ifndef ALIGN
#define ALIGN(_p, _width) (((unsigned long)(_p) + (_width-1)) & (0-_width))
#endif

#ifndef MIN
#define MIN(_a, _b) (_a < _b ? _a : _b)
#endif

#define GET_DPU_ID(r_id, c_id, d_id) (r_id * 64 + c_id * 8 + d_id)
#define GET_DPU_ID_BY_DPU(d, r_id)                  \
        GET_DPU_ID(                           \
            r_id, \
            dpu_get_slice_id(d.dpu),          \
            dpu_get_member_id(d.dpu))

/* ============================================
 * CPU-side callback function argument structure
 * ============================================ */
 typedef struct {
    uint8_t *output_fifo_data;          /* output FIFO data buffer */
    size_t output_fifo_bytes;           /* output FIFO buffer total bytes (for zeroing before flush) */
    void *output_link;                  /* dpu_fifo_link_t* */
    void *searcher;                     /* IVFSearcher* */
    void *queries;                      /* FloatMatrix* */
    void *all_results;                  /* ResultBuffer_C** */
    void *cluster_mapping;              /* ClusterMappingTable* */
    void *base_data;                    /* FloatMatrix* */
    int topk;                           /* Top-K result count */
    int nr_dpus;                        /* number of DPUs */
    int batch_size;                     /* batch size */
    int *queries_sent;                 /* per-DPU independent query count (for diagnostics) */
    uint8_t *dpu_rank_id;              /* DPU rank ID */
} flush_callback_args_t;

/* Data type definitions */
typedef int32_t PID;  /* Point ID */

/* Forward declarations */
typedef struct QuantizedGraph_C QuantizedGraph_C;

/* Index wrapper structure for interacting with the symphonyqg library */
typedef struct {
    QuantizedGraph_C* index;
    size_t cluster_id;
} IndexWrapper;

/* IVF searcher structure */
typedef struct {
    IndexWrapper** indices;       /* index array, indexed by cluster_id (lazy-loaded, all NULL by default) */
    int num_indices;              /* maximum number of clusters */
    FloatMatrix* centroids;       /* centroid data */
    ClusterMappingTable* cluster_to_original;  /* cluster-to-original-index mapping */
    FloatMatrix* base_data;       /* base data */
    int total_num_elements;       /* total number of data points */
    /* Lazy-loading related fields */
    char** index_paths;           /* index file paths per cluster (NULL means not present or cleaned up) */
    size_t* index_num_elements;   /* element count per cluster (recorded at build time, available without loading) */
    int index_degree;             /* index degree at build time (for lazy IndexWrapper creation) */
} IVFSearcher;

/* =============== IndexWrapper Functions =============== */
IndexWrapper* index_wrapper_create(const char* index_type,
                                  const char* metric,
                                  size_t num_elements,
                                  size_t dimension,
                                  size_t degree_bound,
                                  int allocate_vectors);
void index_wrapper_free(IndexWrapper* wrapper);
int index_wrapper_load(IndexWrapper* wrapper, const char* filename);
int index_wrapper_save(IndexWrapper* wrapper, const char* filename);
void index_wrapper_set_ef(IndexWrapper* wrapper, size_t ef_search);
void index_wrapper_set_cluster(IndexWrapper* wrapper, size_t c_id);
void index_wrapper_set_post_ef(IndexWrapper* wrapper, size_t ef_search);
void index_wrapper_enable_profiling(IndexWrapper* wrapper, int enable);
void index_wrapper_report_timings(IndexWrapper* wrapper);
void index_wrapper_get_lut_sumq(IndexWrapper* wrapper, const float* query, float* lut, float* sumq);
void index_wrapper_get_rotated_query(IndexWrapper* wrapper, const float* query,
                                     float* rotated_query_out, float* sumq_out);
IntArray* index_wrapper_search(IndexWrapper* wrapper, const FloatArray* query, int k);

/* Other IndexWrapper functions */
size_t index_wrapper_get_num_elements(const IndexWrapper* wrapper);
size_t index_wrapper_get_dimension(const IndexWrapper* wrapper);
const float* index_wrapper_get_vector(const IndexWrapper* wrapper, uint32_t data_id);
const uint64_t* index_wrapper_get_packed_code(const IndexWrapper* wrapper, uint32_t data_id);
const float* index_wrapper_get_factor(const IndexWrapper* wrapper, uint32_t data_id);
const uint32_t* index_wrapper_get_neighbors(const IndexWrapper* wrapper, uint32_t data_id);
uint32_t index_wrapper_get_entry_point(const IndexWrapper* wrapper);

/* =============== IVFSearcher Functions =============== */
IVFSearcher* ivf_searcher_create(void);
void ivf_searcher_free(IVFSearcher* searcher);

/* Load data and indices */
int ivf_searcher_load_centroids(IVFSearcher* searcher, const char* centroids_path);
int ivf_searcher_load_cluster_mapping(IVFSearcher* searcher, const char* mapping_path);
int ivf_searcher_load_base_data(IVFSearcher* searcher, const char* base_path);
int ivf_searcher_load_indices(IVFSearcher* searcher, const char* dataset, int degree, int C);

/* IVF search */
IntArray* ivf_searcher_search(struct dpu_set_t* dpu_set,
                             IVFSearcher* searcher,
                             const FloatArray* query_vec,
                             int ef,
                             int k,
                             int nprobe,
                             int post_ef);

/* Batch IVF search 
 * @param nr_dpus virtual DPU count for load balancing simulation. When nr_dpus > NR_DPUS,
 *                only the top NR_DPUS virtual DPUs with the highest load are mapped to physical DPUs;
 *                queries for remaining virtual DPUs use random results
 */
IntArray** ivf_searcher_search_batch(struct dpu_set_t* dpu_set,
                                     IVFSearcher* searcher,
                                     const FloatMatrix* queries,
                                     int batch_size,
                                     int ef,
                                     int k,
                                     int nprobe,
                                     int post_ef,
                                     int nr_dpus);

/**
 * Compute query-to-DPU mapping (load balancing)
 * @param searcher IVF searcher
 * @param queries query vector matrix
 * @param batch_size batch size
 * @param nprobe number of centroids to probe per query
 * @param nr_dpus number of DPUs
 * @return query-to-DPU-set mapping (HashTable array, must be freed by caller)
 */
HashTable** compute_query_to_dpu_mapping(
    IVFSearcher* searcher,
    const FloatMatrix* queries,
    int batch_size,
    int nprobe,
    int nr_dpus,
    HashTable*** query_centroid_to_dpu_out);

/* Pure CPU batch IVF search (for verifying DPU code accuracy) */
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
                                HashTable** query_centroid_to_dpu);

/* Pure CPU batch IVF search (for verifying DPU code accuracy) */
IntArray** ivf_searcher_search_batch_only_CPU(
                                                struct dpu_set_t* dpu_set,
                                                IVFSearcher* searcher,
                                              const FloatMatrix* queries,
                                              int batch_size,
                                              int ef,
                                              int topk,
                                              int nprobe,
                                              int post_ef);

/* Free batch search results */
void ivf_searcher_free_batch_results(IntArray** batch_results, int batch_size);

/* Find suitable EF values */
IntArray* ivf_searcher_find_EFS(struct dpu_set_t* dpu_set,
                               IVFSearcher* searcher,
                               const FloatMatrix* query,
                               int NQ,
                               int nprobe,
                               const IntMatrix* gt);

/* Get memory usage */
size_t ivf_searcher_get_memory_usage(const IVFSearcher* searcher);

/* Clean up memory */
void ivf_searcher_clear_indices(IVFSearcher* searcher);

#endif /* IVF_SEARCH_H */

