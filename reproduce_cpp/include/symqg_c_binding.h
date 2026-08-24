#ifndef SYMQG_C_BINDING_H
#define SYMQG_C_BINDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pointer type - hides C++ implementation details */
typedef struct QuantizedGraph_C QuantizedGraph_C;
typedef struct ResultBuffer_C ResultBuffer_C;

/**
 * Create QuantizedGraph object
 * @param num_elements number of elements
 * @param degree_bound degree upper bound
 * @param dimension vector dimension
 * @param allocate_vectors if 0, skip allocating original vector storage (~89% memory savings for search-only use)
 * @return QuantizedGraph object pointer, NULL on failure
 */
QuantizedGraph_C* qg_create(size_t num_elements,
                            size_t degree_bound,
                            size_t dimension,
                            int allocate_vectors);

/**
 * Free QuantizedGraph object
 * @param qg QuantizedGraph object pointer
 */
void qg_free(QuantizedGraph_C* qg);

/**
 * Load index
 * @param qg QuantizedGraph object pointer
 * @param filename index file path
 * @return 0 on success, -1 on failure
 */
int qg_load_index(QuantizedGraph_C* qg, const char* filename);

/**
 * Save index
 * @param qg QuantizedGraph object pointer
 * @param filename index file path
 * @return 0 on success, -1 on failure
 */
int qg_save_index(QuantizedGraph_C* qg, const char* filename);

/**
 * Set search parameter ef
 * @param qg QuantizedGraph object pointer
 * @param ef_search EF parameter value
 */
void qg_set_ef(QuantizedGraph_C* qg, size_t ef_search);

/**
 * Set cluster ID
 * @param qg QuantizedGraph object pointer
 * @param c_id cluster ID
 */
void qg_set_cluster(QuantizedGraph_C* qg, size_t c_id);

/**
 * Set post ef parameter
 * @param qg QuantizedGraph object pointer
 * @param ef_search Post EF parameter value
 */
void qg_set_post_ef(QuantizedGraph_C* qg, size_t ef_search);

/**
 * Enable/disable profiling
 * @param qg QuantizedGraph object pointer
 * @param enable 1 to enable, 0 to disable
 */
void qg_enable_profiling(QuantizedGraph_C* qg, int enable);

/**
 * Report performance statistics
 * @param qg QuantizedGraph object pointer
 */
void qg_report_timings(QuantizedGraph_C* qg);

/**
 * Search for nearest neighbors
 * @param qg QuantizedGraph object pointer
 * @param query query vector
 * @param knn number of nearest neighbors to return
 * @param results result array (allocated by caller, at least knn elements)
 * @return actual number of results returned, -1 on failure
 */
int qg_search(QuantizedGraph_C* qg, 
              const float* query, 
              uint32_t knn, 
              uint32_t* results,
              float ep_dist);


/**
 * Get number of elements in the index
 * @param qg QuantizedGraph object pointer
 * @return number of elements
 */
size_t qg_get_num_elements(const QuantizedGraph_C* qg);

/**
 * Get vector dimension
 * @param qg QuantizedGraph object pointer
 * @return vector dimension
 */
size_t qg_get_dimension(const QuantizedGraph_C* qg);

void qg_get_lut_sumq(QuantizedGraph_C* qg, const float* query, float* lut, float* sumq);

/**
 * Get rotated query vector (with -10 scaling applied) and sumq
 * @param qg             QuantizedGraph object pointer
 * @param query          original query vector (dimension floats)
 * @param rotated_query_out  output: rotated query vector (padded_dim floats)
 * @param sumq_out       output: sum of rotated_query components
 */
void qg_get_rotated_query(QuantizedGraph_C* qg, const float* query,
                          float* rotated_query_out, float* sumq_out);

/**
 * Get original vector for a data point
 * @param qg QuantizedGraph object pointer
 * @param data_id data point ID
 * @return original vector pointer, NULL on failure
 */
const float* qg_get_vector(const QuantizedGraph_C* qg, uint32_t data_id);

/**
 * Get packed code for a data point
 * @param qg QuantizedGraph object pointer
 * @param data_id data point ID
 * @return packed code pointer, NULL on failure
 */
const uint64_t* qg_get_packed_code(const QuantizedGraph_C* qg, uint32_t data_id);

/**
 * Get factor value for a data point
 * @param qg QuantizedGraph object pointer
 * @param data_id data point ID
 * @return factor value pointer, NULL on failure
 */
const float* qg_get_factor(const QuantizedGraph_C* qg, uint32_t data_id);

/**
 * Get neighbor list for a data point
 * @param qg QuantizedGraph object pointer
 * @param data_id data point ID
 * @return neighbor ID list pointer, NULL on failure
 */
const uint32_t* qg_get_neighbors(const QuantizedGraph_C* qg, uint32_t data_id);

/**
 * Get entry point ID of the graph
 * @param qg QuantizedGraph object pointer
 * @return entry point ID, 0 on failure
 */
uint32_t qg_get_entry_point(const QuantizedGraph_C* qg);

/* =============== ResultBuffer Related Functions =============== */

/**
 * Create ResultBuffer object
 * @param capacity result buffer capacity
 * @return ResultBuffer object pointer, NULL on failure
 */
ResultBuffer_C* result_buffer_create(size_t capacity);

/**
 * Free ResultBuffer object
 * @param buffer ResultBuffer object pointer
 */
void result_buffer_free(ResultBuffer_C* buffer);

/**
 * Insert result into ResultBuffer
 * @param buffer ResultBuffer object pointer
 * @param data_id data point ID
 * @param distance distance
 */
void result_buffer_insert(ResultBuffer_C* buffer, uint32_t data_id, float distance);

/**
 * Check if ResultBuffer is full
 * @param buffer ResultBuffer object pointer
 * @return 1 if full, 0 if not full
 */
int result_buffer_is_full(const ResultBuffer_C* buffer);

/**
 * Get ResultBuffer size
 * @param buffer ResultBuffer object pointer
 * @return current number of results
 */
size_t result_buffer_size(const ResultBuffer_C* buffer);

/**
 * Copy results to arrays
 * @param buffer ResultBuffer object pointer
 * @param ids result ID array (allocated by caller)
 * @param distances distance array (allocated by caller, can be NULL)
 * @param max_size max array capacity
 * @return actual number of results copied
 */
size_t result_buffer_copy_results(const ResultBuffer_C* buffer, 
                                  uint32_t* ids, 
                                  float* distances, 
                                  size_t max_size);

/**
 * Clear ResultBuffer
 * @param buffer ResultBuffer object pointer
 */
void result_buffer_clear(ResultBuffer_C* buffer);

/**
 * Merge two ResultBuffers, inserting all elements from source into destination
 * @param dest destination ResultBuffer (merged result)
 * @param src source ResultBuffer (data to merge)
 */
void result_buffer_merge(ResultBuffer_C* dest, ResultBuffer_C* src);


#ifdef __cplusplus
}
#endif

#endif /* SYMQG_C_BINDING_H */

