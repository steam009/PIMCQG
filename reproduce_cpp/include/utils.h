#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Dynamic array structure - replaces std::vector<float> */
typedef struct {
    float* data;
    size_t size;
    size_t capacity;
} FloatArray;

/* Dynamic integer array - replaces std::vector<int> */
#ifndef INT_ARRAY_DEFINED
#define INT_ARRAY_DEFINED
typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} IntArray;
#endif

/* Now include hashtable.h, which needs IntArray */
#include "hashtable.h"

/* Matrix structure - replaces std::vector<std::vector<float>> */
typedef struct {
    FloatArray* rows;
    size_t num_rows;
    size_t capacity;
} FloatMatrix;

/* Integer matrix structure - replaces std::vector<std::vector<int>> */
typedef struct {
    IntArray* rows;
    size_t num_rows;
    size_t capacity;
} IntMatrix;

/* Cluster mapping table - implemented using hash table, replaces std::unordered_map<int, std::vector<int>> */
typedef struct {
    HashTable* hashtable;  /* underlying hash table */
} ClusterMappingTable;

/* Distance-index pair - replaces std::pair<float, int> */
typedef struct {
    float distance;
    int index;
} DistIndexPair;

typedef struct {
    DistIndexPair* data;
    size_t size;
    size_t capacity;
} DistIndexArray;

/* BeamSizeGenerator structure */
typedef struct {
    int k;
    int e;
    int index;
    int bases[10];
    int num_bases;
} BeamSizeGenerator;

/* =============== FloatArray Functions =============== */
FloatArray* float_array_create(size_t initial_capacity);
void float_array_free(FloatArray* arr);
void float_array_push(FloatArray* arr, float value);
void float_array_resize(FloatArray* arr, size_t new_size);
void float_array_clear(FloatArray* arr);

/* =============== IntArray Functions =============== */
IntArray* int_array_create(size_t initial_capacity);
void int_array_free(IntArray* arr);
void int_array_push(IntArray* arr, int value);
void int_array_resize(IntArray* arr, size_t new_size);
void int_array_clear(IntArray* arr);

/* =============== FloatMatrix Functions =============== */
FloatMatrix* float_matrix_create(size_t initial_capacity);
void float_matrix_free(FloatMatrix* mat);
void float_matrix_add_row(FloatMatrix* mat, FloatArray* row);

/* =============== IntMatrix Functions =============== */
IntMatrix* int_matrix_create(size_t initial_capacity);
void int_matrix_free(IntMatrix* mat);
void int_matrix_add_row(IntMatrix* mat, IntArray* row);

/* =============== ClusterMappingTable Functions =============== */
ClusterMappingTable* cluster_mapping_table_create(size_t initial_capacity);
void cluster_mapping_table_free(ClusterMappingTable* table);
void cluster_mapping_table_add(ClusterMappingTable* table, int cluster_id, IntArray* indices);
IntArray* cluster_mapping_table_get(ClusterMappingTable* table, int cluster_id);

/* =============== DistIndexArray Functions =============== */
DistIndexArray* dist_index_array_create(size_t initial_capacity);
void dist_index_array_free(DistIndexArray* arr);
void dist_index_array_push(DistIndexArray* arr, float distance, int index);
void dist_index_array_clear(DistIndexArray* arr);

/* =============== IO-Related Functions =============== */
FloatMatrix* read_fvecs(const char* filename);
IntMatrix* read_ivecs(const char* filename);
FloatMatrix* read_bvecs_as_float(const char* filename);
ClusterMappingTable* load_mapping(const char* filename);

/* SPACE1B binary format readers */
/* query.bin: [count(int32), dim(int32), count*dim int8] -> FloatMatrix */
FloatMatrix* read_space1b_query_bin(const char* filename);
/* truth.bin: [count(int32), topk(int32), count*topk int32 ids, count*topk float32 dists] -> IntMatrix (ids only) */
IntMatrix* read_space1b_truth_bin(const char* filename);
/* base vectors from directory (vectors_1.bin, vectors_2.bin, ...): int8 -> FloatMatrix */
FloatMatrix* read_space1b_base_as_float(const char* dir_path, long long max_count);

/* u8bin: [num_pts: uint32 LE][num_dims: uint32 LE][data: uint8 * num_pts * num_dims] */
int read_u8bin_header(const char* filename, uint32_t* out_num_pts, uint32_t* out_num_dims);
FloatMatrix* read_u8bin_as_float(const char* filename, long long max_vectors);

/* =============== Preprocessing Functions =============== */
void normalize_matrix(FloatMatrix* data);
void normalize_vector(FloatArray* vec);

/* =============== Memory Monitoring =============== */
size_t get_memory_usage(void);

/* =============== BeamSizeGenerator Functions =============== */
BeamSizeGenerator* beam_size_generator_create(int k);
void beam_size_generator_free(BeamSizeGenerator* gen);
int beam_size_generator_next(BeamSizeGenerator* gen);

/* =============== Distance Computation =============== */
float sqr_dist(const FloatArray* vec1, const FloatArray* vec2);
float sqr_dist_raw(const float* vec1, const float* vec2, size_t dim);

/* =============== Find Nearest Centroids =============== */
DistIndexArray* find_nearest_centroids_batch(const FloatMatrix* queries, 
                                      int batch_size,
                                      const FloatMatrix* centroids, 
                                      int nprobe);

/* =============== String Helper Functions =============== */
char* string_duplicate(const char* str);
void build_path(char* dest, size_t dest_size, const char* parts[], int num_parts);

#endif /* UTILS_H */

