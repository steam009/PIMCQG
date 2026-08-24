#define _POSIX_C_SOURCE 200809L  /* enable getline function */
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* =============== FloatArray Implementation =============== */

FloatArray* float_array_create(size_t initial_capacity) {
    FloatArray* arr = (FloatArray*)malloc(sizeof(FloatArray));
    if (!arr) return NULL;
    
    arr->capacity = initial_capacity > 0 ? initial_capacity : 16;
    arr->size = 0;
    arr->data = (float*)malloc(arr->capacity * sizeof(float));
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    return arr;
}

void float_array_free(FloatArray* arr) {
    if (arr) {
        if (arr->data) free(arr->data);
        free(arr);
    }
}

void float_array_push(FloatArray* arr, float value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        float* new_data = (float*)realloc(arr->data, arr->capacity * sizeof(float));
        if (!new_data) {
            fprintf(stderr, "Failed to resize float array\n");
            exit(1);
        }
        arr->data = new_data;
    }
    arr->data[arr->size++] = value;
}

void float_array_resize(FloatArray* arr, size_t new_size) {
    if (new_size > arr->capacity) {
        arr->capacity = new_size;
        float* new_data = (float*)realloc(arr->data, arr->capacity * sizeof(float));
        if (!new_data) {
            fprintf(stderr, "Failed to resize float array\n");
            exit(1);
        }
        arr->data = new_data;
    }
    arr->size = new_size;
}

void float_array_clear(FloatArray* arr) {
    arr->size = 0;
}

/* =============== IntArray Implementation =============== */

IntArray* int_array_create(size_t initial_capacity) {
    IntArray* arr = (IntArray*)malloc(sizeof(IntArray));
    if (!arr) return NULL;
    
    arr->capacity = initial_capacity > 0 ? initial_capacity : 16;
    arr->size = 0;
    arr->data = (int*)malloc(arr->capacity * sizeof(int));
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    return arr;
}

void int_array_free(IntArray* arr) {
    if (arr) {
        if (arr->data) free(arr->data);
        free(arr);
    }
}

void int_array_push(IntArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        int* new_data = (int*)realloc(arr->data, arr->capacity * sizeof(int));
        if (!new_data) {
            fprintf(stderr, "Failed to resize int array\n");
            exit(1);
        }
        arr->data = new_data;
    }
    arr->data[arr->size++] = value;
}

void int_array_resize(IntArray* arr, size_t new_size) {
    if (new_size > arr->capacity) {
        arr->capacity = new_size;
        int* new_data = (int*)realloc(arr->data, arr->capacity * sizeof(int));
        if (!new_data) {
            fprintf(stderr, "Failed to resize int array\n");
            exit(1);
        }
        arr->data = new_data;
    }
    arr->size = new_size;
}

void int_array_clear(IntArray* arr) {
    arr->size = 0;
}

/* =============== FloatMatrix Implementation =============== */

FloatMatrix* float_matrix_create(size_t initial_capacity) {
    FloatMatrix* mat = (FloatMatrix*)malloc(sizeof(FloatMatrix));
    if (!mat) return NULL;
    
    mat->capacity = initial_capacity > 0 ? initial_capacity : 16;
    mat->num_rows = 0;
    mat->rows = (FloatArray*)malloc(mat->capacity * sizeof(FloatArray));
    if (!mat->rows) {
        free(mat);
        return NULL;
    }
    return mat;
}

void float_matrix_free(FloatMatrix* mat) {
    if (mat) {
        if (mat->rows) {
            for (size_t i = 0; i < mat->num_rows; i++) {
                if (mat->rows[i].data) {
                    free(mat->rows[i].data);
                }
            }
            free(mat->rows);
        }
        free(mat);
    }
}

void float_matrix_add_row(FloatMatrix* mat, FloatArray* row) {
    if (mat->num_rows >= mat->capacity) {
        mat->capacity *= 2;
        FloatArray* new_rows = (FloatArray*)realloc(mat->rows, mat->capacity * sizeof(FloatArray));
        if (!new_rows) {
            fprintf(stderr, "Failed to resize float matrix\n");
            exit(1);
        }
        mat->rows = new_rows;
    }
    mat->rows[mat->num_rows++] = *row;
}

/* =============== IntMatrix Implementation =============== */

IntMatrix* int_matrix_create(size_t initial_capacity) {
    IntMatrix* mat = (IntMatrix*)malloc(sizeof(IntMatrix));
    if (!mat) return NULL;
    
    mat->capacity = initial_capacity > 0 ? initial_capacity : 16;
    mat->num_rows = 0;
    mat->rows = (IntArray*)malloc(mat->capacity * sizeof(IntArray));
    if (!mat->rows) {
        free(mat);
        return NULL;
    }
    return mat;
}

void int_matrix_free(IntMatrix* mat) {
    if (mat) {
        if (mat->rows) {
            for (size_t i = 0; i < mat->num_rows; i++) {
                if (mat->rows[i].data) {
                    free(mat->rows[i].data);
                }
            }
            free(mat->rows);
        }
        free(mat);
    }
}

void int_matrix_add_row(IntMatrix* mat, IntArray* row) {
    if (mat->num_rows >= mat->capacity) {
        mat->capacity *= 2;
        IntArray* new_rows = (IntArray*)realloc(mat->rows, mat->capacity * sizeof(IntArray));
        if (!new_rows) {
            fprintf(stderr, "Failed to resize int matrix\n");
            exit(1);
        }
        mat->rows = new_rows;
    }
    mat->rows[mat->num_rows++] = *row;
}

/* =============== ClusterMappingTable Implementation (using hash table) =============== */

ClusterMappingTable* cluster_mapping_table_create(size_t initial_capacity) {
    ClusterMappingTable* table = (ClusterMappingTable*)malloc(sizeof(ClusterMappingTable));
    if (!table) return NULL;
    
    /* Create underlying hash table */
    table->hashtable = hashtable_create(initial_capacity);
    if (!table->hashtable) {
        free(table);
        return NULL;
    }
    
    return table;
}

void cluster_mapping_table_free(ClusterMappingTable* table) {
    if (table) {
        if (table->hashtable) {
            hashtable_free(table->hashtable);
        }
        free(table);
    }
}

void cluster_mapping_table_add(ClusterMappingTable* table, int cluster_id, IntArray* indices) {
    if (!table || !table->hashtable || !indices) {
        fprintf(stderr, "Invalid parameters to cluster_mapping_table_add\n");
        return;
    }
    
    if (hashtable_insert(table->hashtable, cluster_id, indices) != 0) {
        fprintf(stderr, "Failed to insert cluster mapping for cluster_id %d\n", cluster_id);
    }
}

IntArray* cluster_mapping_table_get(ClusterMappingTable* table, int cluster_id) {
    if (!table || !table->hashtable) return NULL;
    return hashtable_get(table->hashtable, cluster_id);
}

/* =============== DistIndexArray Implementation =============== */

DistIndexArray* dist_index_array_create(size_t initial_capacity) {
    DistIndexArray* arr = (DistIndexArray*)malloc(sizeof(DistIndexArray));
    if (!arr) return NULL;
    
    arr->capacity = initial_capacity > 0 ? initial_capacity : 16;
    arr->size = 0;
    arr->data = (DistIndexPair*)malloc(arr->capacity * sizeof(DistIndexPair));
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    return arr;
}

void dist_index_array_free(DistIndexArray* arr) {
    if (arr) {
        if (arr->data) free(arr->data);
        free(arr);
    }
}

void dist_index_array_push(DistIndexArray* arr, float distance, int index) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        DistIndexPair* new_data = (DistIndexPair*)realloc(arr->data, 
                                                          arr->capacity * sizeof(DistIndexPair));
        if (!new_data) {
            fprintf(stderr, "Failed to resize dist index array\n");
            exit(1);
        }
        arr->data = new_data;
    }
    arr->data[arr->size].distance = distance;
    arr->data[arr->size].index = index;
    arr->size++;
}

void dist_index_array_clear(DistIndexArray* arr) {
    arr->size = 0;
}

/* =============== IO-Related Function Implementation =============== */

FloatMatrix* read_fvecs(const char* filename) {
    printf("Reading File - %s\n", filename);
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    FloatMatrix* data = float_matrix_create(1000);
    int dim;
    
    /* Read dimension of first vector */
    if (fread(&dim, sizeof(int), 1, file) != 1) {
        printf("\t%s read (empty file)\n", filename);
        fclose(file);
        return data;
    }
    
    fseek(file, 0, SEEK_SET);  /* return to beginning of file */
    
    while (1) {
        int current_dim;
        if (fread(&current_dim, sizeof(int), 1, file) != 1) break;
        
        if (current_dim != dim) {
            fprintf(stderr, "Non-uniform vector sizes in %s\n", filename);
            break;
        }
        
        FloatArray vec;
        vec.size = dim;
        vec.capacity = dim;
        vec.data = (float*)malloc(dim * sizeof(float));
        if (!vec.data) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }
        
        if (fread(vec.data, sizeof(float), dim, file) != (size_t)dim) {
            free(vec.data);
            break;
        }
        
        float_matrix_add_row(data, &vec);
    }
    
    fclose(file);
    printf("\t%s read\n", filename);
    return data;
}

IntMatrix* read_ivecs(const char* filename) {
    printf("Reading File - %s\n", filename);
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    IntMatrix* data = int_matrix_create(1000);
    int dim;
    
    /* Read dimension of first vector */
    if (fread(&dim, sizeof(int), 1, file) != 1) {
        printf("\t%s read (empty file)\n", filename);
        fclose(file);
        return data;
    }
    
    fseek(file, 0, SEEK_SET);  /* return to beginning of file */
    
    while (1) {
        int current_dim;
        if (fread(&current_dim, sizeof(int), 1, file) != 1) break;
        
        if (current_dim != dim) {
            fprintf(stderr, "Non-uniform vector sizes in %s\n", filename);
            break;
        }
        
        IntArray vec;
        vec.size = dim;
        vec.capacity = dim;
        vec.data = (int*)malloc(dim * sizeof(int));
        if (!vec.data) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }
        
        if (fread(vec.data, sizeof(int), dim, file) != (size_t)dim) {
            free(vec.data);
            break;
        }
        
        int_matrix_add_row(data, &vec);
    }
    
    fclose(file);
    printf("\t%s read\n", filename);
    return data;
}

FloatMatrix* read_bvecs_as_float(const char* filename) {
    printf("Reading from %s\n", filename);
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    FloatMatrix* data = float_matrix_create(1000);
    int dim;
    
    /* Read dimension of first vector */
    if (fread(&dim, sizeof(int), 1, file) != 1) {
        fclose(file);
        return data;
    }
    
    fseek(file, 0, SEEK_SET);  /* return to beginning of file */
    
    while (1) {
        int current_dim;
        if (fread(&current_dim, sizeof(int), 1, file) != 1) break;
        
        if (current_dim != dim) {
            fprintf(stderr, "Non-uniform vector sizes in %s\n", filename);
            break;
        }
        
        uint8_t* raw_vec = (uint8_t*)malloc(dim);
        if (!raw_vec) break;
        
        if (fread(raw_vec, 1, dim, file) != (size_t)dim) {
            free(raw_vec);
            break;
        }
        
        /* Convert to float */
        FloatArray vec;
        vec.size = dim;
        vec.capacity = dim;
        vec.data = (float*)malloc(dim * sizeof(float));
        if (!vec.data) {
            free(raw_vec);
            break;
        }
        
        for (int i = 0; i < dim; i++) {
            vec.data[i] = (float)raw_vec[i];
        }
        
        free(raw_vec);
        float_matrix_add_row(data, &vec);
    }
    
    fclose(file);
    return data;
}

ClusterMappingTable* load_mapping(const char* filename) {
    /* Find file extension */
    const char* ext = strrchr(filename, '.');
    if (!ext) {
        fprintf(stderr, "No file extension found: %s\n", filename);
        return NULL;
    }
    
    ClusterMappingTable* table = cluster_mapping_table_create(4096);
    
    if (strcmp(ext, ".txt") == 0) {
        /* Read text format */
        FILE* file = fopen(filename, "r");
        if (!file) {
            fprintf(stderr, "Cannot open mapping file: %s\n", filename);
            cluster_mapping_table_free(table);
            return NULL;
        }
        
        char* line = NULL;
        size_t line_buf_size = 0;
        ssize_t line_len;
        int num_clusters = 0;
        while ((line_len = getline(&line, &line_buf_size, file)) != -1) {
            int cluster_id;
            char* ptr = line;
            char* endptr;
            
            cluster_id = (int)strtol(ptr, &endptr, 10);
            if (ptr == endptr) continue;  /* skip empty or malformed lines */
            ptr = endptr;
            
            IntArray* indices = int_array_create(100);
            while (*ptr != '\0' && *ptr != '\n') {
                int idx = (int)strtol(ptr, &endptr, 10);
                if (ptr == endptr) break;
                int_array_push(indices, idx);
                ptr = endptr;
            }
            
            cluster_mapping_table_add(table, cluster_id, indices);
            num_clusters++;
        }
        if (line) free(line);
        fclose(file);

        printf("number of clusters: %d\n", num_clusters);
    } else if (strcmp(ext, ".bin") == 0) {
        /* Read binary format */
        FILE* file = fopen(filename, "rb");
        if (!file) {
            fprintf(stderr, "Cannot open mapping file: %s\n", filename);
            cluster_mapping_table_free(table);
            return NULL;
        }
        
        int num_clusters;
        if (fread(&num_clusters, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Failed to read number of clusters from %s\n", filename);
            fclose(file);
            cluster_mapping_table_free(table);
            return NULL;
        }
        
        for (int i = 0; i < num_clusters; i++) {
            int cluster_id;
            int num_indices;
            if (fread(&cluster_id, sizeof(int), 1, file) != 1 ||
                fread(&num_indices, sizeof(int), 1, file) != 1) {
                fprintf(stderr, "Failed to read cluster header from %s\n", filename);
                fclose(file);
                cluster_mapping_table_free(table);
                return NULL;
            }
            
            IntArray* indices = int_array_create(num_indices);
            indices->size = num_indices;
            if (fread(indices->data, sizeof(int), num_indices, file) != (size_t)num_indices) {
                fprintf(stderr, "Failed to read cluster indices from %s\n", filename);
                int_array_free(indices);
                fclose(file);
                cluster_mapping_table_free(table);
                return NULL;
            }
            
            cluster_mapping_table_add(table, cluster_id, indices);
        }
        fclose(file);
        
    } else {
        fprintf(stderr, "Unsupported mapping file format: %s. ", ext);
        fprintf(stderr, "Please convert %s to .txt or .bin format using utils/convert_pickle.py\n", 
                filename);
        cluster_mapping_table_free(table);
        return NULL;
    }
    
    printf("Loaded mapping for %zu clusters from %s\n", 
           hashtable_size(table->hashtable), filename);
    return table;
}

/* =============== SPACE1B binary format readers =============== */

FloatMatrix* read_space1b_query_bin(const char* filename) {
    printf("Reading File (SPACE1B query format) - %s\n", filename);

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }

    int count, dim;
    if (fread(&count, sizeof(int), 1, file) != 1 ||
        fread(&dim, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read header from %s\n", filename);
        fclose(file);
        return NULL;
    }
    printf("  count=%d, dim=%d\n", count, dim);

    FloatMatrix* data = float_matrix_create(count);
    int8_t* raw_vec = (int8_t*)malloc(dim * sizeof(int8_t));
    if (!raw_vec) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        float_matrix_free(data);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        if (fread(raw_vec, sizeof(int8_t), dim, file) != (size_t)dim) {
            fprintf(stderr, "Failed to read vector %d from %s\n", i, filename);
            break;
        }

        FloatArray vec;
        vec.size = dim;
        vec.capacity = dim;
        vec.data = (float*)malloc(dim * sizeof(float));
        if (!vec.data) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }

        for (int j = 0; j < dim; j++) {
            vec.data[j] = (float)raw_vec[j];
        }

        float_matrix_add_row(data, &vec);
    }

    free(raw_vec);
    fclose(file);
    printf("\t%s read (%zu rows)\n", filename, data->num_rows);
    return data;
}

IntMatrix* read_space1b_truth_bin(const char* filename) {
    printf("Reading File (SPACE1B truth format) - %s\n", filename);

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }

    int count, topk;
    if (fread(&count, sizeof(int), 1, file) != 1 ||
        fread(&topk, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read header from %s\n", filename);
        fclose(file);
        return NULL;
    }
    printf("  count=%d, topk=%d\n", count, topk);

    IntMatrix* data = int_matrix_create(count);
    int32_t* ids = (int32_t*)malloc(topk * sizeof(int32_t));
    if (!ids) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        int_matrix_free(data);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        if (fread(ids, sizeof(int32_t), topk, file) != (size_t)topk) {
            fprintf(stderr, "Failed to read truth ids for query %d\n", i);
            break;
        }

        IntArray vec;
        vec.size = topk;
        vec.capacity = topk;
        vec.data = (int*)malloc(topk * sizeof(int));
        if (!vec.data) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }

        for (int j = 0; j < topk; j++) {
            vec.data[j] = (int)ids[j];
        }

        int_matrix_add_row(data, &vec);
    }

    free(ids);
    fclose(file);
    printf("\t%s read (%zu rows)\n", filename, data->num_rows);
    return data;
}

FloatMatrix* read_space1b_base_as_float(const char* dir_path, long long max_count) {
    printf("Reading SPACE1B base vectors from directory: %s (max %lld vectors)\n",
           dir_path, max_count);

    /* Open vectors_1.bin to read the global header */
    char part_path[1024];
    snprintf(part_path, sizeof(part_path), "%s/vectors_1.bin", dir_path);

    FILE* f = fopen(part_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", part_path);
        return NULL;
    }

    int total_count_hdr, vec_dim;
    if (fread(&total_count_hdr, sizeof(int), 1, f) != 1 ||
        fread(&vec_dim, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read header from %s\n", part_path);
        fclose(f);
        return NULL;
    }
    printf("\theader: total_count=%d, vec_dim=%d\n", total_count_hdr, vec_dim);

    long long n_to_read = (max_count > 0 && max_count < (long long)total_count_hdr)
                          ? max_count : (long long)total_count_hdr;

    FloatMatrix* data = float_matrix_create((size_t)n_to_read);

    int8_t* row_buf = (int8_t*)malloc(vec_dim);
    if (!row_buf) {
        fprintf(stderr, "Memory allocation failed for row buffer\n");
        fclose(f);
        float_matrix_free(data);
        return NULL;
    }

    long long total_read = 0;
    int part_idx = 1;

    while (total_read < n_to_read) {
        int filled = 0;
        while (filled < vec_dim) {
            size_t n = fread(row_buf + filled, 1, vec_dim - filled, f);
            if (n > 0) {
                filled += (int)n;
            } else {
                fclose(f);
                f = NULL;
                part_idx++;
                snprintf(part_path, sizeof(part_path), "%s/vectors_%d.bin",
                         dir_path, part_idx);
                f = fopen(part_path, "rb");
                if (!f) goto done;
            }
        }

        FloatArray vec;
        vec.size = vec_dim;
        vec.capacity = vec_dim;
        vec.data = (float*)malloc(vec_dim * sizeof(float));
        if (!vec.data) goto done;

        for (int d = 0; d < vec_dim; d++) {
            vec.data[d] = (float)row_buf[d];
        }
        float_matrix_add_row(data, &vec);
        total_read++;

        if (total_read % 10000000 == 0) {
            printf("\tRead %lld / %lld vectors...\n", total_read, n_to_read);
        }
    }

done:
    free(row_buf);
    if (f) fclose(f);

    printf("\tDone: read %lld vectors\n", total_read);
    return data;
}

static uint32_t read_u32_le(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) {
        return 0;
    }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}


int read_u8bin_header(const char* filename, uint32_t* out_num_pts, uint32_t* out_num_dims) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open u8bin header: %s\n", filename);
        return -1;
    }
    uint32_t n = read_u32_le(file);
    uint32_t d = read_u32_le(file);
    fclose(file);
    if (d == 0) {
        fprintf(stderr, "Invalid u8bin header (num_dims=0): %s\n", filename);
        return -1;
    }
    if (out_num_pts) {
        *out_num_pts = n;
    }
    if (out_num_dims) {
        *out_num_dims = d;
    }
    return 0;
}

FloatMatrix* read_u8bin_as_float(const char* filename, long long max_vectors) {
    printf("Reading u8bin - %s\n", filename);

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }

    uint32_t num_pts = read_u32_le(file);
    uint32_t num_dims = read_u32_le(file);
    if (num_pts == 0 || num_dims == 0) {
        fprintf(stderr, "Invalid u8bin header in %s\n", filename);
        fclose(file);
        return NULL;
    }

    long long n_read = (long long)num_pts;
    if (max_vectors > 0 && max_vectors < n_read) {
        n_read = max_vectors;
    }

    printf("\tnum_pts=%u, num_dims=%u, reading %lld vectors\n", num_pts, num_dims,
           (long long)n_read);

    size_t payload = (size_t)n_read * (size_t)num_dims;
    uint8_t* raw = (uint8_t*)malloc(payload);
    if (!raw) {
        fprintf(stderr, "Memory allocation failed for u8bin payload\n");
        fclose(file);
        return NULL;
    }

    if (fread(raw, 1, payload, file) != payload) {
        fprintf(stderr, "Failed to read u8bin data from %s\n", filename);
        free(raw);
        fclose(file);
        return NULL;
    }
    fclose(file);

    FloatMatrix* data = float_matrix_create((size_t)n_read);
    for (long long i = 0; i < n_read; i++) {
        FloatArray vec;
        vec.size = num_dims;
        vec.capacity = num_dims;
        vec.data = (float*)malloc((size_t)num_dims * sizeof(float));
        if (!vec.data) {
            fprintf(stderr, "Memory allocation failed for u8bin row %lld\n", i);
            free(raw);
            float_matrix_free(data);
            return NULL;
        }
        for (uint32_t d = 0; d < num_dims; d++) {
            vec.data[d] = (float)raw[i * (long long)num_dims + d];
        }
        float_matrix_add_row(data, &vec);
    }

    free(raw);
    printf("\tDone reading u8bin (%lld vectors)\n", (long long)n_read);
    return data;
}

/* =============== Preprocessing Function Implementation =============== */

void normalize_matrix(FloatMatrix* data) {
    for (size_t i = 0; i < data->num_rows; i++) {
        normalize_vector(&data->rows[i]);
    }
}

void normalize_vector(FloatArray* vec) {
    float norm = 0.0f;
    for (size_t i = 0; i < vec->size; i++) {
        norm += vec->data[i] * vec->data[i];
    }
    norm = sqrtf(norm);
    
    if (norm > 0.0f) {
        for (size_t i = 0; i < vec->size; i++) {
            vec->data[i] /= norm;
        }
    }
}

/* =============== Memory Monitoring Implementation =============== */

size_t get_memory_usage(void) {
    /* Read /proc/self/status file to get memory usage */
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            size_t memory_kb;
            sscanf(line + 6, "%zu", &memory_kb);
            fclose(file);
            return memory_kb * 1024;  /* convert to bytes */
        }
    }
    
    fclose(file);
    return 0;  /* return 0 if unable to get */
}

/* =============== BeamSizeGenerator Implementation =============== */

BeamSizeGenerator* beam_size_generator_create(int k) {
    BeamSizeGenerator* gen = (BeamSizeGenerator*)malloc(sizeof(BeamSizeGenerator));
    if (!gen) return NULL;
    
    gen->k = k;
    gen->e = (int)log10(k) - 1;
    if (gen->e < 0) gen->e = 0;
    gen->index = 0;
    
    int bases[] = {10, 15, 20, 25, 30, 40, 50, 60, 70, 80};
    gen->num_bases = 10;
    memcpy(gen->bases, bases, sizeof(bases));
    
    assert(k >= 1);
    return gen;
}

void beam_size_generator_free(BeamSizeGenerator* gen) {
    if (gen) free(gen);
}

int beam_size_generator_next(BeamSizeGenerator* gen) {
    int result = gen->bases[gen->index] * (int)pow(10, gen->e);
    gen->index++;
    if (gen->index >= gen->num_bases) {
        gen->e++;
        gen->index = 0;
    }
    return result;
}

/* =============== Distance Computation Implementation =============== */

float sqr_dist(const FloatArray* vec1, const FloatArray* vec2) {
    float dist = 0.0f;
    size_t min_size = vec1->size < vec2->size ? vec1->size : vec2->size;
    for (size_t i = 0; i < min_size; i++) {
        float diff = vec1->data[i] - vec2->data[i];
        dist += diff * diff;
    }
    return dist;
}

float sqr_dist_raw(const float* vec1, const float* vec2, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    return dist;
}

/* =============== Comparison Function for Sorting =============== */
static int compare_dist_index(const void* a, const void* b) {
    const DistIndexPair* pa = (const DistIndexPair*)a;
    const DistIndexPair* pb = (const DistIndexPair*)b;
    if (pa->distance < pb->distance) return -1;
    if (pa->distance > pb->distance) return 1;
    return 0;
}

/* =============== Find Nearest Centroids Implementation =============== */

DistIndexArray* find_nearest_centroids_batch(const FloatMatrix* queries, 
                                      int batch_size,
                                      const FloatMatrix* centroids, 
                                      int nprobe) {
    DistIndexArray* centroid_dist = dist_index_array_create(centroids->num_rows * batch_size);

    /* Partial sort, only find the nearest nprobe */
    int actual_nprobe = nprobe;
    
    /* Compute distance from query vector to each centroid */
    DistIndexArray* centroid_dist_query = dist_index_array_create(centroids->num_rows);
    for (size_t i = 0; i < (size_t)batch_size; i++) {
        for (size_t j = 0; j < centroids->num_rows; j++) {
            float dist = sqr_dist(&queries->rows[i], &centroids->rows[j]);
            dist_index_array_push(centroid_dist_query, dist, (int)j);
        }
        /* Use quickselect or full sort */
        qsort(centroid_dist_query->data, centroid_dist_query->size, sizeof(DistIndexPair), compare_dist_index);
        
        /* Resize to nprobe */
        centroid_dist_query->size = actual_nprobe;
        for (size_t j = 0; j < centroid_dist_query->size; j++) {
            dist_index_array_push(centroid_dist, centroid_dist_query->data[j].distance, centroid_dist_query->data[j].index);
        }
        dist_index_array_clear(centroid_dist_query);
    }
    dist_index_array_free(centroid_dist_query);
    
    return centroid_dist;
}

/* =============== String Helper Function Implementation =============== */

char* string_duplicate(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str) + 1;
    char* dup = (char*)malloc(len);
    if (dup) {
        memcpy(dup, str, len);
    }
    return dup;
}

void build_path(char* dest, size_t dest_size, const char* parts[], int num_parts) {
    dest[0] = '\0';
    for (int i = 0; i < num_parts; i++) {
        if (i > 0) {
            strncat(dest, "/", dest_size - strlen(dest) - 1);
        }
        strncat(dest, parts[i], dest_size - strlen(dest) - 1);
    }
}

