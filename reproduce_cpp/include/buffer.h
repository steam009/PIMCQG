#ifndef BUFFER_H
#define BUFFER_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* Constant definitions */

#ifndef KPID_MAX
#define KPID_MAX 0xFFFFFFFF
#endif

/* Data type definitions */
typedef int32_t PID;  /* Point ID */
typedef float DIST;

/* Result ID structure */
typedef struct {
    PID id;
    int32_t centroid_id;
} ResultID;

/* Candidate structure */
typedef struct {
    PID id;
    DIST distance;
} candidate_t;

/* SearchBuffer - manages candidate set during search */
typedef struct {
    candidate_t *data;      /* candidate array */
    size_t size;           /* current element count */
    size_t cur;            /* current cursor position */
    size_t capacity;       /* maximum capacity */
} search_buffer_t;

/* ResultBuffer - stores final kNN results */  
typedef struct {
    ResultID *ids;              /* ID array */
    DIST *distances;      /* distance array */
    size_t size;           /* current element count */
    size_t capacity;       /* maximum capacity */
} result_buffer_t;

/* HashBasedBooleanSet - hash-based boolean set for visited tracking */
typedef struct {
    PID *table;            /* main hash table */
    PID *overflow_table;   /* overflow table for collision resolution */
    size_t table_size;     /* main table size */
    size_t overflow_size;  /* overflow table size */
    size_t overflow_count; /* element count in overflow table */
    PID mask;              /* hash mask (table_size - 1) */
} hashset_t;

/* ===========================================
 * SearchBuffer function declarations
 * =========================================== */

/* Initialize SearchBuffer */
static inline void search_buffer_init(search_buffer_t *buf, candidate_t *data_array, size_t capacity);

/* Insert candidate into SearchBuffer */
static inline void search_buffer_insert(search_buffer_t *buf, PID data_id, DIST dist);

/* Check if full and new distance is worse */
static inline bool search_buffer_is_full(const search_buffer_t *buf, DIST dist);

/* Pop closest unchecked point and return distance */
static inline PID search_buffer_pop_with_dist(search_buffer_t *buf, DIST *dist);

/* Clear SearchBuffer */
static inline void search_buffer_clear(search_buffer_t *buf);

/* Get next ID (without popping) */
static inline PID search_buffer_next_id(const search_buffer_t *buf);

/* Check if there is a next element */
static inline bool search_buffer_has_next(const search_buffer_t *buf);

/* Reset capacity */
static inline void search_buffer_resize(search_buffer_t *buf, candidate_t *new_data_array, size_t new_capacity);

/* ===========================================
 * ResultBuffer function declarations  
 * =========================================== */

/* Initialize DPU ResultBuffer */
static inline void dpu_result_buffer_init(result_buffer_t *buf, ResultID *ids_array, DIST *dist_array, size_t capacity);

/* Insert result into DPU ResultBuffer */
static inline void dpu_result_buffer_insert(result_buffer_t *buf, PID data_id, DIST dist, int32_t centroid_id);

/* Check if DPU ResultBuffer is full */
static inline bool dpu_result_buffer_is_full(const result_buffer_t *buf);

/* Get DPU ResultBuffer result count */
static inline size_t dpu_result_buffer_size(const result_buffer_t *buf);

/* Clear DPU ResultBuffer */
static inline void dpu_result_buffer_clear(result_buffer_t *buf);

/* Copy DPU ResultBuffer results to output array */
static inline void dpu_result_buffer_copy_results(const result_buffer_t *buf, ResultID *knn, size_t topk);

/* ===========================================
 * HashBasedBooleanSet function declarations
 * =========================================== */

/* Initialize HashBasedBooleanSet */
static inline void hashset_init(hashset_t *set, PID *table_array, size_t table_size, 
                  PID *overflow_array, size_t overflow_size);

/* Clear HashBasedBooleanSet */
static inline void hashset_clear(hashset_t *set);

/* Check if element exists */
static inline bool hashset_get(const hashset_t *set, PID data_id);

/* Add element to set */
static inline void hashset_set(hashset_t *set, PID data_id);

/* Compute hash value */
static inline size_t hashset_hash1(const hashset_t *set, PID value);

/* ===========================================
 * Helper function declarations
 * =========================================== */

/* Binary search for SearchBuffer */
static inline size_t search_buffer_binary_search(const search_buffer_t *buf, DIST dist);

/* Binary search for DPU ResultBuffer */
static inline size_t dpu_result_buffer_binary_search(const result_buffer_t *buf, DIST dist);

/* Set checked flag */
static inline void set_checked(PID *data_id);

/* Check if already checked */
static inline bool is_checked(PID data_id);

/* ===========================================
 * SearchBuffer function implementation
 * =========================================== */

static inline void search_buffer_init(search_buffer_t *buf, candidate_t *data_array, size_t capacity) {
    buf->data = data_array;
    buf->size = 0;
    buf->cur = 0;
    buf->capacity = capacity;
}

static inline size_t search_buffer_binary_search(const search_buffer_t *buf, DIST dist) {
    size_t lo = 0;
    size_t len = buf->size;
    size_t half;
    
    while (len > 1) {
        half = len >> 1;
        len -= half;
        lo += (buf->data[lo + half - 1].distance < dist) ? half : 0;
    }
    
    return (lo < buf->size && buf->data[lo].distance < dist) ? lo + 1 : lo;
}

static inline void search_buffer_insert(search_buffer_t *buf, PID data_id, DIST dist) {
    size_t lo = search_buffer_binary_search(buf, dist);

    // Calculate number of elements to shift right
    size_t move_cnt = (buf->size > lo) ? (buf->size - lo) : 0;
    if (buf->size == buf->capacity) {
        // When buffer is full, can only overwrite the last position; shift right up to size-1
        if (move_cnt > 0) {
            move_cnt -= 1;
        }
    }

    if (move_cnt > 0) {
        memmove(&buf->data[lo + 1], &buf->data[lo], move_cnt * sizeof(candidate_t));
    }

    buf->data[lo].id = data_id;
    buf->data[lo].distance = dist;

    if (buf->size < buf->capacity) {
        buf->size++;
    }

    if (lo < buf->cur) {
        buf->cur = lo;
    }
}

static inline bool search_buffer_is_full(const search_buffer_t *buf, DIST dist) {
    return buf->size == buf->capacity && dist > buf->data[buf->size - 1].distance;
}

static inline void set_checked(PID *data_id) {
    *data_id |= (1 << 31);
}

static inline bool is_checked(PID data_id) {
    return (bool)(data_id >> 31);
}

static inline PID search_buffer_pop_with_dist(search_buffer_t *buf, DIST *dist) {
    if (buf->cur >= buf->size) {
        return -1;  /* invalid ID */
    }
    
    PID cur_id = buf->data[buf->cur].id;
    *dist = buf->data[buf->cur].distance;
    set_checked(&buf->data[buf->cur].id);
    buf->cur++;
    
    /* Skip already-checked elements */
    while (buf->cur < buf->size && is_checked(buf->data[buf->cur].id)) {
        buf->cur++;
    }
    
    return cur_id;
}

static inline void search_buffer_clear(search_buffer_t *buf) {
    buf->size = 0;
    buf->cur = 0;
}

static inline PID search_buffer_next_id(const search_buffer_t *buf) {
    return (buf->cur < buf->size) ? buf->data[buf->cur].id : -1;
}

static inline bool search_buffer_has_next(const search_buffer_t *buf) {
    return buf->cur < buf->size;
}

static inline void search_buffer_resize(search_buffer_t *buf, candidate_t *new_data_array, size_t new_capacity) {
    buf->capacity = new_capacity;
    buf->data = new_data_array;
    buf->size = 0;
    buf->cur = 0;
}

/* ===========================================
 * ResultBuffer function implementation
 * =========================================== */

 static inline void dpu_result_buffer_clear(result_buffer_t *buf) {
    buf->size = 0;
}

static inline void dpu_result_buffer_init(result_buffer_t *buf, ResultID *ids_array, DIST *dist_array, size_t capacity) {
    buf->ids = ids_array;
    buf->distances = dist_array;
    buf->size = 0;
    buf->capacity = capacity;
}

static inline size_t dpu_result_buffer_binary_search(const result_buffer_t *buf, DIST dist) {
    size_t lo = 0;
    size_t len = buf->size;
    size_t half;
    
    while (len > 1) {
        half = len >> 1;
        len -= half;
        lo += (buf->distances[lo + half - 1] < dist) ? half : 0;
    }
    
    return (lo < buf->size && buf->distances[lo] < dist) ? lo + 1 : lo;
}

static inline void dpu_result_buffer_set_finished(result_buffer_t *buf) {
    if(buf->size == buf->capacity) {
        return;
    }
    buf->ids[buf->size].id = -1;
    buf->ids[buf->size].centroid_id = -1;
    buf->distances[buf->size] = 0;
    buf->size++;
}

static inline void dpu_result_buffer_insert(result_buffer_t *buf, PID data_id, DIST dist, int32_t centroid_id) {
    if (buf->size == buf->capacity && dist > buf->distances[buf->size - 1]) {
        return; // Full and worse; discard directly
    }

    size_t lo = dpu_result_buffer_binary_search(buf, dist);

    // Calculate number of elements to shift right
    size_t move_cnt = (buf->size > lo) ? (buf->size - lo) : 0;
    if (buf->size == buf->capacity) {
        // When buffer is full, can only overwrite the last position; shift right up to size-1
        if (move_cnt > 0) {
            move_cnt -= 1;
        }
    }

    if (move_cnt > 0) {
        memmove(&buf->ids[lo + 1], &buf->ids[lo], move_cnt * sizeof(ResultID));
        memmove(&buf->distances[lo + 1], &buf->distances[lo], move_cnt * sizeof(DIST));
    }

    buf->ids[lo].id = data_id;
    buf->ids[lo].centroid_id = centroid_id;
    buf->distances[lo] = dist;

    if (buf->size < buf->capacity) {
        buf->size++;
    }
}

static inline bool dpu_result_buffer_is_full(const result_buffer_t *buf) {
    return buf->size == buf->capacity;
}

static inline size_t dpu_result_buffer_size(const result_buffer_t *buf) {
    return buf->size;
}

static inline void dpu_result_buffer_copy_results(const result_buffer_t *buf, ResultID *knn, size_t topk) {
    for (size_t i = 0; i < topk; i++) {
        knn[i] = buf->ids[i];
    }
}

/* ===========================================
 * HashBasedBooleanSet function implementation
 * =========================================== */

static inline void hashset_init(hashset_t *set, PID *table_array, size_t table_size, 
                                PID *overflow_array, size_t overflow_size) {
    set->table = table_array;
    set->overflow_table = overflow_array;
    set->table_size = table_size;
    set->overflow_size = overflow_size;
    set->overflow_count = 0;
    set->mask = (PID)(table_size - 1);
    
    /* Check if table size is a power of 2 */
    if ((table_size & (table_size - 1)) != 0) {
        /* Warning: table size is not a power of 2; hash performance may be affected */
    }
    
    /* Initialize main table to empty value */
    for (size_t i = 0; i < table_size; i++) {
        set->table[i] = KPID_MAX;
    }
    
    /* Initialize overflow table */
    if (overflow_array != NULL) {
        for (size_t i = 0; i < overflow_size; i++) {
            set->overflow_table[i] = KPID_MAX;
        }
    }
}

static inline size_t hashset_hash1(const hashset_t *set, PID value) {
    return (size_t)(value & set->mask);
}

static inline void hashset_clear(hashset_t *set) {
    /* Clear main table */
    for (size_t i = 0; i < set->table_size; i++) {
        set->table[i] = KPID_MAX;
    }
    
    /* Clear overflow table */
    if (set->overflow_table != NULL) {
        for (size_t i = 0; i < set->overflow_count; i++) {
            set->overflow_table[i] = KPID_MAX;
        }
    }
    
    set->overflow_count = 0;
}

static inline bool hashset_get(const hashset_t *set, PID data_id) {
    /* First check main table */
    size_t hash_pos = hashset_hash1(set, data_id);
    PID val = set->table[hash_pos];
    
    if (val == data_id) {
        return true;
    }
    
    /* If not matching in main table and position is occupied, check overflow table */
    if ((unsigned int)val != KPID_MAX && set->overflow_table != NULL) {
        for (size_t i = 0; i < set->overflow_count; i++) {
            if (set->overflow_table[i] == data_id) {
                return true;
            }
        }
    }
    
    return false;
}

static inline void hashset_set(hashset_t *set, PID data_id) {
    size_t hash_pos = hashset_hash1(set, data_id);
    PID *val = &set->table[hash_pos];
    
    /* If already exists, return directly */
    if (*val == data_id) {
        return;
    }
    
    /* If position is empty, set directly */
    if ((unsigned int)*val == KPID_MAX) {
        *val = data_id;
        return;
    }
    
    /* Collision occurred; need to use overflow table */
    if (set->overflow_table != NULL) {
        /* First check if already exists in overflow table */
        for (size_t i = 0; i < set->overflow_count; i++) {
            if (set->overflow_table[i] == data_id) {
                return; /* already exists */
            }
        }
        
        /* Add to overflow table */
        if (set->overflow_count < set->overflow_size) {
            set->overflow_table[set->overflow_count] = data_id;
            set->overflow_count++;
        }
        /* If overflow table is also full, consider linear probing or other strategies */
    }
}

#endif /* BUFFER_H */
