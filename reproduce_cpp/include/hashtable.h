#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <stddef.h>

/* IntArray definition (if not already defined) */
#ifndef INT_ARRAY_DEFINED
#define INT_ARRAY_DEFINED
typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} IntArray;
#endif

/* Hash table node */
typedef struct HashNode {
    int key;                    /* cluster ID */
    IntArray value;             /* index array */
    struct HashNode* next;      /* linked list pointer (for collision resolution) */
} HashNode;

/* Hash table structure */
typedef struct {
    HashNode** buckets;         /* bucket array */
    size_t bucket_count;        /* number of buckets */
    size_t size;                /* number of elements */
    size_t collision_count;     /* collision count (for statistics) */
} HashTable;

/* =============== Hash Table Functions =============== */

/**
 * Create a hash table
 * @param initial_bucket_count initial number of buckets (prime recommended)
 * @return newly created hash table, NULL on failure
 */
HashTable* hashtable_create(size_t initial_bucket_count);

/**
 * Free hash table (including stored values)
 * @param table hash table to free
 */
void hashtable_free(HashTable* table);

/**
 * Shallow-free hash table (only free the hash table structure, not stored values)
 * Used when storing shared/externally-owned data
 * @param table hash table to free
 */
void hashtable_free_shallow(HashTable* table);

/**
 * Insert a key-value pair
 * @param table hash table
 * @param key key (cluster ID)
 * @param value value (index array pointer, ownership transferred)
 * @return 0 on success, -1 on failure
 */
int hashtable_insert(HashTable* table, int key, IntArray* value);

/**
 * Look up a value
 * @param table hash table
 * @param key key (cluster ID)
 * @return pointer to the found value, NULL if not found
 */
IntArray* hashtable_get(HashTable* table, int key);

/**
 * Check if a key exists
 * @param table hash table
 * @param key key
 * @return 1 if exists, 0 if not
 */
int hashtable_contains(HashTable* table, int key);

/**
 * Delete a key-value pair
 * @param table hash table
 * @param key key
 * @return 0 on success, -1 on failure (key not found)
 */
int hashtable_remove(HashTable* table, int key);

/**
 * Get hash table size
 * @param table hash table
 * @return number of elements
 */
size_t hashtable_size(const HashTable* table);

/**
 * Get load factor
 * @param table hash table
 * @return load factor (size / bucket_count)
 */
double hashtable_load_factor(const HashTable* table);

/**
 * Print hash table statistics (for debugging)
 * @param table hash table
 */
void hashtable_print_stats(const HashTable* table);

/**
 * Rehash (expand)
 * @param table hash table
 * @param new_bucket_count new number of buckets
 * @return 0 on success, -1 on failure
 */
int hashtable_rehash(HashTable* table, size_t new_bucket_count);

/**
 * Get all keys in the hash table
 * @param table hash table
 * @return array of all keys
 */
int* hashtable_get_all_keys(HashTable* table);

#endif /* HASHTABLE_H */

