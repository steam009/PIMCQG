#include "../include/hashtable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hash function (using simple modulo) */
static size_t hash_function(int key, size_t bucket_count) {
    /* Use FNV-1a hash variant */
    unsigned int hash = 2166136261u;
    unsigned int k = (unsigned int)key;
    
    hash ^= (k & 0xFF);
    hash *= 16777619u;
    hash ^= ((k >> 8) & 0xFF);
    hash *= 16777619u;
    hash ^= ((k >> 16) & 0xFF);
    hash *= 16777619u;
    hash ^= ((k >> 24) & 0xFF);
    hash *= 16777619u;
    
    return hash % bucket_count;
}

/* Check if a number is prime (for selecting bucket count) */
static int is_prime(size_t n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    
    for (size_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

/* Find the next prime >= n */
static size_t next_prime(size_t n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;
    
    while (!is_prime(n)) {
        n += 2;
    }
    return n;
}

/* =============== Public Function Implementation =============== */

HashTable* hashtable_create(size_t initial_bucket_count) {
    if (initial_bucket_count == 0) {
        initial_bucket_count = 1024;  /* default bucket count */
    }
    
    /* Use prime as bucket count to reduce collisions */
    initial_bucket_count = next_prime(initial_bucket_count);
    
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    if (!table) return NULL;
    
    table->buckets = (HashNode**)calloc(initial_bucket_count, sizeof(HashNode*));
    if (!table->buckets) {
        free(table);
        return NULL;
    }
    
    table->bucket_count = initial_bucket_count;
    table->size = 0;
    table->collision_count = 0;
    
    return table;
}

void hashtable_free(HashTable* table) {
    if (!table) return;
    
    /* Free all nodes */
    for (size_t i = 0; i < table->bucket_count; i++) {
        HashNode* node = table->buckets[i];
        while (node) {
            HashNode* next = node->next;
            if (node->value.data) {
                free(node->value.data);
            }
            free(node);
            node = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

void hashtable_free_shallow(HashTable* table) {
    if (!table) return;
    
    /* Free all nodes but not stored values */
    for (size_t i = 0; i < table->bucket_count; i++) {
        HashNode* node = table->buckets[i];
        while (node) {
            HashNode* next = node->next;
            /* Do not free node->value.data because it is shared/externally owned */
            free(node);
            node = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

int hashtable_insert(HashTable* table, int key, IntArray* value) {
    if (!table || !value) return -1;
    
    size_t index = hash_function(key, table->bucket_count);
    
    /* Check if key already exists */
    HashNode* node = table->buckets[index];
    while (node) {
        if (node->key == key) {
            /* Key already exists, update value */
            if (node->value.data) {
                free(node->value.data);
            }
            node->value = *value;
            return 0;
        }
        node = node->next;
    }
    
    /* Create new node */
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    if (!new_node) return -1;
    
    new_node->key = key;
    new_node->value = *value;
    new_node->next = table->buckets[index];
    
    /* If the bucket already has a node, record collision */
    if (table->buckets[index]) {
        table->collision_count++;
    }
    
    table->buckets[index] = new_node;
    table->size++;
    
    /* Check load factor; rehash if too high */
    double load_factor = (double)table->size / table->bucket_count;
    if (load_factor > 0.75) {
        hashtable_rehash(table, table->bucket_count * 2);
    }
    
    return 0;
}

IntArray* hashtable_get(HashTable* table, int key) {
    if (!table) return NULL;
    
    size_t index = hash_function(key, table->bucket_count);
    HashNode* node = table->buckets[index];
    
    while (node) {
        if (node->key == key) {
            return &node->value;
        }
        node = node->next;
    }
    
    return NULL;
}

int hashtable_contains(HashTable* table, int key) {
    return hashtable_get(table, key) != NULL;
}

int hashtable_remove(HashTable* table, int key) {
    if (!table) return -1;
    
    size_t index = hash_function(key, table->bucket_count);
    HashNode* node = table->buckets[index];
    HashNode* prev = NULL;
    
    while (node) {
        if (node->key == key) {
            if (prev) {
                prev->next = node->next;
            } else {
                table->buckets[index] = node->next;
            }
            
            if (node->value.data) {
                free(node->value.data);
            }
            free(node);
            table->size--;
            return 0;
        }
        prev = node;
        node = node->next;
    }
    
    return -1;  /* key does not exist */
}

size_t hashtable_size(const HashTable* table) {
    return table ? table->size : 0;
}

double hashtable_load_factor(const HashTable* table) {
    if (!table || table->bucket_count == 0) return 0.0;
    return (double)table->size / table->bucket_count;
}

void hashtable_print_stats(const HashTable* table) {
    if (!table) {
        printf("HashTable: NULL\n");
        return;
    }
    
    printf("=== HashTable Statistics ===\n");
    printf("Size: %zu\n", table->size);
    printf("Bucket Count: %zu\n", table->bucket_count);
    printf("Load Factor: %.2f\n", hashtable_load_factor(table));
    printf("Collision Count: %zu\n", table->collision_count);
    
    /* Statistics of chain length distribution */
    size_t empty_buckets = 0;
    size_t max_chain_length = 0;
    size_t total_chain_length = 0;
    
    for (size_t i = 0; i < table->bucket_count; i++) {
        size_t chain_length = 0;
        HashNode* node = table->buckets[i];
        
        if (!node) {
            empty_buckets++;
        } else {
            while (node) {
                chain_length++;
                node = node->next;
            }
            total_chain_length += chain_length;
            if (chain_length > max_chain_length) {
                max_chain_length = chain_length;
            }
        }
    }
    
    printf("Empty Buckets: %zu (%.1f%%)\n", 
           empty_buckets, 
           100.0 * empty_buckets / table->bucket_count);
    printf("Max Chain Length: %zu\n", max_chain_length);
    
    if (table->size > 0) {
        printf("Average Chain Length: %.2f\n", 
               (double)total_chain_length / (table->bucket_count - empty_buckets));
    }
    printf("===========================\n");
}

int hashtable_rehash(HashTable* table, size_t new_bucket_count) {
    if (!table) return -1;
    
    new_bucket_count = next_prime(new_bucket_count);
    
    /* Create new bucket array */
    HashNode** new_buckets = (HashNode**)calloc(new_bucket_count, sizeof(HashNode*));
    if (!new_buckets) return -1;
    
    /* Re-insert all nodes */
    for (size_t i = 0; i < table->bucket_count; i++) {
        HashNode* node = table->buckets[i];
        while (node) {
            HashNode* next = node->next;
            
            /* Compute new index */
            size_t new_index = hash_function(node->key, new_bucket_count);
            
            /* Insert into new bucket */
            node->next = new_buckets[new_index];
            new_buckets[new_index] = node;
            
            node = next;
        }
    }
    
    /* Free old bucket array, update hash table */
    free(table->buckets);
    table->buckets = new_buckets;
    table->bucket_count = new_bucket_count;
    table->collision_count = 0;  /* reset collision count */
    
    return 0;
}

int* hashtable_get_all_keys(HashTable* table) {
    if (!table) return NULL;
    int* keys = (int*)malloc(table->size * sizeof(int));
    size_t index = 0;
    for (size_t i = 0; i < table->bucket_count; i++) {
        HashNode* node = table->buckets[i];
        while (node) {
            keys[index++] = node->key;
            node = node->next;
        }
    }
    return keys;
}