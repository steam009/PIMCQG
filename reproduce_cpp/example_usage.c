/**
 * Simplified example: demonstrate how to use reproduce_c (C language version)
 * 
 * Compilation command:
 *   gcc -std=c11 -O3 -fopenmp -I./include example_usage.c src/utils.c src/ivf_search.c -o example_usage -lm
 */

#include "include/settings.h"
#include "include/utils.h"
#include "include/ivf_search.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

/* Helper function: check if a file exists */
static int file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

/* Helper function: compute vector magnitude */
static float vector_norm(const FloatArray* vec) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec->size; i++) {
        sum += vec->data[i] * vec->data[i];
    }
    return sqrtf(sum);
}

int main(void) {
    printf("=== reproduce_c Usage Example (C Language Version) ===\n");
    
    /* 1. Data file reading example */
    printf("\n1. Data file reading test\n");
    
    /* Use relatively small test files here */
    const char* test_base = "../data/sift/sift_base.fvecs";
    const char* test_query = "../data/sift/sift_query.fvecs";
    
    /* Check if file exists */
    if (!file_exists(test_base)) {
        printf("Test data file does not exist: %s\n", test_base);
        printf("Please ensure data files exist before running this example\n");
        return 1;
    }
    
    /* Read a small amount of data for testing */
    FloatMatrix* base = read_fvecs(test_base);
    FloatMatrix* query = read_fvecs(test_query);
    
    if (!base || !query) {
        fprintf(stderr, "Failed to read data file\n");
        if (base) float_matrix_free(base);
        if (query) float_matrix_free(query);
        return 1;
    }
    
    printf("Base dataset size: %zu x %zu\n", 
           base->num_rows, 
           base->num_rows > 0 ? base->rows[0].size : 0);
    printf("Query dataset size: %zu x %zu\n", 
           query->num_rows, 
           query->num_rows > 0 ? query->rows[0].size : 0);
    
    /* 2. Distance computation test */
    printf("\n2. Distance computation test\n");
    if (base->num_rows > 0 && query->num_rows > 0) {
        float dist = sqr_dist(&base->rows[0], &query->rows[0]);
        printf("Squared distance between first base vector and first query vector: %.6f\n", dist);
    }
    
    /* 3. Normalization test */
    printf("\n3. Normalization test\n");
    if (query->num_rows > 0) {
        /* Copy a vector for testing */
        FloatArray test_vec;
        test_vec.size = query->rows[0].size;
        test_vec.capacity = query->rows[0].size;
        test_vec.data = (float*)malloc(test_vec.size * sizeof(float));
        
        if (test_vec.data) {
            /* Copy data */
            for (size_t i = 0; i < test_vec.size; i++) {
                test_vec.data[i] = query->rows[0].data[i];
            }
            
            float norm_before = vector_norm(&test_vec);
            printf("Vector magnitude before normalization: %.6f\n", norm_before);
            
            normalize_vector(&test_vec);
            
            float norm_after = vector_norm(&test_vec);
            printf("Vector magnitude after normalization: %.6f\n", norm_after);
            
            free(test_vec.data);
        }
    }
    
    /* 4. Memory usage monitoring */
    printf("\n4. Memory usage monitoring\n");
    size_t memory_usage = get_memory_usage();
    printf("Current memory usage: %zu MB\n", memory_usage / (1024 * 1024));
    
    /* 5. BeamSize generator test */
    printf("\n5. BeamSize generator test\n");
    BeamSizeGenerator* generator = beam_size_generator_create(10);
    if (generator) {
        printf("First 5 beam size values: ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", beam_size_generator_next(generator));
        }
        printf("\n");
        beam_size_generator_free(generator);
    }
    
    /* 6. Centroid search test */
    printf("\n6. Centroid search test\n");
    if (base->num_rows > 0 && query->num_rows > 0) {
        /* Use first 10 base vectors as "centroids" for testing */
        FloatMatrix* test_centroids = float_matrix_create(10);
        int num_centroids = base->num_rows < 10 ? (int)base->num_rows : 10;
        
        for (int i = 0; i < num_centroids; i++) {
            /* Copy vector */
            FloatArray centroid;
            centroid.size = base->rows[i].size;
            centroid.capacity = base->rows[i].size;
            centroid.data = (float*)malloc(centroid.size * sizeof(float));
            
            if (centroid.data) {
                for (size_t j = 0; j < centroid.size; j++) {
                    centroid.data[j] = base->rows[i].data[j];
                }
                float_matrix_add_row(test_centroids, &centroid);
            }
        }
        
        DistIndexArray* nearest = find_nearest_centroids(&query->rows[0], test_centroids, 3);
        if (nearest) {
            printf("Nearest 3 centroid IDs: ");
            for (size_t i = 0; i < nearest->size; i++) {
                printf("%d(%.6f) ", nearest->data[i].index, nearest->data[i].distance);
            }
            printf("\n");
            dist_index_array_free(nearest);
        }
        
        float_matrix_free(test_centroids);
    }
    
    printf("\n=== Example Complete ===\n");
    printf("If all tests pass, the basic functionality is working correctly\n");
    printf("Next, you can run the complete reproduce_c program\n");
    
    /* Clean up memory */
    float_matrix_free(base);
    float_matrix_free(query);
    
    return 0;
}

