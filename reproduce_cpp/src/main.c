#include "../include/settings.h"
#include "../include/utils.h"
#include "../include/ivf_search.h"
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

/* Helper function: recursively create directories */
static int mkdir_recursive(const char* path) {
    char tmp[1024];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

/* CSV output helper function */
static void write_results_to_csv(const char* filename,
                                 const double* qps_values,
                                 const double* recall_values,
                                 const int* ef_values,
                                 const int* post_ef_values,
                                 int num_values,
                                 const char* method,
                                 size_t memory_usage,
                                 int nprobe) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return;
    }
    
    fprintf(file, "QPS,Recall,EFS,POST_EFS,Method,Memory,NPROBE\n");
    
    for (int i = 0; i < num_values; i++) {
        fprintf(file, "%.2f,%.2f,%d,%d,%s,%zu,%d\n",
               qps_values[i],
               recall_values[i],
               ef_values[i],
               post_ef_values[i],
               method,
               memory_usage,
               nprobe);
    }
    
    fclose(file);
    printf("Results saved to: %s\n", filename);
}

/* Helper function: return the minimum of two values */
static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

int main(int argc, char *argv[]) {
    /* Parse command line arguments to get nprobe value */
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nprobe>\n", argv[0]);
        fprintf(stderr, "Example: %s 32\n", argv[0]);
        return 1;
    }
    
    int nprobe_main = atoi(argv[1]);
    if (nprobe_main <= 0) {
        fprintf(stderr, "Error: nprobe must be a positive integer\n");
        fprintf(stderr, "Usage: %s <nprobe>\n", argv[0]);
        return 1;
    }
    
    printf("Using NPROBE=%d from command line\n", nprobe_main);
    
    /* Iterate over all datasets */
    for (int ds_idx = 0; ds_idx < g_num_datasets; ds_idx++) {
        const Dataset* dataset = &g_datasets[ds_idx];
        printf("Processing dataset: %s\n", dataset->name);
        
        /* Build file paths */
        char query_path[512];
        char gt_path[512];

        /* SPACE1B uses a flat binary format: [count, dim, data] instead of per-vector headers */
        int is_space1b = (strcmp(dataset->name, "SPACE1B") == 0);
        int is_ssn = (strcmp(dataset->name, "SSN") == 0);
        if (is_space1b) {
            snprintf(query_path, sizeof(query_path), "../data/%s/query.bin", dataset->name);
            snprintf(gt_path, sizeof(gt_path), "../data/%s/truth.bin", dataset->name);
        }
        else if (is_ssn) {
            snprintf(query_path, sizeof(query_path), "../data/%s/SSN_queries_subset3000.u8bin", dataset->name);
            snprintf(gt_path, sizeof(gt_path), "../data/%s/SSN_gt_subset3000.ivecs", dataset->name);
        }
        else {
            snprintf(query_path, sizeof(query_path), "../data/%s/%s_query.bvecs",
                    dataset->name, dataset->name);
            snprintf(gt_path, sizeof(gt_path), "../data/%s/%s_groundtruth.ivecs",
                    dataset->name, dataset->name);
        }

        struct dpu_set_t dpu_set;
        uint32_t nr_of_dpus;

        // Allocate DPUs and load binary
        DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
        DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
        DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
        printf("Allocated %d DPU(s)\n", nr_of_dpus);
        
        /* Read data */
        printf("Loading data...\n");
        FloatMatrix* query;
        IntMatrix* gt;
        if(is_ssn) {
            query = read_u8bin_as_float(query_path, 0);
            gt = read_ivecs(gt_path);
        }
        else {
            query = is_space1b
                ? read_space1b_query_bin(query_path)
                : read_bvecs_as_float(query_path);
            gt = is_space1b
                ? read_space1b_truth_bin(gt_path)
                : read_ivecs(gt_path);
        }
        
        if (!query || !gt) {
            fprintf(stderr, "Failed to load data files\n");
            // if (base) float_matrix_free(base);
            if (query) float_matrix_free(query);
            if (gt) int_matrix_free(gt);
            continue;
        }
        
        const int C = 8192;
        char centroids_path[512];
        char mapping_path[512];

        snprintf(centroids_path, sizeof(centroids_path),
                "../data/%s/%s_centroid_%d.fvecs", dataset->name, dataset->name, C);
        if (is_space1b || is_ssn) {
            snprintf(mapping_path, sizeof(mapping_path),
                    "../data/%s/IVF-reduceMem/cluster_to_original_%d.txt", dataset->name, C);
        } else {
            snprintf(mapping_path, sizeof(mapping_path),
                    "../data/%s/IVF-reduceMem-8192/cluster_to_original_%d.txt", dataset->name, C);
        }
        
        FloatMatrix* centroids = read_fvecs(centroids_path);

        if (!centroids) {
            fprintf(stderr, "Failed to load centroid data\n");
            // float_matrix_free(base);
            float_matrix_free(query);
            int_matrix_free(gt);
            continue;
        }
        printf("BATCH_SIZE: %d\n", BATCH_SIZE);
        int NQ = min_int(BATCH_SIZE, (int)query->num_rows);
        int N = 1000000000; // TODO: update to actual base count
        int D = query->num_rows > 0 ? (int)query->rows[0].size : 0;
        
        printf("Dataset info: N=%d, D=%d, NQ=%d\n", N, D, NQ);
        
        /* Normalization (if using angular distance) */
        if (strcmp(dataset->distance, "angular") == 0) {
            normalize_matrix(query);
        }
        
        /* Get degree configuration for this dataset */
        const DatasetDegrees* deg_config = get_dataset_degrees(dataset->name);
        if (!deg_config) {
            fprintf(stderr, "No degree configuration found for dataset: %s\n", dataset->name);
            // float_matrix_free(base);
            float_matrix_free(query);
            int_matrix_free(gt);
            float_matrix_free(centroids);
            continue;
        }


        /* Process each degree configuration */
        for (int deg_idx = 0; deg_idx < deg_config->num_degrees; deg_idx++) {
            int degree = deg_config->degrees[deg_idx];
            printf("Processing degree: %d\n", degree);
            
            size_t m1 = get_memory_usage();
            
            /* Create IVF searcher */
            IVFSearcher* searcher = ivf_searcher_create();
            if (!searcher) {
                fprintf(stderr, "Failed to create IVF searcher\n");
                continue;
            }
            searcher->total_num_elements = N;
            
            ivf_searcher_load_centroids(searcher, centroids_path);
            // ivf_searcher_load_base_data(searcher, base_path);
            
            /* Load cluster mapping */
            if (ivf_searcher_load_cluster_mapping(searcher, mapping_path) != 0) {
                fprintf(stderr, "Warning: Failed to load cluster mapping from %s\n", mapping_path);
                fprintf(stderr, "Skipping this configuration due to missing mapping file.\n");
                ivf_searcher_free(searcher);
                continue;
            }
            
            ivf_searcher_load_indices(searcher, dataset->name, degree, C);
            
            size_t m2 = get_memory_usage();
            size_t memory_usage = m2 - m1;
            printf("Memory usage for DEGREE %d: %zu bytes\n", degree, memory_usage);

            /* Try different nprobe values */
            int NPROBES[] = {nprobe_main};
            int num_nprobes = 1;
            int batch_size = NQ;
            
            for (int np_idx = 0; np_idx < num_nprobes; np_idx++) {
                int nprobe = NPROBES[np_idx];
                printf("Testing with NPROBE=%d\n", nprobe);
                
                /* Configure EF and POST_EF values */
                int EFS[] = {EF};
                int POST_EFS[] = {POST_EF};
                int num_efs = 1;
                int num_post_efs = 1;
                
                for (int round = 0; round < ROUND; round++) {
                    printf("Round %d/%d\n", round + 1, ROUND);
                    
                    
                    for (int ef_idx = 0; ef_idx < num_efs; ef_idx++) {
                        for (int pef_idx = 0; pef_idx < num_post_efs; pef_idx++) {
                            if(EFS[ef_idx] > POST_EFS[pef_idx]) continue;
                            int ef = EFS[ef_idx];
                            int post_ef = POST_EFS[pef_idx];
                            
                            // /* Use batch search function */
                            // Use NR_DPUS as the virtual DPU count; a larger value can also be passed to simulate more DPUs
                            IntArray** results_dpu = ivf_searcher_search_batch(&dpu_set, searcher, query, batch_size,
                                                                          ef, TOPK, nprobe, post_ef, VIRTUAL_DPUS);
                            /* Calculate recall */
                            int total_num = batch_size * TOPK;
                            int total_correct = 0;
                            
                            for (int i = 0; i < batch_size; i++) {
                                /* Create result set for lookup */
                                for (int j = 0; j < TOPK && j < (int)gt->rows[i].size; j++) {
                                    int gt_val = gt->rows[i].data[j];
                                    /* Check if in results */
                                    for (size_t k = 0; k < results_dpu[i]->size; k++) {
                                        if (results_dpu[i]->data[k] == gt_val) {
                                            total_correct++;
                                            break;
                                        }
                                    }
                                }
                            }   
                            double recall = (total_correct * 100.0) / total_num;
                            
                            
                            printf("EF: %d, POST EF: %d, NPROBE: %d\n",
                                  ef, post_ef, nprobe);
                            ivf_searcher_free_batch_results(results_dpu, batch_size);

                            // HashTable** query_centroid_to_dpu = NULL;
                            // HashTable** query_to_dpu_set = compute_query_to_dpu_mapping(searcher, query, batch_size, nprobe, VIRTUAL_DPUS, &query_centroid_to_dpu);    

                            // clock_gettime(CLOCK_MONOTONIC, &start_time);
                            // IntArray** results = calculate_recall_cpu(&dpu_set, searcher, query, batch_size, ef, TOPK, nprobe, post_ef, query_to_dpu_set, query_centroid_to_dpu);
                            // clock_gettime(CLOCK_MONOTONIC, &end_time);
                            // double total_time_cpu = (end_time.tv_sec - start_time.tv_sec) + 
                            //                 (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
                            // /* Calculate recall */
                            // int total_num_cpu = batch_size * TOPK;
                            // int total_correct_cpu = 0;
                            // printf("Calculate recall for topk: %d, gt number: %d\n", TOPK, (int)gt->rows[0].size);
                            // for (int i = 0; i < batch_size; i++) {
                            //     /* Create result set for lookup */
                            //     for (int j = 0; j < TOPK && j < (int)gt->rows[i].size; j++) {
                            //         int gt_val = gt->rows[i].data[j];
                            //         /* Check if in results */
                            //         for (size_t k = 0; k < results[i]->size; k++) {
                            //             if (results[i]->data[k] == gt_val) {
                            //                 total_correct_cpu++;
                            //                 break;
                            //             }
                            //         }
                            //     }
                            // }
                            
                            // double qps_cpu = batch_size / total_time_cpu;
                            // double recall_cpu = (total_correct_cpu * 100.0) / total_num_cpu;
                            
                            
                            // printf("%d DPU:EF: %d, POST EF: %d, NPROBE: %d, QPS: %.2f, Recall: %.2f%%\n",
                            //       VIRTUAL_DPUS,
                            //       ef, post_ef, nprobe, qps_cpu, recall_cpu);
                            
                            // /* Free results */
                            // ivf_searcher_free_batch_results(results, NQ);
                            
                            // /* Free mapping relationships */
                            // if (query_to_dpu_set) {
                            //     for (int i = 0; i < batch_size; i++) {
                            //         if (query_to_dpu_set[i]) {
                            //             hashtable_free_shallow(query_to_dpu_set[i]);
                            //         }
                            //     }
                            //     free(query_to_dpu_set);
                            // }
                            
                            // if (query_centroid_to_dpu) {
                            //     for (int i = 0; i < batch_size; i++) {
                            //         if (query_centroid_to_dpu[i]) {
                            //             // Free IntArray values in hash table
                            //             int* keys = hashtable_get_all_keys(query_centroid_to_dpu[i]);
                            //             if (keys) {
                            //                 for (size_t j = 0; j < query_centroid_to_dpu[i]->size; j++) {
                            //                     IntArray* arr = hashtable_get(query_centroid_to_dpu[i], keys[j]);
                            //                 }
                            //                 free(keys);
                            //             }
                            //             hashtable_free_shallow(query_centroid_to_dpu[i]);
                            //         }
                            //     }
                            //     free(query_centroid_to_dpu);
                            // }
                        }
                    }
                }
            }
            
            /* Clean up memory */
            // ivf_searcher_free(searcher); // has issues, causes errors, investigate later
            printf("Memory cleared for degree %d\n", degree);

        }
        DPU_ASSERT(dpu_free(dpu_set));
        printf("DPUs freed\n");
        /* Free data */
        // float_matrix_free(base);
        float_matrix_free(query);
        int_matrix_free(gt);
        float_matrix_free(centroids);
    }
    
    printf("All processing completed successfully!\n");
    return 0;
}

