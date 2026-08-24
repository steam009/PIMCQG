#include "../include/settings.h"
#include "../include/utils.h"
#include "../include/ivf_search.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

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

        /* Build file paths per dataset */
        char base_path[512];
        char query_path[512];
        char gt_path[512];

        int is_space1b = (strcmp(dataset->name, "SPACE1B") == 0);
        int is_ssn = (strcmp(dataset->name, "SSN") == 0);

        if (is_space1b) {
            /* SPACE1B: base is a directory containing vectors_*.bin;
             * query and truth are flat binary with [count, dim] header */
            snprintf(base_path, sizeof(base_path),
                    "../data/%s/vectors.bin", dataset->name);
            snprintf(query_path, sizeof(query_path),
                    "../data/%s/query.bin", dataset->name);
            snprintf(gt_path, sizeof(gt_path),
                    "../data/%s/truth.bin", dataset->name);
        } else if (is_ssn) {
            /* SSN (FB_ssnpp): base is u8bin; query and gt are u8bin/ivecs */
            snprintf(base_path, sizeof(base_path),
                    "../data/%s/FB_ssnpp_database.u8bin", dataset->name);
            snprintf(query_path, sizeof(query_path),
                    "../data/%s/SSN_queries_subset2500.u8bin", dataset->name);
            snprintf(gt_path, sizeof(gt_path),
                    "../data/%s/SSN_gt_subset2500.ivecs", dataset->name);
        } else {
            /* SIFT1B: base is bvecs, query is bvecs, gt is ivecs */
            snprintf(base_path, sizeof(base_path),
                    "../data/%s/%s_base.bvecs", dataset->name, dataset->name);
            snprintf(query_path, sizeof(query_path),
                    "../data/%s/%s_query.bvecs", dataset->name, dataset->name);
            snprintf(gt_path, sizeof(gt_path),
                    "../data/%s/%s_groundtruth.ivecs", dataset->name, dataset->name);
        }

        /* Read query and ground truth */
        printf("Loading data...\n");
        FloatMatrix* query;
        IntMatrix* gt;
        if (is_ssn) {
            query = read_u8bin_as_float(query_path, 0);
            gt = read_ivecs(gt_path);
        } else if (is_space1b) {
            query = read_space1b_query_bin(query_path);
            gt = read_space1b_truth_bin(gt_path);
        } else {
            query = read_bvecs_as_float(query_path);
            gt = read_ivecs(gt_path);
        }

        if (!query || !gt) {
            fprintf(stderr, "Failed to load data files\n");
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
            float_matrix_free(query);
            int_matrix_free(gt);
            continue;
        }

        printf("BATCH_SIZE: %d\n", BATCH_SIZE);
        int NQ = min_int(BATCH_SIZE, (int)query->num_rows);

        /* Determine N (total base vectors) */
        int N;
        int D = query->num_rows > 0 ? (int)query->rows[0].size : 0;
        if (is_ssn) {
            uint32_t base_pts = 0, base_dims = 0;
            if (read_u8bin_header(base_path, &base_pts, &base_dims) != 0) {
                fprintf(stderr, "Failed to read base u8bin header: %s\n", base_path);
                float_matrix_free(query); int_matrix_free(gt);
                float_matrix_free(centroids);
                continue;
            }
            N = (int)base_pts;
            (void)base_dims;
        } else {
            /* SIFT1B and SPACE1B both have 1B vectors */
            N = 1000000000;
        }

        printf("Dataset info: N=%d, D=%d, NQ=%d\n", N, D, NQ);

        /* Normalization (if using angular distance) */
        if (strcmp(dataset->distance, "angular") == 0) {
            normalize_matrix(query);
        }

        /* Get degree configuration for this dataset */
        const DatasetDegrees* deg_config = get_dataset_degrees(dataset->name);
        if (!deg_config) {
            fprintf(stderr, "No degree configuration found for dataset: %s\n", dataset->name);
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

            ivf_searcher_load_centroids(searcher, centroids_path);

            /* Load cluster mapping */
            if (ivf_searcher_load_cluster_mapping(searcher, mapping_path) != 0) {
                fprintf(stderr, "Warning: Failed to load cluster mapping from %s\n", mapping_path);
                fprintf(stderr, "Skipping this configuration due to missing mapping file.\n");
                ivf_searcher_free(searcher);
                continue;
            }

            ivf_searcher_load_indices(searcher, dataset->name, degree, C);
            if (ivf_searcher_load_base_data(searcher, base_path) != 0) {
                fprintf(stderr, "Failed to load base vectors from %s\n", base_path);
                ivf_searcher_free(searcher);
                continue;
            }
            if (searcher->base_data) {
                searcher->total_num_elements = (int)searcher->base_data->num_rows;
            } else {
                searcher->total_num_elements = N;
            }

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

                /* Allocate storage for each round */
                #define MAX_CONFIGS 1
                double ALL_QPS[ROUND][MAX_CONFIGS];
                double ALL_RECALL[ROUND][MAX_CONFIGS];
                int ALL_EF[ROUND][MAX_CONFIGS];
                int ALL_POST_EF[ROUND][MAX_CONFIGS];

                for (int round = 0; round < ROUND; round++) {
                    printf("Round %d/%d\n", round + 1, ROUND);

                    int config_idx = 0;

                    for (int ef_idx = 0; ef_idx < num_efs; ef_idx++) {
                        for (int pef_idx = 0; pef_idx < num_post_efs; pef_idx++) {
                            if (EFS[ef_idx] > POST_EFS[pef_idx]) continue;
                            int ef = EFS[ef_idx];
                            int post_ef = POST_EFS[pef_idx];

                            struct timespec start_time, end_time;

                            HashTable** query_centroid_to_dpu = NULL;
                            HashTable** query_to_dpu_set = compute_query_to_dpu_mapping(
                                searcher, query, batch_size, nprobe,
                                VIRTUAL_DPUS, &query_centroid_to_dpu);

                            clock_gettime(CLOCK_MONOTONIC, &start_time);
                            IntArray** results = calculate_recall_cpu(
                                NULL, searcher, query, batch_size, ef, TOPK,
                                nprobe, post_ef, query_to_dpu_set, query_centroid_to_dpu);
                            clock_gettime(CLOCK_MONOTONIC, &end_time);
                            double total_time_cpu = (end_time.tv_sec - start_time.tv_sec) +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

                            /* Per-query recall (same denominator as global: TOPK) */
                            int total_num_cpu = batch_size * TOPK;
                            int total_correct_cpu = 0;
                            printf("Calculate recall for topk: %d, gt number: %d\n",
                                   TOPK, (int)gt->rows[0].size);
                            for (int i = 0; i < batch_size; i++) {
                                int correct_i = 0;
                                int gt_lim = min_int(TOPK, (int)gt->rows[i].size);
                                for (int j = 0; j < gt_lim; j++) {
                                    int gt_val = gt->rows[i].data[j];
                                    for (size_t k = 0; k < results[i]->size; k++) {
                                        if (results[i]->data[k] == gt_val) {
                                            correct_i++;
                                            break;
                                        }
                                    }
                                }
                                total_correct_cpu += correct_i;
                            }

                            double qps_cpu = batch_size / total_time_cpu;
                            double recall_cpu = (total_correct_cpu * 100.0) / total_num_cpu;

                            ALL_QPS[round][config_idx] = qps_cpu;
                            ALL_RECALL[round][config_idx] = recall_cpu;
                            ALL_EF[round][config_idx] = ef;
                            ALL_POST_EF[round][config_idx] = post_ef;
                            config_idx++;

                            printf("%d DPU:EF: %d, POST EF: %d, NPROBE: %d, Recall: %.2f%%\n",
                                   VIRTUAL_DPUS, ef, post_ef, nprobe, recall_cpu);

                            /* Free results */
                            ivf_searcher_free_batch_results(results, NQ);

                            /* Free mapping relationships */
                            if (query_to_dpu_set) {
                                for (int i = 0; i < batch_size; i++) {
                                    if (query_to_dpu_set[i]) {
                                        hashtable_free_shallow(query_to_dpu_set[i]);
                                    }
                                }
                                free(query_to_dpu_set);
                            }

                            if (query_centroid_to_dpu) {
                                for (int i = 0; i < batch_size; i++) {
                                    if (query_centroid_to_dpu[i]) {
                                        int* keys = hashtable_get_all_keys(query_centroid_to_dpu[i]);
                                        if (keys) {
                                            for (size_t j = 0; j < query_centroid_to_dpu[i]->size; j++) {
                                                IntArray* arr = hashtable_get(query_centroid_to_dpu[i], keys[j]);
                                            }
                                            free(keys);
                                        }
                                        hashtable_free_shallow(query_centroid_to_dpu[i]);
                                    }
                                }
                                free(query_centroid_to_dpu);
                            }
                        }
                    }
                }

                /* Calculate averages */
                int num_configs = num_efs * num_efs;
                double avg_qps[MAX_CONFIGS];
                double avg_recall[MAX_CONFIGS];
                int avg_ef[MAX_CONFIGS];
                int avg_post_ef[MAX_CONFIGS];

                for (int i = 0; i < num_configs; i++) {
                    avg_qps[i] = 0.0;
                    avg_recall[i] = 0.0;
                    avg_ef[i] = 0;
                    avg_post_ef[i] = 0;

                    for (int round = 0; round < ROUND; round++) {
                        avg_qps[i] += ALL_QPS[round][i];
                        avg_recall[i] += ALL_RECALL[round][i];
                        avg_ef[i] += ALL_EF[round][i];
                        avg_post_ef[i] += ALL_POST_EF[round][i];
                    }

                    avg_qps[i] /= ROUND;
                    avg_recall[i] /= ROUND;
                    avg_ef[i] /= ROUND;
                    avg_post_ef[i] /= ROUND;
                }

                /* Save results */
                char res_dir[512];
                snprintf(res_dir, sizeof(res_dir), "./results/%s/symphonyqg_ivf_reduceMem/",
                        dataset->name);
                mkdir_recursive(res_dir);

                char method[128];
                snprintf(method, sizeof(method), "symphonyqg%d_ivf%d", degree, nprobe);

                char output_file[1024];
                snprintf(output_file, sizeof(output_file), "%s%s_%d.csv",
                        res_dir, method, TOPK);

                write_results_to_csv(output_file, avg_qps, avg_recall, avg_ef, avg_post_ef,
                                   num_configs, method, memory_usage, nprobe);
            }

            /* Clean up memory */
            printf("Memory cleared for degree %d\n", degree);
        }

        /* Free data */
        float_matrix_free(query);
        int_matrix_free(gt);
        float_matrix_free(centroids);
    }

    printf("All processing completed successfully!\n");
    return 0;
}
