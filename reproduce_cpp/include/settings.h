#ifndef SETTINGS_H
#define SETTINGS_H

#include <stddef.h>
#include <string.h>

/* Configuration constants */
#define ROUND 1      /* round of testing query */
#ifndef TOPK
#define TOPK 10      /* knn */
#endif

#ifndef VIRTUAL_DPUS
#define VIRTUAL_DPUS 100
#endif

#ifndef EF
#define EF 400
#endif

/* DPU-related constants */
#ifndef NR_DPUS
#define NR_DPUS 100
#endif

#ifndef VIRTUAL_DPUS
#define VIRTUAL_DPUS NR_DPUS
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 200
#endif

#ifndef DIMM
#define DIMM 420
#endif

#ifndef PADDED_DIMM
#define PADDED_DIMM 512
#endif

#ifndef MAX_SIZE_PER_DPU
#define MAX_SIZE_PER_DPU 30000
#endif

#ifndef DEGREE
#define DEGREE 32
#endif

#ifndef MAX_QUERY_CLUSTER_PER_DPU
#define MAX_QUERY_CLUSTER_PER_DPU 1004
#endif

#ifndef POST_EF
#define POST_EF 30
#endif

/* Dataset names */
#define MAX_DATASETS 8
#define MAX_DATASET_NAME_LEN 64

/* Dataset definitions */
typedef struct {
    char name[MAX_DATASET_NAME_LEN];
    char distance[16];  /* "l2" or "angular" */
} Dataset;

/* Dataset configuration */
static const Dataset g_datasets[] = {
    // {"SSN", "l2"},
    // {"SPACE1B", "l2"},
    {"sift1B", "l2"},
    /* More datasets can be added:
    {"sift1B", "l2"},
    {"deep1M", "l2"},
    {"gist", "l2"},
    {"msong", "l2"},
    {"tiny5m", "l2"},
    {"imagenet", "l2"},
    {"deep100M", "l2"},
    {"msturing100M", "l2"}
    */
};
static const int g_num_datasets = sizeof(g_datasets) / sizeof(g_datasets[0]);

/* Degree configuration */
typedef struct {
    char dataset_name[MAX_DATASET_NAME_LEN];
    int degrees[8];
    int num_degrees;
} DatasetDegrees;

static const DatasetDegrees g_degrees[] = {
    // {"SSN", {32}, 1},
    // {"SPACE1B", {32}, 1},
    {"sift1B", {32}, 1},
    /* More configurations can be added:
    {"sift1B", {32}, 1},
    {"deep1M", {32, 64, 128}, 3},
    {"gist", {32, 64, 128}, 3},
    {"msong", {32, 64, 128}, 3},
    {"tiny5m", {32, 64, 128}, 3},
    {"imagenet", {32, 64, 128}, 3},
    {"deep100M", {32}, 1},
    {"msturing100M", {32}, 1}
    */
};

/* Iteration configuration */
typedef struct {
    char dataset_name[MAX_DATASET_NAME_LEN];
    int iterations;
} DatasetIter;

static const DatasetIter g_iter[] = {
    // {"SSN", 3},
    // {"SPACE1B", 3},
    {"sift1B", 3},
    /* More configurations can be added:
    {"sift1B", 3},
    {"deep1M", 3},
    {"gist", 3},
    {"msong", 3},
    {"tiny5m", 3},
    {"imagenet", 3},
    {"deep100M", 3},
    {"msturing100M", 4}
    */
};

/* Helper function: find distance type by dataset name */
static inline const char* get_dataset_distance(const char* name) {
    for (int i = 0; i < g_num_datasets; i++) {
        if (strcmp(g_datasets[i].name, name) == 0) {
            return g_datasets[i].distance;
        }
    }
    return NULL;
}

/* Helper function: find degree configuration by dataset name */
static inline const DatasetDegrees* get_dataset_degrees(const char* name) {
    int num_configs = sizeof(g_degrees) / sizeof(g_degrees[0]);
    for (int i = 0; i < num_configs; i++) {
        if (strcmp(g_degrees[i].dataset_name, name) == 0) {
            return &g_degrees[i];
        }
    }
    return NULL;
}

#endif /* SETTINGS_H */

