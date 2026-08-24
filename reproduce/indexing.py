from utils.io import (
    fvecs_read,
    ivecs_read,
    read_fvecs,
    read_ivecs,
    read_bvecs2,
    read_spacev1b_vectors,
    read_spacev1b_truth,
    read_u8bin,
)
from utils.preprocess import normalize
import symphonyqg
from time import time
from settings import EF, datasets, degrees, iter
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading
import gc

def process_one_cluster(cluster_id_value, base_data, global_cluster_id):
    # Find indices for this cluster
    indices = np.where(global_cluster_id == cluster_id_value)[0]
    if len(indices) < 2:
        return None, indices
        
    # Extract only this cluster's data
    cluster_samples = base_data[indices].astype(np.float32)
    return cluster_samples, indices


def save_mapping(mapping, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(mapping, f)

if __name__ == "__main__":

    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]
        ITER = iter[DATASET]

        if DATASET == "SSN":
            base = read_u8bin("/datahome/datasets/amelieteam/FB_ssnpp_database.u8bin")
            gt = None
            C = 8192
            centroids_path = os.path.join(f"./data/{DATASET}/{DATASET}_centroid_{C}.fvecs")
            cluster_id_path = os.path.join(f"./data/{DATASET}/{DATASET}_cluster_id_{C}.ivecs")
        elif DATASET == "SPACE1B":
            base = read_spacev1b_vectors(f"./data/{DATASET}/vectors.bin")
            gt, _ = read_spacev1b_truth(f"./data/{DATASET}/truth.bin")
            C = 8192
            centroids_path = os.path.join(f"./data/{DATASET}/SPACE1B_centroid_{C}.fvecs")
            cluster_id_path = os.path.join(f"./data/{DATASET}/SPACE1B_cluster_id_{C}.ivecs")
        else:
            base = read_bvecs2(f"./data/{DATASET}/{DATASET}_base.bvecs")
            # base = base[:100000000]
            # base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
            # query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
            # gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")
            gt = ivecs_read(f"./data/{DATASET}/{DATASET}_idx_500M.ivecs")
            C = 4096
            centroids_path = os.path.join(f"./data/{DATASET}/{DATASET}_centroid_{C}.fvecs")
            cluster_id_path = os.path.join(f"./data/{DATASET}/{DATASET}_cluster_id_{C}.ivecs")
        
        N, D = base.shape
        
        print(f"Dataset: {DATASET}, N: {N}, D: {D}")
        
        base = base.astype(np.float32)
        
        dist_to_centroid_path = os.path.join(f"./data/{DATASET}/{DATASET}_dist_to_centroid_{C}.fvecs")

        centroids  = read_fvecs(centroids_path)
        cluster_id = read_ivecs(cluster_id_path)

        # Create output directory
        ivf_dir = f"./data/{DATASET}/IVF-reduceMem"
        os.makedirs(ivf_dir, exist_ok=True)
        
        # Generate and save mapping (could be optimized further)
        cluster_to_original = {}
        for i in range(C):
            cluster_to_original[i] = np.where(cluster_id == i)[0]
        
        mapping_path = f"./data/{DATASET}/IVF-reduceMem/cluster_to_original_{C}.pkl"
        save_mapping(cluster_to_original, mapping_path)
        
        if DISTANCE == "angular":
            base = normalize(base)

        for DEGREE in degrees[DATASET]:
            mat = None
            for i in range(C):
                # Process only this cluster
                cluster_data, indices = process_one_cluster(i, base, cluster_id)
                
                if cluster_data is None or cluster_data.shape[0] < 2:
                    print(f"Skipping cluster {i}: insufficient data")
                    continue
                
                # Build and save index for this cluster
                index = None
                if i == 0:
                    index = symphonyqg.Index(
                        index_type="QG",
                        metric="L2",
                        num_elements=cluster_data.shape[0],
                        dimension=D,
                        degree_bound=DEGREE,
                    )
                else:
                    index = symphonyqg.Index(
                        index_type="QG",
                        metric="L2",
                        num_elements=cluster_data.shape[0],
                        dimension=D,
                        degree_bound=DEGREE,
                        mat=mat,
                    )
                mat = index.get_mat()
                # 打印mat的前10个元素
                # 检查mat的维度和形状
                print(f"mat shape: {mat.shape}, ndim: {mat.ndim}")
                if mat.ndim == 1:
                    # 如果是1维数组，直接切片
                    print(f"mat前10个元素: {mat[:10]}")
                elif mat.ndim == 2:
                    # 如果是2维数组，访问第一行
                    print(f"mat第一行前10个元素: {mat[0][:10]}")
                else:
                    print(f"mat: {mat}")
                index.set_cluster(i)
                t1 = time()
                index.build_index(cluster_data, EF, num_iter=ITER, centroids=centroids[i])
                t2 = time()
                print(f"The construction time for {DATASET}{DEGREE} is {t2-t1}")
                
                # Save index and explicitly free memory
                index_path = f"./data/{DATASET}/IVF-reduceMem/symphonyqg_{DEGREE}_cluster_{i}.index"
                index.save(index_path)
                del index
                del cluster_data
                gc.collect()
                print(f"Index for cluster {i} saved and memory freed")