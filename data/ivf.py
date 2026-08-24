import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import faiss
import struct
from reproduce.utils.io import (fvecs_write as to_fvecs,
                                ivecs_write as to_ivecs,
                                read_bvecs,
                                read_u8bin)

source = './'

# ── Dataset configurations ───────────────────────────────────────────────────

DATASETS = [
    {
        "name": "SIFT1B",
        "base_file": "SIFT1B_base.bvecs",
        "loader": "bvecs",
    },
    # {
    #     "name": "SPACE1B",
    #     "base_file": "vectors.bin",       # directory with vectors_*.bin parts
    #     "loader": "space1b",
    # },
    # {
    #     "name": "SSN",
    #     "base_file": "FB_ssnpp_database.u8bin",
    #     "loader": "u8bin",
    # },
]

# ── Data loaders ─────────────────────────────────────────────────────────────

def load_bvecs(path):
    """Load SIFT1B base vectors (bvecs format: uint8 rows)."""
    X = read_bvecs(path)
    return X.astype(np.float32)


def load_space1b(vectors_bin_dir):
    """Read SPACE1B vectors from the vectors.bin directory containing multiple
    part files (vectors_1.bin, vectors_2.bin, ...)."""
    part_files = sorted(
        [f for f in os.listdir(vectors_bin_dir)
         if f.startswith('vectors_') and f.endswith('.bin')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    part_count = len(part_files)
    print(f"Found {part_count} part file(s) in {vectors_bin_dir}")

    vec_count = None
    vec_dimension = None
    vecbuf = None
    vecbuf_offset = 0

    for i, part_name in enumerate(part_files):
        part_path = os.path.join(vectors_bin_dir, part_name)
        print(f"Reading part file: {part_path}")
        with open(part_path, 'rb') as fvec:
            if i == 0:
                # Only the first file contains the header
                vec_count = struct.unpack('i', fvec.read(4))[0]
                vec_dimension = struct.unpack('i', fvec.read(4))[0]
                print(f"Total vectors: {vec_count}, Dimension: {vec_dimension}")
                vecbuf = bytearray(vec_count * vec_dimension)
            while True:
                part = fvec.read(1048576)
                if len(part) == 0:
                    break
                vecbuf[vecbuf_offset: vecbuf_offset + len(part)] = part
                vecbuf_offset += len(part)

    X = np.frombuffer(vecbuf, dtype=np.int8).reshape((vec_count, vec_dimension))
    # FAISS requires float32; convert int8 -> float32
    return X.astype(np.float32)


def load_u8bin(path):
    """Load SSN base vectors (u8bin format: [uint32 num_pts][uint32 dims][uint8 data])."""
    X = read_u8bin(path)
    return X.astype(np.float32)


LOADERS = {
    "bvecs":   load_bvecs,
    "space1b": load_space1b,
    "u8bin":   load_u8bin,
}

# ── Cluster redistribution ───────────────────────────────────────────────────

def redistribute_oversized_clusters(X, K, cluster_id, dist_to_centroid,
                                    centroids, max_cluster_size_limit,
                                    max_iterations=100):
    """Iteratively reassign vectors from oversized clusters until all satisfy
    the size limit."""
    cluster_id_flat = cluster_id.flatten()
    iteration = 0

    while iteration < max_iterations:
        cluster_sizes = np.bincount(cluster_id_flat, minlength=K)
        oversized_clusters = np.where(cluster_sizes > max_cluster_size_limit)[0]

        if len(oversized_clusters) == 0:
            print(f"All clusters satisfy the size limit after {iteration} iterations")
            break

        if iteration == 0:
            print(f"Found {len(oversized_clusters)} oversized clusters")

        # For each oversized cluster, find excess vectors and reassign
        for cluster_idx in oversized_clusters:
            cluster_vectors = np.where(cluster_id_flat == cluster_idx)[0]
            excess_count = len(cluster_vectors) - max_cluster_size_limit

            # Compute distances from these vectors to all centroids
            excess_vectors = X[cluster_vectors[:excess_count]]
            k_neighbors = min(100, K)  # search up to 100 nearest neighbors
            distances, candidate_cluster_ids = faiss.knn(
                excess_vectors, centroids, k_neighbors)

            # Reassign: prefer clusters that are under the size limit
            for i, vec_idx in enumerate(cluster_vectors[:excess_count]):
                assigned = False
                for j in range(k_neighbors):
                    candidate_cluster = candidate_cluster_ids[i, j]
                    if candidate_cluster == cluster_idx:
                        continue
                    if cluster_sizes[candidate_cluster] < max_cluster_size_limit:
                        cluster_id_flat[vec_idx] = candidate_cluster
                        dist_to_centroid[vec_idx, 0] = distances[i, j] ** 0.5
                        assigned = True
                        break

                # If all candidates exceed the limit, pick the smallest one
                if not assigned:
                    candidate_sizes = cluster_sizes[candidate_cluster_ids[i, :]]
                    mask = candidate_cluster_ids[i, :] != cluster_idx
                    if np.any(mask):
                        valid_indices = np.where(mask)[0]
                        best_idx = valid_indices[np.argmin(candidate_sizes[valid_indices])]
                        cluster_id_flat[vec_idx] = candidate_cluster_ids[i, best_idx]
                        dist_to_centroid[vec_idx, 0] = distances[i, best_idx] ** 0.5
                    else:
                        if k_neighbors > 1:
                            cluster_id_flat[vec_idx] = candidate_cluster_ids[i, 1]
                            dist_to_centroid[vec_idx, 0] = distances[i, 1] ** 0.5

        iteration += 1
        cluster_sizes = np.bincount(cluster_id_flat, minlength=K)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Max cluster size: {np.max(cluster_sizes)}, "
                  f"Oversized clusters: {np.sum(cluster_sizes > max_cluster_size_limit)}")

    if iteration >= max_iterations:
        print(f"Warning: Reached maximum iterations ({max_iterations}). "
              "Some clusters may still exceed the limit.")

    # Final statistics
    cluster_sizes = np.bincount(cluster_id_flat, minlength=K)
    final_oversized = np.where(cluster_sizes > max_cluster_size_limit)[0]
    if len(final_oversized) > 0:
        print(f"Warning: {len(final_oversized)} clusters still exceed the limit "
              f"after {iteration} iterations")
    print(f"Final statistics - Max cluster size: {np.max(cluster_sizes)}, "
          f"Min cluster size: {np.min(cluster_sizes)}, "
          f"Mean cluster size: {np.mean(cluster_sizes):.2f}")

    return cluster_id_flat.reshape(-1, 1), dist_to_centroid


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    K = 8192
    max_cluster_size_limit = 440000  # None means no limit

    for ds_cfg in DATASETS:
        dataset = ds_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Clustering - {dataset}")
        print(f"{'='*60}")

        # Load base vectors
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, ds_cfg["base_file"])
        loader = LOADERS[ds_cfg["loader"]]
        X = loader(base_path)
        X = X[:1000000000]  # Use only the first 1 billion vectors for clustering
        D = X.shape[1]

        print(f"Number of vectors: {X.shape[0]}, Dimension: {D}, Clusters: {K}")
        if max_cluster_size_limit:
            print(f"Max cluster size limit: {max_cluster_size_limit}")

        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        dist_to_centroid_path = os.path.join(path, f'{dataset}_dist_to_centroid_{K}.fvecs')
        cluster_id_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

        # Cluster data vectors with FAISS IVF
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
        dist_to_centroid = dist_to_centroid ** 0.5

        # Redistribute oversized clusters if a size limit is set
        if max_cluster_size_limit:
            print("Checking and redistributing oversized clusters...")
            cluster_id, dist_to_centroid = redistribute_oversized_clusters(
                X, K, cluster_id, dist_to_centroid, centroids,
                max_cluster_size_limit)

        to_fvecs(dist_to_centroid_path, dist_to_centroid)
        to_ivecs(cluster_id_path, cluster_id)
        to_fvecs(centroids_path, centroids)

        # Print per-cluster size summary
        cluster_to_original = {}
        max_cluster_size = 0
        for i in range(K):
            cluster_to_original[i] = np.where(cluster_id == i)[0]
            if len(cluster_to_original[i]) > max_cluster_size:
                max_cluster_size = len(cluster_to_original[i])
        print(f"Max cluster size: {max_cluster_size}")
