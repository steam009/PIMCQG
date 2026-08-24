import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import faiss
import struct

from reproduce.utils.io import read_fvecs, read_bvecs2, read_ivecs, fvecs_write as to_fvecs, ivecs_write as to_ivecs
source = './'
_SUFFIX_DTYPE = {
    '.fbin':  (np.float32, 4),
    '.u8bin': (np.uint8,   1),
    '.i8bin': (np.int8,    1),
}


def read_bin(filename, start_idx=0, chunk_size=-1):
    """
    Read a SimSearchNet binary file (.fbin / .u8bin / .i8bin) using memory mapping.

    File layout:
        [num_points:     uint32  (4 bytes)]
        [num_dimensions: uint32  (4 bytes)]
        [num_points × num_dimensions × sizeof(type) bytes]

    Parameters
    ----------
    filename   : path to the .fbin / .u8bin / .i8bin file
    start_idx  : first vector index to read (0-based)
    chunk_size : number of vectors to read (-1 = all remaining)

    Returns
    -------
    np.ndarray, shape (nvecs_to_read, dim), dtype float32
        int8 / uint8 data is cast to float32 so FAISS can consume it directly.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _SUFFIX_DTYPE:
        raise ValueError(f"Unsupported file extension '{ext}'. "
                         f"Expected one of {list(_SUFFIX_DTYPE.keys())}")
    dtype, itemsize = _SUFFIX_DTYPE[ext]

    print(f"Reading from {filename}  (dtype={dtype.__name__}) ...")
    with open(filename, 'rb') as f:
        # Header uses uint32, not int32
        nvecs = struct.unpack('I', f.read(4))[0]
        dim   = struct.unpack('I', f.read(4))[0]
    print(f"Header claims: {nvecs} vectors, Dimension: {dim}")

    # Infer actual vector count from real file size to handle partial downloads
    file_size    = os.path.getsize(filename)
    actual_nvecs = (file_size - 8) // (dim * itemsize)
    if actual_nvecs != nvecs:
        print(f"Warning: file only contains {actual_nvecs} vectors "
              f"(header says {nvecs}, file may be a partial download)")
        nvecs = actual_nvecs
    print(f"Actual vectors available: {nvecs}")

    # Clamp to what is actually present
    nvecs_to_read = nvecs - start_idx
    if chunk_size != -1:
        nvecs_to_read = min(chunk_size, nvecs_to_read)

    byte_offset = 8 + start_idx * dim * itemsize

    # Memory-map: no full file load into RAM; OS pages data on demand
    data = np.memmap(filename, dtype=dtype, mode='r',
                     offset=byte_offset, shape=(nvecs_to_read, dim))
    print(f"Loaded {nvecs_to_read} vectors via memmap (offset={byte_offset} bytes)")

    # FAISS requires float32; cast only when necessary (fbin is already float32)
    if dtype == np.float32:
        return data
    return data.astype(np.float32)

def load_base_vectors(path, dataset):
    """
    Read original vectors X.
    Prefer fvecs; fall back to bvecs if fvecs does not exist.
    """
    fvecs_path = os.path.join(path, f"{dataset}_base.fvecs")
    bvecs_path = os.path.join(path, f"{dataset}_base.bvecs")

    if os.path.exists(fvecs_path):
        X = read_fvecs(fvecs_path)
    elif os.path.exists(bvecs_path):
        X = read_bvecs2(bvecs_path)
    else:
        raise FileNotFoundError(
            f"Cannot find base vectors for dataset '{dataset}' under {path} "
            f"(neither '{dataset}_base.fvecs' nor '{dataset}_base.bvecs')."
        )
    return X


def redistribute_clusters(
    X,
    centroids,
    cluster_id,
    dist_to_centroid,
    max_cluster_size_limit,
    max_iterations=100,
    k_neighbors=100,
):

    # Refine an existing clustering so that every cluster size stays within
    # max_cluster_size_limit. Logic is adapted from data/ivf.py and
    # extracted as a standalone function.

    if max_cluster_size_limit is None:
        return cluster_id, dist_to_centroid

    print("Checking and redistributing oversized clusters...")
    cluster_id_flat = cluster_id.flatten()
    N = X.shape[0]

    if cluster_id_flat.shape[0] != N:
        raise ValueError(
            f"cluster_id size ({cluster_id_flat.shape[0]}) "
            f"does not match number of base vectors ({N})"
        )

    # Build a simple L2 quantizer from centroids for re-searching nearest coarse clusters
    D = centroids.shape[1]
    K = centroids.shape[0]
    quantizer = faiss.IndexFlatL2(D)
    quantizer.add(centroids.astype(np.float32))

    k_neighbors = min(k_neighbors, K)

    iteration = 0
    while iteration < max_iterations:
        cluster_sizes = np.bincount(cluster_id_flat, minlength=K)
        oversized_clusters = np.where(cluster_sizes > max_cluster_size_limit)[0]

        if len(oversized_clusters) == 0:
            print(f"All clusters satisfy the size limit after {iteration} iterations")
            break

        if iteration == 0:
            print(f"Found {len(oversized_clusters)} oversized clusters")

        # For each oversized cluster, reassign excess vectors to other clusters
        for cluster_idx in oversized_clusters:
            cluster_vectors = np.where(cluster_id_flat == cluster_idx)[0]
            excess_count = len(cluster_vectors) - max_cluster_size_limit
            if excess_count <= 0:
                continue

            # Extract the vectors that need reassignment
            excess_indices = cluster_vectors[:excess_count]
            excess_vectors = X[excess_indices]

            # Search more nearest neighbors to find clusters under the limit
            distances, candidate_cluster_ids = quantizer.search(
                excess_vectors.astype(np.float32), k_neighbors
            )

            # Reassign: prefer clusters that are under the size limit
            for i, vec_idx in enumerate(excess_indices):
                assigned = False
                # Traverse candidate clusters; pick the first under the limit
                for j in range(k_neighbors):
                    candidate_cluster = int(candidate_cluster_ids[i, j])
                    # Skip the original cluster
                    if candidate_cluster == cluster_idx:
                        continue
                    # If candidate cluster is under the limit, assign to it
                    if cluster_sizes[candidate_cluster] < max_cluster_size_limit:
                        cluster_id_flat[vec_idx] = candidate_cluster
                        dist_to_centroid[vec_idx, 0] = distances[i, j] ** 0.5
                        cluster_sizes[cluster_idx] -= 1
                        cluster_sizes[candidate_cluster] += 1
                        assigned = True
                        break

                # If all candidate clusters exceed the limit, pick the smallest one
                if not assigned:
                    candidate_indices = candidate_cluster_ids[i, :]
                    candidate_sizes = cluster_sizes[candidate_indices]
                    # Exclude the original cluster
                    mask = candidate_indices != cluster_idx
                    if np.any(mask):
                        valid_indices = np.where(mask)[0]
                        best_idx = valid_indices[np.argmin(candidate_sizes[valid_indices])]
                        best_cluster = int(candidate_indices[best_idx])
                        cluster_id_flat[vec_idx] = best_cluster
                        dist_to_centroid[vec_idx, 0] = distances[i, best_idx] ** 0.5
                        cluster_sizes[cluster_idx] -= 1
                        cluster_sizes[best_cluster] += 1
                    else:
                        # Should not happen: all candidates are the original cluster
                        if k_neighbors > 1:
                            fallback_cluster = int(candidate_indices[1])
                            cluster_id_flat[vec_idx] = fallback_cluster
                            dist_to_centroid[vec_idx, 0] = distances[i, 1] ** 0.5
                            cluster_sizes[cluster_idx] -= 1
                            cluster_sizes[fallback_cluster] += 1

        iteration += 1

        if iteration % 10 == 0:
            cluster_sizes = np.bincount(cluster_id_flat, minlength=K)
            oversized_cnt = np.sum(cluster_sizes > max_cluster_size_limit)
            print(
                f"Iteration {iteration}: Max cluster size: {np.max(cluster_sizes)}, "
                f"Oversized clusters: {oversized_cnt}"
            )

    if iteration >= max_iterations:
        print(
            f"Warning: Reached maximum iterations ({max_iterations}). "
            "Some clusters may still exceed the limit."
        )

    # Final statistics
    cluster_sizes = np.bincount(cluster_id_flat, minlength=K)
    final_oversized = np.where(cluster_sizes > max_cluster_size_limit)[0]
    if len(final_oversized) > 0:
        print(
            f"Warning: {len(final_oversized)} clusters still exceed the limit "
            f"after {iteration} iterations"
        )
    print(
        "Final statistics - "
        f"Max cluster size: {np.max(cluster_sizes)}, "
        f"Min cluster size: {np.min(cluster_sizes)}, "
        f"Mean cluster size: {np.mean(cluster_sizes):.2f}"
    )

    return cluster_id_flat.reshape(-1, 1), dist_to_centroid


# ── Dataset configurations ───────────────────────────────────────────────────

DATASETS = [
    {
        "name": "SIFT1B",
        "base_file": "SIFT1B_base.bvecs",
        "loader": "bvecs",
    },
    {
        "name": "SPACE1B",
        "base_file": "vectors.bin",       # directory with vectors_*.bin parts
        "loader": "space1b",
    },
    {
        "name": "SSN",
        "base_file": "FB_ssnpp_database.u8bin",
        "loader": "u8bin",
    },
]


def load_base_vectors_by_dataset(path, dataset_cfg):
    """Load base vectors for a dataset using the configured loader."""
    base_path = os.path.join(path, dataset_cfg["base_file"])
    loader = dataset_cfg["loader"]

    if loader == "bvecs":
        X = read_bvecs2(base_path)
    elif loader == "space1b":
        # read_bin handles .i8bin extension
        X = read_bin(base_path)
    elif loader == "u8bin":
        X = read_bin(base_path)
    else:
        raise ValueError(f"Unknown loader: {loader}")
    return X


if __name__ == "__main__":

    K = 8192                    # Must match K used during training
    max_cluster_size_limit = 400000

    for ds_cfg in DATASETS:
        dataset = ds_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Refine IVF clustering - dataset={dataset}, K={K}")
        print(f"{'='*60}")
        if max_cluster_size_limit:
            print(f"Max cluster size limit: {max_cluster_size_limit}")

        path = os.path.join(source, dataset)

        # Paths match those defined in data/ivf.py
        centroids_path = os.path.join(path, f"{dataset}_centroid_{K}.fvecs")
        dist_to_centroid_path = os.path.join(path, f"{dataset}_dist_to_centroid_{K}.fvecs")
        cluster_id_path = os.path.join(path, f"{dataset}_cluster_id_{K}.ivecs")

        # Read original vectors and IVF results
        X = load_base_vectors_by_dataset(path, ds_cfg)
        centroids = read_fvecs(centroids_path)
        dist_to_centroid = read_fvecs(dist_to_centroid_path)
        cluster_id = read_ivecs(cluster_id_path)

        print(
            f"Loaded: X={X.shape}, centroids={centroids.shape}, "
            f"dist_to_centroid={dist_to_centroid.shape}, cluster_id={cluster_id.shape}"
        )

        # Refine clusters
        cluster_id_refined, dist_to_centroid_refined = redistribute_clusters(
            X,
            centroids,
            cluster_id,
            dist_to_centroid,
            max_cluster_size_limit=max_cluster_size_limit,
        )

        # Overwrite the original files (change to new filenames if you need to keep the originals)
        to_fvecs(dist_to_centroid_path, dist_to_centroid_refined)
        to_ivecs(cluster_id_path, cluster_id_refined)

        print("Refinement finished and files updated.")

