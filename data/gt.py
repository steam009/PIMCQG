import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from reproduce.utils.io import *
from sklearn import preprocessing
import struct

source = './'

def read_spacev1b_vectors(vectors_bin_dir):
    """Read SPACEV1B vectors from the vectors.bin directory containing multiple part files."""
    part_files = sorted(
        [f for f in os.listdir(vectors_bin_dir) if f.startswith('vectors_') and f.endswith('.bin')],
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

    return np.frombuffer(vecbuf, dtype=np.int8).reshape((vec_count, vec_dimension))

def normalize(X):
    X = preprocessing.normalize(X, axis=1, norm='l2')
    return X

datasets = ["SPACE1B"]
dis_type = "l2"

if __name__ == "__main__":
    dataset = 'SPACE1B'
    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    vectors_bin_dir = os.path.join(path, 'vectors.bin')
    X = read_spacev1b_vectors(vectors_bin_dir)
    X = X[:1000000000]  # Use only the first 1 billion vectors for clustering
    D = X.shape[1]
    print(f"Computing groundtruth for {dataset}")
    query_path = os.path.join(path, 'query.bin')
    with open(query_path, 'rb') as fq:
        q_count = struct.unpack('i', fq.read(4))[0]
        q_dimension = struct.unpack('i', fq.read(4))[0]
        print(f"Query vectors: {q_count}, Dimension: {q_dimension}")
        query = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))
    if dis_type == "angular":
        X = normalize(X)
        query = normalize(query)

    gt = []
    for q in query:
        # Cast to int16 to avoid int8 overflow during subtraction,
        # then use squared L2 distance (sufficient for ranking, no sqrt needed)
        diff = X.astype(np.int16) - q.astype(np.int16)
        dist_sq = np.einsum('ij,ij->i', diff.astype(np.int32), diff.astype(np.int32))
        gt.append(np.argsort(dist_sq)[:1000])

    gt = np.array(gt)

    ivecs_write(f"./{dataset}/{dataset}_groundtruth.ivecs", gt)
