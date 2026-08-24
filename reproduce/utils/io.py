import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import struct
import re

def ivecs_read(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename):
    return ivecs_read(filename).view('float32')


def ivecs_write(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(filename)
    print(f"\t{filename} wrote")


def fvecs_write(filename, m):
    m = m.astype('float32')
    ivecs_write(filename, m.view('int32'))


###
# https://github.com/harsha-simhadri/big-ann-benchmarks
# MIT License

# Copyright (c) 2021 Martin Aumüller

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_u8bin(filename, start_idx=0, chunk_size=None):
    """ Read *.u8bin file that contains uint8 vectors.
    Binary format: 8-byte header (num_points: uint32, num_dimensions: uint32),
    followed by num_points * num_dimensions bytes of uint8 data.
    Args:
        :param filename (str): path to *.u8bin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of uint8 vectors (numpy.ndarray)
    """
    print(f"Reading u8bin file: {filename}")
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.uint32)
        nvecs = int(nvecs)
        dim = int(dim)
        read_count = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=read_count * dim, dtype=np.uint8,
                          offset=start_idx * dim)
    print(f"  Loaded {read_count} vectors, dimension {dim}")
    return arr.reshape(read_count, dim)

def read_bvecs2(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    bv = np.fromfile(filename, dtype=np.uint8)
    if bv.size == 0:
        return np.zeros((0, 0))
    
    # First 4 bytes are the dimension (as int32)
    dim = np.frombuffer(bv[:4], dtype=np.int32)[0]
    assert dim > 0
    
    # Reshape the data - each vector has a 4-byte header followed by dim bytes of data
    bv = bv.reshape(-1, 4 + dim)
    
    # Extract the data part (without headers) and convert to float32
    vectors = bv[:, 4:]
    
    if c_contiguous:
        vectors = vectors.copy()
    
    return vectors

def process_batch_bvecs(args):
    start_idx, count, file_path, dimension = args
    
    with open(file_path, 'rb') as f:
        # 计算文件偏移量并移动到正确位置
        offset = start_idx * (4 + dimension)
        f.seek(offset)
        
        # 为当前批次分配内存 - 使用 uint8 类型
        batch_data = np.zeros((count, dimension), dtype=np.uint8)
        
        for i in range(count):
            # 读取维度信息
            dim_check = np.fromfile(f, dtype=np.int32, count=1)[0]
            if dim_check != dimension:
                raise IOError(f"Non-uniform vector sizes at position {start_idx + i}")
            
            # 读取原始字节数据
            batch_data[i] = np.fromfile(f, dtype=np.uint8, count=dimension)
        
        # 不再转换为 float32，直接返回 uint8 数据
    
    return start_idx, batch_data

def read_bvecs(filename, c_contiguous=True, batch_size=100000, num_workers=100):
    print(f"Reading from {filename} using {num_workers} workers.")
    
    # 首先确定文件基本信息
    with open(filename, 'rb') as f:
        # 读取第一个向量的维度
        header = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert header > 0
        dim = header
        
        # 计算文件中的向量数量
        filesize = os.path.getsize(filename)
        num_vectors = filesize // (4 + dim)
        
    print(f"Total vectors: {num_vectors}, Dimension: {dim}")
    
    # 创建结果数组 - 使用 uint8 类型
    result = np.zeros((num_vectors, dim), dtype=np.uint8)
    
    # 划分批处理任务
    tasks = []
    for start_idx in range(0, num_vectors, batch_size):
        count = min(batch_size, num_vectors - start_idx)
        tasks.append((start_idx, count, filename, dim))
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_batch_bvecs, task): task for task in tasks}
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            start_idx, batch_result = future.result()
            count = batch_result.shape[0]
            result[start_idx:start_idx+count] = batch_result
            
            completed += count
            if completed % 1000000 == 0 or completed >= num_vectors:
                print(f"Processed {completed}/{num_vectors} vectors")
    
    if c_contiguous and not result.flags.c_contiguous:
        result = np.ascontiguousarray(result)
    
    return result


def read_fvecs(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def _spacev1b_sorted_vector_parts(vectors_dir):
    pattern = re.compile(r"^vectors_(\d+)\.bin$")
    part_files = []
    for name in os.listdir(vectors_dir):
        match = pattern.match(name)
        if match:
            part_files.append((int(match.group(1)), os.path.join(vectors_dir, name)))
    part_files.sort(key=lambda x: x[0])
    if not part_files:
        raise FileNotFoundError(f"No vector shard found under {vectors_dir}")
    return [path for _, path in part_files]


def read_spacev1b_query(query_path):
    with open(query_path, "rb") as f:
        q_count = struct.unpack("i", f.read(4))[0]
        q_dim = struct.unpack("i", f.read(4))[0]
        query = np.frombuffer(f.read(q_count * q_dim), dtype=np.int8).reshape((q_count, q_dim))
    return query


def read_spacev1b_truth(truth_path):
    with open(truth_path, "rb") as f:
        q_count = struct.unpack("i", f.read(4))[0]
        topk = struct.unpack("i", f.read(4))[0]
        truth_ids = np.frombuffer(f.read(q_count * topk * 4), dtype=np.int32).reshape((q_count, topk))
        truth_dist = np.frombuffer(f.read(q_count * topk * 4), dtype=np.float32).reshape((q_count, topk))
    return truth_ids, truth_dist


def read_spacev1b_vectors(vectors_dir, max_vectors=None, chunk_bytes=1048576):
    """Read spacev1B sharded vectors.

    Each shard file (except the first) contains raw int8 bytes with no header.
    Shard boundaries do NOT align to vector boundaries (file size % dim != 0),
    so we stream all shards into a single byte buffer and reshape at the end.
    """
    part_files = _spacev1b_sorted_vector_parts(vectors_dir)
    with open(part_files[0], "rb") as f:
        vec_count = struct.unpack("i", f.read(4))[0]
        vec_dim = struct.unpack("i", f.read(4))[0]

    target_count = vec_count if max_vectors is None else min(max_vectors, vec_count)
    total_bytes = target_count * vec_dim

    print(f"Reading spacev1B vectors: target {target_count} vectors, {vec_dim} dims, {total_bytes} bytes total")

    buf = bytearray(total_bytes)
    buf_offset = 0

    for idx, part_path in enumerate(part_files):
        if buf_offset >= total_bytes:
            break
        with open(part_path, "rb") as f:
            if idx == 0:
                f.seek(8)
            while buf_offset < total_bytes:
                want = min(chunk_bytes, total_bytes - buf_offset)
                chunk = f.read(want)
                if not chunk:
                    break
                buf[buf_offset:buf_offset + len(chunk)] = chunk
                buf_offset += len(chunk)
        print(f"  Part {idx + 1}/{len(part_files)} done, loaded {buf_offset}/{total_bytes} bytes")

    if buf_offset != total_bytes:
        raise IOError(f"Expected {total_bytes} bytes ({target_count} vectors), but got {buf_offset} bytes")

    return np.frombuffer(buf, dtype=np.int8).reshape(target_count, vec_dim).copy()
