# 导入必要的库
import numpy as np
from time import time
import os
import pickle
import gc
import symphonyqg
import pandas as pd
from utils.io import (
    fvecs_read,
    ivecs_read,
    read_fvecs,
    read_ivecs,
    read_bvecs2,
    read_spacev1b_vectors,
    read_spacev1b_query,
    read_spacev1b_truth,
)
from utils.preprocess import normalize
from utils.beam_size import beam_size_gen
from utils.memory import get_memory_usage
from settings import ROUND, TOPK, datasets, degrees
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# 计算欧氏距离的平方
def sqr_dist(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)

# 查找最近的nprobe个质心
def find_nearest_centroids(query, centroids, nprobe):
    """
    找到查询向量最近的nprobe个质心
    
    参数:
        query: 查询向量，形状为(D,)
        centroids: 质心数组，形状为(C, D)
        nprobe: 要返回的最近质心数量
    
    返回:
        list of (distance, centroid_id) 元组，按距离排序
    """
    C = centroids.shape[0]
    centroid_dist = []
    
    # 计算查询向量到每个质心的距离
    for i in range(C):
        dist = sqr_dist(query, centroids[i])
        centroid_dist.append((dist, i))
    
    # 部分排序，只找出最近的nprobe个
    centroid_dist.sort(key=lambda x: x[0])
    return centroid_dist[:nprobe]

# 加载映射数据的函数
def load_mapping(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def search_single_cluster(args):
    """
    在单个簇中进行搜索的函数
    
    参数:
        args: 包含搜索参数的元组 (centroid_id, query_vec, ef, post_ef, k, cluster_to_original, indices, base)
    
    返回:
        (centroid_id, local_results, original_indices) 或 None
    """
    centroid_id, query_vec, ef, post_ef, k, index, ep_dist = args
    
    # 处理簇大小小于2的情况
    if len(cluster_to_original[centroid_id]) < 2:
        return None
    # if len(cluster_to_original[centroid_id]) < 2:
    #     original_indices = cluster_to_original[centroid_id]
    #     dist_to_q = sqr_dist(query_vec, base[original_indices[0]])
    #     return (centroid_id, None, original_indices, [(dist_to_q, original_indices[0])])
    
    # 获取该簇的索引
    index = indices[centroid_id]
    index.set_ef(ef)
    index.set_post_ef(post_ef)
    index.enable_profiling(True)
    
    # 在该簇中搜索
    search_start = time()
    local_results = index.search(query_vec, post_ef, ep_dist)
    search_end = time()
    search_time = search_end - search_start
    
    return (centroid_id, local_results, None, search_time)

# 修改后的搜索逻辑，使用IVF结构
def ivf_search(query_vec, indices, centroids, cluster_to_original, ef, k, nprobe=10, post_ef=100, max_workers=None, enable_timing=False):
    """
    使用IVF结构进行并行搜索
    
    参数:
        query_vec: 查询向量
        indices: 每个簇的索引字典 {cluster_id: index}
        centroids: 质心数组
        cluster_to_original: 簇内索引到原始索引的映射
        ef: 搜索参数
        k: 返回的最近邻数量
        nprobe: 要搜索的簇数量
        post_ef: post处理的ef参数
        max_workers: 最大并行工作线程数，None表示使用默认值
    
    返回:
        前k个最近邻的原始索引列表
    """
    timing_info = {}
    
    # 1. 找到最近的nprobe个质心
    if enable_timing:
        centroid_start = time()
    
    nearest_centroids = find_nearest_centroids(query_vec, centroids, nprobe)
    
    if enable_timing:
        centroid_end = time()
        timing_info['centroid_selection'] = centroid_end - centroid_start
    
    #准备并行搜索的参数
    if enable_timing:
        prep_start = time()
    
    search_args = []
    for dist, centroid_id in nearest_centroids:
        index = indices[centroid_id]
        entry_point = index.get_entry_point()
        original_indices = cluster_to_original[centroid_id]
        ep_dist = sqr_dist(query_vec, base[original_indices[entry_point]])
        args = (centroid_id, query_vec, ef, post_ef, k, indices, ep_dist)
        search_args.append(args)
    
    if enable_timing:
        prep_end = time()
        timing_info['preparation'] = prep_end - prep_start
    
    all_results = []
    cluster_search_times = {}
    
    # 并行搜索各个簇
    # if enable_timing:
    #     parallel_start = time()
        
    # for dist, centroid_id in nearest_centroids:
    #     if len(cluster_to_original[centroid_id]) < 1:
    #         continue
    #     if len(cluster_to_original[centroid_id]) < 2:
    #         original_indices = cluster_to_original[centroid_id]
    #         dist_to_q = sqr_dist(query_vec, base[original_indices[0]])
    #         all_results.append((dist_to_q, original_indices[0]))
    #     # 获取该簇的索引
    #     if centroid_id in indices:
    #         index = indices[centroid_id]
    #         index.set_ef(ef)
    #         index.set_post_ef(post_ef)
    #         index.enable_profiling(True)

    #         # 在该簇中搜索
    #         # print(f"Searching in cluster {centroid_id} with ef={ef} and k={k}")
    #         local_results = index.search(query_vec, k)
    #         # print(f"Cluster {centroid_id} time breakdown:")
    #         index.report_timings()
            
    #         # 将局部索引转换为原始索引
    #         original_indices = cluster_to_original[centroid_id]
    #         global_results = [original_indices[idx] for idx in local_results]
    #         # gt_res = [932085,934876,561813,708177,706771,695756,435345,701258,455537,872728]
    #         # 存储结果和对应的距离
    #         for idx in global_results:
    #             # 计算到查询向量的实际距离
    #             dist_to_query = sqr_dist(query_vec, base[idx])
    #             all_results.append((dist_to_query, idx))
    
    with ThreadPoolExecutor(max_workers=nprobe) as executor:
        # 提交所有搜索任务
        future_to_centroid = {
            executor.submit(search_single_cluster, args): args[0] 
            for args in search_args
        }
        
        if enable_timing:
            parallel_start = time()
        # 收集结果
        for future in as_completed(future_to_centroid):
            centroid_id = future_to_centroid[future]
            try:
                result = future.result()
                if result is not None:
                    centroid_id, local_results, direct_results, search_time = result
                    cluster_search_times[centroid_id] = search_time
                    
                    if direct_results is not None:
                        # 处理簇大小小于2的情况
                        all_results.extend(direct_results)
                    else:
                        original_indices = cluster_to_original[centroid_id]
                        # 处理正常搜索结果，过滤无效结果 (UINT32_MAX = 4294967295)
                        valid_results = [idx for idx in local_results if idx < len(original_indices)]
                        global_results = [original_indices[idx] for idx in valid_results]
                        
                        # 存储结果和对应的距离
                        for idx in global_results:
                            # 计算到查询向量的实际距离
                            dist_to_query = sqr_dist(query_vec, base[idx])
                            all_results.append((dist_to_query, idx))
                            
            except Exception as exc:
                print(f'Cluster {centroid_id} generated an exception: {exc}')
    
    if enable_timing:
        parallel_end = time()
        timing_info['parallel_search'] = parallel_end - parallel_start
        timing_info['cluster_search_times'] = cluster_search_times
    
    # 合并并排序所有结果
    if enable_timing:
        merge_start = time()
    
    all_results.sort(key=lambda x: x[0])
    final_results = [idx for _, idx in all_results[:k]]
    
    if enable_timing:
        merge_end = time()
        timing_info['merge_sort'] = merge_end - merge_start
        timing_info['total_clusters_searched'] = len(cluster_search_times)
        timing_info['avg_cluster_search_time'] = np.mean(list(cluster_search_times.values())) if cluster_search_times else 0
        timing_info['max_cluster_search_time'] = max(cluster_search_times.values()) if cluster_search_times else 0
        timing_info['min_cluster_search_time'] = min(cluster_search_times.values()) if cluster_search_times else 0
    
    # 返回前k个结果
    if enable_timing:
        return final_results, timing_info
    else:
        return final_results

def print_timing_breakdown(timing_info, query_id=None):
    """打印时间breakdown信息"""
    if query_id is not None:
        print(f"\n=== Query {query_id} Timing Breakdown ===")
    else:
        print(f"\n=== IVF Search Timing Breakdown ===")
    
    total_time = sum([v for k, v in timing_info.items() if k != 'cluster_search_times' and k != 'total_clusters_searched'])
    
    print(f"Centroid Selection: {timing_info.get('centroid_selection', 0)*1000:.2f} ms ({timing_info.get('centroid_selection', 0)/total_time*100:.1f}%)")
    print(f"Preparation:       {timing_info.get('preparation', 0)*1000:.2f} ms ({timing_info.get('preparation', 0)/total_time*100:.1f}%)")
    print(f"Parallel Search:   {timing_info.get('parallel_search', 0)*1000:.2f} ms ({timing_info.get('parallel_search', 0)/total_time*100:.1f}%)")
    print(f"Merge & Sort:      {timing_info.get('merge_sort', 0)*1000:.2f} ms ({timing_info.get('merge_sort', 0)/total_time*100:.1f}%)")
    print(f"Total Time:        {total_time*1000:.2f} ms")
    print(f"Clusters Searched: {timing_info.get('total_clusters_searched', 0)}")
    print(f"Avg Cluster Time:  {timing_info.get('avg_cluster_search_time', 0)*1000:.2f} ms")
    print(f"Max Cluster Time:  {timing_info.get('max_cluster_search_time', 0)*1000:.2f} ms")
    print(f"Min Cluster Time:  {timing_info.get('min_cluster_search_time', 0)*1000:.2f} ms")
    
    # 显示每个簇的搜索时间
    cluster_times = timing_info.get('cluster_search_times', {})
    if cluster_times:
        print("\nPer-cluster search times:")
        sorted_clusters = sorted(cluster_times.items(), key=lambda x: x[1], reverse=True)
        for cluster_id, search_time in sorted_clusters[:5]:  # 只显示最慢的5个
            print(f"  Cluster {cluster_id}: {search_time*1000:.2f} ms")

def find_EFS_for_ivf(indices, query, centroids, cluster_to_original, NQ, nprobe):
    """为IVF搜索找到合适的EF值"""
    print("Finding suitable EF values for IVF search...")
    EFS = []
    W = beam_size_gen(TOPK)  # beam size generator
    prev_recall = 0
    while True:
        EF = next(W)
        EFS.append(EF)
        total_time = 0
        results = []
        
        for i in range(NQ):
            t1 = time()
            pred = ivf_search(
                query[i], indices, centroids, 
                cluster_to_original, EF, TOPK, nprobe
            )
            t2 = time()
            results.append(pred)
            total_time += t2 - t1
            print(f"Query {i+1}/{NQ} completed in {t2 - t1:.4f} seconds")
        
        total_num = NQ * TOPK
        total_correct = 0
        for i in range(NQ):
            res_set = set(results[i])
            for j in range(TOPK):
                if gt[i][j] in res_set:
                    total_correct += 1
        
        qps = NQ / total_time
        recall = total_correct / total_num * 100
        print(f"Recall: {recall:.2f}%, prev_recall: {prev_recall:.2f}, QPS: {qps:.2f}, EF: {EF}, nprobe: {nprobe}")
        
        if recall > 99.8 or (recall - prev_recall) < 0.05 or qps < 10:
            break
        prev_recall = recall
    return EFS

# 在主程序中添加IVF搜索逻辑
if __name__ == "__main__":
    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]

        if DATASET == "SPACE1B":
            base = read_spacev1b_vectors(f"./data/{DATASET}/vectors.bin").astype(np.float32)
            query = read_spacev1b_query(f"./data/{DATASET}/query.bin").astype(np.float32)
            gt, _ = read_spacev1b_truth(f"./data/{DATASET}/truth.bin")
            C = 8192
            centroids_path = os.path.join(f"./data/{DATASET}/SPACE1B_centroid_{C}.fvecs")
            cluster_id_path = os.path.join(f"./data/{DATASET}/SPACE1B_cluster_id_{C}.ivecs")
        else:
            base = read_bvecs2(f"./data/{DATASET}/{DATASET}_base.bvecs").astype(np.float32)
            query = read_bvecs2(f"./data/{DATASET}/{DATASET}_query.bvecs").astype(np.float32)
            gt = ivecs_read(f"./data/{DATASET}/{DATASET}_idx_100M.ivecs")
            # base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
            # query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
            # gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")
            C = 4096
            centroids_path = os.path.join(f"./data/{DATASET}/{DATASET}_centroid_{C}.fvecs")
            cluster_id_path = os.path.join(f"./data/{DATASET}/{DATASET}_cluster_id_{C}.ivecs")

        centroids = read_fvecs(centroids_path)
        cluster_id = read_ivecs(cluster_id_path)
        
        # 加载簇到原始索引的映射
        cluster_to_original = load_mapping(f"./data/{DATASET}/IVF-reduceMem/cluster_to_original_{C}.pkl")
        
        NQ, D = query.shape
        NQ = 100
        N = base.shape[0]
        # gt_res = [932085,934876,561813,708177,706771,695756,435345,701258,455537,872728]

        if DISTANCE == "angular":
            query = normalize(query)
        
        for DEGREE in degrees[DATASET]:
            m1 = get_memory_usage()
            
            # 加载各个簇的索引
            indices = {}
            for i in range(C):
                # 仅加载非空簇的索引
                if i in cluster_to_original and len(cluster_to_original[i]) > 1:
                    index_path = f"./data/{DATASET}/IVF-reduceMem/symphonyqg_{DEGREE}_cluster_{i}.index"
                    if os.path.exists(index_path):
                        # 获取该簇中的数据点数量
                        n_points = len(cluster_to_original[i])
                        
                        # 创建并加载索引
                        indices[i] = symphonyqg.Index(
                            index_type="QG",
                            metric="L2",
                            num_elements=n_points,
                            dimension=D,
                            degree_bound=DEGREE,
                        )
                        indices[i].load(index_path)
                        indices[i].set_cluster(i)
            
            m2 = get_memory_usage()
            MEMORY = m2 - m1
            print(f"Memory usage for DEGREE {DEGREE}: {MEMORY} bytes")
            # break
            
                
            
            # # 找出每个gt_res 所属的cluster 和 cluster中的索引
            # cluster_to_gt = {}
            # for idx in gt_res:
            #     c_id = cluster_id[idx]
            #     original_indices = cluster_to_original[c_id[0]]
            #     local_idx = np.where(original_indices == idx)[0][0]
            #     print(f"Index {idx} belongs to cluster {c_id} with local index {local_idx}")
            #     # 输出gt_res 的 vector 信息
            #     print(f"GT vector: {base[idx]}")
            
            # 尝试不同的nprobe值
            NPROBES = [32]
            
            for NPROBE in NPROBES:
                print(f"Testing with NPROBE={NPROBE}")
                
                # 找到合适的EF值
                # EFS = find_EFS_for_ivf(indices, query, centroids, cluster_to_original, NQ, NPROBE)
                # EFS = [10,15,20,25,30,35,40]
                EFS = [30]
                # POST_EFS = [10,15,20,25,30,35,40]
                POST_EFS = [30]
                
                ALL_QPS = []
                ALL_RECALL = []
                ALL_EF = []
                ALL_POST_EF = []
                
                for _ in range(ROUND):
                    QPS = []
                    RECALL = []
                    A_EF = []
                    A_POST_EF = []
                    
                    for EF in EFS:
                        for POST_EF in POST_EFS:
                            total_time = 0
                            results = []
                            
                            for i in range(NQ):
                                # 输出 query[i] 的信息
                                # print(f"Query vector: {query[i]}")
                                
                                t1 = time()
                                # 使用IVF搜索
                                if i < 5:  # 只为前5个查询显示详细时间信息
                                    pred, timing_info = ivf_search(
                                        query[i], indices, centroids, 
                                        cluster_to_original, EF, TOPK, NPROBE, POST_EF, 
                                        enable_timing=True
                                    )
                                    print_timing_breakdown(timing_info, query_id=i+1)
                                else:
                                    pred = ivf_search(
                                        query[i], indices, centroids, 
                                        cluster_to_original, EF, TOPK, NPROBE, POST_EF, 
                                        enable_timing=False
                                    )
                                t2 = time()
                                results.append(pred)
                                total_time += t2 - t1
                            
                            # 计算召回率
                            total_num = NQ * TOPK
                            total_correct = 0
                            for i in range(NQ):
                                res_set = set(results[i])
                                for j in range(TOPK):
                                    if gt[i][j] in res_set:
                                        total_correct += 1
                                        # print(f"Query {i+1}, Top {j+1} correct: {gt[i][j]} in results")
                            
                            qps = NQ / total_time
                            recall = total_correct / total_num * 100
                            QPS.append(qps)
                            RECALL.append(recall)
                            A_EF.append(EF)
                            A_POST_EF.append(POST_EF)
                            print(f"EF: {EF}, POST EF: {POST_EF}, NPROBE: {NPROBE}, QPS: {qps:.2f}, Recall: {recall:.2f}%")
                    
                    ALL_QPS.append(QPS)
                    ALL_RECALL.append(RECALL)
                    ALL_EF.append(A_EF)
                    ALL_POST_EF.append(A_POST_EF)
                
                # 计算平均值
                ALL_QPS = np.average(np.array(ALL_QPS), axis=0)
                ALL_RECALL = np.average(np.array(ALL_RECALL), axis=0)
                ALL_EF = np.average(np.array(ALL_EF), axis=0)
                ALL_POST_EF = np.average(np.array(ALL_POST_EF), axis=0)
                
                # 保存结果
                df = pd.DataFrame(
                    {
                        "QPS": ALL_QPS,
                        "Recall": ALL_RECALL,
                        "EFS": ALL_EF,
                        "POST_EFS": ALL_POST_EF,
                        "Method": f"symphonyqg{DEGREE}_ivf{NPROBE}",
                        "Memory": MEMORY,
                        "NPROBE": NPROBE,
                    }
                )
                
                res_dir = f"./results/{DATASET}/symphonyqg_ivf_reduceMem/"
                try:
                    os.makedirs(res_dir, exist_ok=True)
                except OSError as e:
                    print(e)
                df.to_csv(res_dir + f"symphonyqg{DEGREE}_ivf{NPROBE}_{TOPK}.csv", index=False)
            
            # 清理内存
            del indices
            gc.collect()
